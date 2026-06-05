"""Performance benchmark for Qwen3VLEngine.

Measures three things per resolution:
  * **ATB stage-by-stage timing** (with sync at boundaries) — isolates each
    pipeline phase: preprocess, vision pos_embed, vision model, text embed +
    inject, position ids, text model.
  * **ATB end-to-end timing** (no sync inside the call) — real throughput.
  * **TF reference end-to-end timing** (optional, ``--no-ref`` skips) —
    apples-to-apples comparison against the transformers implementation.

ATB and TF are loaded sequentially (TF only after ATB is released and the NPU
cache is cleared) to avoid OOM on devices with limited HBM (e.g. 910b).

Stage breakdown reuses ``Qwen3VLEngine`` internal methods directly so the
benchmark cannot drift away from the actual ``engine.forward()`` path.

Usage::

    python tests/benchmark.py
    python tests/benchmark.py --no-ref
    python tests/benchmark.py --iter 10 --warmup 5 --resolutions 416x672,720x1280
"""
# ── Buffer size MUST be set before any engine/graph import ──────────
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size

set_atb_buffer_size(20 * 1024 * 1024 * 1024)  # 20 GB — supports 2560x1440

# ── Standard imports ───────────────────────────────────────────────
import argparse
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR
from atb_python_qwen3vl_embedding.engine_utils import (
    compute_rot_pos_emb, fast_pos_embed_interpolate, get_rope_index,
)
from atb_python_qwen3vl_embedding.vision_model import (
    run_block_npu, run_first_layer_npu, run_merger_npu,
)


# ═══════════════════════════════════════════════════════════════════
# Timing utilities
# ═══════════════════════════════════════════════════════════════════

def sync():
    torch.npu.synchronize()


def now():
    return time.perf_counter()


def ms(seconds_list):
    """Convert seconds → milliseconds numpy array."""
    return np.asarray(seconds_list) * 1000.0


# ═══════════════════════════════════════════════════════════════════
# Input construction (shared between ATB and TF phases)
# ═══════════════════════════════════════════════════════════════════

def make_inputs(engine, processor, w: int, h: int):
    """Construct one benchmark input bundle for a (width, height) image.

    The image goes through:
      * ``engine.preprocess_image`` → ``pv_raw`` / ``grid_thw`` (ATB native)
      * ``processor.apply_chat_template`` → ``input_ids`` (tokenizer-only)
        and ``tf_pixel_values`` / ``tf_grid_thw`` (used by TF reference only).
    """
    img = Image.new('RGB', (w, h), color='blue')
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)

    pv_raw, grid_thw = engine.preprocess_image(img_arr)

    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Describe the image.'}]}]
    tf_in = processor.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True)

    return {
        'w': w, 'h': h, 'img_arr': img_arr,
        'pv_raw': pv_raw, 'grid_thw': grid_thw,
        'input_ids': tf_in['input_ids'],
        'attention_mask': tf_in.get('attention_mask',
                                    torch.ones_like(tf_in['input_ids'])),
        'tf_pixel_values': tf_in.get('pixel_values'),
        'tf_grid_thw': tf_in.get('image_grid_thw'),
    }


# ═══════════════════════════════════════════════════════════════════
# ATB benchmarks
# ═══════════════════════════════════════════════════════════════════

STAGES = ['preprocess', 'vision_pos', 'vision_model',
          'text_embed', 'position_ids', 'text_model']


def _run_vision_stages(engine, pv, gth):
    """Reproduce engine._run_vision but split into (pos+rope) and (NPU graph).

    Mirrors engine.py:167-186 — kept here so the benchmark can sync between the
    two sub-stages without modifying engine.py.
    """
    # ── Stage 2: vision pos_embed + rope (CPU) ─────────────────────
    pos = fast_pos_embed_interpolate(
        gth, engine.v_pos_embed, engine.num_grid, engine.merge_size)
    rope = compute_rot_pos_emb(gth, engine.vis_rotary, engine.merge_size)
    rope = rope.reshape(pv.shape[0], -1)
    emb = torch.cat((rope, rope), dim=-1)
    cos_v, sin_v = emb.cos(), emb.sin()

    # ── Stage 3: vision model on NPU ───────────────────────────────
    h = run_first_layer_npu(engine.g_v_first, pv,
                            engine.v_pe_w, engine.v_pe_b,
                            pos, cos_v, sin_v,
                            engine.v_block_weights[0])
    ds_feats = []
    for li in range(1, engine.v_depth):
        h = run_block_npu(engine.g_v_block, h,
                          engine.v_block_weights[li], cos_v, sin_v)
        if li in engine.ds_indexes:
            ds_idx = engine.ds_indexes.index(li)
            ds_feats.append(
                run_merger_npu(engine.g_v_ds, h, engine.v_ds_w[ds_idx]))
    vis = run_merger_npu(engine.g_v_merger, h, engine.v_merger_w)
    return (pos, cos_v, sin_v), (vis, ds_feats)


def benchmark_atb_staged(engine, inputs, n_warmup, n_iter):
    """Per-stage timing for ATB pipeline (sync between stages)."""
    input_ids = inputs['input_ids']
    img_arr = inputs['img_arr']
    pv_raw = inputs['pv_raw']
    grid_thw = inputs['grid_thw']
    S = input_ids.shape[1]
    engine._ensure_text_graph(S)  # prebuild before timing

    # Warmup with full forward to populate all NPU caches/graphs.
    for _ in range(n_warmup):
        engine.forward(input_ids, pv_raw, grid_thw)
    sync()

    results = {s: [] for s in STAGES}

    for _ in range(n_iter):
        t = {}

        # ── Stage 1: preprocess (CPU) ───────────────────────────────
        sync(); t0 = now()
        pv, gth = engine.preprocess_image(img_arr)
        sync(); t1 = now(); t['preprocess'] = t1 - t0

        # ── Stages 2 & 3: vision pos_embed + NPU model ──────────────
        sync(); t2_start = now()
        pos = fast_pos_embed_interpolate(
            gth, engine.v_pos_embed, engine.num_grid, engine.merge_size)
        rope = compute_rot_pos_emb(gth, engine.vis_rotary, engine.merge_size)
        rope = rope.reshape(pv.shape[0], -1)
        emb = torch.cat((rope, rope), dim=-1)
        cos_v, sin_v = emb.cos(), emb.sin()
        sync(); t2 = now(); t['vision_pos'] = t2 - t2_start

        h = run_first_layer_npu(engine.g_v_first, pv,
                                engine.v_pe_w, engine.v_pe_b,
                                pos, cos_v, sin_v,
                                engine.v_block_weights[0])
        ds_feats = []
        for li in range(1, engine.v_depth):
            h = run_block_npu(engine.g_v_block, h,
                              engine.v_block_weights[li], cos_v, sin_v)
            if li in engine.ds_indexes:
                ds_idx = engine.ds_indexes.index(li)
                ds_feats.append(
                    run_merger_npu(engine.g_v_ds, h, engine.v_ds_w[ds_idx]))
        vis = run_merger_npu(engine.g_v_merger, h, engine.v_merger_w)
        sync(); t3 = now(); t['vision_model'] = t3 - t2

        # ── Stage 4: text embed + vision injection (CPU + NPU copy) ─
        ie = F.embedding(input_ids, engine.embed_w).half().npu()
        vis_mask = input_ids.squeeze(0) == engine.img_tok
        ie[0, vis_mask.npu(), :] = vis
        sync(); t4 = now(); t['text_embed'] = t4 - t3

        # ── Stage 5: position ids (CPU) ─────────────────────────────
        pid, _ = get_rope_index(
            input_ids, grid_thw, None, None,
            image_token_id=engine.img_tok,
            spatial_merge_size=engine.spatial_merge)
        sync(); t5 = now(); t['position_ids'] = t5 - t4

        # ── Stage 6: text model on NPU (reuse engine._run_text) ─────
        _ = engine._run_text(ie, pid, vis_mask,
                             ds_feats if ds_feats else None)
        sync(); t6 = now(); t['text_model'] = t6 - t5

        for s in STAGES:
            results[s].append(t[s])

    return results


def benchmark_atb_e2e(engine, inputs, n_warmup, n_iter):
    """End-to-end wall-clock timing (no sync inside engine.forward)."""
    input_ids = inputs['input_ids']
    pv_raw = inputs['pv_raw']
    grid_thw = inputs['grid_thw']

    for _ in range(n_warmup):
        engine.forward(input_ids, pv_raw, grid_thw)
    sync()

    results = []
    for _ in range(n_iter):
        sync(); t0 = now()
        engine.forward(input_ids, pv_raw, grid_thw)
        sync()
        results.append(now() - t0)
    return results


# ═══════════════════════════════════════════════════════════════════
# Transformers reference benchmark
# ═══════════════════════════════════════════════════════════════════

def load_tf_ref(model_dir: str):
    import safetensors.torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"

    ref = Qwen3VLModel(cfg).eval().half().npu()
    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors",
                                     device="cpu")
    sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    assert not missing and not unexpected
    return ref


def benchmark_tf_e2e(ref, inputs, n_warmup, n_iter):
    input_ids = inputs['input_ids'].npu()
    attn_mask = inputs['attention_mask'].npu()
    pv = inputs['tf_pixel_values'].half().npu()
    gth = inputs['tf_grid_thw'].npu()

    def _call():
        with torch.no_grad():
            ref(use_cache=False, input_ids=input_ids,
                attention_mask=attn_mask,
                pixel_values=pv, image_grid_thw=gth)

    for _ in range(n_warmup):
        _call()
    sync()

    results = []
    for _ in range(n_iter):
        sync(); t0 = now()
        _call()
        sync()
        results.append(now() - t0)

    # Accuracy check: compare last_hidden_state vs ATB output
    with torch.no_grad():
        tf_out = ref(use_cache=False, input_ids=input_ids,
                     attention_mask=attn_mask,
                     pixel_values=pv, image_grid_thw=gth
                     ).last_hidden_state.cpu().float()
    accuracy = None
    if inputs.get('atb_output') is not None:
        a = inputs['atb_output'].flatten()
        t = tf_out.flatten()
        accuracy = F.cosine_similarity(a.unsqueeze(0), t.unsqueeze(0)).item()
    return results, accuracy

    for _ in range(n_warmup):
        _call()
    sync()

    results = []
    for _ in range(n_iter):
        sync(); t0 = now()
        _call()
        sync()
        results.append(now() - t0)
    return results


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

STAGE_LABELS = {
    'preprocess':   'Preprocess',
    'vision_pos':   'Vision PosEmb',
    'vision_model': 'Vision Model',
    'text_embed':   'Text Embed+Inj',
    'position_ids': 'Position IDs',
    'text_model':   'Text Model',
}


def print_atb_report(resolution, S, n_vis_tokens, stages, atb_e2e):
    w, h = resolution
    atb_arr = ms(atb_e2e)
    print(f"\n{'─' * 70}")
    print(f"  {w}x{h}   S={S}   vision_tokens={n_vis_tokens}")
    print(f"{'─' * 70}")
    print(f"{'Stage':<18} {'Mean (ms)':>12} {'% of staged':>14}")
    total = sum(np.mean(stages[s]) * 1000 for s in STAGES)
    for s in STAGES:
        m = np.mean(stages[s]) * 1000
        pct = (m / total * 100) if total > 0 else 0.0
        print(f"{STAGE_LABELS[s]:<18} {m:>12.2f} {pct:>13.1f}%")
    print(f"{'─' * 18} {'─' * 12} {'─' * 14}")
    print(f"{'Staged sum':<18} {total:>12.2f}")
    print(f"{'E2E (no sync)':<18} {atb_arr.mean():>10.2f} ± {atb_arr.std():.2f}")


def print_atb_vs_tf_table(all_results):
    """Final ATB-vs-TF comparison table across all resolutions."""
    print(f"\n{'=' * 90}")
    print(f"{'Resolution':<14}{'S':>6}{'VisTok':>9}"
          f"{'ATB E2E (ms)':>18}{'TF E2E (ms)':>18}{'Ratio':>8}{'Cosine':>9}")
    print(f"{'─' * 90}")
    for (w, h), r in all_results.items():
        atb_arr = ms(r['atb_e2e'])
        tf_arr = ms(r['tf_e2e']) if r.get('tf_e2e') else None
        acc = r.get('accuracy')
        atb_str = f"{atb_arr.mean():>8.1f} ± {atb_arr.std():<5.1f}"
        if tf_arr is not None:
            tf_str = f"{tf_arr.mean():>8.1f} ± {tf_arr.std():<5.1f}"
            ratio = f"{atb_arr.mean() / tf_arr.mean():.2f}x"
        else:
            tf_str = "(skipped)"
            ratio = "—"
        acc_str = f"{acc:.4f}" if acc is not None else "—"
        print(f"{w}x{h:<8}{r['S']:>6}{r['n_vis']:>9}"
              f"{atb_str:>18}{tf_str:>18}{ratio:>8}{acc_str:>9}")
    print(f"{'=' * 90}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_resolutions(s: str):
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        w, h = tok.lower().split('x')
        out.append((int(w), int(h)))
    return out


def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--no-ref', dest='ref', action='store_false',
                   help='Skip transformers reference benchmark')
    p.add_argument('--iter', type=int, default=5,
                   help='Number of timed iterations (default 5)')
    p.add_argument('--warmup', type=int, default=3,
                   help='Number of warmup iterations (default 3)')
    p.add_argument('--resolutions', type=str,
                   default='416x672,720x1280,1080x1920,2560x1440',
                   help='Comma-separated WxH list')
    p.add_argument('--model-dir', default=QWEN3VL_EMB_MODEL_DIR,
                   help='Override QWEN3VL_EMB_MODEL_DIR from .env')
    p.set_defaults(ref=True)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    resolutions = parse_resolutions(args.resolutions)
    print(f"Model dir : {args.model_dir}")
    print(f"Iter/warm : {args.iter} / {args.warmup}")
    print(f"Resolutions: {resolutions}")
    print(f"TF ref    : {'on' if args.ref else 'off'}")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # ── Phase 1: ATB ─────────────────────────────────────────────
    print("\n[ATB] Loading Qwen3VLEngine ...")
    engine = Qwen3VLEngine(args.model_dir)
    print(f"[ATB] {engine.n_layer} text layers, {engine.v_depth} vision blocks")

    all_results = {}
    for (w, h) in resolutions:
        print(f"\n[ATB] Preparing {w}x{h} ...")
        inputs = make_inputs(engine, processor, w, h)
        S = inputs['input_ids'].shape[1]
        n_vis = int(torch.prod(inputs['grid_thw'], dim=1).sum()) \
            // (engine.merge_size ** 2)
        print(f"[ATB] grid={inputs['grid_thw'].tolist()}  S={S}  vis_tok={n_vis}")

        stages = benchmark_atb_staged(engine, inputs, args.warmup, args.iter)
        atb_e2e = benchmark_atb_e2e(engine, inputs, args.warmup, args.iter)

        # Save one ATB output for accuracy comparison against TF
        atb_output = engine.forward(inputs['input_ids'],
                                    inputs['pv_raw'], inputs['grid_thw'])
        inputs['atb_output'] = atb_output

        print_atb_report((w, h), S, n_vis, stages, atb_e2e)

        all_results[(w, h)] = {
            'S': S, 'n_vis': n_vis, 'inputs': inputs,
            'stages': stages, 'atb_e2e': atb_e2e, 'tf_e2e': None,
            'accuracy': None,
        }

    del engine
    torch.npu.empty_cache()

    # ── Phase 2: TF reference (optional) ─────────────────────────
    if args.ref:
        print("\n[TF] Loading transformers reference ...")
        ref = load_tf_ref(args.model_dir)
        print("[TF] Loaded")
        for (w, h), r in all_results.items():
            print(f"\n[TF] Benchmarking {w}x{h} ...")
            r['tf_e2e'], r['accuracy'] = benchmark_tf_e2e(
                ref, r['inputs'], args.warmup, args.iter)
            tf_arr = ms(r['tf_e2e'])
            acc_str = f"  cosine={r['accuracy']:.6f}" if r['accuracy'] is not None else ""
            print(f"[TF]  E2E: {tf_arr.mean():.2f} ± {tf_arr.std():.2f} ms{acc_str}")
        del ref
        torch.npu.empty_cache()

    print_atb_vs_tf_table(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
