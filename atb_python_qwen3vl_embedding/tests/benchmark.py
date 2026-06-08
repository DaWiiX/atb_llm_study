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

For fair C++ vs Python comparison, first run C++ with --mode compare to
save pixel_values, then run Python with --load-pixel-values::

    ./build/benchmark --mode compare --iter 3 --warmup 2
    python tests/benchmark.py --mode all --iter 3 --warmup 2 --load-pixel-values
"""
# ── Buffer size MUST be set before any engine/graph import ──────────
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
set_atb_buffer_size(10 * 1024 * 1024 * 1024)  # 10 GB, matching C++

# ── Standard imports ───────────────────────────────────────────────
import argparse
import struct
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
    compute_posemb_indices, compute_rope_indices, get_rope_index,
)
from atb_python_qwen3vl_embedding.vision_model import (
    run_block_npu, run_first_layer_npu, run_merger_npu,
)
from atb_python_qwen3vl_embedding.vision_pos_embed import run_posemb_npu
from atb_python_qwen3vl_embedding.utils import make_seqlen_tensor, to_npu_half


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

def load_cpp_pixel_values(w: int, h: int, engine) -> tuple:
    """Load C++-preprocessed pixel_values from file and compute matching grid_thw.

    Reads the .bin file saved by C++ benchmark --mode compare
    (SavePixelValues format: [int32_t num_values] [uint16_t * num_values]).
    Returns (pixel_values_float32_tensor, grid_thw_tensor) in the same
    format as engine.preprocess_image(), so they slot directly into
    engine.forward().

    The grid_thw is recomputed via smart_resize using the engine's own
    config so it matches the dimensions C++ PreprocessImage produced.
    """
    from atb_python_qwen3vl_embedding.preprocess import smart_resize

    path = f"/tmp/cpp_pv_{w}x{h}.bin"
    with open(path, "rb") as fh:
        (num_values,) = struct.unpack("<i", fh.read(4))
        raw = fh.read(num_values * 2)
    pv_fp16 = np.frombuffer(raw, dtype=np.float16)
    pv_fp32 = torch.from_numpy(pv_fp16.astype(np.float32))

    # Compute grid_thw using the same parameters as C++ PreprocessImage
    factor = int(engine.patch_size * engine.merge_size)
    new_h, new_w = smart_resize(h, w, factor=factor,
                                min_pixels=int(engine.pp_min_px),
                                max_pixels=int(engine.pp_max_px))
    grid_h = new_h // engine.patch_size
    grid_w = new_w // engine.patch_size
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

    return pv_fp32, grid_thw


def make_inputs(engine, processor, w: int, h: int, load_pv: bool = False):
    """Construct one benchmark input bundle for a (width, height) image.

    The image goes through:
      * ``engine.preprocess_image`` → ``pv_raw`` / ``grid_thw`` (ATB native)
      * ``processor.apply_chat_template`` → ``input_ids`` (tokenizer-only)
        and ``tf_pixel_values`` / ``tf_grid_thw`` (used by TF reference only).

    When load_pv=True, pixel_values are loaded from C++-saved .bin files
    (saved by ``benchmark --mode compare``) instead of generated from a
    solid-blue image, for fair cross-framework comparison.
    """
    # PIL image is always needed for the processor's chat template
    img = Image.new('RGB', (w, h), color='blue')
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)

    if load_pv:
        pv_raw, grid_thw = load_cpp_pixel_values(w, h, engine)
    else:
        pv_raw, grid_thw = engine.preprocess_image(img_arr)

    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Describe the image.'}]}]
    tf_in = processor.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True)

    return {
        'w': w,
        'h': h,
        'img_arr': img_arr,
        'pv_raw': pv_raw,
        'grid_thw': grid_thw,
        'input_ids': tf_in['input_ids'],
        'attention_mask': tf_in.get('attention_mask',
                                    torch.ones_like(tf_in['input_ids'])).npu(),
        'tf_pixel_values': tf_in.get('pixel_values').npu(),
        'tf_grid_thw': tf_in.get('image_grid_thw').npu(),
    }


def save_pooler_bin(atb_output, attention_mask, engine, save_path):
    """Save pooler output as .bin in C++ test_accuracy.cpp format.

    Format: [8 bytes: int64_t dim] [dim * 4 bytes: float32 data]
    """
    pooled = engine.embedding_pooling(atb_output, attention_mask)
    pooled_np = pooled.cpu().float().numpy()
    dim = pooled_np.shape[1]
    data = pooled_np.squeeze(0)
    with open(save_path, 'wb') as f:
        f.write(np.array([dim], dtype=np.int64).tobytes())
        f.write(data.astype(np.float32).tobytes())
    print(f"  Saved pooler output → {save_path}  (dim={dim})")


def make_image_only_inputs(engine, processor, w, h, load_pv: bool = False):
    """Construct an IMAGE_ONLY input bundle: input_ids are all image_token_id.

    Input_ids are all image_token_id; S = merger_tokens (pure image, no extra
    text tokens), matching the vision model output shape.

    merger_tokens = total_patches / (spatial_merge_size ** 2)

    When load_pv=True, pixel_values are loaded from C++ .bin files.
    A dummy img_arr is still created so staged-benchmark's preprocess
    timing step has a valid image (though the actual forward pass uses
    the loaded C++ pixel_values).
    """
    # Always create img_arr — staged benchmark needs it for preprocess timing
    img = Image.new('RGB', (w, h), color='blue')
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)

    if load_pv:
        pv_raw, grid_thw = load_cpp_pixel_values(w, h, engine)
    else:
        pv_raw, grid_thw = engine.preprocess_image(img_arr)

    total_patches = int(torch.prod(grid_thw, dim=1).sum())
    merger_tokens = total_patches // (engine.merge_size ** 2)
    num_tokens = merger_tokens

    input_ids = torch.full((1, num_tokens), engine.img_tok, dtype=torch.long)

    return {
        'w': w,
        'h': h,
        'img_arr': img_arr,
        'pv_raw': pv_raw,
        'grid_thw': grid_thw,
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids).npu(),
        'tf_pixel_values': None,  # no TF comparison for IO mode
        'tf_grid_thw': None,
    }


# ═══════════════════════════════════════════════════════════════════
# ATB benchmarks
# ═══════════════════════════════════════════════════════════════════

STAGES = ['preprocess', 'vision_pos', 'vision_model',
          'text_embed', 'position_ids', 'text_model']


def _run_vision_stages(engine, pv, gth):
    """Reproduce engine._run_vision but split into (pos+rope) and (NPU graph).

    Mirrors engine.py — uses ATB VisPosEmb graph for pos_embed + RoPE.
    """
    # ── Stage 2: vision pos_embed + rope (CPU indices + NPU graph) ──
    idx_wt = compute_posemb_indices(gth, engine.num_grid, engine.merge_size)
    rope_idx = compute_rope_indices(gth, engine.vis_rotary, engine.merge_size)
    freq_npu = to_npu_half(rope_idx['freq_table'])

    pos_npu, cos_npu, sin_npu = run_posemb_npu(
        engine.g_v_posemb, engine.v_pe_w_table,
        idx_wt, rope_idx, freq_npu)

    # ── Stage 3: vision model on NPU ───────────────────────────────
    pv_npu = to_npu_half(pv.reshape(-1) if pv.ndim == 2 else pv)
    npatches = idx_wt['idx00'].shape[0]
    seqlen_v = make_seqlen_tensor(npatches)

    h = run_first_layer_npu(engine.g_v_first, pv_npu,
                            engine.v_pe_w, engine.v_pe_b,
                            pos_npu, cos_npu, sin_npu,
                            engine.v_block_weights[0], seqlen_v)
    ds_feats = []
    for li in range(1, engine.v_depth):
        h = run_block_npu(engine.g_v_block, h,
                          engine.v_block_weights[li], cos_npu, sin_npu, seqlen_v)
        if li in engine.ds_indexes:
            ds_idx = engine.ds_indexes.index(li)
            ds_feats.append(
                run_merger_npu(engine.g_v_ds, h, engine.v_ds_w[ds_idx]))
    vis = run_merger_npu(engine.g_v_merger, h, engine.v_merger_w)
    return (idx_wt, rope_idx), (vis, ds_feats)


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
        idx_wt = compute_posemb_indices(gth, engine.num_grid, engine.merge_size)
        rope_idx = compute_rope_indices(gth, engine.vis_rotary, engine.merge_size)
        freq_npu = to_npu_half(rope_idx['freq_table'])
        # Sync: CPU→NPU transfers for idx_wt/rope_idx/freq must complete
        # before ATB posemb graph reads them (ATB may use a separate stream).
        sync()
        pos_npu, cos_npu, sin_npu = run_posemb_npu(
            engine.g_v_posemb, engine.v_pe_w_table,
            idx_wt, rope_idx, freq_npu)
        sync(); t2 = now(); t['vision_pos'] = t2 - t2_start

        pv_npu = to_npu_half(pv.reshape(-1) if pv.ndim == 2 else pv)
        npatches = idx_wt['idx00'].shape[0]
        seqlen_v = make_seqlen_tensor(npatches)
        # Sync: pv_npu H2D (or NPU dtype conversion) must complete before
        # the ATB first-layer graph reads it.
        sync()

        h = run_first_layer_npu(engine.g_v_first, pv_npu,
                                engine.v_pe_w, engine.v_pe_b,
                                pos_npu, cos_npu, sin_npu,
                                engine.v_block_weights[0], seqlen_v)
        ds_feats = []
        for li in range(1, engine.v_depth):
            sync()  # sync between ATB graph calls (matches engine._run_vision)
            h = run_block_npu(engine.g_v_block, h,
                              engine.v_block_weights[li], cos_npu, sin_npu, seqlen_v)
            if li in engine.ds_indexes:
                ds_idx = engine.ds_indexes.index(li)
                ds_feats.append(
                    run_merger_npu(engine.g_v_ds, h, engine.v_ds_w[ds_idx]))
        sync()
        vis = run_merger_npu(engine.g_v_merger, h, engine.v_merger_w)
        sync(); t3 = now(); t['vision_model'] = t3 - t2

        # ── Stage 4: text embed + vision injection (CPU + NPU copy) ─
        ie = F.embedding(input_ids, engine.embed_w).half().npu()
        vis_mask = input_ids.squeeze(0) == engine.img_tok
        # Sync: ie (H2D), vis (ATB graph output), and vis_mask.npu() (H2D)
        # must all be ready before the NPU scatter operation.
        sync()
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


def parse_seq_lengths(s: str):
    """Parse comma-separated sequence lengths, e.g. '100,512,1024,2048,4096'."""
    return [int(tok.strip()) for tok in s.split(',') if tok.strip()]


def save_token_ids(path, ids):
    """Save token ID list to binary file.

    Format: [int32 count] [int64 * count]
    Consumed by C++ benchmark --mode compare via LoadTokenIds().
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(ids)))
        for tid in ids:
            f.write(struct.pack('<q', int(tid)))


def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--no-ref', dest='ref', action='store_false',
                   help='Skip transformers reference benchmark')
    p.add_argument('--iter', type=int, default=5,
                   help='Number of timed iterations (default 5)')
    p.add_argument('--warmup', type=int, default=3,
                   help='Number of warmup iterations (default 3)')
    p.add_argument('--resolutions', type=str,
                   default='416x672,720x1280,1080x1920,1440x2560',
                   help='Comma-separated WxH list')
    p.add_argument('--seq', type=str, default='100,512,1024,2048,4096',
                   help='Comma-separated sequence lengths for TEXT_ONLY mode')
    p.add_argument('--mode', type=str, default='mm',
                   choices=['text', 'io', 'mm', 'all'],
                   help='Benchmark mode: text (TEXT_ONLY), io (IMAGE_ONLY), '
                        'mm (IMAGE_AND_TEXT, default), all (run all three)')
    p.add_argument('--load-pixel-values', action='store_true',
                   help='Load C++-preprocessed pixel_values from '
                        '/tmp/cpp_pv_{W}x{H}.bin (saved by '
                        'benchmark --mode compare) instead of generating '
                        'solid-blue images. Ensures identical input for '
                        'fair cross-framework comparison.')
    p.add_argument('--save-tokens', action='store_true',
                   help='Save tokenized input_ids to /tmp/tokens_*.bin '
                        'so C++ benchmark --mode compare can load identical '
                        'token sequences.')
    p.add_argument('--model-dir', default=QWEN3VL_EMB_MODEL_DIR,
                   help='Override QWEN3VL_EMB_MODEL_DIR from .env')
    p.set_defaults(ref=True)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    modes_to_run = ['text', 'io', 'mm'] if args.mode == 'all' else [args.mode]
    resolutions = parse_resolutions(args.resolutions)
    seq_lengths = parse_seq_lengths(args.seq)
    print(f"Model dir : {args.model_dir}")
    print(f"Iter/warm : {args.iter} / {args.warmup}")
    print(f"Mode(s)   : {modes_to_run}")
    print(f"Resolutions: {resolutions}")
    print(f"Text seqs : {seq_lengths}")
    print(f"TF ref    : {'on' if args.ref else 'off'}")
    print(f"Load C++ PV: {'on' if args.load_pixel_values else 'off'}")
    print(f"Save tokens: {'on' if args.save_tokens else 'off'}")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # ── Load engine (shared across modes) ──────────────────────────
    print("\n[ATB] Loading Qwen3VLEngine ...")
    engine = Qwen3VLEngine(args.model_dir)
    print(f"[ATB] {engine.n_layer} text layers, {engine.v_depth} vision blocks")

    all_results = {}

    # ═════════════════════════════════════════════════════════════════
    # Mode: TEXT_ONLY
    # ═════════════════════════════════════════════════════════════════
    if 'text' in modes_to_run:
        print(f"\n{'─' * 70}")
        print("  Mode: TEXT_ONLY  (no image)")
        print(f"{'─' * 70}")

        text = "Describe the image."
        base_tokens = processor.tokenizer(text,
                                          add_special_tokens=False)['input_ids']
        print(f"  Base text: \"{text}\"  |tokens|={len(base_tokens)}")
        print(f"  Sequence lengths: {seq_lengths}")

        if args.save_tokens:
            save_token_ids("/tmp/tokens_text_only.bin", base_tokens)
            print(f"  Saved base tokens → /tmp/tokens_text_only.bin")

        torch.npu.synchronize()

        for seq in seq_lengths:
            # Build input_ids by repeating base tokens to reach target length
            if len(base_tokens) >= seq:
                ids = base_tokens[:seq]
            else:
                repeats = (seq + len(base_tokens) - 1) // len(base_tokens)
                ids = (base_tokens * repeats)[:seq]
            text_input_ids = torch.tensor([ids], dtype=torch.long)
            text_attn_mask = torch.ones_like(text_input_ids)
            S = text_input_ids.shape[1]

            print(f"\n  ── S={S} ──")

            # E2E timing (no staged breakdown — no vision stages)
            engine._ensure_text_graph(S)
            sync()
            for _ in range(args.warmup):
                engine.forward(text_input_ids, None, None)
            sync()

            e2e_times = []
            for _ in range(args.iter):
                sync(); t_start = now()
                engine.forward(text_input_ids, None, None)
                sync()
                e2e_times.append(now() - t_start)

            e2e_arr = ms(e2e_times)
            print(f"  Text E2E: {e2e_arr.mean():.2f} ± {e2e_arr.std():.2f} ms")

            # Save pooler output as .bin
            atb_output = engine.forward(text_input_ids, None, None)
            save_pooler_bin(atb_output, text_attn_mask, engine,
                            f"/tmp/py_text_only_{seq}.bin")

    # ═════════════════════════════════════════════════════════════════
    # Mode: IMAGE_ONLY
    # ═════════════════════════════════════════════════════════════════
    if 'io' in modes_to_run:
        print(f"\n{'=' * 70}")
        print("  Mode: IMAGE_ONLY")
        print(f"{'=' * 70}")

        for (w, h) in resolutions:
            print(f"\n[IO] Preparing {w}x{h} ...")
            inputs = make_image_only_inputs(engine, processor, w, h,
                                              load_pv=args.load_pixel_values)
            S = inputs['input_ids'].shape[1]
            n_vis = int(torch.prod(inputs['grid_thw'], dim=1).sum()) \
                // (engine.merge_size ** 2)
            print(f"[IO] grid={inputs['grid_thw'].tolist()}  S={S}  vis_tok={n_vis}")

            torch.npu.synchronize()

            stages = benchmark_atb_staged(engine, inputs, args.warmup, args.iter)
            atb_e2e = benchmark_atb_e2e(engine, inputs, args.warmup, args.iter)

            # Save pooler output as .bin
            atb_output = engine.forward(inputs['input_ids'],
                                        inputs['pv_raw'], inputs['grid_thw'])
            save_pooler_bin(atb_output, inputs['attention_mask'].cpu(), engine,
                            f"/tmp/py_io_{w}x{h}.bin")

            print_atb_report((w, h), S, n_vis, stages, atb_e2e)

    # ═════════════════════════════════════════════════════════════════
    # Mode: IMAGE_AND_TEXT  (multimodal — default, keeps existing behavior)
    # ═════════════════════════════════════════════════════════════════
    if 'mm' in modes_to_run:
        print(f"\n{'=' * 70}")
        print("  Mode: IMAGE_AND_TEXT")
        print(f"{'=' * 70}")

        for (w, h) in resolutions:
            print(f"\n[ATB] Preparing {w}x{h} ...")
            inputs = make_inputs(engine, processor, w, h,
                                   load_pv=args.load_pixel_values)
            S = inputs['input_ids'].shape[1]
            n_vis = int(torch.prod(inputs['grid_thw'], dim=1).sum()) \
                // (engine.merge_size ** 2)
            print(f"[ATB] grid={inputs['grid_thw'].tolist()}  S={S}  vis_tok={n_vis}")

            if args.save_tokens:
                mm_tokens = inputs['input_ids'][0].tolist()
                save_token_ids(f"/tmp/tokens_mm_{w}x{h}.bin", mm_tokens)
                print(f"  Saved MM tokens → /tmp/tokens_mm_{w}x{h}.bin")

            torch.npu.synchronize()

            stages = benchmark_atb_staged(engine, inputs, args.warmup, args.iter)
            atb_e2e = benchmark_atb_e2e(engine, inputs, args.warmup, args.iter)

            # Save one ATB output for accuracy comparison against TF
            atb_output = engine.forward(inputs['input_ids'],
                                        inputs['pv_raw'], inputs['grid_thw'])
            inputs['atb_output'] = atb_output

            # Save pooler output as .bin
            save_pooler_bin(atb_output, inputs['attention_mask'].cpu(), engine,
                            f"/tmp/py_mm_{w}x{h}.bin")

            print_atb_report((w, h), S, n_vis, stages, atb_e2e)

            all_results[(w, h)] = {
                'S': S, 'n_vis': n_vis, 'inputs': inputs,
                'stages': stages, 'atb_e2e': atb_e2e, 'tf_e2e': None,
                'accuracy': None,
            }

    del engine
    torch.npu.empty_cache()

    # ── Phase 2: TF reference (optional, only for mm mode) ─────────
    if args.ref and all_results:
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

    if all_results:
        print_atb_vs_tf_table(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
