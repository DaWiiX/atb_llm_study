"""
Performance benchmark: ATB engine vs torch_npu transformers.

Tests multiple image resolutions with "Describe the image." prompt.

Timing strategy:
    E2E total:      wall-clock, NO synchronize (measure real throughput)
    Per-stage:      synchronize at boundaries ONLY (isolate each stage's NPU time)

Stages:
    1. Preprocess      — CPU image resize + patch extraction
    2. Vision PosEmb   — fast_pos_embed_interpolate + rot_pos_emb (CPU)
    3. Vision Model    — patch_embed + 24 blocks + merger (NPU)
    4. Text Embed      — token embedding + vision injection (CPU)
    5. Position IDs    — get_rope_index + MRoPE rotary_emb (CPU)
    6. Text Model      — 28 decoder layers + norm (NPU)

Usage:
    python benchmark.py  [--iter N]  [--warmup M]
"""
import sys, os, time, argparse
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, '/mnt/workspace/gitCode/atb_python_model')
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')

import torch
import torch_npu
import safetensors.torch
import numpy as np
from PIL import Image

from atb_python_model.utils import set_atb_buffer_size
from atb_python_model.engine import Qwen3VLEngine
set_atb_buffer_size(20000 * 1024 * 1024)  # 20GB for large sequences


# ═════════════════════════════════════════════════════════════════════
# Timing utilities
# ═════════════════════════════════════════════════════════════════════

def sync():
    """Synchronize NPU stream — only call at stage boundaries for per-stage timing."""
    torch.npu.synchronize()

def now():
    """High-resolution wall-clock time (no sync)."""
    return time.perf_counter()


def stats(arr):
    """Mean ± std in milliseconds for a list of second values."""
    a = np.array(arr) * 1000
    return f"{a.mean():.1f} ± {a.std():.1f}"


# ═════════════════════════════════════════════════════════════════════
# Setup: load models, build inputs
# ═════════════════════════════════════════════════════════════════════

def setup(model_dir: str):
    rp = model_dir

    # ── ATB engine ─────────────────────────────────────────────────
    atb = Qwen3VLEngine(rp)

    # ── torch_npu reference ─────────────────────────────────────────
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    cfg = Qwen3VLConfig.from_pretrained(rp, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    ref = Qwen3VLModel(cfg).eval().half().npu()
    sd = safetensors.torch.load_file(f"{rp}/model.safetensors", device="cpu")
    sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
    m, u = ref.load_state_dict(sd, strict=False)
    assert len(m) == 0 and len(u) == 0, f"weights mismatch: {len(m)} missing, {len(u)} unexpected"

    return atb, ref


def make_test_image(w: int, h: int):
    """Create a random RGB image of given dimensions and return preprocessed inputs."""
    img = Image.new('RGB', (w, h), color='blue')
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)

    from atb_python_model.preprocess import preprocess_image
    pv_raw, grid_thw = preprocess_image(img_arr)

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(rp)
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Describe the image.'}]}]
    tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                      return_tensors='pt', add_generation_prompt=True)
    return img_arr, pv_raw, grid_thw, tf_in


# ═════════════════════════════════════════════════════════════════════
# ATB: staged benchmark
# ═════════════════════════════════════════════════════════════════════

def benchmark_atb_staged(atb, input_ids, img_arr, pv_raw, grid_thw,
                          n_warmup=5, n_iter=10):
    """Measure per-stage timing for ATB engine (WITH sync at boundaries)."""
    from atb_python_model.engine_utils import (
        TextRotaryEmbedding, get_rope_index, get_embed_weight,
        fast_pos_embed_interpolate, compute_rot_pos_emb,
        VisionRotaryEmbedding, get_vision_pos_embed,
        get_vision_block_weights, get_patch_embed_weights, get_merger_weights,
    )
    from atb_python_model.preprocess import preprocess_image
    from atb_python_model.text_model import (
        make_causal_mask, run_text_layer, run_text_norm,
    )
    from atb_python_model.vision_model import run_first_layer, run_block, run_merger

    S = input_ids.shape[1]
    atb._ensure_text_graph(S)

    stages = ["preprocess", "vision_pos", "vision_model",
              "text_embed", "position_ids", "text_model"]
    results = {s: [] for s in stages}

    for _ in range(n_warmup):
        atb.forward(input_ids, pv_raw, grid_thw)

    for _ in range(n_iter):
        t = {}

        # ── Preprocess (CPU) ────────────────────────────────────────
        sync(); t0 = now()
        pv, gth = preprocess_image(img_arr)
        sync(); t1 = now(); t["preprocess"] = t1 - t0

        # ── Vision pos_embed (CPU) ──────────────────────────────────
        pos = fast_pos_embed_interpolate(gth, atb.v_pos_embed, atb.num_grid, atb.merge_size)
        rope = compute_rot_pos_emb(gth, atb.vis_rotary, atb.merge_size)
        rope = rope.reshape(pv.shape[0], -1)
        emb = torch.cat((rope, rope), dim=-1)
        cos_v, sin_v = emb.cos(), emb.sin()
        sync(); t2 = now(); t["vision_pos"] = t2 - t1

        # ── Vision model (NPU) ──────────────────────────────────────
        vis, _ = atb._run_vision(pv, gth)
        sync(); t3 = now(); t["vision_model"] = t3 - t2

        # ── Text embed + vision injection (CPU) ─────────────────────
        ie = torch.nn.functional.embedding(input_ids, atb.embed_w).float()
        vis_mask = input_ids.squeeze(0) == atb.img_tok
        ie[0, vis_mask, :] = vis.cpu().float()
        sync(); t4 = now(); t["text_embed"] = t4 - t3

        # ── Position IDs + RoPE (CPU) ───────────────────────────────
        pid, _ = get_rope_index(input_ids, gth, None, None,
                                 image_token_id=atb.img_tok,
                                 spatial_merge_size=atb.spatial_merge)
        sync(); t5 = now(); t["position_ids"] = t5 - t4

        # ── Text model (NPU) ────────────────────────────────────────
        atb._ensure_text_graph(S)
        cos_f, sin_f = atb.text_rope(pid)
        cos_npu = cos_f.reshape(-1, atb.hd_t).half().npu()
        sin_npu = sin_f.reshape(-1, atb.hd_t).half().npu()
        ht = ie.half().npu()
        ntoken = 1 * S
        for li in range(atb.n_layer):
            inputs = [ht]
            inputs.extend(atb.t_layer_weights[li])
            inputs.extend([cos_npu, sin_npu])
            inputs.append(atb._cached_mask)
            inputs.append(torch.tensor([ntoken], dtype=torch.int32))
            ht = atb.g_t_layer.forward(inputs)[0]
        sync(); t6 = now(); t["text_model"] = t6 - t5

        for s in stages:
            results[s].append(t[s])

    return results


def benchmark_atb_e2e(atb, input_ids, pv_raw, grid_thw, n_warmup=5, n_iter=10):
    """Measure E2E wall-clock for ATB engine (NO sync)."""
    for _ in range(n_warmup):
        atb.forward(input_ids, pv_raw, grid_thw)

    results = []
    for _ in range(n_iter):
        t0 = now()
        atb.forward(input_ids, pv_raw, grid_thw)
        t1 = now()
        results.append(t1 - t0)
    return results


# ═════════════════════════════════════════════════════════════════════
# torch_npu: staged benchmark
# ═════════════════════════════════════════════════════════════════════

def benchmark_torch_staged(ref, atb, input_ids, img_arr, pv_raw, grid_thw, tf_in,
                            n_warmup=5, n_iter=10):
    """Measure per-stage timing for torch_npu (WITH sync at boundaries)."""
    from atb_python_model.engine_utils import (
        fast_pos_embed_interpolate, compute_rot_pos_emb, get_rope_index,
    )
    from atb_python_model.preprocess import preprocess_image

    im_tok = ref.config.image_token_id
    S = input_ids.shape[1]
    cm = torch.zeros(S, S)
    cm[torch.triu(torch.ones(S, S), diagonal=1).bool()] = -65504.0

    stages = ["preprocess", "vision_pos", "vision_model",
              "text_embed", "position_ids", "text_model"]
    results = {s: [] for s in stages}

    for _ in range(n_warmup):
        with torch.no_grad():
            ref(input_ids=tf_in['input_ids'].npu(),
                pixel_values=tf_in['pixel_values'].half().npu(),
                image_grid_thw=tf_in['image_grid_thw'].npu())

    for _ in range(n_iter):
        t = {}

        # ── Preprocess (CPU) ────────────────────────────────────────
        sync(); t0 = now()
        pv, gth = preprocess_image(img_arr)
        sync(); t1 = now(); t["preprocess"] = t1 - t0

        # ── Vision pos_embed (CPU) ──────────────────────────────────
        pos = fast_pos_embed_interpolate(gth, atb.v_pos_embed, atb.num_grid, atb.merge_size)
        rope = compute_rot_pos_emb(gth, atb.vis_rotary, atb.merge_size)
        rope = rope.reshape(pv.shape[0], -1)
        emb = torch.cat((rope, rope), dim=-1)
        cos_v, sin_v = emb.cos(), emb.sin()
        sync(); t2 = now(); t["vision_pos"] = t2 - t1

        # ── Vision model (torch_npu) ────────────────────────────────
        with torch.no_grad():
            vis_out = ref.visual(pv.half().npu(), grid_thw=gth.npu())[0]
        sync(); t3 = now(); t["vision_model"] = t3 - t2

        # ── Text embed + vision injection (CPU/NPU) ─────────────────
        ie = ref.language_model.embed_tokens(input_ids.npu()).float()
        v_mask = input_ids.squeeze(0).npu() == im_tok
        ie[0, v_mask, :] = vis_out.float()
        sync(); t4 = now(); t["text_embed"] = t4 - t3

        # ── Position IDs + RoPE (CPU) ───────────────────────────────
        pid, _ = get_rope_index(input_ids, gth, None, None,
                                 image_token_id=im_tok,
                                 spatial_merge_size=ref.config.vision_config.spatial_merge_size)
        sync(); t5 = now(); t["position_ids"] = t5 - t4

        # ── Text model (torch_npu) ──────────────────────────────────
        with torch.no_grad():
            ht = ref.language_model(
                inputs_embeds=ie.half().npu(),
                attention_mask=None,
                position_ids=pid.npu(),
                use_cache=False,
            ).last_hidden_state.float()
        sync(); t6 = now(); t["text_model"] = t6 - t5

        for s in stages:
            results[s].append(t[s])

    return results


def benchmark_torch_e2e(ref, tf_in, n_warmup=5, n_iter=10):
    """Measure E2E wall-clock for torch_npu (NO sync)."""
    for _ in range(n_warmup):
        with torch.no_grad():
            ref(input_ids=tf_in['input_ids'].npu(),
                pixel_values=tf_in['pixel_values'].half().npu(),
                image_grid_thw=tf_in['image_grid_thw'].npu())

    results = []
    for _ in range(n_iter):
        t0 = now()
        with torch.no_grad():
            ref(input_ids=tf_in['input_ids'].npu(),
                pixel_values=tf_in['pixel_values'].half().npu(),
                image_grid_thw=tf_in['image_grid_thw'].npu())
        t1 = now()
        results.append(t1 - t0)
    return results


# ═════════════════════════════════════════════════════════════════════
# Report
# ═════════════════════════════════════════════════════════════════════

def print_report(resolution, S, atb_e2e, torch_e2e, atb_stages, torch_stages):
    labels = {
        "preprocess":    "Preprocess       ",
        "vision_pos":    "Vision PosEmb    ",
        "vision_model":  "Vision Model     ",
        "text_embed":    "Text Embed+Inj   ",
        "position_ids":  "Position IDs     ",
        "text_model":    "Text Model       ",
    }
    stage_order = ["preprocess", "vision_pos", "vision_model",
                   "text_embed", "position_ids", "text_model"]

    print(f"\n{'─'*80}")
    print(f" {resolution}  (S={S})")
    print(f"{'─'*80}")
    print(f"{'Stage':<20} {'ATB staged (ms)':>18} {'torch_npu staged (ms)':>22} {'Ratio':>8}")
    print(f"{'─'*20} {'─'*18} {'─'*22} {'─'*8}")

    atb_sum = 0.0
    torch_sum = 0.0
    for s in stage_order:
        atb_val = np.mean(atb_stages[s]) * 1000
        torch_val = np.mean(torch_stages[s]) * 1000 if torch_stages else 0.0
        ratio = f"{atb_val/torch_val:.2f}x" if torch_val > 0 else "-"
        print(f"{labels[s]:<20} {atb_val:>16.2f}  {torch_val:>20.2f}  {ratio:>6}")
        atb_sum += atb_val
        torch_sum += torch_val

    print(f"{'─'*20} {'─'*18} {'─'*22} {'─'*8}")
    print(f"{'Staged sum':<20} {atb_sum:>16.2f}  {torch_sum:>20.2f}")

    # E2E (no sync for both)
    atb_ms = np.mean(atb_e2e) * 1000
    torch_ms = np.mean(torch_e2e) * 1000
    atb_std = np.std(atb_e2e) * 1000
    torch_std = np.std(torch_e2e) * 1000
    print(f"\n{'E2E (no sync)':<20} {atb_ms:>8.1f} ± {atb_std:<5.1f} ms  "
          f"{'torch_npu':>8} {torch_ms:>8.1f} ± {torch_std:<5.1f} ms  "
          f"({atb_ms/torch_ms:.2f}x)")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rp = '/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B'
    print("Loading ATB engine...")
    atb = Qwen3VLEngine(rp)

    resolutions = [
        (416, 672),
        (720, 1280),
        (1080, 1920),
        (2560, 1440),
    ]

    for w, h in resolutions:
        print(f"\n\nPreparing {w}x{h}...")
        img_arr, pv_raw, grid_thw, tf_in = make_test_image(w, h)
        S = tf_in['input_ids'].shape[1]
        atb._ensure_text_graph(S)
        n_vis_tokens = int(torch.prod(grid_thw, dim=1).sum()) // (atb.merge_size ** 2)
        print(f"  Image: {w}x{h} → grid={grid_thw.tolist()} → {n_vis_tokens} vision tokens, S={S}")

        # ATB staged (with sync at boundaries)
        atb_stages = benchmark_atb_staged(atb, tf_in['input_ids'], img_arr,
                                          pv_raw, grid_thw, n_warmup=3, n_iter=5)
        # ATB E2E (no sync — real throughput)
        atb_e2e = benchmark_atb_e2e(atb, tf_in['input_ids'], pv_raw, grid_thw,
                                     n_warmup=3, n_iter=5)

        # Print ATB-only report
        labels = {
            "preprocess":    "Preprocess       ",
            "vision_pos":    "Vision PosEmb    ",
            "vision_model":  "Vision Model     ",
            "text_embed":    "Text Embed+Inj   ",
            "position_ids":  "Position IDs     ",
            "text_model":    "Text Model       ",
        }
        stage_order = ["preprocess", "vision_pos", "vision_model",
                       "text_embed", "position_ids", "text_model"]

        print(f"\n{'─'*80}")
        print(f" {w}x{h}  (S={S})")
        print(f"{'─'*80}")
        print(f"{'Stage':<20} {'Time (ms)':>12} {'%':>8}")
        print(f"{'─'*20} {'─'*12} {'─'*8}")

        total_staged = 0.0
        for s in stage_order:
            ms = np.mean(atb_stages[s]) * 1000
            total_staged += ms
        for s in stage_order:
            ms = np.mean(atb_stages[s]) * 1000
            pct = ms / total_staged * 100 if total_staged > 0 else 0
            print(f"{labels[s]:<20} {ms:>10.2f}  {pct:>6.1f}%")
        print(f"{'─'*20} {'─'*12} {'─'*8}")
        print(f"{'Staged sum':<20} {total_staged:>10.2f}")

        e2e_ms = np.mean(atb_e2e) * 1000
        e2e_std = np.std(atb_e2e) * 1000
        print(f"\n{'E2E (no sync)':<20} {e2e_ms:>8.1f} ± {e2e_std:<5.1f} ms")
        print(f"  (sync overhead: total_staged - e2e = {total_staged - e2e_ms:.1f} ms)")

    print(f"\n{'='*80}")
    print(" Done.")
