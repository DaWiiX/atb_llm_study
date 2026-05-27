"""
ATB vs torch_npu E2E benchmark.

Runs ATB and torch_npu benchmarks separately to avoid NPU memory contention.

Usage:
    python tests/benchmark_simple.py
"""
import sys, os, time
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')

import torch
import torch_npu  # noqa: required for .npu() ops
import safetensors.torch
import numpy as np
from PIL import Image

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.preprocess import preprocess_image

set_atb_buffer_size(20000 * 1024 * 1024)


def sync():
    torch.npu.synchronize()


def now():
    return time.perf_counter()


def make_test_inputs(img_w, img_h, engine, ref_config):
    img = Image.new('RGB', (img_w, img_h), color='blue')
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    pv_raw, grid_thw = preprocess_image(img_arr)

    vs_id = ref_config.vision_start_token_id
    vi_id = ref_config.image_token_id
    ve_id = ref_config.vision_end_token_id
    pad_id = ref_config.pad_token_id or 0

    n_img_tok = int(torch.prod(grid_thw, dim=1).sum() // (engine.merge_size ** 2))
    S = max(32, n_img_tok + 20)

    input_ids_list = [pad_id] * 5
    input_ids_list.append(vs_id)
    for _ in range(n_img_tok):
        input_ids_list.append(vi_id)
    input_ids_list.append(ve_id)
    for _ in range(S - len(input_ids_list)):
        input_ids_list.append(pad_id)
    input_ids = torch.tensor([input_ids_list[:S]], dtype=torch.long)
    pv_tf = pv_raw.unsqueeze(0) if pv_raw.dim() == 2 else pv_raw

    return img_arr, pv_raw, grid_thw, input_ids, pv_tf


def benchmark_atb_forward(engine, input_ids, pv_raw, grid_thw, n_warmup=5, n_iter=10):
    for _ in range(n_warmup):
        engine.forward(input_ids, pv_raw, grid_thw)
        sync()

    results = []
    for _ in range(n_iter):
        sync()
        t0 = now()
        engine.forward(input_ids, pv_raw, grid_thw)
        sync()
        results.append(now() - t0)
    return results


def benchmark_torch_forward(ref_model, input_ids, pv_tf, grid_thw, n_warmup=5, n_iter=10):
    for _ in range(n_warmup):
        with torch.no_grad():
            ref_model(
                input_ids=input_ids.npu(),
                pixel_values=pv_tf.half().npu(),
                image_grid_thw=grid_thw.npu(),
            )
        sync()

    results = []
    for _ in range(n_iter):
        sync()
        t0 = now()
        with torch.no_grad():
            ref_model(
                input_ids=input_ids.npu(),
                pixel_values=pv_tf.half().npu(),
                image_grid_thw=grid_thw.npu(),
            )
        sync()
        results.append(now() - t0)
    return results


if __name__ == "__main__":
    rp = '/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B'

    resolutions = [
        (416, 672),
        (720, 1280),
        (1080, 1920),
        (2560, 1440),
    ]

    # Load config once for input preparation
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    cfg = Qwen3VLConfig.from_pretrained(rp, trust_remote_code=True)

    # ── Phase 1: ATB benchmarks (no ref model on NPU) ────────────────
    print("Loading ATB engine...")
    atb = Qwen3VLEngine(rp)
    print(f"ATB engine: {atb.n_layer} text layers, {atb.v_depth} vision blocks")

    atb_results = {}
    for w, h in resolutions:
        print(f"\nATB {w}x{h}...")
        img_arr, pv_raw, grid_thw, input_ids, pv_tf = make_test_inputs(w, h, atb, cfg)
        S = input_ids.shape[1]
        atb._ensure_text_graph(S)
        n_vis = int(torch.prod(grid_thw, dim=1).sum() // (atb.merge_size ** 2))
        print(f"  S={S}, vis_tokens={n_vis}")

        e2e = benchmark_atb_forward(atb, input_ids, pv_raw, grid_thw, n_warmup=3, n_iter=5)
        atb_results[(w, h)] = (S, n_vis, e2e)

    # Free ATB engine before loading ref model
    del atb
    torch.npu.empty_cache()

    # ── Phase 2: torch_npu benchmarks ─────────────────────────────────
    print("\n\nLoading torch_npu reference model...")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    cfg_npu = Qwen3VLConfig.from_pretrained(rp, trust_remote_code=True)
    cfg_npu._attn_implementation = "eager"
    cfg_npu.text_config._attn_implementation = "eager"
    ref = Qwen3VLModel(cfg_npu).eval().half().npu()
    sd = safetensors.torch.load_file(f"{rp}/model.safetensors", device="cpu")
    sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
    m, u = ref.load_state_dict(sd, strict=False)
    assert len(m) == 0 and len(u) == 0, f"weights mismatch: {len(m)} missing, {len(u)} unexpected"
    print(f"torch_npu model loaded OK")

    # Minimal object for make_test_inputs — merge_size=2 is fixed for this model
    class _MergeSize:
        merge_size = 2

    torch_results = {}
    for w, h in resolutions:
        print(f"\ntorch_npu {w}x{h}...")
        img_arr, pv_raw, grid_thw, input_ids, pv_tf = make_test_inputs(w, h, _MergeSize(), cfg_npu)

        e2e = benchmark_torch_forward(ref, input_ids, pv_tf, grid_thw, n_warmup=3, n_iter=5)
        torch_results[(w, h)] = e2e

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Resolution':<16} {'S':<6} {'VisTokens':<12} {'ATB (ms)':<16} {'torch_npu (ms)':<18} {'Ratio':<8}")
    print(f"{'─'*80}")

    for w, h in resolutions:
        S, n_vis, atb_e2e = atb_results[(w, h)]
        torch_e2e = torch_results[(w, h)]

        atb_ms = np.mean(atb_e2e) * 1000
        atb_std = np.std(atb_e2e) * 1000
        torch_ms = np.mean(torch_e2e) * 1000
        torch_std = np.std(torch_e2e) * 1000
        ratio = atb_ms / torch_ms if torch_ms > 0 else 0

        print(f"{w}x{h:<10} {S:<6} {n_vis:<12} {atb_ms:>8.1f} ± {atb_std:<5.1f}   {torch_ms:>10.1f} ± {torch_std:<5.1f}     {ratio:>5.2f}x")

    print(f"\n{'='*80}")
    print(" Done.")
