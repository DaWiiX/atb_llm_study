"""
Vision Pipeline Per-Stage Precision Diagnostic (Python side).

Steps through Qwen3VLEngine._run_vision manually, saving intermediate
outputs at each stage to /tmp/diag_*.bin for comparison with C++.

Binary format: int64 dim + raw fp16 data (matching C++ convention).
"""

import sys
import struct
import numpy as np
import torch

# ── Configuration ────────────────────────────────────────────────
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402
IMG_H = 720
IMG_W = 1280
IMG_C = 3


def save_bin(path: str, tensor_fp16: torch.Tensor):
    """Save fp16 tensor to binary file: int64 total_elements + raw fp16 bytes."""
    flat = tensor_fp16.reshape(-1)
    n = flat.numel()
    arr = flat.cpu().numpy()  # np.float16
    with open(path, "wb") as f:
        f.write(struct.pack("q", n))
        f.write(arr.numpy().tobytes() if hasattr(arr, 'numpy') else arr.tobytes())
    print(f"  Saved {path}: {n} fp16 values, shape={list(tensor_fp16.shape)}")


def save_bin_u16(path: str, data_u16):
    """Save uint16 (fp16 bits) array: int64 total_elements + raw uint16 bytes."""
    flat = np.asarray(data_u16, dtype=np.uint16).reshape(-1)
    n = flat.size
    with open(path, "wb") as f:
        f.write(struct.pack("q", n))
        f.write(flat.tobytes())
    print(f"  Saved {path}: {n} fp16 values")


def save_bin_f32(path: str, data_f32):
    """Save float32 array: int64 total_elements + raw float32 bytes."""
    flat = np.asarray(data_f32, dtype=np.float32).reshape(-1)
    n = flat.size
    with open(path, "wb") as f:
        f.write(struct.pack("q", n))
        f.write(flat.tobytes())
    print(f"  Saved {path}: {n} f32 values")


def create_test_image(channels: int, height: int, width: int) -> torch.Tensor:
    """Create the same test image as C++ side: (C, H, W) uint8."""
    image = torch.zeros(channels, height, width, dtype=torch.uint8)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                value = (h * 255 // height + w * 255 // width + c * 85) % 256
                image[c, h, w] = value
    return image


def main():
    print("=" * 60)
    print("Vision Pipeline Per-Stage Diagnostic (Python)")
    print(f"Image: {IMG_H}x{IMG_W}")
    print("=" * 60)

    # ── Setup engine ────────────────────────────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)

    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    from atb_python_qwen3vl_embedding.preprocess import preprocess_image
    from atb_python_qwen3vl_embedding.engine_utils import (
        compute_posemb_indices, compute_rope_indices,
    )
    from atb_python_qwen3vl_embedding.vision_model import (
        run_first_layer_npu, run_block_npu, run_merger_npu,
    )
    from atb_python_qwen3vl_embedding.vision_pos_embed import run_posemb_npu
    from atb_python_qwen3vl_embedding.utils import to_npu_half, to_cpu_float, make_seqlen_tensor

    print("\nLoading Python ATB engine...")
    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine loaded: {engine.n_layer} text layers, {engine.v_depth} vision blocks")
    print(f"  ds_indexes: {engine.ds_indexes}")
    print(f"  merge_size: {engine.merge_size}")

    # ── Step 1: Preprocess image ────────────────────────────────
    print("\n--- Step 1: Preprocessing ---")
    img = create_test_image(IMG_C, IMG_H, IMG_W)
    pv, grid_thw = preprocess_image(img)
    print(f"  pixel_values: shape={list(pv.shape)}, dtype={pv.dtype}")
    print(f"  grid_thw: {grid_thw}")

    num_patches = pv.shape[0]
    vis_hs = engine.v_cfg["hidden_size"]
    vis_hd = engine.hd_v
    merge_size = engine.merge_size
    num_grid = engine.num_grid
    print(f"  num_patches={num_patches}, vis_hs={vis_hs}, vis_hd={vis_hd}")

    # Save pixel_values as fp16
    save_bin("/tmp/diag_pixel_values.bin", pv.half())

    # ── Step 2: Position embedding + RoPE via ATB graph (same as engine) ──
    print("\n--- Step 2: Position Embedding + RoPE (NPU) ---")
    idx_wt = compute_posemb_indices(grid_thw, num_grid, merge_size)
    rope_idx = compute_rope_indices(grid_thw, engine.vis_rotary, merge_size)
    freq_npu = to_npu_half(rope_idx['freq_table'])

    pos_npu, cos_npu, sin_npu = run_posemb_npu(
        engine.g_v_posemb, engine.v_pe_w_table, idx_wt, rope_idx, freq_npu)

    # Save NPU fp16 outputs
    save_bin("/tmp/diag_pos_embed.bin", pos_npu)
    save_bin("/tmp/diag_rope_cos.bin", cos_npu)
    save_bin("/tmp/diag_rope_sin.bin", sin_npu)

    # ── Step 2b: Compute CPU f32 RoPE for comparison ────────────
    print("\n--- Step 2b: CPU f32 RoPE (simulating C++ path) ---")
    # The C++ path computes RoPE on CPU f32, then converts to fp16
    # Python NPU path: Gather freq_table (fp16) → Concat → Cos/Sin
    # Let's compute the same thing on CPU f32 to see the difference
    from atb_python_qwen3vl_embedding.engine_utils import VisionRotaryEmbedding
    vis_rotary_cpu = VisionRotaryEmbedding(dim=vis_hd // 2)
    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table_cpu = vis_rotary_cpu(max_hw)  # (max_hw, dim) f32

    # Gather by position IDs
    pid_row = rope_idx['pid_row']  # (total_tokens,) int64
    pid_col = rope_idx['pid_col']  # (total_tokens,) int64

    row_freq_cpu = freq_table_cpu[pid_row]  # (total_tokens, dim) f32
    col_freq_cpu = freq_table_cpu[pid_col]  # (total_tokens, dim) f32

    # Concat row and col → (total_tokens, dim*2)
    rope_cpu = torch.cat([row_freq_cpu, col_freq_cpu], dim=1)
    # Duplicate → (total_tokens, dim*4) = (total_tokens, vis_hd)
    emb_cpu = torch.cat([rope_cpu, rope_cpu], dim=1)

    cos_cpu_f32 = emb_cpu.cos()
    sin_cpu_f32 = emb_cpu.sin()

    print(f"  cos_cpu_f32: shape={list(cos_cpu_f32.shape)}")
    print(f"  sin_cpu_f32: shape={list(sin_cpu_f32.shape)}")

    # Save CPU f32 RoPE
    save_bin_f32("/tmp/diag_rope_cos_cpu_f32.bin", cos_cpu_f32.numpy())
    save_bin_f32("/tmp/diag_rope_sin_cpu_f32.bin", sin_cpu_f32.numpy())

    # Also save as fp16 (simulating C++ f32→fp16 conversion)
    cos_cpu_fp16 = cos_cpu_f32.half()
    sin_cpu_fp16 = sin_cpu_f32.half()
    save_bin("/tmp/diag_rope_cos_cpu_fp16.bin", cos_cpu_fp16)
    save_bin("/tmp/diag_rope_sin_cpu_fp16.bin", sin_cpu_fp16)

    # Compare NPU fp16 vs CPU f32→fp16
    cos_npu_cpu = cos_npu.cpu().float()
    sin_npu_cpu = sin_npu.cpu().float()
    cos_cosine = torch.nn.functional.cosine_similarity(
        cos_npu_cpu.reshape(1, -1), cos_cpu_f32.reshape(1, -1)).item()
    sin_cosine = torch.nn.functional.cosine_similarity(
        sin_npu_cpu.reshape(1, -1), sin_cpu_f32.reshape(1, -1)).item()
    print(f"  RoPE cos: NPU-fp16 vs CPU-f32 cosine = {cos_cosine:.6f}")
    print(f"  RoPE sin: NPU-fp16 vs CPU-f32 cosine = {sin_cosine:.6f}")

    # ── Step 3: Prepare NPU inputs ──────────────────────────────
    print("\n--- Step 3: Vision Model Execution ---")
    pv_npu = to_npu_half(pv.reshape(-1) if pv.ndim == 2 else pv)
    seqlen_v = make_seqlen_tensor(num_patches)
    torch.npu.synchronize()

    # ── Step 3a: First layer (patch_embed + pos_embed + block 0) ──
    print("\n--- Step 3a: First Layer ---")
    h = run_first_layer_npu(engine.g_v_first, pv_npu,
                            engine.v_pe_w, engine.v_pe_b,
                            pos_npu, cos_npu, sin_npu,
                            engine.v_block_weights[0], seqlen_v)
    torch.npu.synchronize()
    save_bin("/tmp/diag_first_layer_out.bin", h)
    print(f"  first_layer_out: shape={list(h.shape)}")

    # ── Step 3b: Loop through blocks ────────────────────────────
    ds_feats = []
    for li in range(1, engine.v_depth):
        torch.npu.synchronize()
        h = run_block_npu(engine.g_v_block, h, engine.v_block_weights[li],
                          cos_npu, sin_npu, seqlen_v)

        # Save block 1 output for early-divergence diagnosis
        if li == 1:
            save_bin("/tmp/diag_block_1_out.bin", h)
            print(f"  block 1: shape={list(h.shape)}")

        # Save outputs at deepstack extraction points and last block
        if li in engine.ds_indexes:
            ds_idx = engine.ds_indexes.index(li)
            ds_out = run_merger_npu(engine.g_v_ds, h, engine.v_ds_w[ds_idx])
            ds_feats.append(ds_out)
            save_bin(f"/tmp/diag_block_{li}_out.bin", h)
            save_bin(f"/tmp/diag_deepstack_{li}_out.bin", ds_out)
            print(f"  block {li}: shape={list(h.shape)} (deepstack point)")

        if li == engine.v_depth - 1:
            save_bin("/tmp/diag_last_block_out.bin", h)
            print(f"  block {li} (last): shape={list(h.shape)}")

    # ── Step 3c: Merger ─────────────────────────────────────────
    print("\n--- Step 3c: Merger ---")
    vis = run_merger_npu(engine.g_v_merger, h, engine.v_merger_w)
    torch.npu.synchronize()
    save_bin("/tmp/diag_merger_out.bin", vis)
    print(f"  merger_out: shape={list(vis.shape)}")

    # Also save deepstack features
    for i, ds in enumerate(ds_feats):
        save_bin(f"/tmp/diag_deepstack_final_{i}.bin", ds)
        print(f"  deepstack_final_{i}: shape={list(ds.shape)}")

    print("\n" + "=" * 60)
    print("Python diagnostic complete. Files saved to /tmp/diag_*.bin")
    print("=" * 60)


if __name__ == "__main__":
    main()
