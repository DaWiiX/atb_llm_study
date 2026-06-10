"""
Generate per-stage reference data for precision verification between
Python and C++ implementations of the Qwen3VL vision pipeline.

Saves binary reference files for Levels 0-3 to /tmp/stage_L{0-3}_*.bin.

Binary format for all tensors:
    int64       ndim
    int64[ndim] shape
    raw bytes   data (float16 for fp16, float32 for fp32, int64 for int tensors)

Usage:
    python atb_cpp_llm/tests/gen_stage_reference.py
"""

import os
import sys
import struct
from pathlib import Path
import torch
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────────────────


# Unified test inputs
IMG_H, IMG_W, IMG_C = 720, 1280, 3
TEXT_TOKENS = [74785, 279, 2168, 13]  # "Describe the image."
IMAGE_TOKEN_ID = 151655
MERGE_SIZE = 2
SPATIAL_MERGE = 2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size

# Must be called exactly once before any ATB graph build.
set_atb_buffer_size(10 * 1024 * 1024 * 1024)

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.preprocess import preprocess_image
from atb_python_qwen3vl_embedding.vision_pos_embed import (
    build_vision_posemb_graph,
    run_posemb_npu,
)
from atb_python_qwen3vl_embedding.vision_patch_embed import build_patch_embed
from atb_python_qwen3vl_embedding.engine_utils import (
    compute_posemb_indices,
    compute_rope_indices,
    fast_pos_embed_interpolate,
)
from atb_python_qwen3vl_embedding.utils import (
    to_npu_half,
    to_cpu_float,
    make_seqlen_tensor,
    make_elewise_add,
    get_atb_builder,
)


# ── Binary save helper ──────────────────────────────────────────────

def save_tensor(path, tensor, dtype='fp16'):
    """Save tensor to binary file.

    Binary format: [ndim: int64] [shape: int64[ndim]] [data: raw bytes]

    Args:
        path:  output file path
        tensor: torch.Tensor to save
        dtype: 'fp16' -> save as float16, 'fp32' -> save as float32,
               'int64' -> save as int64
    """
    if dtype == 'fp16':
        data = tensor.flatten().half().cpu()
    elif dtype == 'fp32':
        data = tensor.flatten().float().cpu()
    elif dtype == 'int64':
        data = tensor.flatten().long().cpu()
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    ndim = len(tensor.shape)
    shape = list(tensor.shape)
    with open(path, 'wb') as f:
        f.write(struct.pack('q', ndim))
        for s in shape:
            f.write(struct.pack('q', s))
        f.write(data.numpy().tobytes())
    print(f"  Saved {path}: shape={shape}, dtype={dtype}")


# ── Gradient image (matches C++ side) ──────────────────────────────

def create_gradient_image(channels, height, width):
    """Create a 3-channel gradient pattern image.

    image[c, h, w] = (h * 255 // height + w * 255 // width + c * 85) % 256
    """
    image = torch.zeros(channels, height, width, dtype=torch.uint8)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                value = (h * 255 // height + w * 255 // width + c * 85) % 256
                image[c, h, w] = value
    return image


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Per-Stage Reference Generator (Levels 0-3)")
    print("=" * 60)

    # ── Load engine ────────────────────────────────────────────
    print("\nLoading Python ATB engine...")
    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine loaded: {engine.n_layer} text layers, "
          f"{engine.v_depth} vision blocks")
    print(f"  image_token_id={engine.img_tok}, "
          f"spatial_merge={engine.spatial_merge}")
    print(f"  num_grid={engine.num_grid}, "
          f"nh_v={engine.nh_v}, hd_v={engine.hd_v}")

    # ── Create test image ──────────────────────────────────────
    img = create_gradient_image(IMG_C, IMG_H, IMG_W)

    # ═══════════════════════════════════════════════════════════
    # Level 0: Preprocessing
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Level 0: Preprocessing")
    print("=" * 60)

    pixel_values, grid_thw = preprocess_image(img)
    print(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw}")
    print(f"  first 8 pixel_values[0]: {pixel_values[0, :8].tolist()}")

    save_tensor("/tmp/stage_L0_pixel_values.bin", pixel_values, dtype='fp16')
    save_tensor("/tmp/stage_L0_grid_thw.bin", grid_thw, dtype='int64')

    # ═══════════════════════════════════════════════════════════
    # Level 1: Patch Embedding
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Level 1: Patch Embedding")
    print("=" * 60)

    # Build standalone patch_embed graph to isolate the Linear output
    # BEFORE adding position embedding.
    v_cfg = engine.v_cfg
    in_channels = v_cfg["in_channels"]       # 3
    temporal_patch_size = v_cfg["temporal_patch_size"]  # 2
    patch_size = v_cfg["patch_size"]         # 16
    embed_dim = v_cfg["hidden_size"]         # 1280

    _, patch_graph, _ = build_patch_embed(
        in_channels, temporal_patch_size, patch_size, embed_dim,
        name="PatchEmbed_ref")

    # Prepare patch_embed inputs: [x (pixels), w (weight), b (bias)]
    pv_flat = pixel_values.reshape(-1)
    pv_npu = to_npu_half(pv_flat)
    pe_w_npu = engine.v_pe_w   # already NPU fp16
    pe_b_npu = engine.v_pe_b   # already NPU fp16

    torch.npu.synchronize()
    patch_embed_out_npu = patch_graph.forward([pv_npu, pe_w_npu, pe_b_npu])[0]
    torch.npu.synchronize()

    patch_embed_out = to_cpu_float(patch_embed_out_npu)
    print(f"  patch_embed_out: shape={patch_embed_out.shape}, dtype={patch_embed_out.dtype}")
    print(f"  first 8: {patch_embed_out[0, :8].tolist()}")

    save_tensor("/tmp/stage_L1_patch_embed_out.bin", patch_embed_out_npu, dtype='fp16')

    # ═══════════════════════════════════════════════════════════
    # Level 2: Position Embedding
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Level 2: Position Embedding")
    print("=" * 60)

    # Compute pos_embed indices and RoPE indices (CPU tensors)
    idx_wt = compute_posemb_indices(grid_thw, engine.num_grid, MERGE_SIZE)
    rope_idx = compute_rope_indices(grid_thw, engine.vis_rotary, MERGE_SIZE)
    freq_npu = to_npu_half(rope_idx['freq_table'])

    # Run the posemb ATB graph to get NPU fp16 position embedding + RoPE cos/sin
    pos_npu, cos_npu, sin_npu = run_posemb_npu(
        engine.g_v_posemb, engine.v_pe_w_table, idx_wt, rope_idx, freq_npu)
    torch.npu.synchronize()

    # (a) NPU fp16 position embedding — what Python's ATB graph produces
    pos_embed_npu_cpu = to_cpu_float(pos_npu)
    print(f"  pos_embed_npu: shape={pos_npu.shape}")
    print(f"  first 8: {pos_embed_npu_cpu[0, :8].tolist()}")
    save_tensor("/tmp/stage_L2_pos_embed_npu.bin", pos_npu, dtype='fp16')

    # (b) CPU f32 position embedding — what C++ should produce
    #     (bilinear interpolation on CPU in f32, saved as fp16 for binary format)
    pos_embed_cpu = fast_pos_embed_interpolate(
        grid_thw, engine.v_pos_embed, engine.num_grid, MERGE_SIZE)
    print(f"  pos_embed_cpu: shape={pos_embed_cpu.shape}")
    print(f"  first 8: {pos_embed_cpu[0, :8].tolist()}")

    # Verify NPU vs CPU consistency
    cos_pos = F.cosine_similarity(
        pos_embed_npu_cpu.flatten(), pos_embed_cpu.flatten(), dim=0).item()
    max_diff_pos = (pos_embed_npu_cpu - pos_embed_cpu).abs().max().item()
    print(f"  NPU vs CPU pos_embed: cosine={cos_pos:.6f}, max_diff={max_diff_pos:.6f}")

    # Save as fp16 — C++ can load and compare
    save_tensor("/tmp/stage_L2_pos_embed_cpu.bin", pos_embed_cpu, dtype='fp16')

    # (c) first_layer_after_pos = patch_embed_out + pos_embed (before block 0)
    #     patch_embed_out is (npatches, hidden_size) fp16 on NPU
    #     pos_npu is (npatches, hidden_size) fp16 on NPU
    #     We need ElewiseAdd on NPU to match the actual computation.
    #     Build a tiny Add graph to compute this on NPU in fp16.
    add_builder = get_atb_builder("PosAdd_ref")
    a_in = add_builder.add_input("a")
    b_in = add_builder.add_input("b")
    add_node = add_builder.add_node([a_in, b_in], make_elewise_add())
    add_builder.mark_output(add_node.get_output(0))
    add_graph = add_builder.build()

    after_pos_npu = add_graph.forward([patch_embed_out_npu, pos_npu])[0]
    torch.npu.synchronize()

    after_pos_cpu = to_cpu_float(after_pos_npu)
    print(f"  first_layer_after_pos: shape={after_pos_npu.shape}")
    print(f"  first 8: {after_pos_cpu[0, :8].tolist()}")
    save_tensor("/tmp/stage_L2_first_layer_after_pos.bin", after_pos_npu, dtype='fp16')

    # ═══════════════════════════════════════════════════════════
    # Level 3: Vision RoPE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Level 3: Vision RoPE")
    print("=" * 60)

    cos_cpu = to_cpu_float(cos_npu)
    sin_cpu = to_cpu_float(sin_npu)

    print(f"  rope_cos: shape={cos_npu.shape}")
    print(f"  first 8: {cos_cpu[0, :8].tolist()}")
    print(f"  rope_sin: shape={sin_npu.shape}")
    print(f"  first 8: {sin_cpu[0, :8].tolist()}")

    save_tensor("/tmp/stage_L3_rope_cos.bin", cos_npu, dtype='fp16')
    save_tensor("/tmp/stage_L3_rope_sin.bin", sin_npu, dtype='fp16')

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Reference data saved:")
    print("  Level 0: /tmp/stage_L0_pixel_values.bin")
    print("           /tmp/stage_L0_grid_thw.bin")
    print("  Level 1: /tmp/stage_L1_patch_embed_out.bin")
    print("  Level 2: /tmp/stage_L2_pos_embed_npu.bin")
    print("           /tmp/stage_L2_pos_embed_cpu.bin")
    print("           /tmp/stage_L2_first_layer_after_pos.bin")
    print("  Level 3: /tmp/stage_L3_rope_cos.bin")
    print("           /tmp/stage_L3_rope_sin.bin")
    print("=" * 60)


if __name__ == "__main__":
    main()
