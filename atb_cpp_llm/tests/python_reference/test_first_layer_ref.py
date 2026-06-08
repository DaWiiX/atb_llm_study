#!/usr/bin/env python3
"""Save Python First Layer (PatchEmbed + PosEmbed + VisionBlock_0) inputs and output.

Produces reference binary files so the C++ First Layer implementation can be
compared against the Python ATB path with *exactly the same* inputs.

Binary format for all tensor files:  int64 dim  +  raw data (fp16 or int32).
For seqlen: int64(=1) + int32 value.

Usage:
    cd /mnt/workspace/gitCode/atb_llm
    python3 atb_cpp_llm/tests/test_first_layer_ref.py
"""
import sys, os, struct, pathlib
import torch
import torch.nn.functional as F

# ── Add the Python package to sys.path ────────────────────────────────
sys.path.insert(0, "/mnt/workspace/gitCode/atb_llm")

from atb_python_qwen3vl_embedding.utils import (
    set_atb_buffer_size, to_npu_half, to_cpu_float, make_seqlen_tensor,
)
from atb_python_qwen3vl_embedding.engine_utils import (
    load_config, load_preprocessor_config, load_weights,
    get_vision_block_weights, get_patch_embed_weights,
    get_vision_pos_embed, VisionRotaryEmbedding,
    compute_posemb_indices, compute_rope_indices,
)
from atb_python_qwen3vl_embedding.preprocess import preprocess_image
from atb_python_qwen3vl_embedding.vision_model import (
    build_vision_first_layer,
    run_first_layer_npu,
)
from atb_python_qwen3vl_embedding.vision_pos_embed import (
    build_vision_posemb_graph, run_posemb_npu,
)

MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B"
OUT_DIR   = "/tmp"


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def save_tensor_bin(path: str, tensor: torch.Tensor, dtype=None):
    """Save tensor as: int64(numel) + raw data.

    dtype: target dtype for saving (e.g. torch.float16, torch.int32).
           If None, saves in the tensor's current dtype.
    """
    t = tensor.detach().cpu()
    if dtype is not None:
        t = t.to(dtype)
    numel = t.numel()
    raw = t.numpy().tobytes()
    with open(path, "wb") as f:
        f.write(struct.pack("<q", numel))
        f.write(raw)
    print(f"  saved {path}  shape={list(tensor.shape)}  numel={numel}  "
          f"dtype={t.dtype}  first4={t.flatten()[:4].tolist()}")


def save_seqlen_bin(path: str, value: int):
    """Save seqlen as: int64(=1) + int32 value."""
    with open(path, "wb") as f:
        f.write(struct.pack("<q", 1))
        f.write(struct.pack("<i", value))
    print(f"  saved {path}  seqlen={value}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import torch_npu  # noqa — NPU runtime

    # ── 1. Load config & weights ──────────────────────────────────────
    cfg = load_config(MODEL_DIR)
    v_cfg = cfg["vision_config"]
    pp = load_preprocessor_config(MODEL_DIR)

    weights = load_weights(MODEL_DIR)

    nh_v    = v_cfg["num_heads"]
    hd_v    = v_cfg["hidden_size"] // nh_v
    merge   = v_cfg["spatial_merge_size"]
    num_grid = int(v_cfg["num_position_embeddings"] ** 0.5)

    # ── 2. Create gradient image & preprocess ─────────────────────────
    H, W = 720, 1280
    # Gradient image: each pixel = (c*255/H, r*255/W, (c+r)*128/(H+W))
    image = torch.zeros(3, H, W, dtype=torch.uint8)
    for c in range(3):
        for row in range(H):
            for col in range(W):
                if c == 0:
                    image[c, row, col] = int(col * 255 / W)
                elif c == 1:
                    image[c, row, col] = int(row * 255 / H)
                else:
                    image[c, row, col] = int((col + row) * 128 / (H + W))

    pixel_values, grid_thw = preprocess_image(
        image,
        patch_size=pp["patch_size"],
        temporal_patch_size=pp["temporal_patch_size"],
        merge_size=merge,
        min_pixels=pp["min_pixels"],
        max_pixels=pp["max_pixels"],
    )
    print(f"pixel_values shape: {pixel_values.shape}  grid_thw: {grid_thw}")

    # ── 3. Extract weights (CPU float32 first) ───────────────────────
    # Patch embed weights
    pe_w_cpu, pe_b_cpu = get_patch_embed_weights(weights, v_cfg["hidden_size"])
    # Block 0 weights (12 tensors)
    block0_weights_cpu = get_vision_block_weights(weights, 0)
    # Position embedding table
    pos_embed_table = get_vision_pos_embed(weights)

    # ── 4. Build ATB graphs ───────────────────────────────────────────
    set_atb_buffer_size(5 * 1024 * 1024 * 1024)

    from atb_python_qwen3vl_embedding.engine import VisionConfigWrapper
    v_config = VisionConfigWrapper(v_cfg)
    g_v_first = build_vision_first_layer(v_config)
    g_v_posemb = build_vision_posemb_graph()

    # ── 5. Compute pos_embed + RoPE via NPU path ─────────────────────
    vis_rotary = VisionRotaryEmbedding(dim=hd_v // 2)

    idx_wt = compute_posemb_indices(grid_thw, num_grid, merge)
    rope_idx = compute_rope_indices(grid_thw, vis_rotary, merge)

    pe_w_table_npu = to_npu_half(pos_embed_table)
    freq_npu = to_npu_half(rope_idx["freq_table"])

    pos_npu, cos_npu, sin_npu = run_posemb_npu(
        g_v_posemb, pe_w_table_npu, idx_wt, rope_idx, freq_npu)

    # ── 6. Convert all weights to NPU fp16 ────────────────────────────
    pe_w_npu = to_npu_half(pe_w_cpu)
    pe_b_npu = to_npu_half(pe_b_cpu)
    block0_weights_npu = [to_npu_half(w) for w in block0_weights_cpu]

    pv_npu = to_npu_half(pixel_values.reshape(-1)
                         if pixel_values.ndim == 2 else pixel_values)

    npatches = idx_wt["idx00"].shape[0]
    seqlen_v = make_seqlen_tensor(npatches)

    # Sync before first-layer execution
    torch.npu.synchronize()

    # ── 7. Run Python first layer ─────────────────────────────────────
    output_npu = run_first_layer_npu(
        g_v_first, pv_npu,
        pe_w_npu, pe_b_npu,
        pos_npu, cos_npu, sin_npu,
        block0_weights_npu,
        seqlen_v,
    )
    torch.npu.synchronize()
    output_cpu = to_cpu_float(output_npu)
    print(f"\nFirst layer output shape: {output_cpu.shape}")
    print(f"  first4: {output_cpu.flatten()[:4].tolist()}")
    print(f"  mean={output_cpu.mean().item():.6f}  std={output_cpu.std().item():.6f}")

    # ── 8. Save all inputs and output ─────────────────────────────────
    print("\n=== Saving reference tensors ===")

    # Save pixel_values (fp16)
    save_tensor_bin(f"{OUT_DIR}/ref_fl_pixels.bin", pv_npu.cpu(), torch.float16)

    # Save patch_embed weight & bias (fp16)
    save_tensor_bin(f"{OUT_DIR}/ref_fl_patch_w.bin", pe_w_npu.cpu(), torch.float16)
    save_tensor_bin(f"{OUT_DIR}/ref_fl_patch_b.bin", pe_b_npu.cpu(), torch.float16)

    # Save position embedding (fp16)
    save_tensor_bin(f"{OUT_DIR}/ref_fl_pos_embed.bin", pos_npu.cpu(), torch.float16)

    # Save RoPE cos & sin (fp16)
    save_tensor_bin(f"{OUT_DIR}/ref_fl_cos.bin", cos_npu.cpu(), torch.float16)
    save_tensor_bin(f"{OUT_DIR}/ref_fl_sin.bin", sin_npu.cpu(), torch.float16)

    # Save seqlen (int32)
    save_seqlen_bin(f"{OUT_DIR}/ref_fl_seqlen.bin", npatches)

    # Save block_0 individual weights (fp16) — 12 tensors in order:
    # qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b, n1_w, n1_b, n2_w, n2_b
    block_names = [
        "qkv_w", "qkv_b", "proj_w", "proj_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        "n1_w", "n1_b", "n2_w", "n2_b",
    ]
    for i, name in enumerate(block_names):
        save_tensor_bin(f"{OUT_DIR}/ref_fl_{name}.bin",
                        block0_weights_npu[i].cpu(), torch.float16)

    # Save all 12 block_0 weights concatenated (fp16)
    all_block0_cat = torch.cat([w.cpu().flatten() for w in block0_weights_npu])
    save_tensor_bin(f"{OUT_DIR}/ref_fl_block0_weights.bin", all_block0_cat, torch.float16)

    # Save first-layer output (fp16)
    save_tensor_bin(f"{OUT_DIR}/ref_fl_output.bin", output_npu.cpu(), torch.float16)

    # ── 9. Compare with C++ output if available ───────────────────────
    cpp_out_path = f"{OUT_DIR}/cpp_first_layer_out.bin"
    if os.path.exists(cpp_out_path):
        print(f"\n=== Comparing with C++ output: {cpp_out_path} ===")
        with open(cpp_out_path, "rb") as f:
            cpp_numel = struct.unpack("<q", f.read(8))[0]
            cpp_data = f.read()
        cpp_out = torch.frombuffer(bytearray(cpp_data), dtype=torch.float16).reshape(cpp_numel)
        cpp_out = cpp_out.float()
        py_out = output_cpu.flatten()

        # Align sizes — they might differ slightly
        min_len = min(cpp_out.numel(), py_out.numel())
        print(f"  Python output numel: {py_out.numel()}")
        print(f"  C++    output numel: {cpp_out.numel()}")
        print(f"  Comparing first {min_len} elements")

        cos_sim = F.cosine_similarity(py_out[:min_len].unsqueeze(0),
                                       cpp_out[:min_len].unsqueeze(0), dim=1).item()
        max_diff = (py_out[:min_len] - cpp_out[:min_len]).abs().max().item()
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Max abs diff:      {max_diff:.6f}")
        if cos_sim > 0.99:
            print("  PASS (cos > 0.99)")
        else:
            print("  FAIL (cos <= 0.99) — investigate!")
    else:
        print(f"\nC++ output not found at {cpp_out_path} — skipping comparison")

    # ── 10. Print summary of input order ──────────────────────────────
    print("\n=== Python First Layer input order (19 inputs) ===")
    input_names = [
        "pixels", "pe_w", "pe_b", "pos",
        "cos", "sin", "seqlen",
        "qkv_w", "qkv_b", "proj_w", "proj_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        "n1_w", "n1_b", "n2_w", "n2_b",
    ]
    for i, name in enumerate(input_names):
        print(f"  [{i:2d}] {name}")

    # Print shapes of all inputs for reference
    print("\n=== Input tensor shapes ===")
    all_inputs = [
        ("pixels", pv_npu),
        ("pe_w", pe_w_npu),
        ("pe_b", pe_b_npu),
        ("pos", pos_npu),
        ("cos", cos_npu),
        ("sin", sin_npu),
        ("seqlen", seqlen_v),
    ]
    for i, name in enumerate(block_names):
        all_inputs.append((name, block0_weights_npu[i]))

    for name, t in all_inputs:
        dev = t.device if isinstance(t, torch.Tensor) else "scalar"
        if isinstance(t, torch.Tensor):
            print(f"  {name:12s} shape={str(list(t.shape)):20s} dtype={t.dtype}  device={t.device}")
        else:
            print(f"  {name:12s} value={t}")

    print("\nDone.")


if __name__ == "__main__":
    main()
