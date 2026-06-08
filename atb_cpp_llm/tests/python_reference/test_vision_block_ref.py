"""
Vision Block Python vs C++ Comparison Test.

Runs the Python vision pipeline step-by-step to capture:
  1. First-layer output (input to block 1)
  2. Block 1 weights
  3. RoPE cos/sin
  4. Block 1 output

Saves all as binary files in /tmp/ref_vb_*.bin for C++ comparison,
then computes cosine similarity between Python and C++ block 1 outputs.

Binary format: int64 total_elements + raw fp16 bytes (same as diag_vision_stages.py).

Usage:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
    source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
    source /home/developer/Ascend/nnal/atb/9.0.0/atb/set_env.sh --cxx_abi=1 2>/dev/null
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/developer/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib

    cd /mnt/workspace/gitCode/atb_llm && python3 atb_cpp_llm/tests/test_vision_block_ref.py
"""

import sys
import struct
import os
import numpy as np
import torch
import torch.nn.functional as F

# ── Configuration ────────────────────────────────────────────────
MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B"
IMG_H = 720
IMG_W = 1280
IMG_C = 3


def save_bin(path: str, tensor_fp16: torch.Tensor):
    """Save fp16 tensor to binary file: int64 total_elements + raw fp16 bytes."""
    flat = tensor_fp16.reshape(-1)
    n = flat.numel()
    # Convert to numpy uint16 (raw fp16 bits)
    arr_u16 = flat.cpu().view(torch.uint16).numpy()
    with open(path, "wb") as f:
        f.write(struct.pack("q", n))
        f.write(arr_u16.tobytes())
    print(f"  Saved {path}: {n} fp16 values, shape={list(tensor_fp16.shape)}")


def save_seqlen_bin(path: str, seqlen_value: int):
    """Save seqlen as: int64 dim=1 + int32 value."""
    with open(path, "wb") as f:
        f.write(struct.pack("q", 1))  # dim = 1
        f.write(struct.pack("i", seqlen_value))  # int32 value
    print(f"  Saved {path}: seqlen={seqlen_value}")


def load_cpp_block1_out(path: str):
    """Load C++ block 1 output: int64 total + uint16 fp16 data.

    Returns (numpy float32 array, shape_as_2d) or None.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        total = struct.unpack("q", f.read(8))[0]
        raw = f.read(total * 2)  # uint16 bytes
    arr_u16 = np.frombuffer(raw, dtype=np.uint16)
    arr_f16 = arr_u16.astype(np.float32)  # numpy auto-converts fp16 bits? No.
    # Need to use torch for proper fp16 -> fp32 conversion
    t = torch.from_numpy(arr_u16).view(torch.float16).float()
    return t.numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D float32 arrays."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def create_test_image(channels: int, height: int, width: int) -> torch.Tensor:
    """Create the same gradient test image as C++ side: (C, H, W) uint8."""
    image = torch.zeros(channels, height, width, dtype=torch.uint8)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                value = (h * 255 // height + w * 255 // width + c * 85) % 256
                image[c, h, w] = value
    return image


def main():
    print("=" * 70)
    print("Vision Block Python vs C++ Comparison Test")
    print(f"Image: {IMG_H}x{IMG_W}")
    print("=" * 70)

    # ── Setup engine ────────────────────────────────────────────────
    sys.path.insert(0, "/mnt/workspace/gitCode/atb_llm")
    from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)

    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    from atb_python_qwen3vl_embedding.preprocess import preprocess_image
    from atb_python_qwen3vl_embedding.engine_utils import (
        compute_posemb_indices, compute_rope_indices,
        get_vision_block_weights,
    )
    from atb_python_qwen3vl_embedding.vision_model import (
        run_first_layer_npu, run_block_npu,
    )
    from atb_python_qwen3vl_embedding.vision_pos_embed import run_posemb_npu
    from atb_python_qwen3vl_embedding.utils import (
        to_npu_half, to_cpu_float, make_seqlen_tensor,
    )

    print("\nLoading Python ATB engine...")
    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine loaded: {engine.n_layer} text layers, {engine.v_depth} vision blocks")
    print(f"  ds_indexes: {engine.ds_indexes}")
    print(f"  merge_size: {engine.merge_size}")
    print(f"  num_heads: {engine.nh_v}, head_dim: {engine.hd_v}")

    # ── Step 1: Preprocess image ────────────────────────────────────
    print("\n--- Step 1: Preprocessing ---")
    img = create_test_image(IMG_C, IMG_H, IMG_W)
    pv, grid_thw = preprocess_image(img)
    print(f"  pixel_values: shape={list(pv.shape)}, dtype={pv.dtype}")
    print(f"  grid_thw: {grid_thw}")

    num_patches = pv.shape[0]
    vis_hs = engine.v_cfg["hidden_size"]
    vis_hd = engine.hd_v
    vis_nh = engine.nh_v
    merge_size = engine.merge_size
    num_grid = engine.num_grid
    print(f"  num_patches={num_patches}, vis_hs={vis_hs}, vis_hd={vis_hd}, vis_nh={vis_nh}")

    # ── Step 2: Position embedding + RoPE via ATB graph ─────────────
    print("\n--- Step 2: Position Embedding + RoPE (NPU) ---")
    idx_wt = compute_posemb_indices(grid_thw, num_grid, merge_size)
    rope_idx = compute_rope_indices(grid_thw, engine.vis_rotary, merge_size)
    freq_npu = to_npu_half(rope_idx['freq_table'])

    pos_npu, cos_npu, sin_npu = run_posemb_npu(
        engine.g_v_posemb, engine.v_pe_w_table, idx_wt, rope_idx, freq_npu)

    print(f"  pos_npu: shape={list(pos_npu.shape)}, dtype={pos_npu.dtype}")
    print(f"  cos_npu: shape={list(cos_npu.shape)}, dtype={cos_npu.dtype}")
    print(f"  sin_npu: shape={list(sin_npu.shape)}, dtype={sin_npu.dtype}")

    # ── Step 3: Run first layer (patch_embed + pos_embed + block 0) ──
    print("\n--- Step 3: First Layer ---")
    pv_npu = to_npu_half(pv.reshape(-1) if pv.ndim == 2 else pv)
    seqlen_v = make_seqlen_tensor(num_patches)
    torch.npu.synchronize()

    h = run_first_layer_npu(engine.g_v_first, pv_npu,
                            engine.v_pe_w, engine.v_pe_b,
                            pos_npu, cos_npu, sin_npu,
                            engine.v_block_weights[0], seqlen_v)
    torch.npu.synchronize()
    print(f"  first_layer_out: shape={list(h.shape)}, dtype={h.dtype}")

    # ── Step 4: Run block 1 ─────────────────────────────────────────
    print("\n--- Step 4: Block 1 ---")
    block1_weights = engine.v_block_weights[1]  # 12 weights in ATB order
    torch.npu.synchronize()
    block1_out = run_block_npu(engine.g_v_block, h, block1_weights,
                               cos_npu, sin_npu, seqlen_v)
    torch.npu.synchronize()
    print(f"  block1_out: shape={list(block1_out.shape)}, dtype={block1_out.dtype}")

    # ── Step 5: Save all reference data ─────────────────────────────
    print("\n--- Step 5: Saving Reference Data ---")

    # Input hidden states (first_layer output) - this is the input to block 1
    save_bin("/tmp/ref_vb_hidden.bin", h)

    # Block 1 weights (12 tensors in ATB order):
    # [qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b, n1_w, n1_b, n2_w, n2_b]
    weight_names = [
        "qkv_w", "qkv_b", "proj_w", "proj_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        "n1_w", "n1_b", "n2_w", "n2_b",
    ]
    for i, name in enumerate(weight_names):
        save_bin(f"/tmp/ref_vb_{name}.bin", block1_weights[i])

    # RoPE cos/sin
    save_bin("/tmp/ref_vb_cos.bin", cos_npu)
    save_bin("/tmp/ref_vb_sin.bin", sin_npu)

    # Sequence length
    save_seqlen_bin("/tmp/ref_vb_seqlen.bin", num_patches)

    # Block 1 output (the ground truth from Python)
    save_bin("/tmp/ref_vb_output.bin", block1_out)

    # ── Step 6: Compare Python vs C++ block 1 output ────────────────
    print("\n--- Step 6: Python vs C++ Comparison ---")

    # Load Python block 1 output as float32
    py_out_f32 = block1_out.cpu().float().numpy()
    print(f"  Python block1_out: shape={list(py_out_f32.shape)}, "
          f"first 4: {py_out_f32.flatten()[:4].tolist()}")

    # Try to load C++ block 1 output
    cpp_out_path = "/tmp/cpp_block_1_out.bin"
    cpp_out = load_cpp_block1_out(cpp_out_path)

    if cpp_out is not None:
        print(f"  C++ block1_out: {cpp_out.shape[0]} elements, "
              f"first 4: {cpp_out.flatten()[:4].tolist()}")

        # Both should be (num_patches * vis_hs) elements
        py_flat = py_out_f32.flatten()
        cpp_flat = cpp_out.flatten()

        n_cmp = min(len(py_flat), len(cpp_flat))
        py_cmp = py_flat[:n_cmp]
        cpp_cmp = cpp_flat[:n_cmp]

        cos_sim = cosine_sim(py_cmp, cpp_cmp)
        max_diff = float(np.max(np.abs(py_cmp - cpp_cmp)))
        mse = float(np.mean((py_cmp - cpp_cmp) ** 2))

        print(f"\n  === Python vs C++ Vision Block 1 Output ===")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  MSE:              {mse:.6e}")
        print(f"  Max abs diff:     {max_diff:.6e}")
        print(f"  Elements compared: {n_cmp}")
        print(f"  Python elements:   {len(py_flat)}")
        print(f"  C++ elements:      {len(cpp_flat)}")

        if cos_sim > 0.999:
            print(f"\n  [PASS] Python vs C++ vision block 1 match (cos={cos_sim:.6f} > 0.999)")
        elif cos_sim > 0.99:
            print(f"\n  [WARN] Python vs C++ vision block 1 close but below 0.999 (cos={cos_sim:.6f})")
        else:
            print(f"\n  [FAIL] Python vs C++ vision block 1 diverge (cos={cos_sim:.6f})")

        # Also compare first_layer output (input to block 1) if C++ saved it
        cpp_first_path = "/tmp/cpp_first_layer_out.bin"
        if os.path.exists(cpp_first_path):
            cpp_first = load_cpp_block1_out(cpp_first_path)
            py_first = h.cpu().float().numpy().flatten()
            cpp_first_flat = cpp_first.flatten()
            n2 = min(len(py_first), len(cpp_first_flat))
            cos_input = cosine_sim(py_first[:n2], cpp_first_flat[:n2])
            print(f"\n  === Python vs C++ First Layer Output (Block Input) ===")
            print(f"  Cosine similarity: {cos_input:.6f}")
            if cos_input > 0.999:
                print(f"  [PASS] First layer inputs match (cos={cos_input:.6f})")
            else:
                print(f"  [WARN/FAIL] First layer inputs differ (cos={cos_input:.6f})")
        else:
            print(f"\n  Note: /tmp/cpp_first_layer_out.bin not found -- skipping input comparison")
            print(f"  (Set ATB_DEBUG_VISION=1 when running C++ to save intermediate outputs)")

    else:
        print(f"  C++ block1_out not found at {cpp_out_path}")
        print(f"  To generate it: run the C++ engine with ATB_DEBUG_VISION=1")
        print(f"  Reference data has been saved to /tmp/ref_vb_*.bin for C++ comparison")

    # ── Step 7: Also compare with diag_block_1_out.bin if it exists ──
    print("\n--- Step 7: Cross-check with diag_block_1_out.bin ---")
    diag_path = "/tmp/diag_block_1_out.bin"
    if os.path.exists(diag_path):
        diag_out = load_cpp_block1_out(diag_path)
        py_flat = py_out_f32.flatten()
        diag_flat = diag_out.flatten()
        n3 = min(len(py_flat), len(diag_flat))
        cos_diag = cosine_sim(py_flat[:n3], diag_flat[:n3])
        print(f"  Python vs diag_block_1_out: cosine={cos_diag:.6f}")
        if cos_diag > 0.9999:
            print(f"  [PASS] Python block1 outputs are self-consistent (cos={cos_diag:.6f})")
        else:
            print(f"  [WARN] Python block1 outputs differ between runs (cos={cos_diag:.6f})")
    else:
        print(f"  {diag_path} not found -- skipping cross-check")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Reference data saved to /tmp/ref_vb_*.bin")
    print("  ref_vb_hidden.bin  - input hidden states (first_layer output)")
    print("  ref_vb_qkv_w/b.bin - block 1 attention QKV weight/bias")
    print("  ref_vb_proj_w/b.bin - block 1 attention proj weight/bias")
    print("  ref_vb_fc1_w/b.bin  - block 1 MLP fc1 weight/bias")
    print("  ref_vb_fc2_w/b.bin  - block 1 MLP fc2 weight/bias")
    print("  ref_vb_n1_w/b.bin   - block 1 norm1 weight/bias")
    print("  ref_vb_n2_w/b.bin   - block 1 norm2 weight/bias")
    print("  ref_vb_cos.bin      - RoPE cosine")
    print("  ref_vb_sin.bin      - RoPE sine")
    print("  ref_vb_seqlen.bin   - sequence length")
    print("  ref_vb_output.bin   - block 1 output (Python ground truth)")
    print("=" * 70)


if __name__ == "__main__":
    main()
