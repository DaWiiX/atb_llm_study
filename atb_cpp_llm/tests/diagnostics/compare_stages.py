"""
Compare Python vs C++ intermediate outputs at each vision pipeline stage.

Loads /tmp/diag_*.bin (Python) and /tmp/cpp_*.bin (C++) files,
computes cosine similarity and max abs diff at each stage,
and prints a summary table showing which stage first diverges below cosine=0.99.
"""

import struct
import numpy as np
import torch
import torch.nn.functional as F
import os
import glob


def load_bin(path: str) -> tuple:
    """Load binary file: int64 dim + raw data (fp16 or f32).

    Returns (num_elements, numpy_array) where array dtype is inferred from file size.
    """
    if not os.path.exists(path):
        return None, None

    with open(path, "rb") as f:
        n = struct.unpack("q", f.read(8))[0]
        raw = f.read()

    # Determine dtype based on file size
    bytes_per_elem = len(raw) / n if n > 0 else 0
    if abs(bytes_per_elem - 2.0) < 0.1:
        data = np.frombuffer(raw, dtype=np.float16)
    elif abs(bytes_per_elem - 4.0) < 0.1:
        data = np.frombuffer(raw, dtype=np.float32)
    elif abs(bytes_per_elem - 8.0) < 0.1:
        data = np.frombuffer(raw, dtype=np.int64)
    else:
        print(f"  WARNING: Unknown dtype for {path} ({bytes_per_elem:.1f} bytes/elem)")
        return n, None

    return n, data


def compare_stage(name: str, py_path: str, cpp_path: str,
                  py_is_f32: bool = False, cpp_is_f32: bool = False):
    """Compare Python and C++ outputs at a single stage.

    Returns (cosine, max_abs_diff, status_str) or None if files missing.
    """
    py_n, py_data = load_bin(py_path)
    cpp_n, cpp_data = load_bin(cpp_path)

    if py_data is None or cpp_data is None:
        missing = []
        if py_data is None:
            missing.append(py_path)
        if cpp_data is None:
            missing.append(cpp_path)
        print(f"  SKIP {name}: missing {', '.join(missing)}")
        return None

    # Convert to float32 for comparison
    if py_is_f32:
        py_f32 = py_data.astype(np.float32)
    else:
        py_f32 = py_data.astype(np.float32)  # float16 -> float32

    if cpp_is_f32:
        cpp_f32 = cpp_data.astype(np.float32)
    else:
        cpp_f32 = cpp_data.astype(np.float32)  # float16 -> float32

    if py_f32.shape[0] != cpp_f32.shape[0]:
        print(f"  MISMATCH {name}: Python has {py_f32.shape[0]} elements, C++ has {cpp_f32.shape[0]}")
        return None

    py_t = torch.from_numpy(py_f32)
    cpp_t = torch.from_numpy(cpp_f32)

    cos = F.cosine_similarity(py_t.unsqueeze(0), cpp_t.unsqueeze(0)).item()
    max_diff = (py_t - cpp_t).abs().max().item()
    mse = F.mse_loss(py_t, cpp_t).item()

    status = "OK" if cos >= 0.99 else "DIVERGE"
    if cos < 0.90:
        status = "BAD"

    print(f"  [{status}] {name}:")
    print(f"    Cosine:  {cos:.6f}")
    print(f"    MaxDiff: {max_diff:.6f}")
    print(f"    MSE:     {mse:.6e}")
    print(f"    Elements: {py_f32.shape[0]}")

    # Print first 5 values
    print(f"    Py  first5: {py_f32[:5].tolist()}")
    print(f"    C++ first5: {cpp_f32[:5].tolist()}")

    return (cos, max_diff, status)


def main():
    print("=" * 70)
    print("Vision Pipeline Stage-by-Stage Comparison: Python vs C++")
    print("=" * 70)

    results = []

    # ── Stage 0: Pixel Values ──────────────────────────────────
    print("\n--- Stage 0: Pixel Values (preprocessing output) ---")
    r = compare_stage("pixel_values",
                      "/tmp/diag_pixel_values.bin",
                      "/tmp/cpp_pixel_values.bin")
    if r:
        results.append(("pixel_values", r))

    # ── Stage 1: Position Embedding ────────────────────────────
    print("\n--- Stage 1: Position Embedding (NPU interp) ---")
    r = compare_stage("pos_embed",
                      "/tmp/diag_pos_embed.bin",
                      # C++ doesn't save pos_embed yet, skip
                      "/tmp/cpp_pos_embed.bin")
    if r:
        results.append(("pos_embed", r))
    else:
        print("  (C++ pos_embed not saved — Python-only, already verified OK)")

    # ── Stage 2: RoPE cos/sin (NPU fp16 vs CPU f32→fp16) ──────
    print("\n--- Stage 2: Vision RoPE ---")

    # 2a: Python NPU fp16 vs Python CPU f32→fp16 (quantization error only)
    print("\n  2a: Python NPU fp16 vs Python CPU f32→fp16:")
    py_npu_n, py_npu_cos = load_bin("/tmp/diag_rope_cos.bin")
    py_cpu_n, py_cpu_cos = load_bin("/tmp/diag_rope_cos_cpu_fp16.bin")
    if py_npu_cos is not None and py_cpu_cos is not None:
        py_npu_t = torch.from_numpy(py_npu_cos.astype(np.float32))
        py_cpu_t = torch.from_numpy(py_cpu_cos.astype(np.float32))
        cos_val = F.cosine_similarity(py_npu_t.unsqueeze(0), py_cpu_t.unsqueeze(0)).item()
        max_diff = (py_npu_t - py_cpu_t).abs().max().item()
        print(f"    Cos: {cos_val:.6f}, MaxDiff: {max_diff:.6f}")

    # 2b: Python CPU f32 RoPE vs C++ CPU f32 RoPE
    print("\n  2b: Python CPU f32 RoPE vs C++ CPU f32 RoPE:")
    r = compare_stage("rope_cos_cpu_f32",
                      "/tmp/diag_rope_cos_cpu_f32.bin",
                      "/tmp/cpp_rope_cos_f32.bin",
                      py_is_f32=True, cpp_is_f32=True)
    if r:
        results.append(("rope_cos_cpu_f32", r))

    # 2c: Python CPU fp16 RoPE vs C++ CPU f32→fp16 RoPE
    print("\n  2c: Python CPU fp16 RoPE vs C++ CPU f32→fp16 RoPE:")
    r = compare_stage("rope_cos_cpu_fp16",
                      "/tmp/diag_rope_cos_cpu_fp16.bin",
                      "/tmp/cpp_rope_cos.bin")
    if r:
        results.append(("rope_cos_cpu_fp16", r))

    # 2d: Python NPU fp16 RoPE vs C++ CPU f32→fp16 RoPE
    print("\n  2d: Python NPU fp16 RoPE vs C++ CPU f32→fp16 RoPE:")
    r = compare_stage("rope_cos_npu_vs_cpp",
                      "/tmp/diag_rope_cos.bin",
                      "/tmp/cpp_rope_cos.bin")
    if r:
        results.append(("rope_cos_npu_vs_cpp", r))

    # ── Stage 3: First Layer (patch_embed + pos_embed + block 0) ──
    print("\n--- Stage 3: First Layer Output ---")
    r = compare_stage("first_layer",
                      "/tmp/diag_first_layer_out.bin",
                      "/tmp/cpp_first_layer_out.bin")
    if r:
        results.append(("first_layer", r))

    # ── Stage 4: Deepstack blocks ──────────────────────────────
    for block_idx in [5, 11, 17]:
        print(f"\n--- Stage 4.{block_idx}: Block {block_idx} Output (deepstack) ---")
        r = compare_stage(f"block_{block_idx}",
                          f"/tmp/diag_block_{block_idx}_out.bin",
                          f"/tmp/cpp_block_{block_idx}_out.bin")
        if r:
            results.append((f"block_{block_idx}", r))

    # ── Stage 5: Last Block ────────────────────────────────────
    print("\n--- Stage 5: Last Block Output ---")
    r = compare_stage("last_block",
                      "/tmp/diag_last_block_out.bin",
                      "/tmp/cpp_last_block_out.bin")
    if r:
        results.append(("last_block", r))

    # ── Stage 6: Merger Output ─────────────────────────────────
    print("\n--- Stage 6: Merger Output (final vision embeddings) ---")
    r = compare_stage("merger",
                      "/tmp/diag_merger_out.bin",
                      "/tmp/cpp_vision_merged.bin")
    if r:
        results.append(("merger", r))

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Vision Pipeline Stage Comparison")
    print("=" * 70)
    print(f"{'Stage':<30} {'Cosine':>10} {'MaxDiff':>12} {'Status':>8}")
    print("-" * 70)

    first_diverge = None
    for name, (cos, max_diff, status) in results:
        print(f"{name:<30} {cos:>10.6f} {max_diff:>12.6f} {status:>8}")
        if first_diverge is None and cos < 0.99:
            first_diverge = name

    print("-" * 70)
    if first_diverge:
        print(f"\n*** FIRST DIVERGENCE below 0.99: {first_diverge} ***")
        print("This is the stage where C++ and Python outputs first differ significantly.")
        print("Investigate this stage for the root cause of the precision gap.")
    else:
        print("\nAll stages match with cosine >= 0.99. No significant divergence found.")

    # ── Divergence propagation analysis ────────────────────────
    if len(results) >= 2:
        print("\n--- Divergence Propagation Analysis ---")
        for i in range(1, len(results)):
            prev_name, (prev_cos, prev_diff, _) = results[i-1]
            curr_name, (curr_cos, curr_diff, _) = results[i]
            if prev_cos >= 0.99 and curr_cos < 0.99:
                print(f"  Divergence introduced between:")
                print(f"    {prev_name} (cos={prev_cos:.6f})")
                print(f"    {curr_name} (cos={curr_cos:.6f})")
                print(f"  MaxAbsDiff change: {prev_diff:.6f} → {curr_diff:.6f}")
            elif prev_cos < 0.99 and curr_cos < 0.99:
                delta_cos = curr_cos - prev_cos
                if delta_cos < -0.01:
                    print(f"  Divergence AMPLIFIED: {prev_name}→{curr_name}")
                    print(f"    cos: {prev_cos:.6f} → {curr_cos:.6f} (Δ={delta_cos:.6f})")


if __name__ == "__main__":
    main()
