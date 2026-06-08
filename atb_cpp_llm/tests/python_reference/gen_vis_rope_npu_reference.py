"""
Generate binary reference data for the Vision RoPE NPU graph.

The NPU implementation splits compute_rot_pos_emb's cos/sin generation
into two stages:

  Stage A (CPU, O(N)):
    - Build flat row_idx, col_idx tensors that capture the spatial-merge
      shuffle and temporal repeat for the (T, H, W) grid.
    - Build the (max_hw, half) frequency table  (`freq_table[i][j] = i * inv_freq[j]`).

  Stage B (NPU, O(N * vis_hd)):
    - Gather(freq_table, row_idx)  → row_freq (N, half)
    - Gather(freq_table, col_idx)  → col_freq (N, half)
    - Concat([row, col, row, col], axis=1)  → emb (N, vis_hd=half*4)
    - Cos(emb) → cos_out
    - Sin(emb) → sin_out

Layout of each .bin bundle:
  i64 n                              (= T*H*W per image, summed)
  i64 half
  i64 vis_hd                         (= half * 4)
  i64 max_hw
  i32 row_idx[n]
  i32 col_idx[n]
  f32 freq_table[max_hw * half]
  i64 expected_ndim                  (must be 2)
  i64 expected_shape[2]              (n, vis_hd)
  f32 expected_cos[n * vis_hd]
  f32 expected_sin[n * vis_hd]
"""

import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch

PROJ_DIR = Path(__file__).resolve().parent.parent.parent.parent / "atb_python_qwen3vl_embedding"
sys.path.insert(0, str(PROJ_DIR))

from engine_utils import VisionRotaryEmbedding, compute_rot_pos_emb

OUTPUT_DIR = "/tmp"


# ─── Stage A: build per-token (row, col) indices in spatial-merge order ───
def build_row_col_indices(grid_thw: torch.Tensor, merge_size: int):
    """Returns (row_idx, col_idx) — each int32 tensor of length sum(T*H*W).

    The order matches compute_rot_pos_emb exactly:
      view(merged_h, merge_size, merged_w, merge_size) -> permute(0,2,1,3) -> flat
    plus temporal repeat by T.
    """
    row_chunks, col_chunks = [], []
    for img in range(grid_thw.shape[0]):
        t_i, h_i, w_i = (int(x) for x in grid_thw[img])
        ms = merge_size
        merged_h, merged_w = h_i // ms, w_i // ms

        row_idx = torch.arange(h_i).view(merged_h, ms, 1, 1)
        col_idx = torch.arange(w_i).view(1, 1, merged_w, ms)

        row_grid = row_idx.expand(merged_h, ms, merged_w, ms)
        col_grid = col_idx.expand(merged_h, ms, merged_w, ms)

        row_flat = row_grid.permute(0, 2, 1, 3).reshape(-1)
        col_flat = col_grid.permute(0, 2, 1, 3).reshape(-1)

        if t_i > 1:
            row_flat = row_flat.repeat(t_i)
            col_flat = col_flat.repeat(t_i)
        row_chunks.append(row_flat)
        col_chunks.append(col_flat)

    return torch.cat(row_chunks).int(), torch.cat(col_chunks).int()


# ─── Stage B reference: NPU graph emulator ──────────────────────────────
def stage_b_reference(freq_table: torch.Tensor,
                      row_idx: torch.Tensor, col_idx: torch.Tensor):
    """Returns (cos, sin), each (N, half*4) float32."""
    row_freq = freq_table[row_idx.long()]   # (N, half)
    col_freq = freq_table[col_idx.long()]   # (N, half)
    emb = torch.cat([row_freq, col_freq, row_freq, col_freq], dim=1)  # (N, 4*half)
    return torch.cos(emb), torch.sin(emb)


def write_bundle(path: str, n, half, vis_hd, max_hw,
                 row_idx, col_idx, freq_table, expected_cos, expected_sin):
    with open(path, "wb") as f:
        f.write(np.int64(n).tobytes())
        f.write(np.int64(half).tobytes())
        f.write(np.int64(vis_hd).tobytes())
        f.write(np.int64(max_hw).tobytes())
        f.write(row_idx.astype(np.int32).tobytes())
        f.write(col_idx.astype(np.int32).tobytes())
        f.write(freq_table.astype(np.float32).tobytes())
        f.write(np.int64(2).tobytes())
        f.write(np.array([n, vis_hd], dtype=np.int64).tobytes())
        f.write(expected_cos.astype(np.float32).tobytes())
        f.write(expected_sin.astype(np.float32).tobytes())


def make_case(tag, grid_thw, vis_head_dim, merge_size=2):
    """vis_head_dim is the FULL vision head_dim (e.g. 64). Internal `dim` = head_dim // 2."""
    dim = vis_head_dim // 2          # what VisionRotaryEmbedding takes
    half = dim // 2                  # half of THAT — matches C++ "half = dim / 2"
    vis_hd = half * 4                # = dim * 2 = the output cos/sin column count

    # Build freq table for the canonical reference
    rotary = VisionRotaryEmbedding(dim)
    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = rotary(max_hw)      # (max_hw, half)  fp32

    # Stage A
    row_idx, col_idx = build_row_col_indices(grid_thw, merge_size)
    n = row_idx.shape[0]

    # Stage B reference
    cos_ref, sin_ref = stage_b_reference(freq_table, row_idx, col_idx)

    # Cross-check against canonical compute_rot_pos_emb:
    #   freq_table[pos_ids].flatten(1) gives  (N, half*2) of [row_freq, col_freq]
    # The C++ code then uses cos/sin of [row_freq, col_freq, row_freq, col_freq],
    # so canonical = first half of cos_ref / sin_ref.
    canonical_freq = compute_rot_pos_emb(grid_thw, rotary, merge_size)  # (N, half*2)
    half_cols = vis_hd // 2
    cos_ref_half = cos_ref[:, :half_cols]
    sin_ref_half = sin_ref[:, :half_cols]
    canonical_cos = torch.cos(canonical_freq)
    canonical_sin = torch.sin(canonical_freq)
    max_err = max(
        (cos_ref_half - canonical_cos).abs().max().item(),
        (sin_ref_half - canonical_sin).abs().max().item(),
    )
    print(f"[gen] case {tag}: grid_thw={grid_thw.tolist()} n={n} half={half} vis_hd={vis_hd} max_hw={max_hw} canonical_max_err={max_err:.6g}")
    if max_err > 1e-5:
        raise RuntimeError(f"Stage A/B decomposition does not match canonical compute_rot_pos_emb (err={max_err})")

    path = f"{OUTPUT_DIR}/visrope_npu_case_{tag}.bin"
    write_bundle(path, n, half, vis_hd, max_hw,
                 row_idx.numpy(), col_idx.numpy(),
                 freq_table.numpy(),
                 cos_ref.numpy(), sin_ref.numpy())
    print(f"    -> {path}")


def main():
    # The vision head_dim for Qwen3VL-Embedding-2B:
    # vis_hidden_size=1024, vis_num_heads=16  →  head_dim=64
    VIS_HEAD_DIM = 64

    cases = [
        ("tiny_4x4",  torch.tensor([[1,  4,  4]],  dtype=torch.long)),
        ("t1_8x8",    torch.tensor([[1,  8,  8]],  dtype=torch.long)),
        ("224x224",   torch.tensor([[2, 16, 16]],  dtype=torch.long)),
        ("416x672",   torch.tensor([[2, 26, 42]],  dtype=torch.long)),
        ("896x896",   torch.tensor([[2, 56, 56]],  dtype=torch.long)),
    ]
    for tag, gthw in cases:
        make_case(tag, gthw, vis_head_dim=VIS_HEAD_DIM, merge_size=2)

    print("\nAll reference bundles generated. Run C++ tests:")
    print("  ./build/test_vis_rope_npu_graph")


if __name__ == "__main__":
    main()
