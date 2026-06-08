"""
Generate binary reference data for the Vision PosEmbed NPU graph.

The NPU implementation factors the original `fast_pos_embed_interpolate`
into two stages:

  Stage A (CPU, O(t*h*w)):
    Pre-compute four (flattened) index tensors and four weight tensors
    that capture the bilinear interpolation coefficients AND the
    spatial-merge shuffle AND the temporal repeat — all baked into one
    flat permutation that maps output-row i to four source rows in the
    pos_embed table.

  Stage B (NPU, O(t*h*w * hidden_size)):
    Just gather + weighted sum:
        out[i] = sum_k pos_embed_w[idx_k[i]] * wt_k[i]   for k in 0..3

This script writes:
  /tmp/posembed_npu_pos_embed_w.bin  : (G*G, vis_hs) fp16  source table
  /tmp/posembed_npu_case_<tag>.bin   : per-case bundle (inputs + expected output)
  /tmp/posembed_npu_expected_<tag>.bin : (T*H*W, vis_hs) fp16 expected output

Bundle layout (per case):
  i64 num_grid
  i64 vis_hs
  i64 merge_size
  i64 num_images
  i64 grid_thw[num_images * 3]       (T, H, W per image)
  i64 idx_count                       (= sum over images of T*H*W)
  i32 idx00[idx_count]
  i32 idx01[idx_count]
  i32 idx10[idx_count]
  i32 idx11[idx_count]
  fp16 wt00[idx_count]
  fp16 wt01[idx_count]
  fp16 wt10[idx_count]
  fp16 wt11[idx_count]
"""

import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch

# Path setup so we can reuse Python project helpers
PROJ_DIR = Path(__file__).resolve().parent.parent.parent.parent / "atb_python_qwen3vl_embedding"
sys.path.insert(0, str(PROJ_DIR))

from engine_utils import fast_pos_embed_interpolate

MODEL_DIR = os.environ.get(
    "QWEN3VL_EMB_MODEL_DIR",
    "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B",
)

OUTPUT_DIR = "/tmp"


# ═══════════════════════════════════════════════════════════════════
# Stage A: pre-compute idx/wt on host. MUST match what C++ Stage A
# would produce, since C++ tests load these binaries directly.
# ═══════════════════════════════════════════════════════════════════

def build_indices_and_weights(grid_thw: torch.Tensor,
                              num_grid: int, merge_size: int):
    """Returns (idx00, idx01, idx10, idx11, wt00, wt01, wt10, wt11) — each
    flat 1-D tensor with the SAME row order as the final spatial-merge +
    temporal-repeat output. Length = sum over images of T*H*W.

    Algorithm:
      For each image (T, H, W):
        1. Compute bilinear interp tables (size H*W) for the (H, W) grid
           — idx into the (num_grid*num_grid) embedding table, plus weights.
        2. Apply spatial-merge shuffle to those (H*W) entries:
             view(H//ms, ms, W//ms, ms) -> permute(0,2,1,3) -> flatten
        3. Repeat the shuffled (H*W) block T times along the leading dim.
      Concat all images.
    """
    all_chunks = [[] for _ in range(8)]  # idx00..wt11

    for img in range(grid_thw.shape[0]):
        t_i = int(grid_thw[img, 0])
        h_i = int(grid_thw[img, 1])
        w_i = int(grid_thw[img, 2])

        h_idxs = torch.linspace(0, num_grid - 1, h_i)
        w_idxs = torch.linspace(0, num_grid - 1, w_i)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clamp(max=num_grid - 1)
        w_ceil = (w_floor + 1).clamp(max=num_grid - 1)

        dh = h_idxs - h_floor.float()
        dw = w_idxs - w_floor.float()

        base_h = h_floor * num_grid
        base_h_ceil = h_ceil * num_grid

        # (H, W) index grids
        idx00 = (base_h.unsqueeze(1) + w_floor.unsqueeze(0))      # (H, W)
        idx01 = (base_h.unsqueeze(1) + w_ceil.unsqueeze(0))
        idx10 = (base_h_ceil.unsqueeze(1) + w_floor.unsqueeze(0))
        idx11 = (base_h_ceil.unsqueeze(1) + w_ceil.unsqueeze(0))

        wt00 = (1 - dh).unsqueeze(1) * (1 - dw).unsqueeze(0)       # (H, W)
        wt01 = (1 - dh).unsqueeze(1) * dw.unsqueeze(0)
        wt10 = dh.unsqueeze(1) * (1 - dw).unsqueeze(0)
        wt11 = dh.unsqueeze(1) * dw.unsqueeze(0)

        # Apply spatial-merge shuffle: (H, W) -> (H//ms, ms, W//ms, ms)
        # -> permute(0, 2, 1, 3) -> flatten to (H*W,)
        ms = merge_size
        def shuffle(t):
            return (t.view(h_i // ms, ms, w_i // ms, ms)
                     .permute(0, 2, 1, 3)
                     .reshape(-1))

        idx00 = shuffle(idx00).int()
        idx01 = shuffle(idx01).int()
        idx10 = shuffle(idx10).int()
        idx11 = shuffle(idx11).int()
        wt00 = shuffle(wt00)
        wt01 = shuffle(wt01)
        wt10 = shuffle(wt10)
        wt11 = shuffle(wt11)

        # Repeat T times
        if t_i > 1:
            idx00 = idx00.repeat(t_i)
            idx01 = idx01.repeat(t_i)
            idx10 = idx10.repeat(t_i)
            idx11 = idx11.repeat(t_i)
            wt00 = wt00.repeat(t_i)
            wt01 = wt01.repeat(t_i)
            wt10 = wt10.repeat(t_i)
            wt11 = wt11.repeat(t_i)

        all_chunks[0].append(idx00)
        all_chunks[1].append(idx01)
        all_chunks[2].append(idx10)
        all_chunks[3].append(idx11)
        all_chunks[4].append(wt00)
        all_chunks[5].append(wt01)
        all_chunks[6].append(wt10)
        all_chunks[7].append(wt11)

    return tuple(torch.cat(c) for c in all_chunks)


def stage_b_reference(pos_embed_w: torch.Tensor, idx_tensors, wt_tensors,
                      dtype=torch.float16) -> torch.Tensor:
    """Pure-Python emulation of the NPU Stage B graph: 4× Gather + 4× weighted Add."""
    wt_tensors = [w.to(dtype) for w in wt_tensors]
    parts = [pos_embed_w[idx] * wt[:, None] for idx, wt in zip(idx_tensors, wt_tensors)]
    return parts[0] + parts[1] + parts[2] + parts[3]


# ═══════════════════════════════════════════════════════════════════
# Binary writers
# ═══════════════════════════════════════════════════════════════════

def write_fp16(path: str, data: np.ndarray):
    ndim = np.int64(data.ndim)
    shape = np.array(data.shape, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(ndim.tobytes())
        f.write(shape.tobytes())
        f.write(data.astype(np.float16).tobytes())


def write_bundle(path: str, num_grid, vis_hs, merge_size, grid_thw,
                 idx_arrays, wt_arrays, expected_fp16):
    grid_thw_arr = np.array(grid_thw, dtype=np.int64).reshape(-1)
    num_images = grid_thw_arr.size // 3
    idx_count = idx_arrays[0].shape[0]

    with open(path, "wb") as f:
        f.write(np.int64(num_grid).tobytes())
        f.write(np.int64(vis_hs).tobytes())
        f.write(np.int64(merge_size).tobytes())
        f.write(np.int64(num_images).tobytes())
        f.write(grid_thw_arr.tobytes())
        f.write(np.int64(idx_count).tobytes())
        for arr in idx_arrays:
            f.write(arr.astype(np.int32).tobytes())
        for arr in wt_arrays:
            f.write(arr.astype(np.float16).tobytes())
        # expected output for self-contained verification
        exp_shape = np.array(expected_fp16.shape, dtype=np.int64)
        f.write(np.int64(exp_shape.size).tobytes())
        f.write(exp_shape.tobytes())
        f.write(expected_fp16.astype(np.float16).tobytes())


# ═══════════════════════════════════════════════════════════════════
# Case generators
# ═══════════════════════════════════════════════════════════════════

def load_pos_embed_table():
    """Load the real pos_embed weights from the checkpoint, fp16."""
    from safetensors import safe_open
    safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        w = f.get_tensor("model.visual.pos_embed.weight")
    return w.to(torch.float16)


def make_case(tag: str, grid_thw, pos_embed_w, num_grid: int,
              merge_size: int = 2):
    print(f"[gen] case {tag}: grid_thw={grid_thw.tolist()}")

    # Stage A: indices/weights
    idx00, idx01, idx10, idx11, wt00, wt01, wt10, wt11 = \
        build_indices_and_weights(grid_thw, num_grid, merge_size)

    # Stage B reference (fp16 path matches NPU)
    expected = stage_b_reference(
        pos_embed_w,
        (idx00.long(), idx01.long(), idx10.long(), idx11.long()),
        (wt00, wt01, wt10, wt11),
        dtype=torch.float16,
    )

    # Also verify against the canonical Python implementation.
    canonical = fast_pos_embed_interpolate(
        grid_thw, pos_embed_w, num_grid, merge_size).to(torch.float16)
    # Both should be elementwise equal (they ARE the same math).
    mismatch = (canonical != expected).sum().item()
    cos = torch.nn.functional.cosine_similarity(
        canonical.float().flatten().unsqueeze(0),
        expected.float().flatten().unsqueeze(0)).item()
    print(f"    canonical vs reference: mismatch={mismatch} cosine={cos:.6f}")
    if cos < 0.9999:
        raise RuntimeError(
            f"Stage A/B decomposition does not match canonical fast_pos_embed_interpolate "
            f"(cosine={cos:.6f}). Bug in build_indices_and_weights.")

    # Write bundle
    bundle_path = f"{OUTPUT_DIR}/posembed_npu_case_{tag}.bin"
    vis_hs = int(pos_embed_w.shape[1])
    write_bundle(
        bundle_path, num_grid, vis_hs, merge_size, grid_thw.numpy(),
        idx_arrays=(idx00.numpy(), idx01.numpy(), idx10.numpy(), idx11.numpy()),
        wt_arrays=(wt00.numpy(), wt01.numpy(), wt10.numpy(), wt11.numpy()),
        expected_fp16=expected.numpy(),
    )
    print(f"    -> {bundle_path}  bundle_size={Path(bundle_path).stat().st_size} bytes")

    # Also write expected separately for easy debugging
    write_fp16(f"{OUTPUT_DIR}/posembed_npu_expected_{tag}.bin", expected.numpy())


def main():
    print(f"Loading pos_embed weights from {MODEL_DIR}")
    pos_embed_w = load_pos_embed_table()      # (num_grid*num_grid, vis_hs) fp16
    num_pos, vis_hs = pos_embed_w.shape
    num_grid = int(np.sqrt(num_pos))
    assert num_grid * num_grid == num_pos, \
        f"pos_embed has {num_pos} rows; expected a perfect square."
    print(f"  num_grid={num_grid}  vis_hs={vis_hs}")

    # Write the weight table for the C++ test to load.
    write_fp16(f"{OUTPUT_DIR}/posembed_npu_pos_embed_w.bin",
               pos_embed_w.numpy())
    print(f"  -> {OUTPUT_DIR}/posembed_npu_pos_embed_w.bin  shape={tuple(pos_embed_w.shape)}")

    # ── Cases that mirror our benchmark resolutions ──
    cases = [
        ("224x224",   torch.tensor([[2, 16, 16]],  dtype=torch.long)),  # 224/14=16
        ("416x672",   torch.tensor([[2, 26, 42]],  dtype=torch.long)),  # 416/14=29.7→26, 672/14=48→42... real impl rounds
        ("896x896",   torch.tensor([[2, 56, 56]],  dtype=torch.long)),  # actual benchmark grid
        ("tiny_4x4",  torch.tensor([[1,  4,  4]],  dtype=torch.long)),  # minimum 1-frame, 4x4 grid
        ("t1_8x8",    torch.tensor([[1,  8,  8]],  dtype=torch.long)),  # single frame
        # Note: multi-image not currently used by the engine; skip for now.
    ]
    for tag, gthw in cases:
        make_case(tag, gthw, pos_embed_w, num_grid, merge_size=2)

    print("\nAll reference bundles generated. Run C++ tests:")
    print("  ./build/test_pos_embed_npu")


if __name__ == "__main__":
    main()
