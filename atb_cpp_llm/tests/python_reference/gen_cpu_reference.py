"""
Generate binary reference data for C++ Level 1 CPU pure-function tests.

Each test stage writes a .bin file with known inputs and expected outputs.
C++ tests read these files and validate exact/cosine agreement.

Usage:
    python tests/gen_cpu_reference.py              # all stages
    python tests/gen_cpu_reference.py --stage mrope  # single stage

Output files (in /tmp/):
    cpu_mrope_pid_simple.bin       — GetRopeIndex: [vision_start + 4 img tokens] case
    cpu_mrope_pid_no_img.bin       — GetRopeIndex: text-only sequential fallback
    cpu_mrope_pid_image_text.bin   — GetRopeIndex: text + image + text case
    cpu_mrope_cos_sin.bin          — MRoPE::Compute: cos/sin for position_ids [3, 1, 8]
    cpu_vision_rope_cos_sin.bin    — VisionRotaryEmbedding: cos/sin for grid [1, 4, 6]
    cpu_smart_resize.bin           — SmartResize: height, width pairs (in/out)
"""

import os
import sys
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Add the Python project to path ────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent.parent.parent.parent / "atb_python_qwen3vl_embedding"
sys.path.insert(0, str(PROJ_DIR))

from engine_utils import (
    get_rope_index,
    TextRotaryEmbedding,
    VisionRotaryEmbedding,
    compute_rot_pos_emb,
    fast_pos_embed_interpolate,
)

# ── Model dir (can be overridden) ─────────────────────────────────
MODEL_DIR = os.environ.get(
    "QWEN3VL_EMB_MODEL_DIR",
    "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B",
)

OUTPUT_DIR = "/tmp"


# ═══════════════════════════════════════════════════════════════════
# Binary format helpers
# ═══════════════════════════════════════════════════════════════════

def write_f32(path: str, data: np.ndarray):
    """Write float32 array as: [ndim: int64] [shape: int64[ndim]] [data: f32]."""
    ndim = np.int64(data.ndim)
    shape = np.array(data.shape, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(ndim.tobytes())
        f.write(shape.tobytes())
        f.write(data.astype(np.float32).tobytes())


def write_i64(path: str, data: np.ndarray):
    """Write int64 array as: [ndim: int64] [shape: int64[ndim]] [data: i64]."""
    ndim = np.int64(data.ndim)
    shape = np.array(data.shape, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(ndim.tobytes())
        f.write(shape.tobytes())
        f.write(data.astype(np.int64).tobytes())


def write_fp16(path: str, data: np.ndarray):
    """Write float16 array as: [ndim: int64] [shape: int64[ndim]] [data: fp16 bytes]."""
    ndim = np.int64(data.ndim)
    shape = np.array(data.shape, dtype=np.int64)
    with open(path, "wb") as f:
        f.write(ndim.tobytes())
        f.write(shape.tobytes())
        f.write(data.astype(np.float16).tobytes())


def write_bf16(path: str, data: np.ndarray):
    """Write bf16 array as: [ndim: int64] [shape: int64[ndim]] [data: uint16 bytes].
    bf16 = upper 16 bits of fp32. We round-to-nearest-even via numpy's view.
    """
    ndim = np.int64(data.ndim)
    shape = np.array(data.shape, dtype=np.int64)
    f32 = data.astype(np.float32)
    # Round-to-nearest-even by adding 0x7FFF + LSB-of-upper before truncating.
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    rounding_bias = np.uint32(0x7FFF) + lsb
    u32 = u32 + rounding_bias
    bf16_bits = (u32 >> 16).astype(np.uint16)
    with open(path, "wb") as f:
        f.write(ndim.tobytes())
        f.write(shape.tobytes())
        f.write(bf16_bits.tobytes())


def _bf16_roundtrip(arr: np.ndarray) -> np.ndarray:
    """Round-trip a float32 array through bf16 (mirrors NPU bf16 storage)."""
    f32 = arr.astype(np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    rounding_bias = np.uint32(0x7FFF) + lsb
    u32 = u32 + rounding_bias
    bf16_bits = (u32 >> 16).astype(np.uint16)
    # Decode back: shift up to fp32
    u32_back = (bf16_bits.astype(np.uint32) << 16)
    return u32_back.view(np.float32).reshape(arr.shape)


def write_i32s(path: str, values: list):
    """Write a list of int32 values as: [count: int64] [data: i32]."""
    arr = np.array(values, dtype=np.int32)
    count = np.int64(len(arr))
    with open(path, "wb") as f:
        f.write(count.tobytes())
        f.write(arr.tobytes())


# ═══════════════════════════════════════════════════════════════════
# Stage: GetRopeIndex — position_ids
# ═══════════════════════════════════════════════════════════════════

def gen_get_rope_index_simple():
    """vision_start_token + image_tokens only (no text)."""
    print("[gen] GetRopeIndex — simple image-only case")

    VISION_START = 151652
    IMAGE_TOKEN = 151655
    MERGE_SIZE = 2

    # Input: [vision_start, img, img, img, img, img, img, img, img]
    # grid_thw: [1, 4, 4] -> after merge: h=2, w=2, tokens=4
    input_ids = torch.tensor([[VISION_START, IMAGE_TOKEN, IMAGE_TOKEN,
                                IMAGE_TOKEN, IMAGE_TOKEN]], dtype=torch.long)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    position_ids, _ = get_rope_index(
        input_ids, grid_thw, None, None,
        image_token_id=IMAGE_TOKEN,
        vision_start_token_id=VISION_START,
        spatial_merge_size=MERGE_SIZE,
    )
    # position_ids: (3, 1, S)
    write_i64(f"{OUTPUT_DIR}/cpu_mrope_pid_simple.bin", position_ids.numpy())
    print(f"  → {OUTPUT_DIR}/cpu_mrope_pid_simple.bin  shape={position_ids.shape}")


def gen_get_rope_index_no_image():
    """Text-only input — sequential positions."""
    print("[gen] GetRopeIndex — text-only sequential case")

    input_ids = torch.tensor([[151643, 15339, 1879, 119, 258]], dtype=torch.long)

    position_ids, _ = get_rope_index(
        input_ids, None, None, None,
        image_token_id=151655,
        vision_start_token_id=151652,
        spatial_merge_size=2,
    )
    write_i64(f"{OUTPUT_DIR}/cpu_mrope_pid_no_img.bin", position_ids.numpy())
    print(f"  → {OUTPUT_DIR}/cpu_mrope_pid_no_img.bin  shape={position_ids.shape}")


def gen_get_rope_index_image_text():
    """Text + image + trailing text case."""
    print("[gen] GetRopeIndex — image+text mixed case")

    VISION_START = 151652
    IMAGE_TOKEN = 151655

    # Describe [image: 2x2 grid -> 4 tokens] the image.
    input_ids = torch.tensor([[
        151643,                      # <|im_start|>
        VISION_START,                # vision_start
        IMAGE_TOKEN, IMAGE_TOKEN,    # 2 image tokens (merged grid 2x1)
        IMAGE_TOKEN, IMAGE_TOKEN,    # 2 more image tokens
        15339, 1879,                 # "the image" text
    ]], dtype=torch.long)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)  # 4x4 → 2x2 merged = 4 tokens

    position_ids, _ = get_rope_index(
        input_ids, grid_thw, None, None,
        image_token_id=IMAGE_TOKEN,
        vision_start_token_id=VISION_START,
        spatial_merge_size=2,
    )
    write_i64(f"{OUTPUT_DIR}/cpu_mrope_pid_image_text.bin", position_ids.numpy())
    print(f"  → {OUTPUT_DIR}/cpu_mrope_pid_image_text.bin  shape={position_ids.shape}")


# ═══════════════════════════════════════════════════════════════════
# Stage: MRoPE::Compute — cos/sin
# ═══════════════════════════════════════════════════════════════════

def gen_mrope_cos_sin():
    """Compute MRoPE cos/sin from known position_ids."""
    print("[gen] MRoPE::Compute — cos/sin for position_ids [3, 1, 8]")

    rope = TextRotaryEmbedding(head_dim=128, rope_theta=5000000.0,
                               mrope_section=(24, 20, 20))

    # Use a mix of sequential and grid positions
    position_ids = torch.tensor([
        # T dim:  0  0  1  2  3  4  5  6
        [0, 0, 1, 2, 3, 4, 5, 6],
        # H dim:  0  0  0  0  1  1  1  1
        [0, 0, 0, 0, 1, 1, 1, 1],
        # W dim:  0  0  1  2  0  1  2  3
        [0, 0, 1, 2, 0, 1, 2, 3],
    ], dtype=torch.long).unsqueeze(1)  # (3, 1, 8)

    cos, sin = rope(position_ids)
    # cos, sin: (1, 8, 128)
    cos_np = cos.numpy()   # (1, 8, 128)
    sin_np = sin.numpy()   # (1, 8, 128)

    write_f32(f"{OUTPUT_DIR}/cpu_mrope_cos.bin", cos_np)
    write_f32(f"{OUTPUT_DIR}/cpu_mrope_sin.bin", sin_np)
    # Keep 3D shape: (3, 1, 8) for B=1
    write_i64(f"{OUTPUT_DIR}/cpu_mrope_pid.bin", position_ids.numpy())
    print(f"  → {OUTPUT_DIR}/cpu_mrope_cos.bin  shape={cos_np.shape}")
    print(f"  → {OUTPUT_DIR}/cpu_mrope_sin.bin  shape={sin_np.shape}")


# ═══════════════════════════════════════════════════════════════════
# Stage: VisionRotaryEmbedding — cos/sin
# ═══════════════════════════════════════════════════════════════════

def gen_vision_rope():
    """Compute vision RoPE cos/sin for grid [1, 4, 6]."""
    print("[gen] VisionRotaryEmbedding — cos/sin for grid [1, 4, 6] (merge=2)")

    head_dim = 64  # 128 hidden // 2
    vis_rope = VisionRotaryEmbedding(dim=head_dim // 2)  # dim=32

    grid_thw = torch.tensor([[1, 4, 6]], dtype=torch.long)
    merge_size = 2

    rope = compute_rot_pos_emb(grid_thw, vis_rope, merge_size)
    # rope shape: (total_tokens, dim*2) = (24, 64)
    # This is the raw rope (freqs), NOT cos/sin yet
    emb = torch.cat((rope, rope), dim=-1)  # (24, 128)
    cos = emb.cos()
    sin = emb.sin()

    write_f32(f"{OUTPUT_DIR}/cpu_vision_rope_cos.bin", cos.numpy())
    write_f32(f"{OUTPUT_DIR}/cpu_vision_rope_sin.bin", sin.numpy())
    print(f"  → {OUTPUT_DIR}/cpu_vision_rope_cos.bin  shape={cos.shape}")
    print(f"  → {OUTPUT_DIR}/cpu_vision_rope_sin.bin  shape={sin.shape}")

    # Also save freq_table for testing ComputeFreqTable separately
    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = vis_rope(max_hw)
    write_f32(f"{OUTPUT_DIR}/cpu_vision_freq_table.bin", freq_table.numpy())
    print(f"  → {OUTPUT_DIR}/cpu_vision_freq_table.bin  shape={freq_table.shape}")


# ═══════════════════════════════════════════════════════════════════
# Stage: ComputePosEmbedInterp — position embeddings
# ═══════════════════════════════════════════════════════════════════

def gen_pos_embed_interp():
    """Compute pos_embed interpolation for multiple test cases.

    Test 1: typical case grid [1, 8, 12], merge=2 — uses real model weights
    Test 2: small/boundary case grid [1, 2, 2], merge=2 — uses real model weights
    Test 3: multi-image grid [[1,4,4], [1,6,8]], merge=2 — uses real model weights

    To match the C++ pipeline (which takes fp16 source and converts via Fp16→fp32→
    accumulate→Fp32→fp16), we run the Python reference with the *same* fp16 source
    upcast to fp32 for the gather/accumulate, then cast the result back to fp16.
    This keeps the C++ test fair (no fp32 vs fp16 weight precision drift).
    """
    print("[gen] ComputePosEmbedInterp — multiple cases")

    # Must load real pos_embed weights from model
    model_dir = MODEL_DIR
    if not os.path.isdir(model_dir):
        print(f"  SKIP: model dir not found: {model_dir}")
        return

    import safetensors.torch
    import json

    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)

    vis_cfg = cfg.get("vision_config", {})
    num_pos = vis_cfg.get("num_position_embeddings", 2304)
    num_grid = int(np.sqrt(num_pos))

    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors", device="cpu")
    pe_candidates = [k for k in sd.keys()
                     if "position_embed" in k.lower() or "pos_embed" in k.lower()]
    if not pe_candidates:
        print(f"  SKIP: no position_embed key found in model")
        return

    pe_key = pe_candidates[0]
    # Cast bf16 → fp16, then upcast to fp32 for the Python interpolation.
    # The C++ test feeds the same fp16 source.
    pe_fp16 = sd[pe_key].to(torch.float16)             # (num_grid^2, hidden)
    pe_for_interp = pe_fp16.to(torch.float32)          # fp32 view of fp16 values

    # Save shared source once
    write_fp16(f"{OUTPUT_DIR}/cpu_pos_embed_src.bin", pe_fp16.numpy())
    print(f"  → {OUTPUT_DIR}/cpu_pos_embed_src.bin  shape={pe_fp16.shape}  num_grid={num_grid}")

    merge_size = 2

    def _run_case(name: str, grid_thw_list: list):
        grid_thw = torch.tensor(grid_thw_list, dtype=torch.long)
        pos_ref_f32 = fast_pos_embed_interpolate(
            grid_thw, pe_for_interp, num_grid, merge_size)
        # C++ writes fp16 result; round-trip through fp16 for fair comparison.
        pos_ref_fp16_roundtrip = pos_ref_f32.to(torch.float16).to(torch.float32)
        write_f32(f"{OUTPUT_DIR}/cpu_pos_embed_ref_{name}.bin",
                  pos_ref_fp16_roundtrip.numpy())
        write_i64(f"{OUTPUT_DIR}/cpu_pos_embed_grid_{name}.bin", grid_thw.numpy())
        print(f"  → {OUTPUT_DIR}/cpu_pos_embed_ref_{name}.bin "
              f"shape={pos_ref_f32.shape}  grid={grid_thw_list}")

    # Case 1: typical 720x1280-equivalent grid
    _run_case("typical", [[1, 8, 12]])
    # Case 2: small/boundary grid (single merged block)
    _run_case("small", [[1, 2, 2]])
    # Case 3: multi-image
    _run_case("multi", [[1, 4, 4], [1, 6, 8]])

    # Back-compat: also write the un-suffixed files used by older consumers.
    grid_typical = torch.tensor([[1, 8, 12]], dtype=torch.long)
    pos_ref_f32 = fast_pos_embed_interpolate(
        grid_typical, pe_for_interp, num_grid, merge_size)
    write_f32(f"{OUTPUT_DIR}/cpu_pos_embed_ref.bin", pos_ref_f32.numpy())
    write_i64(f"{OUTPUT_DIR}/cpu_pos_embed_grid.bin", grid_typical.numpy())


# ═══════════════════════════════════════════════════════════════════
# Stage: SmartResize — dimension computation
# ═══════════════════════════════════════════════════════════════════

def gen_smart_resize():
    """Generate SmartResize test cases (input h,w → output h,w).

    Uses the ACTUAL preprocess.smart_resize from the Python project
    to avoid any reimplementation errors.
    """
    print("[gen] SmartResize — dimension computation")

    # Import the REAL implementation — no reimplementation risk
    from preprocess import smart_resize

    factor = 32  # patch_size * merge_size = 16 * 2
    min_px = 4096
    max_px = 1310720

    # Cases: (name, in_h, in_w)
    cases = [
        ("720x1280",    720, 1280),
        ("1080x1920",  1080, 1920),
        ("64x64",        64,   64),
        ("200x120",     200,  120),
        ("3000x4000",  3000, 4000),  # exceed max_pixels
        ("4x4",           4,    4),  # below min_pixels
    ]

    results = []
    for name, h, w in cases:
        nh, nw = smart_resize(h, w, factor=factor,
                              min_pixels=min_px, max_pixels=max_px)
        print(f"  {name}: ({h},{w}) → ({nh},{nw})")
        results.extend([h, w, nh, nw])

    write_i32s(f"{OUTPUT_DIR}/cpu_smart_resize.bin", results)
    print(f"  → {OUTPUT_DIR}/cpu_smart_resize.bin  {len(cases)} cases")


# ═══════════════════════════════════════════════════════════════════
# Stage: float_utils — bf16/fp16/fp32 conversion ground truth
# ═══════════════════════════════════════════════════════════════════

def gen_float_utils():
    """Generate bf16/fp16 conversion ground truth.

    Output binary format (little-endian):
        [count: int64]
        [bf16_bits[count]: uint16]
        [fp16_expected_from_bf16[count]: uint16]
        [fp32_values[count]: float32]   — original float32 source values
        [fp16_expected_from_fp32[count]: uint16]
        [fp32_roundtrip_from_fp16[count]: float32]

    The bf16→fp16 path: bf16 raw bits → reinterpret as upper-16 of fp32 →
    cast down to fp16 (round-to-nearest-even). This matches the CANN
    aclFloatToFloat16 behavior. Historical bug: the old C++ path used
    bit truncation which zeroed low fp16 mantissa bits.
    """
    print("[gen] float_utils — bf16/fp16 conversion ground truth")

    # Include "interesting" values — those that don't represent exactly in
    # bf16 or fp16, exposing rounding-bias bugs.
    test_values = np.array([
        0.0, -0.0,
        1.0, -1.0,
        0.5, -0.5,
        0.1, 0.3, 0.123456789,
        100.0, -100.0,
        0.0001, 1e-5,
        2.0**16, 65504.0,                # near fp16 max
        6.103515625e-05,                 # smallest normal fp16
        5.960464477539063e-08,           # smallest subnormal fp16
        3.14159265, -2.71828,
    ], dtype=np.float32)

    # bf16: take upper 16 bits of fp32 (truncate, but for these constants
    # numpy promotes via float, so we explicitly round-down by bit shifting).
    # Use ml_dtypes if available for exact RTE bf16, else truncate.
    try:
        import ml_dtypes
        bf16_arr = test_values.astype(ml_dtypes.bfloat16)
        bf16_bits = bf16_arr.view(np.uint16)
        # Recover the float32 value as represented by bf16 (lower 16 bits zero)
        bf16_as_f32 = bf16_arr.astype(np.float32)
    except ImportError:
        # Fall back to bit-level truncation
        bf16_bits = (test_values.view(np.uint32) >> 16).astype(np.uint16)
        f32_recon = (bf16_bits.astype(np.uint32) << 16).view(np.float32)
        bf16_as_f32 = f32_recon

    # bf16 → fp16: take the reconstructed fp32 value, then cast to fp16 (RTE)
    fp16_from_bf16 = bf16_as_f32.astype(np.float16).view(np.uint16)

    # Direct fp32 → fp16
    fp16_from_fp32 = test_values.astype(np.float16).view(np.uint16)

    # fp16 → fp32 roundtrip (for Fp16ToF32 verification)
    fp16_back_to_f32 = test_values.astype(np.float16).astype(np.float32)

    count = np.int64(len(test_values))
    path = f"{OUTPUT_DIR}/cpu_float_utils.bin"
    with open(path, "wb") as f:
        f.write(count.tobytes())
        f.write(bf16_bits.tobytes())
        f.write(fp16_from_bf16.tobytes())
        f.write(test_values.astype(np.float32).tobytes())
        f.write(fp16_from_fp32.tobytes())
        f.write(fp16_back_to_f32.astype(np.float32).tobytes())
    print(f"  → {path}  count={int(count)}")
    print(f"     bf16_bits sample: {bf16_bits[:4]}")
    print(f"     fp16_from_bf16 sample: {fp16_from_bf16[:4]}")
    print(f"     fp16_from_fp32 sample: {fp16_from_fp32[:4]}")


# ═══════════════════════════════════════════════════════════════════
# Stage: RmsNorm — fp32 ground truth for 3 LayerType variants
# ═══════════════════════════════════════════════════════════════════

def gen_rms_norm():
    """Generate fp32 reference for RmsNorm precision tests.

    Three cases:
      * "typical"  — Qwen3VL hidden=2048, seq=16
      * "small"    — seq=4, hidden=64 for debug
      * "prenorm"  — same shape as small, but the test exercises PRENORM/POSTNORM
                     variants which share the same NORM math for the y output.

    File layout (per case): one bin for x (fp32), one for w (fp32),
    one for expected output (fp32).
    """
    print("[gen] RmsNorm — fp32 reference")

    eps = 1e-6

    def _make(name: str, S: int, H: int):
        torch.manual_seed(42 + S + H)
        x = torch.randn(S, H, dtype=torch.float32)
        w = (torch.rand(H, dtype=torch.float32) - 0.5) * 0.4 + 1.0
        # RMSNorm: x / sqrt(mean(x^2) + eps) * w
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        out = norm * w
        write_f32(f"{OUTPUT_DIR}/cpu_op_rms_norm_{name}_input.bin",  x.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_rms_norm_{name}_weight.bin", w.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_rms_norm_{name}_output.bin", out.numpy())
        print(f"  → {name}: x{tuple(x.shape)} w{tuple(w.shape)} out{tuple(out.shape)}")

    _make("typical", 16,  2048)
    _make("small",    4,    64)
    _make("medium",   8,   256)


# ═══════════════════════════════════════════════════════════════════
# Stage: LayerNorm — fp32 ground truth (axis variants)
# ═══════════════════════════════════════════════════════════════════

def gen_layer_norm():
    """Generate fp32 reference for LayerNorm precision tests.

    Cases:
      * "typical"      — vision hidden=1280, seq=16,   axis=-1 (per-token norm)
      * "small"        — seq=4, hidden=64,             axis=-1
      * "axis_minus1"  — explicit axis=-1 case (mathematically identical to
                         axis=1 for 2D tensors), included to document the
                         parameter. The reference is the same per-row norm.
    """
    print("[gen] LayerNorm — fp32 reference")

    eps = 1e-5

    def _make(name: str, S: int, H: int):
        torch.manual_seed(123 + S + H)
        x  = torch.randn(S, H, dtype=torch.float32)
        g  = (torch.rand(H, dtype=torch.float32) - 0.5) * 0.4 + 1.0
        b  = (torch.rand(H, dtype=torch.float32) - 0.5) * 0.2
        # PyTorch reference: F.layer_norm normalizes over the last `len(shape)` dims.
        out = F.layer_norm(x, [H], weight=g, bias=b, eps=eps)
        write_f32(f"{OUTPUT_DIR}/cpu_op_layer_norm_{name}_input.bin",  x.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_layer_norm_{name}_gamma.bin",  g.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_layer_norm_{name}_beta.bin",   b.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_layer_norm_{name}_output.bin", out.numpy())
        print(f"  → {name}: x{tuple(x.shape)} g{tuple(g.shape)} b{tuple(b.shape)} out{tuple(out.shape)}")

    _make("typical",     16, 1280)
    _make("small",        4,   64)
    _make("axis_minus1",  4,   64)


# ═══════════════════════════════════════════════════════════════════
# Stage: Linear — fp32 ground truth (has_bias / transpose variants)
# ═══════════════════════════════════════════════════════════════════

def gen_linear():
    """Generate fp32 reference for Linear precision tests.

    The ATB Linear default is transpose_b=True (weight shape [n, k]), so the
    Python reference uses `x @ W.T` to match. The "no_transpose" case writes
    a weight in [k, n] layout and uses `x @ W`.

    Cases:
      * "typical"       — seq=16, in=2048, out=2048, no bias, transpose_b=True
      * "small_bias"    — seq=4, in=64, out=32, has_bias=True, transpose_b=True
      * "no_transpose"  — seq=4, in=64, out=32, no bias, transpose_b=False
    """
    print("[gen] Linear — fp32 reference")

    def _make_tb(name: str, S: int, K: int, N: int, has_bias: bool):
        torch.manual_seed(7 + S + K + N)
        x = torch.randn(S, K, dtype=torch.float32) * 0.1
        w = torch.randn(N, K, dtype=torch.float32) * 0.1  # [N, K], transpose_b=True
        out = x @ w.T
        write_f32(f"{OUTPUT_DIR}/cpu_op_linear_{name}_input.bin",  x.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_linear_{name}_weight.bin", w.numpy())
        if has_bias:
            b = torch.randn(N, dtype=torch.float32) * 0.05
            out = out + b
            write_f32(f"{OUTPUT_DIR}/cpu_op_linear_{name}_bias.bin", b.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_linear_{name}_output.bin", out.numpy())
        print(f"  → {name}: x{tuple(x.shape)} w{tuple(w.shape)} "
              f"{'+bias' if has_bias else ''} out{tuple(out.shape)}")

    def _make_no_tb(name: str, S: int, K: int, N: int):
        torch.manual_seed(99 + S + K + N)
        x = torch.randn(S, K, dtype=torch.float32) * 0.1
        w = torch.randn(K, N, dtype=torch.float32) * 0.1  # [K, N], transpose_b=False
        out = x @ w
        write_f32(f"{OUTPUT_DIR}/cpu_op_linear_{name}_input.bin",  x.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_linear_{name}_weight.bin", w.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_linear_{name}_output.bin", out.numpy())
        print(f"  → {name}: x{tuple(x.shape)} w{tuple(w.shape)} (no transpose) "
              f"out{tuple(out.shape)}")

    _make_tb("typical",    16, 2048, 2048, False)
    _make_tb("small_bias",  4,   64,   32, True)
    _make_no_tb("no_transpose", 4, 64, 32)


# ═══════════════════════════════════════════════════════════════════
# Stage: Activation — fp32 ground truth (SiLU / GELU / FastGELU)
# ═══════════════════════════════════════════════════════════════════

def gen_activation():
    """Generate fp32 reference for ActivationOp precision tests.

    Three cases (one per activation):
      * "silu"      — x * sigmoid(x)
      * "gelu"      — GELU (PyTorch's "tanh" approximation, matching ATB)
      * "fast_gelu" — x * sigmoid(1.702 * x)  (the formula ATB FastGELU uses)

    Each writes a typical-size (16, 2048) and a small-size (4, 64) input/output.
    """
    print("[gen] Activation — fp32 reference (SiLU / GELU / FastGELU)")

    def _save_case(activation: str, name: str, S: int, H: int):
        torch.manual_seed(11 + S + H + hash(activation) % 1000)
        x = torch.randn(S, H, dtype=torch.float32)
        if activation == "silu":
            out = F.silu(x)
        elif activation == "gelu":
            # ATB GELU is the tanh approximation
            out = F.gelu(x, approximate="tanh")
        elif activation == "fast_gelu":
            out = x * torch.sigmoid(1.702 * x)
        else:
            raise ValueError(activation)
        write_f32(f"{OUTPUT_DIR}/cpu_op_activation_{activation}_{name}_input.bin",  x.numpy())
        write_f32(f"{OUTPUT_DIR}/cpu_op_activation_{activation}_{name}_output.bin", out.numpy())
        print(f"  → {activation}/{name}: x{tuple(x.shape)} out{tuple(out.shape)}")

    for act in ("silu", "gelu", "fast_gelu"):
        _save_case(act, "typical", 16, 2048)
        _save_case(act, "small",    4,   64)


# ═══════════════════════════════════════════════════════════════════
# Stage: Level-2 ATB op references (Elewise / Split / Concat / Softmax /
#        Gather / Reduce / Transpose / SetValue)
#
# All inputs are stored as fp16 binaries (write_fp16) so the C++ side can
# upload them verbatim to the NPU. The reference outputs are also stored
# in fp16 (after the Python ground-truth is round-tripped through fp16)
# so the comparison is fair against fp16 ATB execution.
# ═══════════════════════════════════════════════════════════════════

def _fp16_roundtrip(arr: np.ndarray) -> np.ndarray:
    """Round-trip a float32 array through fp16 (mirrors NPU fp16 storage)."""
    return arr.astype(np.float16).astype(np.float32)


def gen_op_elewise():
    """Reference data for ElewiseOp Add/Mul/Muls/Sub/Cast."""
    print("[gen] op_elewise — Add/Mul/Muls/Sub/Cast")
    rng = np.random.default_rng(42)

    # Add — two fp16 [4, 8]
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    a16 = _fp16_roundtrip(a); b16 = _fp16_roundtrip(b)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_add_a.bin", a16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_add_b.bin", b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_add_ref.bin", _fp16_roundtrip(a16 + b16))

    # Mul — two fp16 [3, 5]
    am = rng.standard_normal((3, 5)).astype(np.float32)
    bm = rng.standard_normal((3, 5)).astype(np.float32)
    am16 = _fp16_roundtrip(am); bm16 = _fp16_roundtrip(bm)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_mul_a.bin", am16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_mul_b.bin", bm16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_mul_ref.bin", _fp16_roundtrip(am16 * bm16))

    # Muls — scale 2.5
    scale = 2.5
    s_in = rng.standard_normal((6,)).astype(np.float32)
    s_in16 = _fp16_roundtrip(s_in)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_muls_a.bin", s_in16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_muls_ref.bin", _fp16_roundtrip(s_in16 * scale))

    # Sub — two fp16 [2, 4]
    sa = rng.standard_normal((2, 4)).astype(np.float32)
    sb = rng.standard_normal((2, 4)).astype(np.float32)
    sa16 = _fp16_roundtrip(sa); sb16 = _fp16_roundtrip(sb)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_sub_a.bin", sa16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_sub_b.bin", sb16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_sub_ref.bin", _fp16_roundtrip(sa16 - sb16))

    # Cast fp16→fp32 — values preserved exactly
    c_in = rng.standard_normal((4,)).astype(np.float32)
    c_in16 = _fp16_roundtrip(c_in)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_elewise_cast_a.bin", c_in16)
    write_f32(f"{OUTPUT_DIR}/cpu_op_elewise_cast_ref.bin", c_in16)

    print("  → wrote 5 elewise cases (Add/Mul/Muls/Sub/Cast)")


def gen_op_split_concat():
    """Reference data for SplitOp and ConcatOp."""
    print("[gen] op_split_concat — Split/Concat")
    rng = np.random.default_rng(7)

    # Split dim=-1 num=2 on [4, 8]
    s1 = rng.standard_normal((4, 8)).astype(np.float32)
    s1_16 = _fp16_roundtrip(s1)
    p0, p1 = np.split(s1_16, 2, axis=-1)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_split2_in.bin", s1_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_split2_p0.bin", _fp16_roundtrip(p0))
    write_fp16(f"{OUTPUT_DIR}/cpu_op_split2_p1.bin", _fp16_roundtrip(p1))

    # Split dim=-1 num=3 on [2, 9]
    s2 = rng.standard_normal((2, 9)).astype(np.float32)
    s2_16 = _fp16_roundtrip(s2)
    q0, q1, q2 = np.split(s2_16, 3, axis=-1)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_split3_in.bin", s2_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_split3_p0.bin", _fp16_roundtrip(q0))
    write_fp16(f"{OUTPUT_DIR}/cpu_op_split3_p1.bin", _fp16_roundtrip(q1))
    write_fp16(f"{OUTPUT_DIR}/cpu_op_split3_p2.bin", _fp16_roundtrip(q2))

    # Concat dim=0 of two [3, 4]
    ca = rng.standard_normal((3, 4)).astype(np.float32)
    cb = rng.standard_normal((3, 4)).astype(np.float32)
    ca16 = _fp16_roundtrip(ca); cb16 = _fp16_roundtrip(cb)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_concat0_a.bin", ca16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_concat0_b.bin", cb16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_concat0_ref.bin",
               _fp16_roundtrip(np.concatenate([ca16, cb16], axis=0)))

    # Concat dim=-1 of two [4, 3]
    da = rng.standard_normal((4, 3)).astype(np.float32)
    db = rng.standard_normal((4, 3)).astype(np.float32)
    da16 = _fp16_roundtrip(da); db16 = _fp16_roundtrip(db)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_concat1_a.bin", da16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_concat1_b.bin", db16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_concat1_ref.bin",
               _fp16_roundtrip(np.concatenate([da16, db16], axis=-1)))

    print("  → wrote split (2) + concat (2) cases")


def gen_op_softmax():
    """Reference data for SoftmaxOp."""
    print("[gen] op_softmax — 1D / 2D / numerical stability")
    rng = np.random.default_rng(11)

    # 1D softmax axis=-1 on [1, 16]
    s1 = rng.standard_normal((1, 16)).astype(np.float32)
    s1_16 = _fp16_roundtrip(s1)
    sm1 = F.softmax(torch.from_numpy(s1_16), dim=-1).numpy()
    write_fp16(f"{OUTPUT_DIR}/cpu_op_softmax1d_in.bin", s1_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_softmax1d_ref.bin", _fp16_roundtrip(sm1))

    # 2D softmax axis=-1 on [4, 8]
    s2 = rng.standard_normal((4, 8)).astype(np.float32)
    s2_16 = _fp16_roundtrip(s2)
    sm2 = F.softmax(torch.from_numpy(s2_16), dim=-1).numpy()
    write_fp16(f"{OUTPUT_DIR}/cpu_op_softmax2d_in.bin", s2_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_softmax2d_ref.bin", _fp16_roundtrip(sm2))

    # Numerical stability — large/small mix
    s3 = np.array([[10.0, 0.0, -5.0, 1.0, 2.0, 8.0, 9.0, -1.0]], dtype=np.float32)
    s3_16 = _fp16_roundtrip(s3)
    sm3 = F.softmax(torch.from_numpy(s3_16), dim=-1).numpy()
    write_fp16(f"{OUTPUT_DIR}/cpu_op_softmax_stab_in.bin", s3_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_softmax_stab_ref.bin", _fp16_roundtrip(sm3))

    print("  → wrote softmax 3 cases")


def gen_op_gather_reduce():
    """Reference data for GatherOp and ReduceOp."""
    print("[gen] op_gather_reduce")
    rng = np.random.default_rng(13)

    # Gather axis=0 on [8, 4] with indices [2, 5, 0]
    g_in = rng.standard_normal((8, 4)).astype(np.float32)
    g_in16 = _fp16_roundtrip(g_in)
    idx = np.array([2, 5, 0], dtype=np.int64)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_gather_in.bin", g_in16)
    write_i64(f"{OUTPUT_DIR}/cpu_op_gather_idx.bin", idx)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_gather_ref.bin", _fp16_roundtrip(g_in16[idx]))

    # Reduce MAX/MIN/SUM on [3, 5] axis=1
    r_in = rng.standard_normal((3, 5)).astype(np.float32)
    r_in16 = _fp16_roundtrip(r_in)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_reduce_in.bin", r_in16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_reduce_max_ref.bin",
               _fp16_roundtrip(np.max(r_in16, axis=1)))
    write_fp16(f"{OUTPUT_DIR}/cpu_op_reduce_min_ref.bin",
               _fp16_roundtrip(np.min(r_in16, axis=1)))
    write_fp16(f"{OUTPUT_DIR}/cpu_op_reduce_sum_ref.bin",
               _fp16_roundtrip(np.sum(r_in16, axis=1)))

    # bf16 versions for MAX/MIN (ATB ReduceOp MAX/MIN only supports int32 / bf16,
    # not fp16). Use the same fp32 input rounded to bf16 storage.
    r_inbf = _bf16_roundtrip(r_in)
    write_bf16(f"{OUTPUT_DIR}/cpu_op_reduce_bf16_in.bin", r_inbf)
    write_bf16(f"{OUTPUT_DIR}/cpu_op_reduce_bf16_max_ref.bin",
               _bf16_roundtrip(np.max(r_inbf, axis=1)))
    write_bf16(f"{OUTPUT_DIR}/cpu_op_reduce_bf16_min_ref.bin",
               _bf16_roundtrip(np.min(r_inbf, axis=1)))

    print("  → wrote gather (1) + reduce MAX/MIN/SUM (3) + bf16 MAX/MIN (2)")


def gen_op_transpose_set_value():
    """Reference data for TransposeOp and SetValueOp."""
    print("[gen] op_transpose_set_value")
    rng = np.random.default_rng(17)

    # Transpose perm=[1,0] on [3, 5]
    t1 = rng.standard_normal((3, 5)).astype(np.float32)
    t1_16 = _fp16_roundtrip(t1)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_transpose2d_in.bin", t1_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_transpose2d_ref.bin",
               _fp16_roundtrip(np.transpose(t1_16, (1, 0))))

    # Transpose perm=[0,2,1,3] on [2, 4, 3, 5]
    t2 = rng.standard_normal((2, 4, 3, 5)).astype(np.float32)
    t2_16 = _fp16_roundtrip(t2)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_transpose4d_in.bin", t2_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_transpose4d_ref.bin",
               _fp16_roundtrip(np.transpose(t2_16, (0, 2, 1, 3))))

    # SetValue — embed a [2, 3] tile into a [4, 6] zero canvas at [1:3, 2:5]
    dst_init = np.zeros((4, 6), dtype=np.float32)
    src = rng.standard_normal((2, 3)).astype(np.float32)
    src_16 = _fp16_roundtrip(src)
    expected = dst_init.copy()
    expected[1:3, 2:5] = src_16
    write_fp16(f"{OUTPUT_DIR}/cpu_op_setvalue_dst_init.bin",
               _fp16_roundtrip(dst_init))
    write_fp16(f"{OUTPUT_DIR}/cpu_op_setvalue_src.bin", src_16)
    write_fp16(f"{OUTPUT_DIR}/cpu_op_setvalue_ref.bin", _fp16_roundtrip(expected))

    print("  → wrote transpose (2) + set_value (1)")


# ═══════════════════════════════════════════════════════════════════
# Stage: RopeOp — fp16 ground truth for rotaryCoeff=2 (LLAMA-style)
#
# ATB RopeOp inputs:  q (B*S, nh*hd), k (B*S, kvh*hd), cos (B*S, hd), sin (B*S, hd), seqlen
# Outputs: ropeQ, ropeK (same shapes as q, k)
#
# rotaryCoeff=2 (Qwen3VL default) applies the contiguous-half rotation:
#   x1 = x[..., :hd//2]; x2 = x[..., hd//2:]
#   rotated = concat([-x2, x1], dim=-1)
#   out = x * cos + rotated * sin
# ═══════════════════════════════════════════════════════════════════

def _apply_rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Contiguous-half rotation matching ATB rotaryCoeff=2 (LLAMA style).

    x:    (B*S, n_heads, hd) fp32
    cos:  (B*S, hd) fp32
    sin:  (B*S, hd) fp32
    """
    hd = x.shape[-1]
    half = hd // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    # broadcast cos/sin: (B*S, 1, hd) over (B*S, n_heads, hd)
    cos_b = cos.unsqueeze(-2)
    sin_b = sin.unsqueeze(-2)
    return x * cos_b + rotated * sin_b


def gen_op_rope():
    """Reference data for RopeOp (rotaryCoeff=2, LLAMA-style contiguous-half).

    Cases:
      * "basic" — MHA: nh=4, kvh=4, hd=64, seq=8 (B=1)
      * "gqa"   — GQA: nh=12, kvh=4, hd=64, seq=8 (B=1)
    """
    print("[gen] op_rope — RopeOp rotaryCoeff=2 (basic + gqa)")

    def _make(name: str, S: int, nh: int, kvh: int, hd: int, seed: int):
        rng = np.random.default_rng(seed)
        # q: (B*S, nh, hd) → flatten to (B*S, nh*hd)
        q3 = rng.standard_normal((S, nh, hd)).astype(np.float32) * 0.2
        k3 = rng.standard_normal((S, kvh, hd)).astype(np.float32) * 0.2
        # cos/sin: deterministic from random pos angles
        ang = rng.standard_normal((S, hd // 2)).astype(np.float32) * 0.5
        # Build full cos/sin like Python uses: emb = cat(freqs, freqs); cos = emb.cos()
        emb = np.concatenate([ang, ang], axis=-1)  # (S, hd)
        cos = np.cos(emb)
        sin = np.sin(emb)

        # Round-trip everything through fp16 to mirror NPU storage
        q3_16 = _fp16_roundtrip(q3)
        k3_16 = _fp16_roundtrip(k3)
        cos_16 = _fp16_roundtrip(cos)
        sin_16 = _fp16_roundtrip(sin)

        # Compute reference in fp32 using fp16 inputs
        q_t = torch.from_numpy(q3_16)
        k_t = torch.from_numpy(k3_16)
        cos_t = torch.from_numpy(cos_16)
        sin_t = torch.from_numpy(sin_16)
        rq = _apply_rope_half(q_t, cos_t, sin_t).numpy()
        rk = _apply_rope_half(k_t, cos_t, sin_t).numpy()
        # Round-trip ref through fp16
        rq_ref = _fp16_roundtrip(rq)
        rk_ref = _fp16_roundtrip(rk)

        # Flatten q/k to (B*S, nh*hd) for storage matching ATB input layout
        q_flat = q3_16.reshape(S, nh * hd)
        k_flat = k3_16.reshape(S, kvh * hd)
        rq_flat = rq_ref.reshape(S, nh * hd)
        rk_flat = rk_ref.reshape(S, kvh * hd)

        write_fp16(f"{OUTPUT_DIR}/cpu_op_rope_{name}_q.bin",   q_flat)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_rope_{name}_k.bin",   k_flat)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_rope_{name}_cos.bin", cos_16)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_rope_{name}_sin.bin", sin_16)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_rope_{name}_rq_ref.bin", rq_flat)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_rope_{name}_rk_ref.bin", rk_flat)
        # Store [S, nh, kvh, hd] metadata as int32s for the C++ side to read
        write_i32s(f"{OUTPUT_DIR}/cpu_op_rope_{name}_meta.bin", [S, nh, kvh, hd])
        print(f"  → {name}: q{q_flat.shape} k{k_flat.shape} cos{cos_16.shape}")

    _make("basic", S=8, nh=4,  kvh=4, hd=64, seed=2024)
    _make("gqa",   S=8, nh=12, kvh=4, hd=64, seed=2025)


# ═══════════════════════════════════════════════════════════════════
# Stage: SelfAttentionOp — fp16 ground truth vs PyTorch SDPA
#
# ATB SelfAttentionOp (BSND, PA_ENCODER):
#   Inputs: q (B*S, nh, hd), k (B*S, kvh, hd), v (B*S, kvh, hd),
#           [mask (S, S)], seqlen (int32)
#   Output: attn_out (B*S, nh, hd)
#   Internally: softmax(q @ k.T / sqrt(hd) + mask) @ v
#
# Reference uses torch.nn.functional.scaled_dot_product_attention
# which gives the exact same math. We use B=1 so B*S = S.
# ═══════════════════════════════════════════════════════════════════

def gen_op_self_attention():
    """Reference data for SelfAttentionOp (MHA / GQA, with/without mask).

    Cases:
      * "mha_nomask"   — nh=4,  kvh=4, hd=32, S=8, no mask
      * "gqa_nomask"   — nh=12, kvh=4, hd=64, S=8, no mask
      * "mha_causal"   — nh=4,  kvh=4, hd=32, S=8, with causal mask
    """
    print("[gen] op_self_attention — MHA/GQA, with and without mask")

    def _run_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  nh: int, kvh: int, attn_mask: torch.Tensor = None,
                  is_causal: bool = False) -> torch.Tensor:
        """Reference SDPA in fp32. Inputs are (S, n_heads, hd) shaped.

        GQA: replicate K/V heads to match Q head count.
        Returns: (S, nh, hd)
        """
        S, _, hd = q.shape
        if kvh != nh:
            assert nh % kvh == 0
            repeat = nh // kvh
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        # SDPA wants (batch, n_heads, seq, hd); add a batch dim then transpose
        q_b = q.unsqueeze(0).transpose(1, 2)  # (1, nh, S, hd)
        k_b = k.unsqueeze(0).transpose(1, 2)
        v_b = v.unsqueeze(0).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_b, k_b, v_b,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        # (1, nh, S, hd) → (S, nh, hd)
        return out.transpose(1, 2).squeeze(0)

    def _make(name: str, S: int, nh: int, kvh: int, hd: int, seed: int,
              use_mask: bool, causal: bool):
        rng = np.random.default_rng(seed)
        # Use small magnitudes so fp16 softmax is stable
        q = rng.standard_normal((S, nh,  hd)).astype(np.float32) * 0.1
        k = rng.standard_normal((S, kvh, hd)).astype(np.float32) * 0.1
        v = rng.standard_normal((S, kvh, hd)).astype(np.float32) * 0.1

        q16 = _fp16_roundtrip(q)
        k16 = _fp16_roundtrip(k)
        v16 = _fp16_roundtrip(v)

        # Build mask (S, S) — additive: 0 attend, large-negative mask
        mask_np = None
        attn_mask = None
        if use_mask:
            if causal:
                # Upper triangle (strict) is masked → -inf
                m = torch.zeros(S, S, dtype=torch.float32)
                idx = torch.triu(torch.ones(S, S), diagonal=1).bool()
                m.masked_fill_(idx, float("-inf"))
                # ATB MASK_TYPE_NORM expects additive mask in fp16 storage;
                # we store as fp16 with a large negative constant for masked
                # positions (fp16 cannot represent -inf in additive form
                # without producing NaN). Use -65504 (most negative fp16).
                mask_np_fp32 = m.numpy()
                mask_np = mask_np_fp32.copy()
                mask_np[np.isinf(mask_np)] = -65504.0
                attn_mask = torch.from_numpy(mask_np)  # for the reference
            else:
                # Non-causal but uses mask input: all-zero mask (attend all)
                mask_np = np.zeros((S, S), dtype=np.float32)
                attn_mask = torch.from_numpy(mask_np)

        # Reference (fp32 SDPA, with fp16 inputs upcast)
        q_t = torch.from_numpy(q16)
        k_t = torch.from_numpy(k16)
        v_t = torch.from_numpy(v16)
        out = _run_sdpa(q_t, k_t, v_t, nh, kvh,
                        attn_mask=attn_mask,
                        is_causal=False)  # mask is built into attn_mask
        # When no-mask requested → just regular SDPA without mask
        if not use_mask:
            out = _run_sdpa(q_t, k_t, v_t, nh, kvh, attn_mask=None, is_causal=False)
        out_ref = _fp16_roundtrip(out.numpy())

        # Persist (q/k/v stored as 3D for the C++ side, output as 3D too)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_sa_{name}_q.bin", q16)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_sa_{name}_k.bin", k16)
        write_fp16(f"{OUTPUT_DIR}/cpu_op_sa_{name}_v.bin", v16)
        if mask_np is not None:
            write_fp16(f"{OUTPUT_DIR}/cpu_op_sa_{name}_mask.bin",
                       _fp16_roundtrip(mask_np))
        write_fp16(f"{OUTPUT_DIR}/cpu_op_sa_{name}_out_ref.bin", out_ref)
        # Meta: [S, nh, kvh, hd, has_mask]
        write_i32s(f"{OUTPUT_DIR}/cpu_op_sa_{name}_meta.bin",
                   [S, nh, kvh, hd, 1 if use_mask else 0])
        kind = "causal" if causal else ("with-mask" if use_mask else "no-mask")
        print(f"  → {name}: S={S} nh={nh} kvh={kvh} hd={hd} ({kind})")

    _make("mha_nomask", S=8, nh=4,  kvh=4, hd=32, seed=4001,
          use_mask=False, causal=False)
    _make("gqa_nomask", S=8, nh=12, kvh=4, hd=64, seed=4002,
          use_mask=False, causal=False)
    _make("mha_causal", S=8, nh=4,  kvh=4, hd=32, seed=4003,
          use_mask=True,  causal=True)


# ═══════════════════════════════════════════════════════════════════
# Stage: SwiGluMlpGraph — fp16 ground truth vs PyTorch reference
#
# ATB SwiGluMlpGraph (4 inputs):
#   hidden_states (S, H) fp16
#   gate_weight   (I, H) fp16     — Linear default transpose_b=True
#   up_weight     (I, H) fp16
#   down_weight   (H, I) fp16
#   → mlp_output  (S, H) fp16
#
# Math (mirrors text_mlp.py):
#   gate = SiLU(x @ gate_w.T)
#   up   = x @ up_w.T
#   out  = (gate * up) @ down_w.T
# ═══════════════════════════════════════════════════════════════════

def gen_swiglu_mlp():
    """Reference data for SwiGluMlpGraph (small + typical-Qwen3VL shape).

    Cases:
      * "small"   — S=4,  H=64,    I=128
      * "typical" — S=16, H=2048,  I=6144 (Qwen3VL hidden/intermediate)
    """
    print("[gen] swiglu_mlp — SwiGLU MLP fp16 ground truth")

    def _make(name: str, S: int, H: int, I: int, seed: int):
        rng = np.random.default_rng(seed)
        # Use modest magnitudes — fp16 SiLU saturates for |x| > ~16
        x  = rng.standard_normal((S, H)).astype(np.float32) * 0.1
        gw = rng.standard_normal((I, H)).astype(np.float32) * 0.05
        uw = rng.standard_normal((I, H)).astype(np.float32) * 0.05
        dw = rng.standard_normal((H, I)).astype(np.float32) * 0.05

        # Round-trip inputs through fp16 (mirror NPU storage)
        x16  = _fp16_roundtrip(x)
        gw16 = _fp16_roundtrip(gw)
        uw16 = _fp16_roundtrip(uw)
        dw16 = _fp16_roundtrip(dw)

        # Compute reference in fp32 from fp16 inputs (matches what NPU sees)
        x_t  = torch.from_numpy(x16)
        gw_t = torch.from_numpy(gw16)
        uw_t = torch.from_numpy(uw16)
        dw_t = torch.from_numpy(dw16)
        gate = F.silu(F.linear(x_t, gw_t))
        up   = F.linear(x_t, uw_t)
        out  = F.linear(gate * up, dw_t)
        out_ref = _fp16_roundtrip(out.numpy())

        write_fp16(f"{OUTPUT_DIR}/cpu_swiglu_mlp_{name}_x.bin",      x16)
        write_fp16(f"{OUTPUT_DIR}/cpu_swiglu_mlp_{name}_gate_w.bin", gw16)
        write_fp16(f"{OUTPUT_DIR}/cpu_swiglu_mlp_{name}_up_w.bin",   uw16)
        write_fp16(f"{OUTPUT_DIR}/cpu_swiglu_mlp_{name}_down_w.bin", dw16)
        write_fp16(f"{OUTPUT_DIR}/cpu_swiglu_mlp_{name}_out_ref.bin", out_ref)
        write_i32s(f"{OUTPUT_DIR}/cpu_swiglu_mlp_{name}_meta.bin", [S, H, I])
        print(f"  → {name}: S={S} H={H} I={I}")

    _make("small",   S=4,  H=64,   I=128,  seed=5001)
    _make("typical", S=16, H=2048, I=6144, seed=5002)


# ═══════════════════════════════════════════════════════════════════
# Stage: TextDecoderLayerGraph — fp16 ground truth vs PyTorch reference
#
# Composes: RMSNorm → attention(+residual) → RMSNorm → SwiGLU(+residual)
#   attention = q/k/v Linear → reshape → q/k RMSNorm → RoPE (LLAMA half)
#               → SDPA (with optional mask) → flatten → o Linear
#
# Inputs (15 no-mask, 16 with mask):
#   hidden (S, H), q/k/v/o_w (heads*hd, H/H), qn/kn_w (hd),
#   gate_w/up_w (I, H), down_w (H, I), input_ln_w (H), post_ln_w (H),
#   cos (S, hd), sin (S, hd), [mask (S, S)], seqlen (int32)
#
# The C++ DecoderLayer accepts hidden as 2D (S, H). We follow that here.
# ═══════════════════════════════════════════════════════════════════

def _rms_norm_torch(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * w. Normalizes over last dim."""
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def _sdpa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              nh: int, kvh: int, attn_mask: torch.Tensor = None) -> torch.Tensor:
    """Reference SDPA in fp32. Inputs are (S, n_heads, hd) shaped.

    Replicates K/V heads for GQA so we can use a single SDPA call.
    Returns: (S, nh, hd)
    """
    if kvh != nh:
        assert nh % kvh == 0
        repeat = nh // kvh
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    q_b = q.unsqueeze(0).transpose(1, 2)  # (1, nh, S, hd)
    k_b = k.unsqueeze(0).transpose(1, 2)
    v_b = v.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask)
    return out.transpose(1, 2).squeeze(0)


def gen_text_decoder_layer():
    """Reference data for TextDecoderLayerGraph (no-mask + GQA-with-mask).

    Cases:
      * "small_nomask" — nh=4, kvh=4, hd=32, I=64,  S=4   (small debug)
      * "gqa_mask"     — nh=12, kvh=4, hd=64, I=256, S=8  (GQA + causal mask)
    """
    print("[gen] text_decoder_layer — full layer fp16 ground truth")
    eps = 1e-6

    def _make(name: str, S: int, nh: int, kvh: int, hd: int, I: int,
              use_mask: bool, seed: int):
        rng = np.random.default_rng(seed)
        H = nh * hd
        Hkv = kvh * hd

        # ── Inputs (all small magnitudes for fp16 stability) ──
        x      = rng.standard_normal((S, H)).astype(np.float32) * 0.1
        q_w    = rng.standard_normal((H,   H)).astype(np.float32) * 0.05
        k_w    = rng.standard_normal((Hkv, H)).astype(np.float32) * 0.05
        v_w    = rng.standard_normal((Hkv, H)).astype(np.float32) * 0.05
        o_w    = rng.standard_normal((H,   H)).astype(np.float32) * 0.05
        qn_w   = (rng.standard_normal((hd,)).astype(np.float32) * 0.1 + 1.0)
        kn_w   = (rng.standard_normal((hd,)).astype(np.float32) * 0.1 + 1.0)
        gate_w = rng.standard_normal((I, H)).astype(np.float32) * 0.05
        up_w   = rng.standard_normal((I, H)).astype(np.float32) * 0.05
        down_w = rng.standard_normal((H, I)).astype(np.float32) * 0.05
        iln_w  = (rng.standard_normal((H,)).astype(np.float32) * 0.1 + 1.0)
        pln_w  = (rng.standard_normal((H,)).astype(np.float32) * 0.1 + 1.0)

        # cos/sin built like the engine does: emb = cat(freqs, freqs); cos=emb.cos
        ang = rng.standard_normal((S, hd // 2)).astype(np.float32) * 0.5
        emb = np.concatenate([ang, ang], axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)

        # Mask
        mask_np = None
        attn_mask_t = None
        if use_mask:
            m = torch.zeros(S, S, dtype=torch.float32)
            idx = torch.triu(torch.ones(S, S), diagonal=1).bool()
            m.masked_fill_(idx, -65504.0)
            mask_np = m.numpy()
            attn_mask_t = m  # additive

        # ── Round-trip all fp16-side inputs ──
        x16    = _fp16_roundtrip(x)
        q_w16  = _fp16_roundtrip(q_w);   k_w16 = _fp16_roundtrip(k_w)
        v_w16  = _fp16_roundtrip(v_w);   o_w16 = _fp16_roundtrip(o_w)
        qn_w16 = _fp16_roundtrip(qn_w);  kn_w16 = _fp16_roundtrip(kn_w)
        gate_w16 = _fp16_roundtrip(gate_w); up_w16 = _fp16_roundtrip(up_w)
        down_w16 = _fp16_roundtrip(down_w)
        iln_w16 = _fp16_roundtrip(iln_w); pln_w16 = _fp16_roundtrip(pln_w)
        cos16 = _fp16_roundtrip(cos);     sin16 = _fp16_roundtrip(sin)
        mask16 = _fp16_roundtrip(mask_np) if mask_np is not None else None

        # ── Reference (fp32 from fp16 inputs) ──
        xt   = torch.from_numpy(x16)
        iln  = torch.from_numpy(iln_w16);  pln  = torch.from_numpy(pln_w16)
        qw_t = torch.from_numpy(q_w16);    kw_t = torch.from_numpy(k_w16)
        vw_t = torch.from_numpy(v_w16);    ow_t = torch.from_numpy(o_w16)
        qn_t = torch.from_numpy(qn_w16);   kn_t = torch.from_numpy(kn_w16)
        gw_t = torch.from_numpy(gate_w16); uw_t = torch.from_numpy(up_w16)
        dw_t = torch.from_numpy(down_w16)
        cos_t = torch.from_numpy(cos16);   sin_t = torch.from_numpy(sin16)

        # 1) input_layernorm
        normed = _rms_norm_torch(xt, iln, eps)

        # 2) attention
        q_lin = F.linear(normed, qw_t).view(S, nh,  hd)
        k_lin = F.linear(normed, kw_t).view(S, kvh, hd)
        v_3d  = F.linear(normed, vw_t).view(S, kvh, hd)
        # q/k RMSNorm on head_dim (last) with per-head_dim weight
        q_n = _rms_norm_torch(q_lin, qn_t, eps)
        k_n = _rms_norm_torch(k_lin, kn_t, eps)
        # RoPE (LLAMA-style half rotation)
        q_rope = _apply_rope_half(q_n, cos_t, sin_t)
        k_rope = _apply_rope_half(k_n, cos_t, sin_t)
        # SDPA (replicate kv for GQA inside helper)
        attn_out = _sdpa_gqa(q_rope, k_rope, v_3d, nh, kvh,
                             attn_mask=attn_mask_t)
        # Flatten (S, nh, hd) → (S, H), then o_proj
        attn_flat = attn_out.reshape(S, H)
        o_out = F.linear(attn_flat, ow_t)

        # 3) residual + post_layernorm + SwiGLU + residual
        h1 = xt + o_out
        normed2 = _rms_norm_torch(h1, pln, eps)
        gate = F.silu(F.linear(normed2, gw_t))
        up   = F.linear(normed2, uw_t)
        mlp_out = F.linear(gate * up, dw_t)
        out = h1 + mlp_out

        out_ref = _fp16_roundtrip(out.numpy())

        # ── Persist ──
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_x.bin",       x16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_q_w.bin",     q_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_k_w.bin",     k_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_v_w.bin",     v_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_o_w.bin",     o_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_qn_w.bin",    qn_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_kn_w.bin",    kn_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_gate_w.bin",  gate_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_up_w.bin",    up_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_down_w.bin",  down_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_iln_w.bin",   iln_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_pln_w.bin",   pln_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_cos.bin",     cos16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_sin.bin",     sin16)
        if mask16 is not None:
            write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_mask.bin", mask16)
        write_fp16(f"{OUTPUT_DIR}/cpu_dec_{name}_out_ref.bin", out_ref)
        # Meta: [S, nh, kvh, hd, I, has_mask]
        write_i32s(f"{OUTPUT_DIR}/cpu_dec_{name}_meta.bin",
                   [S, nh, kvh, hd, I, 1 if use_mask else 0])
        kind = "with-mask" if use_mask else "no-mask"
        print(f"  → {name}: S={S} nh={nh} kvh={kvh} hd={hd} I={I} ({kind})")

    _make("small_nomask", S=4, nh=4,  kvh=4, hd=32, I=64,  use_mask=False, seed=6001)
    _make("gqa_mask",     S=8, nh=12, kvh=4, hd=64, I=256, use_mask=True,  seed=6002)


# ═══════════════════════════════════════════════════════════════════
# Stage: Vision component references (Level 2 — composed graphs)
#
# These tests exercise the full vision component pipelines on NPU and
# compare against PyTorch reference computations. Each generator writes
# the fp16 inputs and an fp16-roundtrip reference output.
#
# Convention: cos=1, sin=0 to disable rotary rotation (identity), which
# lets us validate the linear/attention/projection arithmetic without
# coupling to ATB's RopeOp half-rotation. RoPE precision is covered
# separately by gen_op_rope.
# ═══════════════════════════════════════════════════════════════════

def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """tanh-approximation GELU (matches ATB ACTIVATION_GELU)."""
    return F.gelu(torch.from_numpy(x), approximate="tanh").numpy()


def gen_vision_attention():
    """VisionAttention precision reference.

    Pipeline (with cos=1, sin=0):
        hidden -> qkv Linear(+bias) -> reshape(N, 3, nh, hd) -> Split
        Q,K,V (identity RoPE) -> SDPA -> proj Linear(+bias)

    Small case: N=16, nh=4, hd=32 (hidden=128).
    """
    print("[gen] vision_attention — full attention pipeline (identity RoPE)")
    rng = np.random.default_rng(5101)

    N, nh, hd = 16, 4, 32
    hidden = nh * hd

    x      = rng.standard_normal((N, hidden)).astype(np.float32) * 0.2
    qkv_w  = rng.standard_normal((3 * hidden, hidden)).astype(np.float32) * 0.1
    qkv_b  = rng.standard_normal((3 * hidden,)).astype(np.float32) * 0.05
    proj_w = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.1
    proj_b = rng.standard_normal((hidden,)).astype(np.float32) * 0.05
    cos    = np.ones((N, hd), dtype=np.float32)
    sin    = np.zeros((N, hd), dtype=np.float32)

    x16    = _fp16_roundtrip(x)
    qw16   = _fp16_roundtrip(qkv_w)
    qb16   = _fp16_roundtrip(qkv_b)
    pw16   = _fp16_roundtrip(proj_w)
    pb16   = _fp16_roundtrip(proj_b)
    cos16  = _fp16_roundtrip(cos)
    sin16  = _fp16_roundtrip(sin)

    # Reference: linear → reshape → SDPA → linear
    qkv = x16 @ qw16.T + qb16
    qkv = qkv.reshape(N, 3, nh, hd)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q_t = torch.from_numpy(q.astype(np.float32)).unsqueeze(0).transpose(1, 2)
    k_t = torch.from_numpy(k.astype(np.float32)).unsqueeze(0).transpose(1, 2)
    v_t = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q_t, k_t, v_t).transpose(1, 2).squeeze(0).numpy()
    attn = attn.reshape(N, hidden)
    out = attn @ pw16.T + pb16
    ref = _fp16_roundtrip(out)

    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_x.bin",       x16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_qkv_w.bin",   qw16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_qkv_b.bin",   qb16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_proj_w.bin",  pw16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_proj_b.bin",  pb16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_cos.bin",     cos16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_sin.bin",     sin16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_attn_ref.bin",     ref)
    write_i32s(f"{OUTPUT_DIR}/cpu_vision_attn_meta.bin",    [N, nh, hd])
    print(f"  → vision_attn: N={N} nh={nh} hd={hd}")


def gen_vision_mlp():
    """VisionMLP precision reference.

    Pipeline: fc1 Linear(+bias) -> GELU(tanh) -> fc2 Linear(+bias)
    Small case: N=8, hidden=64, inter=128.
    """
    print("[gen] vision_mlp — fc1 -> GELU -> fc2")
    rng = np.random.default_rng(5201)

    N, hidden, inter = 8, 64, 128
    x     = rng.standard_normal((N, hidden)).astype(np.float32) * 0.2
    fc1_w = rng.standard_normal((inter, hidden)).astype(np.float32) * 0.1
    fc1_b = rng.standard_normal((inter,)).astype(np.float32) * 0.05
    fc2_w = rng.standard_normal((hidden, inter)).astype(np.float32) * 0.1
    fc2_b = rng.standard_normal((hidden,)).astype(np.float32) * 0.05

    x16    = _fp16_roundtrip(x)
    fc1w16 = _fp16_roundtrip(fc1_w)
    fc1b16 = _fp16_roundtrip(fc1_b)
    fc2w16 = _fp16_roundtrip(fc2_w)
    fc2b16 = _fp16_roundtrip(fc2_b)

    h1 = x16 @ fc1w16.T + fc1b16
    h1 = _gelu_tanh(h1)
    out = h1 @ fc2w16.T + fc2b16
    ref = _fp16_roundtrip(out)

    write_fp16(f"{OUTPUT_DIR}/cpu_vision_mlp_x.bin",     x16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_mlp_fc1_w.bin", fc1w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_mlp_fc1_b.bin", fc1b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_mlp_fc2_w.bin", fc2w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_mlp_fc2_b.bin", fc2b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_mlp_ref.bin",   ref)
    write_i32s(f"{OUTPUT_DIR}/cpu_vision_mlp_meta.bin",  [N, hidden, inter])
    print(f"  → vision_mlp: N={N} hidden={hidden} inter={inter}")


def gen_vision_block():
    """VisionBlock precision reference.

    Pipeline (with cos=1, sin=0):
        x -> LN1 -> Attn (+x residual) -> LN2 -> MLP (+attn residual) -> out

    Small case: N=16, nh=4, hd=32 (hidden=128), inter=256.
    """
    print("[gen] vision_block — LN -> Attn -> LN -> MLP (identity RoPE)")
    rng = np.random.default_rng(5301)

    N, nh, hd = 16, 4, 32
    hidden = nh * hd
    inter = 4 * hidden
    eps = 1e-6

    x      = rng.standard_normal((N, hidden)).astype(np.float32) * 0.2
    qkv_w  = rng.standard_normal((3 * hidden, hidden)).astype(np.float32) * 0.1
    qkv_b  = rng.standard_normal((3 * hidden,)).astype(np.float32) * 0.05
    proj_w = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.1
    proj_b = rng.standard_normal((hidden,)).astype(np.float32) * 0.05
    fc1_w  = rng.standard_normal((inter, hidden)).astype(np.float32) * 0.1
    fc1_b  = rng.standard_normal((inter,)).astype(np.float32) * 0.05
    fc2_w  = rng.standard_normal((hidden, inter)).astype(np.float32) * 0.1
    fc2_b  = rng.standard_normal((hidden,)).astype(np.float32) * 0.05
    n1_w = (rng.standard_normal((hidden,)).astype(np.float32) * 0.1) + 1.0
    n1_b =  rng.standard_normal((hidden,)).astype(np.float32) * 0.05
    n2_w = (rng.standard_normal((hidden,)).astype(np.float32) * 0.1) + 1.0
    n2_b =  rng.standard_normal((hidden,)).astype(np.float32) * 0.05
    cos = np.ones((N, hd), dtype=np.float32)
    sin = np.zeros((N, hd), dtype=np.float32)

    x16     = _fp16_roundtrip(x)
    qkv_w16 = _fp16_roundtrip(qkv_w);  qkv_b16  = _fp16_roundtrip(qkv_b)
    proj_w16 = _fp16_roundtrip(proj_w); proj_b16 = _fp16_roundtrip(proj_b)
    fc1_w16 = _fp16_roundtrip(fc1_w);  fc1_b16  = _fp16_roundtrip(fc1_b)
    fc2_w16 = _fp16_roundtrip(fc2_w);  fc2_b16  = _fp16_roundtrip(fc2_b)
    n1_w16  = _fp16_roundtrip(n1_w);   n1_b16   = _fp16_roundtrip(n1_b)
    n2_w16  = _fp16_roundtrip(n2_w);   n2_b16   = _fp16_roundtrip(n2_b)
    cos16   = _fp16_roundtrip(cos);    sin16    = _fp16_roundtrip(sin)

    # LN1
    ln1 = F.layer_norm(torch.from_numpy(x16), [hidden],
                       weight=torch.from_numpy(n1_w16),
                       bias=torch.from_numpy(n1_b16), eps=eps).numpy()
    # Attention (identity RoPE)
    qkv = ln1 @ qkv_w16.T + qkv_b16
    qkv = qkv.reshape(N, 3, nh, hd)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q_t = torch.from_numpy(q.astype(np.float32)).unsqueeze(0).transpose(1, 2)
    k_t = torch.from_numpy(k.astype(np.float32)).unsqueeze(0).transpose(1, 2)
    v_t = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q_t, k_t, v_t).transpose(1, 2).squeeze(0).numpy()
    attn = attn.reshape(N, hidden)
    attn_out = attn @ proj_w16.T + proj_b16
    r1 = x16 + attn_out
    # LN2
    ln2 = F.layer_norm(torch.from_numpy(r1.astype(np.float32)), [hidden],
                       weight=torch.from_numpy(n2_w16),
                       bias=torch.from_numpy(n2_b16), eps=eps).numpy()
    # MLP
    h1 = ln2 @ fc1_w16.T + fc1_b16
    h1 = _gelu_tanh(h1)
    mlp_out = h1 @ fc2_w16.T + fc2_b16
    out = r1 + mlp_out
    ref = _fp16_roundtrip(out)

    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_x.bin",      x16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_qkv_w.bin",  qkv_w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_qkv_b.bin",  qkv_b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_proj_w.bin", proj_w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_proj_b.bin", proj_b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_fc1_w.bin",  fc1_w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_fc1_b.bin",  fc1_b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_fc2_w.bin",  fc2_w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_fc2_b.bin",  fc2_b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_n1_w.bin",   n1_w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_n1_b.bin",   n1_b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_n2_w.bin",   n2_w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_n2_b.bin",   n2_b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_cos.bin",    cos16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_sin.bin",    sin16)
    write_fp16(f"{OUTPUT_DIR}/cpu_vision_block_ref.bin",    ref)
    write_i32s(f"{OUTPUT_DIR}/cpu_vision_block_meta.bin",   [N, nh, hd, inter])
    print(f"  → vision_block: N={N} nh={nh} hd={hd} inter={inter}")


def gen_patch_embed():
    """PatchEmbed precision reference.

    Pipeline: pixels (N*K,) -> reshape (N, K) -> Linear(+bias) -> (N, embed_dim)
    where K = in_channels * tp * p * p.

    Small case: N=4, in_ch=3, tp=2, p=14 → K=1176, embed_dim=64.
    """
    print("[gen] patch_embed — flatten + Linear(+bias)")
    rng = np.random.default_rng(5401)

    N = 4
    in_channels, tp, p = 3, 2, 14
    embed_dim = 64
    K = in_channels * tp * p * p  # 1176

    pixels = rng.standard_normal((N, K)).astype(np.float32) * 0.2
    w      = rng.standard_normal((embed_dim, K)).astype(np.float32) * 0.05
    b      = rng.standard_normal((embed_dim,)).astype(np.float32) * 0.05

    pixels16 = _fp16_roundtrip(pixels)
    w16      = _fp16_roundtrip(w)
    b16      = _fp16_roundtrip(b)

    out = pixels16 @ w16.T + b16
    ref = _fp16_roundtrip(out)

    # C++ feeds pixels as a flat (N*K,) tensor — store likewise
    write_fp16(f"{OUTPUT_DIR}/cpu_patch_embed_pixels.bin", pixels16.reshape(-1))
    write_fp16(f"{OUTPUT_DIR}/cpu_patch_embed_w.bin",      w16)
    write_fp16(f"{OUTPUT_DIR}/cpu_patch_embed_b.bin",      b16)
    write_fp16(f"{OUTPUT_DIR}/cpu_patch_embed_ref.bin",    ref)
    write_i32s(f"{OUTPUT_DIR}/cpu_patch_embed_meta.bin",
               [N, in_channels, tp, p, embed_dim])
    print(f"  → patch_embed: N={N} K={K} embed_dim={embed_dim}")


def gen_vision_merger():
    """VisionMerger and Deepstack precision reference (2 cases).

    Both share fc1->GELU->fc2 but differ in LayerNorm ordering:
      * "main"      — LayerNorm(hidden)  -> reshape(group_4) -> fc1 -> GELU -> fc2
      * "deepstack" — reshape(group_4)   -> LayerNorm(grouped) -> fc1 -> GELU -> fc2

    Small case: N=8 (= 2 merged groups of 4), hidden=64,
                merge=2 → mer_hidden=64*4=256, out_hidden=32.
    """
    print("[gen] vision_merger — main + deepstack ordering")

    merge = 2
    hidden = 64
    mer_hidden = hidden * merge * merge  # 256
    out_hidden = 32
    N = 8  # 8 * 64 = 512 elements, / mer_hidden(256) = 2 merged groups
    eps = 1e-6

    def _make(name: str, is_deepstack: bool, seed: int):
        local = np.random.default_rng(seed)
        x_pre = local.standard_normal((N, hidden)).astype(np.float32) * 0.2

        # main: LN on per-row hidden=64; deepstack: LN on per-group mer_hidden=256
        norm_dim = mer_hidden if is_deepstack else hidden
        n_w = (local.standard_normal((norm_dim,)).astype(np.float32) * 0.1) + 1.0
        n_b =  local.standard_normal((norm_dim,)).astype(np.float32) * 0.05
        f1_w = local.standard_normal((mer_hidden, mer_hidden)).astype(np.float32) * 0.05
        f1_b = local.standard_normal((mer_hidden,)).astype(np.float32) * 0.05
        f2_w = local.standard_normal((out_hidden, mer_hidden)).astype(np.float32) * 0.05
        f2_b = local.standard_normal((out_hidden,)).astype(np.float32) * 0.05

        x16    = _fp16_roundtrip(x_pre)
        n_w16  = _fp16_roundtrip(n_w);  n_b16  = _fp16_roundtrip(n_b)
        f1_w16 = _fp16_roundtrip(f1_w); f1_b16 = _fp16_roundtrip(f1_b)
        f2_w16 = _fp16_roundtrip(f2_w); f2_b16 = _fp16_roundtrip(f2_b)

        if is_deepstack:
            grouped = x16.reshape(N * hidden // mer_hidden, mer_hidden)
            ln = F.layer_norm(torch.from_numpy(grouped.astype(np.float32)),
                              [mer_hidden],
                              weight=torch.from_numpy(n_w16),
                              bias=torch.from_numpy(n_b16), eps=eps).numpy()
        else:
            ln_pre = F.layer_norm(torch.from_numpy(x16.astype(np.float32)),
                                  [hidden],
                                  weight=torch.from_numpy(n_w16),
                                  bias=torch.from_numpy(n_b16), eps=eps).numpy()
            ln = ln_pre.reshape(N * hidden // mer_hidden, mer_hidden)

        h1 = ln @ f1_w16.T + f1_b16
        h1 = _gelu_tanh(h1)
        out = h1 @ f2_w16.T + f2_b16
        ref = _fp16_roundtrip(out)

        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_x.bin",    x16)
        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_n_w.bin",  n_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_n_b.bin",  n_b16)
        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_f1_w.bin", f1_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_f1_b.bin", f1_b16)
        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_f2_w.bin", f2_w16)
        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_f2_b.bin", f2_b16)
        write_fp16(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_ref.bin",  ref)
        write_i32s(f"{OUTPUT_DIR}/cpu_vision_merger_{name}_meta.bin",
                   [N, hidden, merge, out_hidden, 1 if is_deepstack else 0])
        print(f"  → {name}: N={N} hidden={hidden} merge={merge} "
              f"out_hidden={out_hidden} norm_dim={norm_dim}")

    _make("main",      is_deepstack=False, seed=5501)
    _make("deepstack", is_deepstack=True,  seed=5502)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

STAGES = {
    "mrope_pid_simple":    gen_get_rope_index_simple,
    "mrope_pid_no_img":    gen_get_rope_index_no_image,
    "mrope_pid_image_text": gen_get_rope_index_image_text,
    "mrope_cos_sin":       gen_mrope_cos_sin,
    "vision_rope":         gen_vision_rope,
    "pos_embed":           gen_pos_embed_interp,
    "smart_resize":        gen_smart_resize,
    "float_utils":         gen_float_utils,
    "op_rms_norm":         gen_rms_norm,
    "op_layer_norm":       gen_layer_norm,
    "op_linear":           gen_linear,
    "op_activation":       gen_activation,
    "op_elewise":          gen_op_elewise,
    "op_split_concat":     gen_op_split_concat,
    "op_softmax":          gen_op_softmax,
    "op_gather_reduce":    gen_op_gather_reduce,
    "op_transpose_set_value": gen_op_transpose_set_value,
    "op_rope":             gen_op_rope,
    "op_self_attention":   gen_op_self_attention,
    "swiglu_mlp":          gen_swiglu_mlp,
    "text_decoder_layer":  gen_text_decoder_layer,
    "vision_attention":    gen_vision_attention,
    "vision_mlp":          gen_vision_mlp,
    "vision_block":        gen_vision_block,
    "patch_embed":         gen_patch_embed,
    "vision_merger":       gen_vision_merger,
}


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=list(STAGES.keys()) + ["all"],
                   default="all", help="Which stage to generate (default: all)")
    args = p.parse_args()

    if args.stage == "all":
        for name, fn in STAGES.items():
            fn()
        print("\nAll reference files generated.")
    else:
        STAGES[args.stage]()

    return 0


if __name__ == "__main__":
    sys.exit(main())
