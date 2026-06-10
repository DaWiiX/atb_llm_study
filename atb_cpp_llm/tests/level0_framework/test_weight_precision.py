"""Verify C++ weight precision matches Python fp16 conversion.

Reads the hex dump written by the C++ test_weight_precision and compares
every fp16 bit pattern against Python's own conversion from the same
safetensors file.

This test prevents the B9 bug class: double-truncation where f32 weights
were converted bf16 then to fp16, losing precision compared to the correct
direct conversion done by Python.

Tolerance:
  - bf16 -> fp16:  +/- 1 ULP  (both go through fp32 intermediate, rounding
                                differences are possible but rare)
  - fp32 -> fp16:  +/- 1 ULP  (round-to-nearest-even may differ slightly
                                between CANN's aclFloatToFloat16 and numpy)
  - fp16 -> fp16:  EXACT match (0 ULP) — should always be bit-exact

If mismatches exceed these tolerances, there is a real bug in the C++
weight conversion path.

Run: python3 tests/level0_framework/test_weight_precision.py
     (requires the C++ ./test_weight_precision to have been run first)
"""

import json
import os
import struct
import sys

import numpy as np

# ── Configuration ────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR  # noqa: E402
ST_PATH = os.path.join(MODEL_DIR, "model.safetensors")
CPP_DUMP_PATH = "/tmp/cpp_weight_dump.txt"

# Must match the C++ kWeightKeys array exactly
WEIGHT_KEYS = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.14.self_attn.q_proj.weight",
    "model.language_model.layers.27.mlp.down_proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "model.visual.blocks.0.mlp.linear_fc1.weight",
    "model.visual.blocks.23.attn.qkv.weight",
    "model.visual.merger.linear_fc1.weight",
]


# ── SafeTensor reader (reads JSON header + raw data) ─────────
def read_tensor_from_safetensors(filepath, key, max_elements=100):
    """Read a tensor from a safetensors file by parsing the JSON header.

    Returns (dtype_str, num_elements_total, fp16_uint16_flat_array).
    dtype_str is one of 'F32', 'F16', 'BF16'.

    Raises KeyError if the key is not found.
    """
    with open(filepath, "rb") as f:
        # Read header length (8 bytes, little-endian uint64)
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        # Read and parse JSON header
        header_json = f.read(header_len)
        header = json.loads(header_json)

        tensor_info = header.get(key)
        if tensor_info is None:
            raise KeyError(f"Tensor '{key}' not found in safetensors header")

        dtype_str = tensor_info["dtype"]
        shape = tensor_info["shape"]
        offsets = tensor_info["data_offsets"]
        data_start, data_end = offsets[0], offsets[1]

        # Compute total elements
        total = 1
        for s in shape:
            total *= s
        n_read = min(max_elements, total)

        # Seek to data region and read raw bytes
        data_offset = 8 + header_len + data_start
        raw_len = data_end - data_start
        f.seek(data_offset)
        raw = f.read(raw_len)

    return dtype_str, total, raw, n_read


# ── Convert raw bytes to fp16 uint16 (matching C++ logic) ────
def convert_to_fp16_uint16(dtype_str, raw, n):
    """Convert raw bytes to fp16 uint16 array.

    Matches the conversion logic in CopyWeightToFp16Host:
      - F16: direct copy (bit-exact)
      - BF16: reconstruct fp32 from upper 16 bits, then RNE to fp16
      - F32: RNE to fp16 via numpy

    Returns numpy array of dtype uint16, length n.
    """
    if dtype_str not in ("F32", "F16", "BF16"):
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    if dtype_str == "F16":
        # fp16 stored directly — no conversion needed
        arr = np.frombuffer(raw, dtype=np.float16)
        return arr[:n].view(np.uint16)

    elif dtype_str == "BF16":
        # bf16 -> fp32 reconstruction -> fp16 (matching C++ Bf16ToFp16Buffer)
        # bf16 is upper 16 bits of fp32; shift left by 16 to get fp32 bits
        bf16_u16 = np.frombuffer(raw, dtype=np.uint16)
        bf16_u16 = bf16_u16[:n]
        f32_bits = bf16_u16.astype(np.uint32) << 16
        f32 = f32_bits.view(np.float32)
        f16 = f32.astype(np.float16)
        return f16.view(np.uint16)

    elif dtype_str == "F32":
        # fp32 -> fp16 via RNE
        f32 = np.frombuffer(raw, dtype=np.float32)
        f32 = f32[:n]
        f16 = f32.astype(np.float16)
        return f16.view(np.uint16)


# ── Dump file parser ─────────────────────────────────────────
def parse_cpp_dump(filepath):
    """Parse the C++ weight dump into {key: [uint16 values]} dict.

    The C++ dump format is:
        # C++ weight dump: <key>
        # dtype: <str>  nelem: <N>  first100_hex: <hex...>
        <key>: <hex1> <hex2> ...

    We parse only the data lines (key: hex1 hex2 ...).
    Comment lines starting with # are ignored.
    """
    result = {}
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Format: <key>: <hex1> <hex2> ...
            # Use split(':', 1) because keys never contain colons
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue

            key = parts[0].strip()
            hex_str = parts[1].strip()
            if not hex_str:
                result[key] = []
                continue

            hex_vals = [int(h, 16) for h in hex_str.split()]
            result[key] = hex_vals

    return result


# ── ULP comparison ───────────────────────────────────────────
def ulp_distance(a, b):
    """Unsigned ULP distance between two uint16 fp16 bit patterns.

    Handles NaN/inf as infinite distance (never a false match).
    """
    if a == b:
        return 0

    # Check for NaN or inf: exponent bits all 1s (bits 14..10 = 11111)
    if ((a & 0x7C00) == 0x7C00) or ((b & 0x7C00) == 0x7C00):
        return 999999

    return abs(int(a) - int(b))


def max_ulp_tolerance(dtype_str):
    """Return max allowed ULP distance based on conversion path."""
    if dtype_str == "F16":
        return 0  # direct copy, must be bit-exact
    # BF16 and F32 both go through fp32->fp16 RNE, allow +/- 1 ULP
    return 1


# ══════════════════════════════════════════════════════════════
# Main comparison
# ══════════════════════════════════════════════════════════════

def compare_weight_precision():
    """Compare C++ and Python fp16 conversions for all weight keys."""
    # 1. Check prerequisites
    if not os.path.exists(ST_PATH):
        print(f"SKIP: safetensors not found at {ST_PATH}")
        return True

    cpp_dump = parse_cpp_dump(CPP_DUMP_PATH)
    if cpp_dump is None:
        print(
            f"SKIP: C++ dump not found at {CPP_DUMP_PATH} — "
            f"run ./test_weight_precision first"
        )
        # Return True (skip) if C++ hasn't been built/run yet — CI-friendly
        return True

    print("=" * 70)
    print("Weight Precision Test: C++ vs Python fp16 conversion")
    print("=" * 70)

    # 2. Compare each key
    total_keys = 0
    total_mismatches = 0
    bit_exact_keys = 0
    ulp_mismatches = {}  # key -> [(idx, cpp_val, py_val, ulp), ...]

    for key in WEIGHT_KEYS:
        if key not in cpp_dump:
            print(f"  SKIP  {key}  (not in C++ dump)")
            continue

        cpp_vals = cpp_dump[key]
        if not cpp_vals:
            print(f"  SKIP  {key}  (empty C++ dump)")
            continue

        try:
            dtype_str, total_n, raw, n_read = read_tensor_from_safetensors(
                ST_PATH, key, max_elements=len(cpp_vals)
            )
            py_vals = convert_to_fp16_uint16(dtype_str, raw, n_read)
        except KeyError as e:
            print(f"  SKIP  {key}  ({e})")
            continue

        tolerance = max_ulp_tolerance(dtype_str)
        n_compare = min(len(cpp_vals), len(py_vals))

        if n_compare == 0:
            print(f"  SKIP  {key}  (0 elements)")
            continue

        mismatches = []
        for i in range(n_compare):
            ulp = ulp_distance(cpp_vals[i], int(py_vals[i]))
            if ulp > tolerance:
                mismatches.append(
                    (i, cpp_vals[i], int(py_vals[i]), ulp)
                )

        total_keys += 1
        if not mismatches:
            bit_exact_keys += 1
            print(
                f"  OK    {key}  ({n_compare}/{total_n} elements, "
                f"dtype={dtype_str}, exact match)"
            )
        else:
            # Check if all are within 1 ULP
            all_ulp1 = all(m[3] <= 1 for m in mismatches)
            if all_ulp1 and tolerance == 0:
                # fp16 storage but 1 ULP diff — this shouldn't happen for direct copy
                total_mismatches += len(mismatches)
                ulp_mismatches[key] = mismatches
                print(
                    f"  FAIL  {key}  "
                    f"{len(mismatches)}/{n_compare} beyond tolerance "
                    f"(dtype={dtype_str}, tolerance={tolerance} ULP)"
                )
                for idx, cv, pv, ulp in mismatches[:5]:
                    print(
                        f"         [{idx}] C++=0x{cv:04X}  Py=0x{pv:04X}  "
                        f"ULP={ulp}"
                    )
                if len(mismatches) > 5:
                    print(f"         ... and {len(mismatches) - 5} more")
            elif all_ulp1:
                # Within 1 ULP — this is OK (rounding differences)
                print(
                    f"  OK    {key}  ({n_compare}/{total_n} elements, "
                    f"dtype={dtype_str}, {len(mismatches)} are ±1 ULP — "
                    f"rounding difference, acceptable)"
                )
            else:
                total_mismatches += len(mismatches)
                ulp_mismatches[key] = mismatches
                print(
                    f"  FAIL  {key}  "
                    f"{len(mismatches)}/{n_compare} beyond tolerance "
                    f"(dtype={dtype_str}, tolerance={tolerance} ULP)"
                )
                for idx, cv, pv, ulp in mismatches[:5]:
                    print(
                        f"         [{idx}] C++=0x{cv:04X}  Py=0x{pv:04X}  "
                        f"ULP={ulp}"
                    )
                if len(mismatches) > 5:
                    print(f"         ... and {len(mismatches) - 5} more")

    # 3. Summary
    print()
    print("=" * 70)
    print(
        f"Summary: {total_keys} keys checked, "
        f"{bit_exact_keys} bit-exact, "
        f"{total_mismatches} total element mismatches beyond tolerance"
    )
    print("=" * 70)

    if ulp_mismatches:
        print()
        print("FAIL: Weight precision mismatch detected!")
        print(
            "This likely indicates a B9-class bug: the C++ weight conversion"
        )
        print(
            "path does not match Python's conversion. Check the conversion "
            "logic"
        )
        print("in CopyWeightToFp16Host() in src/io/weight_helpers.cpp.")
        print()
        print("Mismatched keys:")
        for key, mismatches in ulp_mismatches.items():
            print(f"  {key}: {len(mismatches)} mismatches")
        return False

    print()
    print("PASS: All weight precision checks passed.")
    return True


if __name__ == "__main__":
    ok = compare_weight_precision()
    sys.exit(0 if ok else 1)
