#!/usr/bin/env python3
"""
Level 1 CPU pure-function tests: .bin format round-trip (Python side).

Companion to test_bin_format.cpp.  The C++ test writes three files to
/tmp/ — this script loads them, cross-verifies the values, performs
Python-side self round-trips, and includes an anti-regression check for
the historical .astype() vs .view() bug.

Formats:
  Format A (tokens, int64):    [int32 count][int64 * count]
  Format B (pixel_values):     [int32 count][uint16 * count]
  Format C (embedding output): [int64 dim][float32 * dim]

Historical bugs this test prevents:
  1. Using .astype(np.float16) on raw uint16 data instead of .view(np.float16)
     → uint16 15360 (0x3C00) is VALUE-CAST to fp16 15360.0 instead of
       BIT-REINTERPRETED to fp16 1.0.
  2. Filename mismatches between C++ and Python causing silent cosine drop.

Usage:
  # Run C++ test first to generate /tmp/test_bin_*.bin, then:
  python tests/level1_cpu_pure/test_bin_format.py

  # Or run standalone (cross-language checks will be skipped if C++
  # output files don't exist).
"""

import numpy as np
import struct
import sys
import os


# ═══════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════

def load_bin_fp16(path):
    """Load Format B: [int32 count][uint16 * count], reinterpret as float16.

    Uses np.frombuffer + .view(np.float16) for BIT REINTERPRETATION.
    This is the CORRECT way — do NOT use .astype(np.float16).
    """
    with open(path, 'rb') as f:
        raw = f.read()
    count = struct.unpack('<i', raw[:4])[0]
    data = np.frombuffer(raw[4:4 + count * 2], dtype=np.uint16).view(np.float16).copy()
    assert len(data) == count, f"Expected {count} elements, got {len(data)}"
    return data


def load_bin_fp16_as_uint16(path):
    """Load Format B as raw uint16 (no reinterpretation)."""
    with open(path, 'rb') as f:
        raw = f.read()
    count = struct.unpack('<i', raw[:4])[0]
    data = np.frombuffer(raw[4:4 + count * 2], dtype=np.uint16).copy()
    assert len(data) == count
    return data


def load_bin_int64(path):
    """Load Format A: [int32 count][int64 * count]."""
    with open(path, 'rb') as f:
        raw = f.read()
    count = struct.unpack('<i', raw[:4])[0]
    data = np.frombuffer(raw[4:4 + count * 8], dtype=np.int64).copy()
    assert len(data) == count, f"Expected {count} elements, got {len(data)}"
    return data


def load_bin_fp32(path):
    """Load Format C: [int64 dim][float32 * dim]."""
    with open(path, 'rb') as f:
        raw = f.read()
    dim = struct.unpack('<q', raw[:8])[0]
    data = np.frombuffer(raw[8:8 + dim * 4], dtype=np.float32).copy()
    assert len(data) == dim, f"Expected {dim} elements, got {len(data)}"
    return data


# ═══════════════════════════════════════════════════════════════════
# Savers
# ═══════════════════════════════════════════════════════════════════

def save_bin_fp16(path, data):
    """Save Format B with fp16 values."""
    fp16_vals = np.asarray(data, dtype=np.float16)
    raw = fp16_vals.view(np.uint16).tobytes()
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(fp16_vals)))
        f.write(raw)


def save_bin_int64(path, data):
    """Save Format A with int64 values."""
    arr = np.asarray(data, dtype=np.int64)
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(arr)))
        f.write(arr.tobytes())


def save_bin_fp32(path, data):
    """Save Format C with float32 values."""
    arr = np.asarray(data, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<q', len(arr)))
        f.write(arr.tobytes())


# ═══════════════════════════════════════════════════════════════════
# Test: cross-language fp16  (Format B)
# ═══════════════════════════════════════════════════════════════════

def test_cross_language_fp16():
    """Read C++-written fp16 .bin, verify each value, and run the
    anti-regression check that .view() != .astype() for non-trivial
    values."""
    path = '/tmp/test_bin_fp16.bin'
    if not os.path.exists(path):
        print("  [SKIP] fp16 cross-language: C++ output not found (run test_bin_format first)")
        return False

    # Load as float16 via correct bit-reinterpretation
    data_f16 = load_bin_fp16(path)
    print(f"  fp16 cross-language: count={len(data_f16)}")

    # Expected values matching the C++ test payload:
    # idx 0: +0.0
    # idx 1:  1.0
    # idx 2: -1.0
    # idx 3: +inf
    # idx 4: -inf
    # idx 5:  NaN (quiet)
    # idx 6: smallest positive subnormal  (~5.96e-8)
    # idx 7: smallest positive normal     (~6.10e-5)
    # idx 8: largest finite              (65504.0)
    # idx 9:  3.0
    # idx10:  4.0

    assert data_f16[0] == np.float16(0.0),  f"idx 0: +0.0 mismatch: {data_f16[0]}"
    assert data_f16[1] == np.float16(1.0),  f"idx 1: 1.0 mismatch: {data_f16[1]}"
    assert data_f16[2] == np.float16(-1.0), f"idx 2: -1.0 mismatch: {data_f16[2]}"
    assert np.isposinf(data_f16[3]),        f"idx 3: expected +inf, got {data_f16[3]}"
    assert np.isneginf(data_f16[4]),        f"idx 4: expected -inf, got {data_f16[4]}"
    assert np.isnan(data_f16[5]),           f"idx 5: expected NaN, got {data_f16[5]}"

    # Denorm, min normal, max normal should all be finite and non-NaN
    for i in [6, 7, 8]:
        assert np.isfinite(data_f16[i]), f"idx {i}: expected finite, got {data_f16[i]} (bit reinterpret error?)"
    assert data_f16[6] > 0, f"idx 6: subnormal should be positive"
    assert data_f16[7] > 0, f"idx 7: min normal should be positive"
    assert data_f16[8] > 0, f"idx 8: max normal should be positive"
    assert data_f16[6] < data_f16[7], f"subnormal ({data_f16[6]}) should be < min normal ({data_f16[7]})"
    assert data_f16[7] < data_f16[8], f"min normal ({data_f16[7]}) should be < max ({data_f16[8]})"
    assert data_f16[9] == np.float16(3.0),  f"idx 9: 3.0 mismatch: {data_f16[9]}"
    assert data_f16[10] == np.float16(4.0), f"idx 10: 4.0 mismatch: {data_f16[10]}"

    # ── Anti-regression: prove that .view() != .astype() ──────────
    # This is THE key check.  Load raw uint16 bits separately, then
    # compare view() (correct bit reinterpretation) vs astype()
    # (incorrect integer-to-float VALUE CAST).
    raw_u16 = load_bin_fp16_as_uint16(path)
    view_f16 = raw_u16.view(np.float16)
    cast_f16 = raw_u16.astype(np.float16)

    # idx 1 = 0x3C00 = 15360 (decimal).  view() → 1.0.  astype() → 15360.0.
    # These MUST differ for the test to be meaningful.
    sample_bit = raw_u16[1]  # 0x3C00 = 15360
    v = view_f16[1]
    c = cast_f16[1]
    assert v != c, (
        f"ANTI-REGRESSION FAILURE: view(0x{sample_bit:04X}) = {v} "
        f"equals cast(0x{sample_bit:04X}) = {c}. "
        f"This means the test value does NOT expose the .view() vs .astype() bug. "
        f"0x{sample_bit:04X} should map to fp16=1.0 via view, not {c} via cast."
    )
    print(f"    Anti-regression: view(0x{sample_bit:04X})={v}, cast(0x{sample_bit:04X})={c} -- DIFFERENT (correct)")

    # Additional: verify view matches direct float16 loading
    assert np.array_equal(view_f16, data_f16), "view() path must match direct load"

    print("  [PASS] fp16 cross-language round-trip")
    return True


# ═══════════════════════════════════════════════════════════════════
# Test: cross-language int64  (Format A)
# ═══════════════════════════════════════════════════════════════════

def test_cross_language_int64():
    """Read C++-written int64 .bin and verify each token ID."""
    path = '/tmp/test_bin_int64.bin'
    if not os.path.exists(path):
        print("  [SKIP] int64 cross-language: C++ output not found (run test_bin_format first)")
        return False

    data = load_bin_int64(path)
    print(f"  int64 cross-language: count={len(data)}")

    # Must match C++ test: [151643, 151652, 151655, 0, -1, INT64_MAX, INT64_MIN]
    expected = [
        151643,
        151652,
        151655,
        0,
        -1,
        np.iinfo(np.int64).max,
        np.iinfo(np.int64).min,
    ]
    assert len(data) == len(expected), f"count mismatch: {len(data)} vs {len(expected)}"
    for i, exp in enumerate(expected):
        assert data[i] == exp, f"idx {i}: expected {exp}, got {data[i]}"

    print("  [PASS] int64 cross-language round-trip")
    return True


# ═══════════════════════════════════════════════════════════════════
# Test: cross-language fp32  (Format C)
# ═══════════════════════════════════════════════════════════════════

def test_cross_language_fp32():
    """Read C++-written fp32 .bin and verify each value."""
    path = '/tmp/test_bin_fp32.bin'
    if not os.path.exists(path):
        print("  [SKIP] fp32 cross-language: C++ output not found (run test_bin_format first)")
        return False

    data = load_bin_fp32(path)
    print(f"  fp32 cross-language: dim={len(data)}")

    expected = [0.0, 1.0, -1.0, 3.14159, 1e-7, 1e7]
    assert len(data) == len(expected), f"count mismatch: {len(data)} vs {len(expected)}"
    for i, exp in enumerate(expected):
        assert abs(data[i] - exp) < 1e-6, f"idx {i}: expected {exp}, got {data[i]}"

    print("  [PASS] fp32 cross-language round-trip")
    return True


# ═══════════════════════════════════════════════════════════════════
# Test: Python fp16 self round-trip
# ═══════════════════════════════════════════════════════════════════

def test_python_round_trip_fp16():
    """Python write -> read -> verify fp16."""
    original = np.array([0.0, 1.0, -1.0, 3.0, 4.0, -3.0, 65504.0],
                        dtype=np.float16)
    save_bin_fp16('/tmp/test_bin_py_fp16.bin', original)
    loaded = load_bin_fp16('/tmp/test_bin_py_fp16.bin')
    assert np.array_equal(original, loaded), (
        f"Python fp16 self round-trip failed\n  orig: {original}\n  loaded: {loaded}"
    )
    print("  [PASS] Python fp16 self round-trip")
    return True


# ═══════════════════════════════════════════════════════════════════
# Test: Python int64 self round-trip
# ═══════════════════════════════════════════════════════════════════

def test_python_round_trip_int64():
    """Python write -> read -> verify int64."""
    original = np.array([151643, 151652, 151655, 0, -1,
                         np.iinfo(np.int64).max, np.iinfo(np.int64).min],
                        dtype=np.int64)
    save_bin_int64('/tmp/test_bin_py_int64.bin', original)
    loaded = load_bin_int64('/tmp/test_bin_py_int64.bin')
    assert np.array_equal(original, loaded), (
        f"Python int64 self round-trip failed\n  orig: {original}\n  loaded: {loaded}"
    )
    print("  [PASS] Python int64 self round-trip")
    return True


# ═══════════════════════════════════════════════════════════════════
# Test: Python fp32 self round-trip
# ═══════════════════════════════════════════════════════════════════

def test_python_round_trip_fp32():
    """Python write -> read -> verify fp32."""
    original = np.array([0.0, 1.0, -1.0, 3.14159, 1e-7, 1e7, -2.71828],
                        dtype=np.float32)
    save_bin_fp32('/tmp/test_bin_py_fp32.bin', original)
    loaded = load_bin_fp32('/tmp/test_bin_py_fp32.bin')
    assert np.array_equal(original, loaded), (
        f"Python fp32 self round-trip failed\n  orig: {original}\n  loaded: {loaded}"
    )
    print("  [PASS] Python fp32 self round-trip")
    return True


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=== G1: .bin format round-trip tests ===\n")

    results = []

    # Cross-language (requires C++ test to have run first)
    for name, fn in [
        ('fp16 cross-language', test_cross_language_fp16),
        ('int64 cross-language', test_cross_language_int64),
        ('fp32 cross-language', test_cross_language_fp32),
    ]:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results.append((name, False))

    # Python self round-trips (always run)
    for name, fn in [
        ('fp16 self round-trip', test_python_round_trip_fp16),
        ('int64 self round-trip', test_python_round_trip_int64),
        ('fp32 self round-trip', test_python_round_trip_fp32),
    ]:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results.append((name, False))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n=== Summary: {passed}/{total} passed ===")
    if passed < total:
        failed = [name for name, ok in results if not ok]
        print(f"  Failed: {', '.join(failed)}")
    sys.exit(0 if passed == total else 1)
