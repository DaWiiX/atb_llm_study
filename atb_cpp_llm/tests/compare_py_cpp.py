#!/usr/bin/env python3
"""
Compare C++ and Python ATB embedding output vectors.

Loads pre-generated .bin files from /tmp/, computes cosine similarity
between C++ and Python embeddings, and prints a summary comparison table.

Usage:
    python tests/compare_py_cpp.py
"""

import math
import os
import struct
import sys
from typing import Dict
from typing import Optional
from typing import Tuple


# ---------------------------------------------------------------------------
# Timing data — C++ (P8: fp16 mask + NPU cache, 5 warmup + 3 iter mean)
# "cold" = first inference (cache miss), "warm" = cached inference
# E2E numbers are warm-cache (benchmark mean after warmup).
# ---------------------------------------------------------------------------
cpp_times: Dict[str, float] = {
    # TEXT_ONLY — warm cache
    "text_only_100":      12.20,
    "text_only_512":      21.56,
    "text_only_1024":     34.76,
    "text_only_2048":     62.32,
    "text_only_4096":    126.29,
    # IMAGE_ONLY — warm cache
    "io_416x672":    38.08,
    "io_720x1280":   82.90,
    "io_1080x1920": 115.50,
    "io_1440x2560": 115.44,
    # IMAGE_AND_TEXT — warm cache
    "mm_416x672":    38.50,
    "mm_720x1280":   84.05,
    "mm_1080x1920": 114.42,
    "mm_1440x2560": 115.17,
}

# Cold-start H2D prep times (first warmup iteration, no cache)
# These show the cost of mask/cos/sin generation + upload when seq_len changes.
cpp_cold_h2d: Dict[str, float] = {
    "text_only_100":      0.36,   # mask 100² = tiny
    "text_only_512":      0.81,   # mask 512² = 262K
    "text_only_1024":     1.17,   # mask 1024² = 1M
    "text_only_2048":     3.67,   # mask 2048² = 4M
    "text_only_4096":    43.13,   # mask 4096² = 16.8M  ← first-frame cost
    "io_416x672":    0.45,   # S=273, mask small
    "io_720x1280":   0.64,   # S=880, mask medium
    "io_1080x1920":  3.37,   # S=1222, mask larger
    "io_1440x2560":  3.25,   # S=1222, mask larger
    "mm_416x672":    0.60,   # S=299  (273 vis + 26 text)
    "mm_720x1280":   0.64,   # S=906  (880 vis + 26 text)
    "mm_1080x1920":  3.87,   # S=1248 (1222 vis + 26 text)
    "mm_1440x2560":  3.98,   # S=1248 (1222 vis + 26 text)
}

py_times: Dict[str, float] = {
    # TEXT_ONLY (Python ATB)
    "text_only_100":      22.56,
    "text_only_512":      36.92,
    "text_only_1024":     48.34,
    "text_only_2048":     75.66,
    "text_only_4096":    152.67,
    # IMAGE_ONLY (Python ATB)
    "io_416x672":    58.00,
    "io_720x1280":  102.35,
    "io_1080x1920": 134.05,
    "io_1440x2560": 139.04,
    # IMAGE_AND_TEXT (Python ATB)
    "mm_416x672":    56.86,
    "mm_720x1280":  105.18,
    "mm_1080x1920": 134.31,
    "mm_1440x2560": 134.32,
}

# ---------------------------------------------------------------------------
# Test case definitions: (display_label, cpp_key, py_key)
# ---------------------------------------------------------------------------
test_cases = [
    ("TEXT 100",      "text_only_100",      "text_only_100"),
    ("TEXT 512",      "text_only_512",      "text_only_512"),
    ("TEXT 1024",     "text_only_1024",     "text_only_1024"),
    ("TEXT 2048",     "text_only_2048",     "text_only_2048"),
    ("TEXT 4096",     "text_only_4096",     "text_only_4096"),
    ("IO 416x672",    "io_416x672",    "io_416x672"),
    ("IO 720x1280",   "io_720x1280",   "io_720x1280"),
    ("IO 1080x1920",  "io_1080x1920",  "io_1080x1920"),
    ("IO 1440x2560",  "io_1440x2560",  "io_1440x2560"),
    ("MM 416x672",    "mm_416x672",    "mm_416x672"),
    ("MM 720x1280",   "mm_720x1280",   "mm_720x1280"),
    ("MM 1080x1920",  "mm_1080x1920",  "mm_1080x1920"),
    ("MM 1440x2560",  "mm_1440x2560",  "mm_1440x2560"),
]

BIN_DIR = "/tmp"
PASS_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embedding(path: str) -> Tuple[list, int]:
    """Load a float32 embedding vector from a .bin file.

    File format:
        int64 dim (little-endian)
        float32[dim] data (little-endian)

    Returns (data_list, dim).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        (dim,) = struct.unpack("<q", f.read(8))
        fmt = f"<{dim}f"
        data = list(struct.unpack(fmt, f.read(dim * 4)))

    return data, dim


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two equal-length float lists.

    Returns 0.0 if either vector has zero norm.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    header_width = 85

    print("=" * header_width)
    print("Python ATB vs C++ ATB — Embedding Cosine Comparison (P8 baseline)")
    print("=" * header_width)
    print()

    fmt = "{:<18s} {:>9s}  {:>8s}  {:>7s}  {:>10s}  {:>10s}"
    print(fmt.format("Mode", "C++ warm", "Py ATB", "Speedup", "Cosine", "Cold H2D"))
    print("-" * header_width)

    results = []
    passed_count = 0
    total_count = 0

    for label, cpp_key, py_key in test_cases:
        cpp_path = os.path.join(BIN_DIR, f"cpp_{cpp_key}.bin")
        py_path = os.path.join(BIN_DIR, f"py_{py_key}.bin")

        cpp_ms = cpp_times.get(cpp_key)
        py_ms = py_times.get(py_key)
        cold_h2d = cpp_cold_h2d.get(cpp_key, 0)

        try:
            cpp_emb, cpp_dim = load_embedding(cpp_path)
        except FileNotFoundError:
            results.append((label, cpp_ms, py_ms, None, "MISSING", None, cold_h2d))
            total_count += 1
            continue

        try:
            py_emb, py_dim = load_embedding(py_path)
        except FileNotFoundError:
            results.append((label, cpp_ms, py_ms, None, "MISSING", None, cold_h2d))
            total_count += 1
            continue

        if cpp_dim != py_dim:
            results.append((label, cpp_ms, py_ms, None, "DIM MISMATCH", None, cold_h2d))
            total_count += 1
            continue

        cos_sim = cosine_similarity(cpp_emb, py_emb)
        passed = cos_sim >= PASS_THRESHOLD
        total_count += 1
        if passed:
            passed_count += 1

        cpp_str = f"{cpp_ms:.2f}" if cpp_ms is not None else "N/A"
        py_str = f"{py_ms:.2f}" if py_ms is not None else "N/A"
        if cpp_ms is not None and py_ms is not None and py_ms > 0:
            speedup = cpp_ms / py_ms
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup = None
            speedup_str = "N/A"

        cosine_str = f"{cos_sim:.6f}"
        cold_str = f"{cold_h2d:.2f}ms"
        status_flag = "" if passed else "  <<< FAIL"

        results.append((label, cpp_str, py_str, speedup_str, cosine_str, passed, cold_h2d))
        print(fmt.format(label, cpp_str, py_str, speedup_str, cosine_str, cold_str) + status_flag)

    # -- Cross-framework E2E summary table --
    print()
    print("=" * header_width)
    print("E2E Comparison Summary (warm cache)")
    print("=" * header_width)

    e2e_fmt = "{:<18s} {:>8s} {:>8s}  {:>9s}  {:>9s}  {:>7s}  {:>10s}"
    print(e2e_fmt.format("Mode", "S", "VisTok", "C++ (ms)", "Python (ms)", "C++/Py", "Cold H2D"))
    print("-" * header_width)

    # Order: TEXT x5, IO x4, MM x4
    order = [
        ("TEXT 100",     100,    0, "text_only_100"),
        ("TEXT 512",     512,    0, "text_only_512"),
        ("TEXT 1024",   1024,    0, "text_only_1024"),
        ("TEXT 2048",   2048,    0, "text_only_2048"),
        ("TEXT 4096",   4096,    0, "text_only_4096"),
        ("IO 416x672",   273,  273, "io_416x672"),
        ("IO 720x1280",  880,  880, "io_720x1280"),
        ("IO 1080x1920", 1222, 1222, "io_1080x1920"),
        ("IO 1440x2560", 1222, 1222, "io_1440x2560"),
        ("MM 416x672",   299,  273, "mm_416x672"),
        ("MM 720x1280",  906,  880, "mm_720x1280"),
        ("MM 1080x1920", 1248, 1222, "mm_1080x1920"),
        ("MM 1440x2560", 1248, 1222, "mm_1440x2560"),
    ]

    for label, seq, vis, key in order:
        cpp_ms = cpp_times.get(key)
        py_ms = py_times.get(key)
        cold_h2d = cpp_cold_h2d.get(key, 0)
        cpp_s = f"{cpp_ms:.2f}" if cpp_ms else "N/A"
        py_s = f"{py_ms:.2f}" if py_ms else "N/A"
        if cpp_ms and py_ms and py_ms > 0:
            ratio = f"{cpp_ms/py_ms:.2f}x"
        else:
            ratio = "N/A"
        cold_s = f"{cold_h2d:.2f}ms"
        print(e2e_fmt.format(label, str(seq), str(vis), cpp_s, py_s, ratio, cold_s))

    # -- Summary --
    print()
    print("=" * header_width)
    print("Cosine Summary")
    print("=" * header_width)

    if passed_count == total_count:
        print(f"All {total_count}/{total_count} tests PASSED (cosine >= {PASS_THRESHOLD})")
    else:
        failed = total_count - passed_count
        print(f"{passed_count}/{total_count} tests PASSED, {failed} FAILED (cosine < {PASS_THRESHOLD})")
        print()
        print("Failing tests:")
        for label, _, _, _, cos_str, passed, _ in results:
            if passed is False:
                print(f"  - {label}: cosine = {cos_str}")

    # -- Cold-start note --
    print()
    print("Note: 'Cold H2D' = first-inference mask/cos/sin generation + NPU upload time.")
    print("On subsequent inferences with same seq_len, this becomes ~0.1-0.5ms (cache hit).")
    print("C++ E2E times are warm-cache (post-warmup mean).")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
