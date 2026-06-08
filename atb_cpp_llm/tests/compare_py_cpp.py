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
# Known timing data (hardcoded)
# ---------------------------------------------------------------------------
cpp_times: Dict[str, float] = {
    "text_only": 10.39,
    "text_only_100": 13.08,
    "text_only_512": 29.07,
    "text_only_1024": 56.83,
    "text_only_2048": 133.91,
    "text_only_4096": 425.51,
    "io_416x672": 41.17,
    "io_720x1280": 99.31,
    "io_1080x1920": 144.86,
    "io_1440x2560": 145.52,
    "mm_416x672": 41.75,
    "mm_720x1280": 99.73,
    "mm_1080x1920": 146.76,
    "mm_1440x2560": 145.29,
}

py_times: Dict[str, float] = {
    "text_only": 15.13,
    "text_only_100": 21.00,
    "text_only_512": 34.71,
    "text_only_1024": 51.09,
    "text_only_2048": 76.56,
    "text_only_4096": 157.39,
    "io_416x672": 57.62,
    "io_720x1280": 101.46,
    "io_1080x1920": 141.22,
    "io_1440x2560": 135.70,
    "mm_416x672": 57.4,
    "mm_720x1280": 100.7,
    "mm_1080x1920": 133.8,
    "mm_1440x2560": 133.0,
}

# ---------------------------------------------------------------------------
# Test case definitions: (display_label, cpp_key, py_key)
# ---------------------------------------------------------------------------
test_cases = [
    ("TEXT 100",      "text_only_100",   "text_only_100"),
    ("TEXT 512",      "text_only_512",   "text_only_512"),
    ("TEXT 1024",     "text_only_1024",  "text_only_1024"),
    ("TEXT 2048",     "text_only_2048",  "text_only_2048"),
    ("TEXT 4096",     "text_only_4096",  "text_only_4096"),
    ("TEXT_ONLY",     "text_only",    "text_only"),
    ("IO 416x672",    "io_416x672",   "io_416x672"),
    ("IO 720x1280",   "io_720x1280",  "io_720x1280"),
    ("IO 1080x1920",  "io_1080x1920", "io_1080x1920"),
    ("IO 1440x2560",  "io_1440x2560", "io_1440x2560"),
    ("MM 416x672",    "mm_416x672",   "mm_416x672"),
    ("MM 720x1280",   "mm_720x1280",  "mm_720x1280"),
    ("MM 1080x1920",  "mm_1080x1920", "mm_1080x1920"),
    ("MM 1440x2560",  "mm_1440x2560", "mm_1440x2560"),
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
    header_width = 70

    print("=" * header_width)
    print("Python ATB vs C++ ATB — Embedding Comparison")
    print("=" * header_width)
    print()

    fmt = "{:<20s} {:>9s}  {:>7s}  {:>7s}  {:>10s}"
    print(fmt.format("Mode", "C++(ms)", "Py(ms)", "Speedup", "Cosine"))
    print("-" * header_width)

    results = []  # (label, cpp_ms, py_ms, speedup, cosine_str, passed)
    passed_count = 0
    total_count = 0

    for label, cpp_key, py_key in test_cases:
        cpp_path = os.path.join(BIN_DIR, f"cpp_{cpp_key}.bin")
        py_path = os.path.join(BIN_DIR, f"py_{py_key}.bin")

        cpp_ms = cpp_times.get(cpp_key)
        py_ms = py_times.get(py_key)

        try:
            cpp_emb, cpp_dim = load_embedding(cpp_path)
        except FileNotFoundError:
            results.append((label, cpp_ms, py_ms, None, "MISSING", None))
            total_count += 1
            continue

        try:
            py_emb, py_dim = load_embedding(py_path)
        except FileNotFoundError:
            results.append((label, cpp_ms, py_ms, None, "MISSING", None))
            total_count += 1
            continue

        if cpp_dim != py_dim:
            print(f"[WARN] {label}: dimension mismatch (C++ {cpp_dim} vs Python {py_dim})")
            results.append((label, cpp_ms, py_ms, None, "DIM MISMATCH", None))
            total_count += 1
            continue

        cos_sim = cosine_similarity(cpp_emb, py_emb)
        passed = cos_sim >= PASS_THRESHOLD
        total_count += 1
        if passed:
            passed_count += 1

        # Format columns
        cpp_str = f"{cpp_ms:.2f}" if cpp_ms is not None else "N/A"
        py_str = f"{py_ms:.2f}" if py_ms is not None else "N/A"
        if cpp_ms is not None and py_ms is not None and py_ms > 0:
            speedup = cpp_ms / py_ms
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup = None
            speedup_str = "N/A"

        cosine_str = f"{cos_sim:.6f}"
        status_flag = "" if passed else "  <<< FAIL"

        results.append((label, cpp_str, py_str, speedup_str, cosine_str, passed))
        print(fmt.format(label, cpp_str, py_str, speedup_str, cosine_str) + status_flag)

    # -- Summary --
    print()
    print("=" * header_width)
    print("Summary")
    print("=" * header_width)

    if passed_count == total_count:
        print(f"All {total_count}/{total_count} tests PASSED (cosine >= {PASS_THRESHOLD})")
    else:
        failed = total_count - passed_count
        print(f"{passed_count}/{total_count} tests PASSED, {failed} FAILED (cosine < {PASS_THRESHOLD})")
        print()
        print("Failing tests:")
        for label, _, _, _, cos_str, passed in results:
            if passed is False:
                print(f"  - {label}: cosine = {cos_str}")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
