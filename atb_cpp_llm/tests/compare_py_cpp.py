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
from typing import Tuple


# ---------------------------------------------------------------------------
# Timing data is no longer hardcoded. To populate timing columns, parse
# benchmark output files. The BENCH_RESULT line format is:
#   BENCH_RESULT <key> <warm_e2e_ms>
# The BENCH_COLD line format is:
#   BENCH_COLD <key> <h2d_ms>
# See atb_cpp_llm/tests/benchmark.cpp and atb_python_qwen3vl_embedding/tests/benchmark.py
# for the benchmark programs that produce these lines.
# ---------------------------------------------------------------------------

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
    header_width = 65

    print("=" * header_width)
    print("Python ATB vs C++ ATB — Embedding Cosine Comparison")
    print("=" * header_width)
    print()

    fmt = "{:<18s} {:>10s}"
    print(fmt.format("Mode", "Cosine"))
    print("-" * header_width)

    results = []
    cosine_by_label: Dict[str, str] = {}
    passed_count = 0
    total_count = 0

    for label, cpp_key, py_key in test_cases:
        cpp_path = os.path.join(BIN_DIR, f"cpp_{cpp_key}.bin")
        py_path = os.path.join(BIN_DIR, f"py_{py_key}.bin")

        try:
            cpp_emb, cpp_dim = load_embedding(cpp_path)
        except FileNotFoundError:
            results.append((label, "MISSING", None))
            cosine_by_label[label] = "MISSING"
            total_count += 1
            continue

        try:
            py_emb, py_dim = load_embedding(py_path)
        except FileNotFoundError:
            results.append((label, "MISSING", None))
            cosine_by_label[label] = "MISSING"
            total_count += 1
            continue

        if cpp_dim != py_dim:
            results.append((label, "DIM MISMATCH", None))
            cosine_by_label[label] = "DIM MISMATCH"
            total_count += 1
            continue

        cos_sim = cosine_similarity(cpp_emb, py_emb)
        passed = cos_sim >= PASS_THRESHOLD
        total_count += 1
        if passed:
            passed_count += 1

        cosine_str = f"{cos_sim:.6f}"
        status_flag = "" if passed else "  <<< FAIL"

        results.append((label, cosine_str, passed))
        cosine_by_label[label] = cosine_str
        print(fmt.format(label, cosine_str) + status_flag)

    # -- E2E Summary table (Mode / Sequence Length / Visual Tokens / Cosine) --
    print()
    print("=" * header_width)
    print("E2E Summary (Mode / Sequence Length / Visual Tokens / Cosine)")
    print("=" * header_width)

    e2e_fmt = "{:<18s} {:>8s} {:>8s}  {:>10s}"
    print(e2e_fmt.format("Mode", "S", "VisTok", "Cosine"))
    print("-" * header_width)

    # Order: TEXT x5, IO x4, MM x4
    order = [
        ("TEXT 100",     100,    0),
        ("TEXT 512",     512,    0),
        ("TEXT 1024",   1024,    0),
        ("TEXT 2048",   2048,    0),
        ("TEXT 4096",   4096,    0),
        ("IO 416x672",   273,  273),
        ("IO 720x1280",  880,  880),
        ("IO 1080x1920", 1222, 1222),
        ("IO 1440x2560", 1222, 1222),
        ("MM 416x672",   299,  273),
        ("MM 720x1280",  906,  880),
        ("MM 1080x1920", 1248, 1222),
        ("MM 1440x2560", 1248, 1222),
    ]

    for label, seq, vis in order:
        cos_str = cosine_by_label.get(label, "N/A")
        print(e2e_fmt.format(label, str(seq), str(vis), cos_str))

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
        for label, cos_str, passed in results:
            if passed is False:
                print(f"  - {label}: cosine = {cos_str}")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
