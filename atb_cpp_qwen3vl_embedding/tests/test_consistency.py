"""
C++ vs Python consistency test.

Loads the C++ engine output from /tmp/cpp_embedding.bin and compares
it against the Python ATB engine output using cosine similarity.

Prerequisites:
    1. Run ./test_consistency (C++ side) first to generate /tmp/cpp_embedding.bin
    2. Model checkpoint at /mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B/

Usage:
    python tests/test_consistency.py
"""

import struct
import sys
import numpy as np
import torch
import torch.nn.functional as F

# ── Configuration ────────────────────────────────────────────────
CPP_OUTPUT_PATH = "/tmp/cpp_embedding.bin"
MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B"
COSINE_THRESHOLD = 0.99


def load_cpp_embedding(path: str) -> tuple:
    """Load C++ embedding from binary file.

    Returns (dim: int, data: np.ndarray[float32])
    """
    with open(path, "rb") as f:
        dim = struct.unpack("q", f.read(8))[0]
        data = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return dim, data


def run_python_engine() -> torch.Tensor:
    """Run Python ATB engine with the same input and return embedding."""
    # Set buffer size before importing engine
    sys.path.insert(0, "/mnt/workspace/gitCode/atb_llm")
    from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)

    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine

    engine = Qwen3VLEngine(MODEL_DIR)

    # Same input_ids as C++ test: [151643, 15339, 1879]
    input_ids = torch.tensor([[151643, 15339, 1879]], dtype=torch.long)

    # encode() does: forward -> pool last token -> L2 normalize
    result = engine.encode(input_ids, normalize=True)
    return result.flatten()


def main():
    print("=" * 60)
    print("C++ vs Python Consistency Test")
    print("=" * 60)

    # ── Load C++ output ──────────────────────────────────────────
    print(f"\n[1] Loading C++ embedding from {CPP_OUTPUT_PATH}")
    try:
        cpp_dim, cpp_data = load_cpp_embedding(CPP_OUTPUT_PATH)
    except FileNotFoundError:
        print(f"ERROR: {CPP_OUTPUT_PATH} not found.")
        print("Run the C++ test first: ./test_consistency")
        sys.exit(1)

    cpp_emb = torch.from_numpy(cpp_data).float()
    print(f"    C++ embedding dim: {cpp_dim}")

    # ── Run Python engine ────────────────────────────────────────
    print(f"\n[2] Running Python ATB engine")
    py_emb = run_python_engine().float()
    print(f"    Python embedding dim: {py_emb.shape[0]}")

    # ── Compare ──────────────────────────────────────────────────
    print(f"\n[3] Comparison:")
    print(f"    dim: C++={cpp_dim}, Python={py_emb.shape[0]}")

    if cpp_dim != py_emb.shape[0]:
        print(f"FAIL: dimension mismatch")
        sys.exit(1)

    cos = F.cosine_similarity(cpp_emb, py_emb, dim=0).item()
    mse = F.mse_loss(cpp_emb, py_emb).item()
    max_diff = (cpp_emb - py_emb).abs().max().item()

    print(f"    cosine similarity: {cos:.6f}")
    print(f"    MSE:               {mse:.2e}")
    print(f"    max abs diff:      {max_diff:.2e}")

    # ── Print first 8 values ─────────────────────────────────────
    print(f"\n[4] First 8 values:")
    cpp_first8 = cpp_data[:8].tolist()
    py_first8 = py_emb[:8].tolist()
    print(f"    C++:    {[f'{v:.6f}' for v in cpp_first8]}")
    print(f"    Python: {[f'{v:.6f}' for v in py_first8]}")

    # ── Verdict ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if cos > COSINE_THRESHOLD:
        print(f"PASS: cosine similarity {cos:.6f} > {COSINE_THRESHOLD}")
        sys.exit(0)
    else:
        print(f"FAIL: cosine similarity {cos:.6f} < {COSINE_THRESHOLD}")
        sys.exit(1)


if __name__ == "__main__":
    main()
