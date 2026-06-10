"""
C++ Engine Accuracy Test — Python comparison side.

Uses identical inputs as C++ test_accuracy:
  - Image: 720x1280 gradient pattern (triggers resize path)
  - Text: "Describe the image." → tokens [74785, 279, 2168, 13]

Tests:
  1. TEXT_ONLY: "Describe the image."
  2. IMAGE_ONLY: 720x1280 image only
  3. IMAGE_AND_TEXT: "Describe" + [image] + " the image."

All modes must achieve cosine similarity ≥ 0.99.

Prerequisites:
    1. Run ./test_accuracy (C++ side) first to generate /tmp/cpp_*.bin
    2. Model checkpoint at $QWEN3VL_EMB_MODEL_DIR (set in .env)

Usage:
    python tests/test_accuracy.py
"""

import struct
import sys
import numpy as np
import torch
import torch.nn.functional as F

# ── Configuration ────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402
COSINE_THRESHOLD = 0.99  # Same threshold for ALL modes, no exceptions

# Shared test constants (must match C++ test_accuracy.cpp exactly)
IMG_H = 720
IMG_W = 1280
IMG_C = 3

# "Describe the image." token IDs (from Qwen3-VL tokenizer)
TOK_DESCRIBE = 74785
TOK_THE = 279
TOK_IMAGE = 2168
TOK_DOT = 13


def load_cpp_embedding(path: str) -> tuple:
    """Load C++ embedding from binary file.

    Returns (dim: int, data: np.ndarray[float32])
    """
    with open(path, "rb") as f:
        dim = struct.unpack("q", f.read(8))[0]
        data = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return dim, data


def create_test_image(channels: int, height: int, width: int) -> torch.Tensor:
    """Create the same test image as C++ side.

    Returns (C, H, W) uint8 tensor
    """
    image = torch.zeros(channels, height, width, dtype=torch.uint8)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                value = (h * 255 // height + w * 255 // width + c * 85) % 256
                image[c, h, w] = value
    return image


def run_python_text_only(engine) -> torch.Tensor:
    """Run Python ATB engine with text-only input: 'Describe the image.'"""
    input_ids = torch.tensor([[TOK_DESCRIBE, TOK_THE, TOK_IMAGE, TOK_DOT]], dtype=torch.long)
    result = engine.encode(input_ids, normalize=True)
    return result.flatten()


def run_python_image_only(engine, preprocess_fn) -> torch.Tensor:
    """Run Python ATB engine with image-only input: 720x1280 image."""
    img = create_test_image(IMG_C, IMG_H, IMG_W)
    pv, grid_thw = preprocess_fn(img)

    num_patches = pv.shape[0]
    merged_tokens = num_patches // (engine.spatial_merge ** 2)
    image_token_id = engine.img_tok
    input_ids = torch.tensor([[image_token_id] * merged_tokens], dtype=torch.long)

    result = engine.encode(input_ids, pixel_values=pv, image_grid_thw=grid_thw, normalize=True)
    return result.flatten()


def run_python_image_text(engine, preprocess_fn) -> torch.Tensor:
    """Run Python ATB engine: 'Describe' + [image] + ' the image.'"""
    img = create_test_image(IMG_C, IMG_H, IMG_W)
    pv, grid_thw = preprocess_fn(img)

    num_patches = pv.shape[0]
    merged_tokens = num_patches // (engine.spatial_merge ** 2)
    image_token_id = engine.img_tok
    input_ids_list = [TOK_DESCRIBE] + [image_token_id] * merged_tokens + [TOK_THE, TOK_IMAGE, TOK_DOT]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    result = engine.encode(input_ids, pixel_values=pv, image_grid_thw=grid_thw, normalize=True)
    return result.flatten()


def compare(name: str, cpp_data: np.ndarray, py_emb: torch.Tensor,
            threshold: float = COSINE_THRESHOLD) -> bool:
    """Compare C++ and Python embeddings."""
    cpp_emb = torch.from_numpy(cpp_data).float()
    py_emb = py_emb.float()

    if cpp_emb.shape[0] != py_emb.shape[0]:
        print(f"  FAIL: dimension mismatch C++={cpp_emb.shape[0]}, Python={py_emb.shape[0]}")
        return False

    cos = F.cosine_similarity(cpp_emb, py_emb, dim=0).item()
    mse = F.mse_loss(cpp_emb, py_emb).item()
    max_diff = (cpp_emb - py_emb).abs().max().item()

    status = "PASS" if cos > threshold else "FAIL"
    print(f"  [{status}] {name}")
    print(f"    Cosine similarity: {cos:.6f}")
    print(f"    MSE:               {mse:.2e}")
    print(f"    Max abs diff:      {max_diff:.2e}")

    # Print first 8 values
    cpp_first8 = cpp_data[:8].tolist()
    py_first8 = py_emb[:8].tolist()
    print(f"    C++ first 8:    {[f'{v:.6f}' for v in cpp_first8]}")
    print(f"    Python first 8: {[f'{v:.6f}' for v in py_first8]}")

    return cos > threshold


def main():
    print("=" * 60)
    print("C++ Engine Accuracy Test — Python Comparison")
    print(f"Image: {IMG_H}x{IMG_W}, Text: 'Describe the image.'")
    print(f"Cosine threshold: {COSINE_THRESHOLD}")
    print("=" * 60)

    # ── Setup Python engine ────────────────────────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)

    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    from atb_python_qwen3vl_embedding.preprocess import preprocess_image

    print("\nLoading Python ATB engine...")
    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine loaded: {engine.n_layer} text layers, {engine.v_depth} vision blocks")
    print(f"image_token_id={engine.img_tok}, spatial_merge={engine.spatial_merge}")

    results = []

    # ── Test 1: TEXT_ONLY ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Test 1: TEXT_ONLY — 'Describe the image.'")
    print("=" * 60)
    try:
        cpp_dim, cpp_data = load_cpp_embedding("/tmp/cpp_text_only.bin")
        py_emb = run_python_text_only(engine)
        results.append(compare("TEXT_ONLY", cpp_data, py_emb))
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)

    # ── Test 2: IMAGE_ONLY ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Test 2: IMAGE_ONLY — {IMG_H}x{IMG_W} image")
    print("=" * 60)
    try:
        cpp_dim, cpp_data = load_cpp_embedding("/tmp/cpp_image_only.bin")
        py_emb = run_python_image_only(engine, preprocess_image)
        results.append(compare("IMAGE_ONLY", cpp_data, py_emb))
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)

    # ── Test 3: IMAGE_AND_TEXT ─────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Test 3: IMAGE_AND_TEXT — 'Describe' + [{IMG_H}x{IMG_W}] + ' the image.'")
    print("=" * 60)
    try:
        cpp_dim, cpp_data = load_cpp_embedding("/tmp/cpp_image_text.bin")
        py_emb = run_python_image_text(engine, preprocess_image)
        results.append(compare("IMAGE_AND_TEXT", cpp_data, py_emb))
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(False)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED — cosine < 0.99 indicates a bug, not acceptable")
        sys.exit(1)


if __name__ == "__main__":
    main()
