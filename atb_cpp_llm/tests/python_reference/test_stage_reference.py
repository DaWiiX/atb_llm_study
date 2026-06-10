"""
Stage-by-stage reference value generator for C++/Python precision comparison.

Computes intermediate values at each pipeline stage and saves them to
/tmp/stage_*.bin for comparison with the C++ test_stage_precision binary.

Usage:
    python tests/test_stage_reference.py

Then run: ./test_stage_precision
"""

import os
import sys
import struct
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.preprocess import preprocess_image, smart_resize


def save_float32(path: str, data: np.ndarray, shape: list[int] | None = None):
    """Save float32 array to binary file: [ndim, shape..., data...]"""
    if shape is None:
        shape = list(data.shape)
    with open(path, "wb") as f:
        f.write(struct.pack("q", len(shape)))
        for s in shape:
            f.write(struct.pack("q", s))
        f.write(data.astype(np.float32).tobytes())


def save_int64(path: str, data: np.ndarray, shape: list[int] | None = None):
    """Save int64 array to binary file: [ndim, shape..., data...]"""
    if shape is None:
        shape = list(data.shape)
    with open(path, "wb") as f:
        f.write(struct.pack("q", len(shape)))
        for s in shape:
            f.write(struct.pack("q", s))
        f.write(data.astype(np.int64).tobytes())


def create_gradient_image(channels: int, height: int, width: int) -> torch.Tensor:
    """Same gradient image as C++ side."""
    image = torch.zeros(channels, height, width, dtype=torch.uint8)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                value = (h * 255 // height + w * 255 // width + c * 85) % 256
                image[c, h, w] = value
    return image


def main():
    print("=" * 60)
    print("Stage Reference Generator")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)
    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    print("Loading Python ATB engine...")
    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine loaded: {engine.n_layer} text layers, {engine.v_depth} vision blocks")

    img_h, img_w = 672, 476

    # ═══════════════════════════════════════════════════════════
    # Stage 1: Preprocessing
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 1: Preprocessing ---")
    img = create_gradient_image(3, img_h, img_w)
    pixel_values, grid_thw = preprocess_image(img)
    print(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw}")
    print(f"  first 8: {pixel_values[0, :8].tolist()}")

    save_float32("/tmp/stage_pixels.bin", pixel_values.numpy())
    save_int64("/tmp/stage_grid_thw.bin", grid_thw.numpy())

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Vision Position Embedding + RoPE
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 2: Vision Position Embedding ---")
    # We can get the vision RoPE from the engine's internal state
    # by calling _run_vision and capturing intermediates.
    # For now, we'll compute the position IDs that the text model uses.

    # ═══════════════════════════════════════════════════════════
    # Stage 3: Vision Model Output (merged embeddings)
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 3: Vision Model (merged embeddings) ---")
    # Run vision model through engine internals
    pv_npu = pixel_values.reshape(-1).half().npu()
    vis_embeds, ds_feats = engine._run_vision(pv_npu, grid_thw)
    torch.npu.synchronize()
    print(f"  vis_embeds: shape={vis_embeds.shape}, dtype={vis_embeds.dtype}")

    vis_cpu = vis_embeds.float().cpu().numpy()
    save_float32("/tmp/stage_vis_embeds.bin", vis_cpu)
    print(f"  first 8: {vis_cpu[0, :8].tolist()}")

    # ═══════════════════════════════════════════════════════════
    # Stage 4: Text Embedding Lookup (for IMAGE_ONLY)
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 4: Text Embedding Lookup ---")
    merged_tokens = pixel_values.shape[0] // (engine.spatial_merge ** 2)
    image_token_id = engine.img_tok
    input_ids_io = torch.tensor([[image_token_id] * merged_tokens], dtype=torch.long)
    embeds_io = F.embedding(input_ids_io, engine.embed_w).half()
    print(f"  IMAGE_ONLY embeds: shape={embeds_io.shape}")
    embeds_io_cpu = embeds_io.float().cpu().numpy()
    save_float32("/tmp/stage_embeds_image_only.bin", embeds_io_cpu.reshape(-1, embeds_io.shape[-1]))
    print(f"  first 8 (pos 0): {embeds_io_cpu[0, 0, :8].tolist()}")

    # For IMAGE_AND_TEXT
    input_ids_mt = torch.tensor([[151643] + [image_token_id] * merged_tokens + [15339, 1879]],
                                 dtype=torch.long)
    embeds_mt = F.embedding(input_ids_mt, engine.embed_w).half()
    embeds_mt_cpu = embeds_mt.float().cpu().numpy()
    save_float32("/tmp/stage_embeds_image_text.bin", embeds_mt_cpu.reshape(-1, embeds_mt.shape[-1]))
    print(f"  IMAGE_AND_TEXT embeds: shape={embeds_mt.shape}")

    # ═══════════════════════════════════════════════════════════
    # Stage 5: Position Encoding (MRoPE)
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 5: Position Encoding (MRoPE) ---")
    from atb_python_qwen3vl_embedding.engine import get_rope_index

    # IMAGE_ONLY
    pos_ids_io, _ = get_rope_index(
        input_ids_io, grid_thw, None, None,
        image_token_id=image_token_id,
        spatial_merge_size=engine.spatial_merge)
    print(f"  IMAGE_ONLY position_ids: shape={pos_ids_io.shape}")
    save_int64("/tmp/stage_pos_ids_image_only.bin", pos_ids_io.numpy())

    # IMAGE_AND_TEXT
    pos_ids_mt, _ = get_rope_index(
        input_ids_mt, grid_thw, None, None,
        image_token_id=image_token_id,
        spatial_merge_size=engine.spatial_merge)
    print(f"  IMAGE_AND_TEXT position_ids: shape={pos_ids_mt.shape}")
    save_int64("/tmp/stage_pos_ids_image_text.bin", pos_ids_mt.numpy())

    # ═══════════════════════════════════════════════════════════
    # Stage 6: Full Pipeline Outputs
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 6: Full Pipeline ---")

    # IMAGE_ONLY
    result_io = engine.encode(input_ids_io, pixel_values=pixel_values,
                               image_grid_thw=grid_thw, normalize=True)
    emb_io = result_io.flatten().float().cpu().numpy()
    save_float32("/tmp/stage_final_image_only.bin", emb_io)
    print(f"  IMAGE_ONLY final: shape={emb_io.shape}, first 8: {emb_io[:8].tolist()}")

    # IMAGE_AND_TEXT
    result_mt = engine.encode(input_ids_mt, pixel_values=pixel_values,
                               image_grid_thw=grid_thw, normalize=True)
    emb_mt = result_mt.flatten().float().cpu().numpy()
    save_float32("/tmp/stage_final_image_text.bin", emb_mt)
    print(f"  IMAGE_AND_TEXT final: shape={emb_mt.shape}, first 8: {emb_mt[:8].tolist()}")

    # TEXT_ONLY
    input_ids_t = torch.tensor([[151643, 15339, 1879]], dtype=torch.long)
    result_t = engine.encode(input_ids_t, normalize=True)
    emb_t = result_t.flatten().float().cpu().numpy()
    save_float32("/tmp/stage_final_text_only.bin", emb_t)
    print(f"  TEXT_ONLY final: shape={emb_t.shape}, first 8: {emb_t[:8].tolist()}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Reference values saved to /tmp/stage_*.bin")
    print("Now run: ./test_stage_precision")
    print("=" * 60)


if __name__ == "__main__":
    main()
