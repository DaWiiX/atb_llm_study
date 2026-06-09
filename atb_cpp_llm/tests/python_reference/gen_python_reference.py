"""
Generate Python reference data for per-stage precision diagnosis.

Saves to /tmp/:
  - python_pixel_values.bin: preprocessed pixel_values (fp16)
  - python_grid_thw.bin: grid_thw (3 x int64)
  - python_text_emb.bin: text-only embedding (fp32)
  - python_image_emb.bin: image-only embedding (fp32)
  - python_image_text_emb.bin: image+text embedding (fp32)
  - python_vision_out.bin: vision model output before merger (fp16)
"""

import sys
import struct
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/workspace/gitCode/atb_llm")
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
set_atb_buffer_size(10 * 1024 * 1024 * 1024)

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.preprocess import preprocess_image

MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B"
IMG_H, IMG_W, IMG_C = 720, 1280, 3
TOK_DESCRIBE, TOK_THE, TOK_IMAGE, TOK_DOT = 74785, 279, 2168, 13


def save_fp16_tensor(path, tensor):
    """Save tensor as fp16 binary: (dim int64) + (data fp16 bytes)"""
    flat = tensor.flatten()
    dim = flat.shape[0]
    fp16 = flat.half()
    with open(path, 'wb') as f:
        f.write(struct.pack('q', dim))
        f.write(fp16.numpy().tobytes())
    print(f"  Saved {path}: dim={dim}")


def save_fp32_tensor(path, tensor):
    """Save tensor as fp32 binary: (dim int64) + (data fp32 bytes)"""
    flat = tensor.flatten().float()
    dim = flat.shape[0]
    with open(path, 'wb') as f:
        f.write(struct.pack('q', dim))
        f.write(flat.numpy().tobytes())
    print(f"  Saved {path}: dim={dim}")


def create_test_image(channels, height, width):
    image = torch.zeros(channels, height, width, dtype=torch.uint8)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                value = (h * 255 // height + w * 255 // width + c * 85) % 256
                image[c, h, w] = value
    return image


def main():
    print("=" * 60)
    print("Generating Python reference data for per-stage diagnosis")
    print("=" * 60)

    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine: {engine.n_layer} text layers, {engine.v_depth} vision blocks")
    print(f"image_token_id={engine.img_tok}, spatial_merge={engine.spatial_merge}")

    # ── Step 1: Preprocessing ──
    print("\n[1] Preprocessing")
    img = create_test_image(IMG_C, IMG_H, IMG_W)
    pv, grid_thw = preprocess_image(img)
    print(f"  pixel_values shape: {pv.shape}")
    print(f"  grid_thw: {grid_thw}")
    save_fp16_tensor("/tmp/python_pixel_values.bin", pv)

    # Save grid_thw
    with open("/tmp/python_grid_thw.bin", 'wb') as f:
        for i in range(3):
            f.write(struct.pack('q', grid_thw[0, i].item()))
    print(f"  Saved /tmp/python_grid_thw.bin: {grid_thw[0].tolist()}")

    merged_tokens = pv.shape[0] // (engine.spatial_merge ** 2)
    print(f"  merged_tokens: {merged_tokens}")

    # ── Step 2: TEXT_ONLY ──
    print("\n[2] TEXT_ONLY")
    input_ids = torch.tensor([[TOK_DESCRIBE, TOK_THE, TOK_IMAGE, TOK_DOT]], dtype=torch.long)
    text_emb = engine.encode(input_ids, normalize=True)
    save_fp32_tensor("/tmp/python_text_emb.bin", text_emb)

    # ── Step 3: IMAGE_ONLY ──
    print("\n[3] IMAGE_ONLY")
    img_tok = engine.img_tok
    input_ids_img = torch.tensor([[img_tok] * merged_tokens], dtype=torch.long)
    image_emb = engine.encode(input_ids_img, pixel_values=pv, image_grid_thw=grid_thw, normalize=True)
    save_fp32_tensor("/tmp/python_image_emb.bin", image_emb)

    # ── Step 4: IMAGE_AND_TEXT ──
    print("\n[4] IMAGE_AND_TEXT")
    ids_list = [TOK_DESCRIBE] + [img_tok] * merged_tokens + [TOK_THE, TOK_IMAGE, TOK_DOT]
    input_ids_it = torch.tensor([ids_list], dtype=torch.long)
    it_emb = engine.encode(input_ids_it, pixel_values=pv, image_grid_thw=grid_thw, normalize=True)
    save_fp32_tensor("/tmp/python_image_text_emb.bin", it_emb)

    print("\nDone. All reference data saved to /tmp/python_*.bin")


if __name__ == "__main__":
    main()
