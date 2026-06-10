"""
Step-by-step Python engine diagnostic — saves intermediate outputs.
"""
import sys
import struct
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
set_atb_buffer_size(10 * 1024 * 1024 * 1024)

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.preprocess import preprocess_image

IMG_H, IMG_W, IMG_C = 720, 1280, 3


def save_fp16(path, tensor):
    flat = tensor.flatten().half().cpu()
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
    engine = Qwen3VLEngine(MODEL_DIR)
    img_tok = engine.img_tok
    spatial_merge = engine.spatial_merge

    img = create_test_image(IMG_C, IMG_H, IMG_W)
    pv, grid_thw = preprocess_image(img)
    num_patches = pv.shape[0]
    merged_tokens = num_patches // (spatial_merge ** 2)

    print(f"grid_thw={grid_thw[0].tolist()}, patches={num_patches}, merged={merged_tokens}")

    # ── Run vision model ──
    print("\n[1] Vision model output")
    vis_result = engine._run_vision(pv, grid_thw)
    vis_embeds = vis_result[0]  # (880, 2048) fp16 NPU tensor
    ds_features = vis_result[1]  # list of deepstack features
    print(f"  Merged shape: {vis_embeds.shape}, dtype: {vis_embeds.dtype}")
    print(f"  Deepstack features: {len(ds_features)} layers")
    for i, ds in enumerate(ds_features):
        print(f"    DS[{i}]: {ds.shape}")
    save_fp16("/tmp/py_vision_merged.bin", vis_embeds)

    # Save deepstack features
    for i, ds in enumerate(ds_features):
        save_fp16(f"/tmp/py_ds_{i}.bin", ds)

    # ── Run full IMAGE_ONLY pipeline with step-by-step tracking ──
    print("\n[2] IMAGE_ONLY inputs_embeds after scatter")
    input_ids = torch.tensor([[img_tok] * merged_tokens], dtype=torch.long)
    inputs_embeds = F.embedding(input_ids.cpu(), engine.embed_w.cpu()).half().npu()
    vis_mask = input_ids.squeeze(0) == img_tok
    inputs_embeds[0, vis_mask.npu(), :] = vis_embeds
    save_fp16("/tmp/py_inputs_embeds_scatter.bin", inputs_embeds)

    # ── Position IDs ──
    print("\n[3] Position IDs")
    from atb_python_qwen3vl_embedding.engine_utils import get_rope_index
    position_ids = get_rope_index(input_ids, grid_thw, img_tok, spatial_merge)
    with open("/tmp/py_position_ids.bin", 'wb') as f:
        pid_total = position_ids.numel()
        f.write(struct.pack('q', pid_total))
        f.write(position_ids.long().cpu().numpy().tobytes())
    print(f"  position_ids: {position_ids.shape}")
    print(f"  D=0: first 5 = {position_ids[0, :5].tolist()}")
    print(f"  D=1: first 5 = {position_ids[1, :5].tolist()}")
    print(f"  D=2: first 5 = {position_ids[2, :5].tolist()}")


if __name__ == "__main__":
    main()
