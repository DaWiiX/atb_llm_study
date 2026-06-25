#!/usr/bin/env python3
"""Generate official-transformers pixel_values reference for the 4 production
resolutions, for the C++ ``PreprocessImageNpu vs official transformers`` gate
in test_aclnn_bicubic_spike.cpp.

Why this generator exists
-------------------------
The C++ ``PreprocessImageNpu`` pipeline (SmartResize + NPU bicubic + rescale +
normalize + patchify) must match the *official* Qwen3VLEmbedder preprocessor.
This generator reproduces the genuine official chain (mirroring
``Qwen3VLEmbedder._preprocess_inputs`` / ``format_model_input`` in
qwen3_vl_embedding.py) so the C++ test can gate on
``cos(NPU, official) >= 0.99`` at all 4 production resolutions.

Input-image sharing (life line)
-------------------------------
The C++ test and this generator MUST consume the byte-exact same input image.
We therefore READ the already-generated shared input bins produced by
``gen_cpu_reference.py`` (``/tmp/bicubic_prod_<H>x<W>_input.bin``, fixed
``np.random.seed(2026)``) instead of regenerating images here. If those bins
are missing, we error out with a clear hint to run ``gen_cpu_reference.py``
first. This guarantees the cosine is meaningful.

Official chain (mirrors Qwen3VLEmbedder._preprocess_inputs)
-----------------------------------------------------------
``Qwen3VLProcessor.from_pretrained(MODEL_DIR, padding_side='right')`` loads the
processor. We build the same conversation ``format_model_input`` produces
(system + user), injecting ``"min_pixels": 4096, "max_pixels": 1800*32*32``
into the image content dict — the official Qwen3VLEmbedder MAX_PIXELS constant
(``1800 * IMAGE_FACTOR * IMAGE_FACTOR``, IMAGE_FACTOR=32). Then:

    text   = processor.apply_chat_template(conversations,
                add_generation_prompt=True, tokenize=False)
    images, _, _ = process_vision_info(conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True)
    inputs = processor(text=text, images=images, do_resize=False,
                return_tensors='pt')

``process_vision_info`` → ``fetch_image`` runs ``smart_resize`` with
``factor = image_patch_size * SPATIAL_MERGE_SIZE = 16*2 = 32`` and the injected
``max_pixels=1843200`` — byte-identical to the C++ ``SmartResize(factor=32,
max_pixels=1843200)``. ``do_resize=False`` tells the processor NOT to re-resize;
it only rescale(1/255) + normalize(0.5/0.5) + patchify. The model's
preprocessor_config.json writes ``max_pixels=1310720`` but the official embedder
ignores it (do_resize=False), so we do too.

Output format
-------------
``/tmp/official_pv_prod_<H>x<W>.bin`` ::
    [count: int32]            # == num_patches * patch_dim (== * 1536)
    [data: float32 * count]   # C-contiguous flatten of pixel_values (N, 1536)

Usage::
    python atb_cpp_llm/tests/python_reference/gen_official_pixel_values.py
"""
import os
import sys
from pathlib import Path

import numpy as np

# Bootstrap sys.path: repo root + local transformers checkout (if configured).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                       # tests/  -> _tests_env
from _tests_env import MODEL_DIR, TRANSFORMERS_SRC  # noqa: E402
if TRANSFORMERS_SRC:
    sys.path.insert(0, TRANSFORMERS_SRC)
sys.path.insert(0, str(HERE.parent.parent.parent))         # repo root

from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402
from qwen_vl_utils.vision_process import process_vision_info  # noqa: E402

OUTPUT_DIR = "/tmp"

# Official Qwen3VLEmbedder constants (qwen3_vl_embedding.py).
MAX_PIXELS = 1800 * 32 * 32   # 1800 * IMAGE_FACTOR * IMAGE_FACTOR, IMAGE_FACTOR=32
MIN_PIXELS = 4096

# The 4 production resolutions (H, W). Must match gen_cpu_reference.py's
# prod bicubic cases and the C++ spike test's `cases` table exactly, so the
# shared input bin names line up: /tmp/bicubic_prod_<H>x<W>_input.bin.
PROD_CASES = [(416, 672), (720, 1280), (1080, 1920), (1440, 2560)]


def read_shared_input_bin(h: int, w: int) -> np.ndarray:
    """Read /tmp/bicubic_prod_<H>x<W>_input.bin -> (3,H,W) uint8.

    The bin format (written by gen_cpu_reference.write_f32) is:
        [ndim: int64][shape: int64[ndim]][data: float32]
    Values are whole numbers in [0, 255] (quantized uint8 stored as f32), so
    casting back to uint8 is lossless and byte-exact with what the C++ test
    feeds to PreprocessImageNpu.
    """
    path = f"{OUTPUT_DIR}/bicubic_prod_{h}x{w}_input.bin"
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Shared input bin missing: {path}\n"
            f"Run `python atb_cpp_llm/tests/python_reference/gen_cpu_reference.py` "
            f"first (it generates the bicubic_prod_*_input.bin files with "
            f"np.random.seed(2026)).")
    with open(path, "rb") as f:
        ndim = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        shape = tuple(int(x) for x in np.frombuffer(f.read(8 * ndim), dtype=np.int64))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)
    # f32 (whole 0..255) -> uint8, lossless.
    return np.clip(np.round(data), 0, 255).astype(np.uint8)


def write_official_pv(path: str, pv_flat: np.ndarray):
    """Write [count: int32][data: float32 * count]."""
    count = np.int32(pv_flat.size)
    with open(path, "wb") as f:
        f.write(count.tobytes())
        f.write(pv_flat.astype(np.float32).tobytes())


def Image_from_array(u8_chw: np.ndarray):
    """(3,H,W) uint8 -> PIL RGB image (H,W,3)."""
    from PIL import Image
    u8_hwc = u8_chw.transpose(1, 2, 0)  # (H,W,3)
    return Image.fromarray(u8_hwc, mode="RGB")


def build_conversation(pil_image) -> list:
    """Mirror Qwen3VLEmbedder.format_model_input for a single image.

    Returns a batch-of-1 conversations list: [[system, user]]. The user image
    content dict injects the official min_pixels/max_pixels (the keys
    process_vision_info.fetch_image reads for smart_resize).
    """
    conversations = [[
        {"role": "system", "content": [
            {"type": "text", "text": "Represent the given image."}]},
        {"role": "user", "content": [
            {"type": "image", "image": pil_image,
             "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": "t"}]},
    ]]
    return conversations


def main():
    print("[gen] official pixel_values reference (official Qwen3VLEmbedder chain)")
    print(f"  chain: Qwen3VLProcessor.from_pretrained(padding_side='right') +")
    print(f"         process_vision_info(image_patch_size=16) + processor(do_resize=False)")
    print(f"  max_pixels={MAX_PIXELS} (official embedder constant, config ignored)")

    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR, padding_side='right')
    print(f"  processor: {type(processor).__name__}")

    for h, w in PROD_CASES:
        u8_chw = read_shared_input_bin(h, w)            # (3,H,W) uint8
        pil = Image_from_array(u8_chw)                  # PIL RGB (H,W,3)

        conversations = build_conversation(pil)

        # Official chain: apply_chat_template -> process_vision_info -> processor
        text = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False)
        images, _, _ = process_vision_info(
            conversations, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True)
        inputs = processor(
            text=text, images=images, do_resize=False, return_tensors='pt')

        pv = inputs["pixel_values"]                     # (N, 1536) float32
        gth = inputs["image_grid_thw"].tolist()         # [[t, h, w]]

        out_path = f"{OUTPUT_DIR}/official_pv_prod_{h}x{w}.bin"
        write_official_pv(out_path, pv.detach().cpu().numpy().reshape(-1))
        print(f"  -> {out_path}  in={pil.size}  pv={tuple(pv.shape)}  "
              f"grid_thw={gth}  count={pv.numel()}")

    print("[gen] official pixel_values reference done.")


if __name__ == "__main__":
    main()
