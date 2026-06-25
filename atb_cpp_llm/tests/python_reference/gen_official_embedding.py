#!/usr/bin/env python3
"""Generate official Qwen3VLEmbedder full-embedding references for C++ path C.

For each production resolution, this script reads the shared image bin produced
by gen_cpu_reference.py, calls the *real* official Qwen3VLEmbedder public
``process`` API to produce the pooled normalized embedding, and writes:

  /tmp/official_embed_mm_<H>x<W>.bin   [int64 dim][float32 * dim]
  /tmp/official_tokens_mm_<H>x<W>.bin  [int32 count][int64 * count]

The token ids are taken from the same official preprocess chain and conversation
as the embedding, so the C++ IMAGE_AND_TEXT request uses byte-identical text/chat
template input. The default instruction is intentionally not overridden: the
official default is "Represent the user's input.".
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401 — registers the NPU backend
import torch_npu.contrib.transfer_to_npu  # noqa: F401 — maps official cuda device to npu
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/ -> _tests_env
from _tests_env import MODEL_DIR, QWEN3VL_EMB_SRC, TRANSFORMERS_SRC  # noqa: E402

if TRANSFORMERS_SRC:
    sys.path.insert(0, TRANSFORMERS_SRC)

OFFICIAL_MODEL_SRC = Path(QWEN3VL_EMB_SRC or "/mnt/workspace/gitCode/Qwen3-VL-Embedding")
OFFICIAL_SRC = OFFICIAL_MODEL_SRC / "src"
if str(OFFICIAL_SRC) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SRC))

from models.qwen3_vl_embedding import Qwen3VLEmbedder, MAX_PIXELS, MIN_PIXELS  # noqa: E402

OUTPUT_DIR = "/tmp"
PROD_CASES = [(416, 672), (720, 1280), (1080, 1920), (1440, 2560)]
EXPECTED_DEFAULT_INSTRUCTION = "Represent the user's input."


def read_shared_input_bin(h: int, w: int) -> np.ndarray:
    """Read /tmp/bicubic_prod_<H>x<W>_input.bin -> (3,H,W) uint8."""
    path = f"{OUTPUT_DIR}/bicubic_prod_{h}x{w}_input.bin"
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Shared input bin missing: {path}\n"
            "Run `python atb_cpp_llm/tests/python_reference/gen_cpu_reference.py` first.")
    with open(path, "rb") as f:
        ndim_buf = f.read(8)
        if len(ndim_buf) != 8:
            raise ValueError(f"Malformed input bin (missing ndim): {path}")
        ndim = int(np.frombuffer(ndim_buf, dtype=np.int64)[0])
        shape_buf = f.read(8 * ndim)
        if len(shape_buf) != 8 * ndim:
            raise ValueError(f"Malformed input bin (missing shape): {path}")
        shape = tuple(int(x) for x in np.frombuffer(shape_buf, dtype=np.int64))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)
    if shape != (3, h, w):
        raise ValueError(f"Unexpected shared input shape for {path}: {shape}, expected {(3, h, w)}")
    return np.clip(np.round(data), 0, 255).astype(np.uint8)


def image_from_array(u8_chw: np.ndarray):
    from PIL import Image
    return Image.fromarray(u8_chw.transpose(1, 2, 0), mode="RGB")


def write_embedding(path: str, emb: np.ndarray) -> None:
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    dim = np.int64(emb.size)
    with open(path, "wb") as f:
        f.write(dim.tobytes())
        f.write(emb.tobytes())


def write_tokens(path: str, input_ids: np.ndarray) -> None:
    ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    count = np.int32(ids.size)
    with open(path, "wb") as f:
        f.write(count.tobytes())
        f.write(ids.tobytes())


def run_public_process_capture(embedder: Qwen3VLEmbedder, inputs):
    """Call public process() and capture its internal processed_inputs."""
    captured = {}
    orig_preprocess = embedder._preprocess_inputs

    def wrapped_preprocess(conversations):
        processed = orig_preprocess(conversations)
        captured["inputs"] = {k: v.detach().cpu().clone() for k, v in processed.items()}
        return processed

    embedder._preprocess_inputs = wrapped_preprocess
    try:
        emb = embedder.process(inputs, normalize=True)
    finally:
        embedder._preprocess_inputs = orig_preprocess

    if "inputs" not in captured:
        raise RuntimeError("Qwen3VLEmbedder.process() did not call _preprocess_inputs")
    return emb, captured["inputs"]


def pooled_embedding_from_processed(embedder: Qwen3VLEmbedder, processed_inputs) -> torch.Tensor:
    processed_inputs = {k: v.to(embedder.model.device) for k, v in processed_inputs.items()}
    outputs = embedder.forward(processed_inputs)
    emb = embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
    return F.normalize(emb, p=2, dim=-1)


def main() -> None:
    print("[gen] official full embedding reference (real Qwen3VLEmbedder)")
    print(f"  official src: {OFFICIAL_SRC}")
    print(f"  model dir:    {MODEL_DIR}")
    print(f"  default prompt: {EXPECTED_DEFAULT_INSTRUCTION!r} (not overridden)")
    print(f"  max_pixels={MAX_PIXELS} min_pixels={MIN_PIXELS}")

    embedder = Qwen3VLEmbedder(MODEL_DIR)
    if embedder.default_instruction != EXPECTED_DEFAULT_INSTRUCTION:
        raise RuntimeError(
            f"Official default instruction changed: {embedder.default_instruction!r} "
            f"!= {EXPECTED_DEFAULT_INSTRUCTION!r}")
    if embedder.max_pixels != 1800 * 32 * 32:
        raise RuntimeError(f"Unexpected official max_pixels={embedder.max_pixels}")

    for h, w in PROD_CASES:
        pil = image_from_array(read_shared_input_bin(h, w))
        public_inputs = [{"image": pil}]

        # Public API reference: this is the official embedding gate target. The
        # wrapper captures input_ids from the same preprocess call inside process().
        public_emb, processed = run_public_process_capture(embedder, public_inputs)

        # Re-run only forward/pooling from the captured official preprocessed inputs
        # to prove the saved embedding and saved tokens are the same conversation.
        internal_emb = pooled_embedding_from_processed(embedder, processed)

        max_diff = (public_emb - internal_emb).abs().max().item()
        if max_diff > 1e-6:
            raise RuntimeError(
                f"Public process() and explicit official chain diverged for {h}x{w}: "
                f"max_diff={max_diff:.6e}")

        input_ids = processed["input_ids"].detach().cpu().numpy().reshape(-1)
        image_token_count = int((input_ids == embedder.model.config.image_token_id).sum())
        grid_thw = processed.get("image_grid_thw")
        grid_list = grid_thw.detach().cpu().tolist() if grid_thw is not None else None

        emb_np = public_emb.detach().cpu().numpy().astype(np.float32).reshape(-1)
        embed_path = f"{OUTPUT_DIR}/official_embed_mm_{h}x{w}.bin"
        token_path = f"{OUTPUT_DIR}/official_tokens_mm_{h}x{w}.bin"
        write_embedding(embed_path, emb_np)
        write_tokens(token_path, input_ids)

        print(
            f"  -> {h}x{w}: dim={emb_np.size} tokens={input_ids.size} "
            f"image_tokens={image_token_count} grid={grid_list} "
            f"public_vs_internal_max_diff={max_diff:.3e}")
        print(f"     {embed_path}")
        print(f"     {token_path}")

    print("[gen] official full embedding reference done.")


if __name__ == "__main__":
    main()
