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
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/ -> _tests_env
from _tests_env import (  # noqa: E402
    MODEL_DIR,
    OFFICIAL_EMBED_CASES,
    QWEN3VL_EMB_SRC,
    TRANSFORMERS_SRC,
)

if TRANSFORMERS_SRC:
    sys.path.insert(0, TRANSFORMERS_SRC)


def resolve_official_src(path_value: str) -> Path:
    """Resolve QWEN3VL_EMB_SRC as either repo root or its src/ directory."""
    base = Path(path_value or "/mnt/workspace/gitCode/Qwen3-VL-Embedding").expanduser().resolve()
    direct = base / "models" / "qwen3_vl_embedding.py"
    nested = base / "src" / "models" / "qwen3_vl_embedding.py"
    if direct.is_file():
        return base
    if nested.is_file():
        return base / "src"
    raise RuntimeError(
        "Cannot locate official Qwen3-VL-Embedding source. Set QWEN3VL_EMB_SRC "
        "to either the repo root or its src directory. "
        f"Got: {base}; checked {direct} and {nested}")


OFFICIAL_SRC = resolve_official_src(QWEN3VL_EMB_SRC)
if str(OFFICIAL_SRC) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SRC))

from models.qwen3_vl_embedding import Qwen3VLEmbedder, MAX_PIXELS, MIN_PIXELS  # noqa: E402

OUTPUT_DIR = "/tmp"
EXPECTED_DEFAULT_INSTRUCTION = "Represent the user's input."


@contextlib.contextmanager
def force_cuda_unavailable(enabled: bool):
    """Temporarily force official code to choose CPU instead of cuda/NPU."""
    if not enabled:
        yield
        return
    orig = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        yield
    finally:
        torch.cuda.is_available = orig


def empty_npu_cache_safe() -> None:
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception:
        pass


def enable_npu_cuda_mapping() -> None:
    """Import torch_npu transfer shim lazily so CPU fallback can avoid it."""
    import torch_npu  # noqa: F401 — registers the NPU backend
    import torch_npu.contrib.transfer_to_npu  # noqa: F401 — maps official cuda device to npu


def parse_cases(value: str):
    """Parse OFFICIAL_EMBED_CASES as a comma-separated HxW list.

    Accepts forms like:
      416x672,720x1280
    """
    raw = (value or "").strip()
    if not raw:
        raw = "416x672,720x1280,1080x1920,1440x2560"

    out = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if 'x' not in item:
            raise ValueError(
                f"Invalid OFFICIAL_EMBED_CASES item {item!r}; expected HxW, e.g. 416x672")
        h_s, w_s = item.lower().split('x', 1)
        case = (int(h_s), int(w_s))
        if case not in out:
            out.append(case)
    if not out:
        raise ValueError("OFFICIAL_EMBED_CASES resolved to an empty case list")
    return out


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


def build_embedder(backend: str) -> Qwen3VLEmbedder:
    with force_cuda_unavailable(backend == "official-cpu"):
        embedder = Qwen3VLEmbedder(MODEL_DIR)
    if backend == "official-cpu":
        embedder.model.to(torch.device("cpu"))
        embedder.model.float()
    return embedder


def validate_embedder(embedder: Qwen3VLEmbedder) -> None:
    if embedder.default_instruction != EXPECTED_DEFAULT_INSTRUCTION:
        raise RuntimeError(
            f"Official default instruction changed: {embedder.default_instruction!r} "
            f"!= {EXPECTED_DEFAULT_INSTRUCTION!r}")
    if embedder.max_pixels != 1800 * 32 * 32:
        raise RuntimeError(f"Unexpected official max_pixels={embedder.max_pixels}")


def run_case(embedder: Qwen3VLEmbedder, h: int, w: int, backend: str) -> None:
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
        f"  -> {h}x{w}: backend={backend} dim={emb_np.size} tokens={input_ids.size} "
        f"image_tokens={image_token_count} grid={grid_list} "
        f"public_vs_internal_max_diff={max_diff:.3e}")
    print(f"     {embed_path}")
    print(f"     {token_path}")


def run_all_cases(backend: str) -> None:
    if backend == "official-npu":
        enable_npu_cuda_mapping()
    embedder = build_embedder("official-cpu" if backend == "official-cpu-fallback" else backend)
    validate_embedder(embedder)
    cases = parse_cases(OFFICIAL_EMBED_CASES)
    print(f"[gen] OFFICIAL_EMBED_CASES={OFFICIAL_EMBED_CASES!r} -> {cases}")
    try:
        for h, w in cases:
            run_case(embedder, h, w, backend)
    finally:
        del embedder
        if backend == "official-npu":
            empty_npu_cache_safe()


def main() -> None:
    print("[gen] official full embedding reference (real Qwen3VLEmbedder)")
    print(f"  official src: {OFFICIAL_SRC}")
    print(f"  model dir:    {MODEL_DIR}")
    print(f"  default prompt: {EXPECTED_DEFAULT_INSTRUCTION!r} (not overridden)")
    print(f"  max_pixels={MAX_PIXELS} min_pixels={MIN_PIXELS}")

    force_cpu = os.environ.get("OFFICIAL_EMBED_FORCE_CPU") == "1"
    no_cpu_fallback = os.environ.get("OFFICIAL_EMBED_NO_CPU_FALLBACK") == "1"

    if force_cpu:
        print("[gen] OFFICIAL_EMBED_FORCE_CPU=1 — using CPU float32 official reference.")
        run_all_cases("official-cpu-fallback")
        print("[gen] official full embedding reference done.")
        return

    try:
        print("[gen] Trying backend=official-npu")
        run_all_cases("official-npu")
    except (RuntimeError, OSError, ValueError) as e:
        if no_cpu_fallback:
            raise
        print(f"[gen] NPU official Qwen3VLEmbedder failed ({type(e).__name__}: {e}); "
              "falling back to CPU float32 official reference.")
        empty_npu_cache_safe()
        run_all_cases("official-cpu-fallback")

    print("[gen] official full embedding reference done.")


if __name__ == "__main__":
    main()
