"""
Qwen3VL Chat Template tokenizer — offline token ID generation for C++ deployment.

Produces token_ids that match Qwen3VLEmbedder.process() exactly:
  1. Constructs conversation dict (system + user + vision markers)
  2. Calls processor.apply_chat_template() → text string
  3. Tokenizes → token_ids (with <|image_pad|> expanded to N tokens
     where N = grid_t * grid_h * grid_w / spatial_merge_size^2)

Usage:
    from chat_tokenizer import apply_qwen3vl_chat_template
    token_ids = apply_qwen3vl_chat_template(
        text="Describe the image.",
        grid_thw=(1, 42, 26),
        instruction="Represent the user's input.",
    )
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from transformers import Qwen3VLProcessor


# ── Model directory ─────────────────────────────────────────────
# Default to whatever .env / shell sets QWEN3VL_EMB_MODEL_DIR to. Callers
# can still override via the `model_dir=...` argument on individual entry
# points; falling through with neither set raises at the env.py import line.
from .env import QWEN3VL_EMB_MODEL_DIR as _DEFAULT_MODEL_DIR

# ── Cached processor (lazy-init) ─────────────────────────────────
_processor: Optional[Qwen3VLProcessor] = None
_model_dir: Optional[str] = None


def _get_processor(model_dir: Optional[str] = None) -> Qwen3VLProcessor:
    global _processor, _model_dir
    md = model_dir or _DEFAULT_MODEL_DIR
    if _processor is None or _model_dir != md:
        _processor = Qwen3VLProcessor.from_pretrained(md, padding_side="right")
        _model_dir = md
    return _processor


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def apply_qwen3vl_chat_template(
    text: str = "",
    image: Optional[Image.Image] = None,
    grid_thw: Optional[Tuple[int, int, int]] = None,
    instruction: str = "Represent the user's input.",
    model_dir: Optional[str] = None,
) -> List[int]:
    """Generate chat-templated token IDs matching Qwen3VLEmbedder.process().

    Args:
        text:        User text query.
        image:       PIL Image (used to compute grid_thw if not provided).
        grid_thw:    Override grid (T, H, W) — useful for preprocessed images.
        instruction: System instruction.
        model_dir:   Model directory (default Qwen3-VL-Embedding-2B).

    Returns:
        List of int64 token IDs with <|image_pad|> tokens expanded
        to the correct count for the given grid.
    """
    processor = _get_processor(model_dir)

    # ── Build conversation ────────────────────────────────────
    content: List[Dict[str, Any]] = []
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]

    # Image
    if image is not None:
        content.append({"type": "image", "image": image})
    elif grid_thw is not None:
        # We have a precomputed grid — use a dummy pixel_values shape
        # just to get the processor to expand image_pad correctly.
        content.append({"type": "image", "image": "dummy"})

    # Text
    if text:
        content.append({"type": "text", "text": text})
    elif not content:
        # Empty input: use NULL (matches Qwen3VLEmbedder)
        content.append({"type": "text", "text": "NULL"})

    # ── Apply chat template ──────────────────────────────────
    text_str = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    # ── Tokenize (text-only first) ───────────────────────────
    tokenized = processor.tokenizer(text_str)
    token_ids = tokenized["input_ids"]

    # ── Expand <|image_pad|> tokens for each image ───────────
    if grid_thw is not None:
        image_token_id = processor.image_token_id
        merge_size = processor.image_processor.merge_size
        t, h, w = grid_thw
        num_image_tokens = t * h * w // (merge_size ** 2)

        expanded = []
        for tid in token_ids:
            if tid == image_token_id:
                expanded.extend([image_token_id] * num_image_tokens)
            else:
                expanded.append(tid)
    else:
        expanded = list(token_ids)

    return expanded


def apply_qwen3vl_chat_template_image_only(
    grid_thw: Tuple[int, int, int],
    instruction: str = "Represent the user's input.",
    model_dir: Optional[str] = None,
) -> List[int]:
    """Image-only chat template (no text)."""
    return apply_qwen3vl_chat_template(
        text="", grid_thw=grid_thw, instruction=instruction, model_dir=model_dir
    )


def apply_qwen3vl_chat_template_text_only(
    text: str,
    instruction: str = "Represent the user's input.",
    model_dir: Optional[str] = None,
) -> List[int]:
    """Text-only chat template (no image)."""
    return apply_qwen3vl_chat_template(
        text=text, image=None, grid_thw=None, instruction=instruction,
        model_dir=model_dir
    )


def get_image_token_id(model_dir: Optional[str] = None) -> int:
    return _get_processor(model_dir).image_token_id


def get_vision_start_token_id(model_dir: Optional[str] = None) -> int:
    return _get_processor(model_dir).tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )


# -------------------------------------------------------------------
# Save/Load helpers (binary format matching C++)
# -------------------------------------------------------------------

def save_token_ids(path: str, token_ids: List[int]):
    """Save token ID list in C++-compatible format: [int32 count] [int64 * count]."""
    import struct
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(token_ids)))
        for tid in token_ids:
            f.write(struct.pack("<q", int(tid)))


def load_token_ids(path: str) -> List[int]:
    """Load C++-compatible token ID binary."""
    import struct
    with open(path, "rb") as f:
        (count,) = struct.unpack("<i", f.read(4))
        fmt = f"<{count}q"
        return list(struct.unpack(fmt, f.read(count * 8)))
