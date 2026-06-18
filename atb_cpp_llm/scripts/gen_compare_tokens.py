#!/usr/bin/env python3
"""
Generate all token files required by C++ benchmark --mode compare.

Produces 13 files under /tmp/:
  5 text-only:  tokens_chat_text_only_{100,512,1024,2048,4096}.bin
  4 image-only: tokens_chat_io_{416x672,720x1280,1080x1920,1440x2560}.bin
  4 multimodal: tokens_chat_mm_{416x672,720x1280,1080x1920,1440x2560}.bin

All tokens are chat-templated via atb_python_qwen3vl_embedding.chat_tokenizer
and match what Qwen3VLEmbedder.process() would produce for the same inputs.

Binary format (matching C++ LoadTokenIds / save_token_ids):
    [int32 count][int64 * count]

Idempotent by default — use --force to regenerate existing files.
"""

import argparse
import os
import sys
from pathlib import Path

# ── Ensure the repo root is on sys.path so atb_python_qwen3vl_embedding
#    can be imported from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Constants (matching C++ benchmark.cpp:591-610) ────────────────────

TEXT_SEQ_LENS = [100, 512, 1024, 2048, 4096]
RESOLUTIONS = [(416, 672), (720, 1280), (1080, 1920), (1440, 2560)]

# Long text (~500 base tokens) used for TEXT_ONLY and MM modes.
# Same content as tests/gen_baseline_tokens.py for consistency.
LONG_TEXT = (
    "Please provide a comprehensive analysis of the image. "
    "Identify all visible objects, people, animals, and their spatial relationships, "
    "including foreground and background elements, lighting conditions, color palette, "
    "mood, and any notable artistic style or composition techniques. "
) * 14

OUTPUT_DIR = "/tmp"


def main():
    parser = argparse.ArgumentParser(
        description="Generate token files for C++ benchmark --mode compare"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate files even if they already exist"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model directory (default: from .env or QWEN3VL_EMB_MODEL_DIR env var)"
    )
    args = parser.parse_args()

    # ── Resolve model directory ──────────────────────────────────────
    # Set os.environ BEFORE importing env-dependent modules so env.py
    # picks up the override via its os.environ > .env > default precedence.
    if args.model_dir:
        os.environ["QWEN3VL_EMB_MODEL_DIR"] = args.model_dir

    try:
        from atb_python_qwen3vl_embedding.chat_tokenizer import (
            apply_qwen3vl_chat_template,
            apply_qwen3vl_chat_template_image_only,
            apply_qwen3vl_chat_template_text_only,
            _get_processor,
            save_token_ids,
        )
        from atb_python_qwen3vl_embedding.preprocess import smart_resize
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Set QWEN3VL_EMB_MODEL_DIR in .env or use --model-dir "
            "to specify the model path.",
            file=sys.stderr,
        )
        return 1

    model_dir = args.model_dir  # None means "use default from env"
    generated = 0
    skipped = 0
    failed = 0

    # ── Helper ──────────────────────────────────────────────────────
    def _write_if_needed(path, tokens):
        nonlocal generated, skipped, failed
        if os.path.exists(path) and not args.force:
            print(f"  SKIP  {path}  (already exists, {len(tokens)} tokens)")
            skipped += 1
            return
        try:
            save_token_ids(path, tokens)
            print(f"  WROTE {path}  ({len(tokens)} tokens)")
            generated += 1
        except Exception as e:
            print(f"  ERROR {path}  ({e})", file=sys.stderr)
            failed += 1

    # ══════════════════════════════════════════════════════════════════
    # 1. Text-only tokens (chat-templated, padded to exact lengths)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("[1/3] Text-only tokens (chat-templated)")
    print("=" * 60)

    try:
        base_tokens = apply_qwen3vl_chat_template_text_only(
            LONG_TEXT, model_dir=model_dir
        )
    except Exception as e:
        print(f"ERROR generating base text tokens: {e}", file=sys.stderr)
        return 1

    print(f"  Base chat-templated text token count: {len(base_tokens)}")

    for S in TEXT_SEQ_LENS:
        path = f"{OUTPUT_DIR}/tokens_chat_text_only_{S}.bin"
        repeat = (S + len(base_tokens) - 1) // len(base_tokens)
        padded = (base_tokens * repeat)[:S]
        _write_if_needed(path, padded)

    # ══════════════════════════════════════════════════════════════════
    # 2. Image-only tokens (chat-templated, image only, no text)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("[2/3] Image-only tokens (chat-templated)")
    print("=" * 60)

    processor = _get_processor(model_dir)
    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size

    for W, H in RESOLUTIONS:
        path = f"{OUTPUT_DIR}/tokens_chat_io_{W}x{H}.bin"
        new_h, new_w = smart_resize(H, W, factor=factor,
                                     min_pixels=ip.min_pixels,
                                     max_pixels=ip.max_pixels)
        grid_thw = (1, new_h // ip.patch_size, new_w // ip.patch_size)
        try:
            tokens = apply_qwen3vl_chat_template_image_only(
                grid_thw=grid_thw,
                model_dir=model_dir,
            )
        except Exception as e:
            print(f"  ERROR {path}  ({e})", file=sys.stderr)
            failed += 1
            continue
        _write_if_needed(path, tokens)

    # ══════════════════════════════════════════════════════════════════
    # 3. Multi-modal tokens (chat-templated, image + long text)
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("[3/3] Multi-modal tokens (chat-templated, image + text)")
    print("=" * 60)

    from PIL import Image as PILImage

    for W, H in RESOLUTIONS:
        path = f"{OUTPUT_DIR}/tokens_chat_mm_{W}x{H}.bin"
        new_h, new_w = smart_resize(H, W, factor=factor,
                                     min_pixels=ip.min_pixels,
                                     max_pixels=ip.max_pixels)
        grid_thw = (1, new_h // ip.patch_size, new_w // ip.patch_size)
        try:
            img = PILImage.new("RGB", (W, H), color="blue")
            tokens = apply_qwen3vl_chat_template(
                image=img,
                grid_thw=grid_thw,
                text=LONG_TEXT,
                model_dir=model_dir,
            )
        except Exception as e:
            print(f"  ERROR {path}  ({e})", file=sys.stderr)
            failed += 1
            continue
        _write_if_needed(path, tokens)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    total = generated + skipped
    print(f"Total: {total}/13 files ready  "
          f"({generated} generated, {skipped} skipped{f', {failed} FAILED' if failed else ''})")
    if failed:
        print("Some files could not be generated — see errors above.")
        return 1
    if generated == 0:
        print("All files already exist. Use --force to regenerate.")
    else:
        print(f"Output directory: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
