"""Generate all token .bin files for the baseline benchmark.

Produces:
  /tmp/tokens_text_only.bin          — raw text tokens (no chat template)
  /tmp/tokens_chat_text_only_{S}.bin — padded text tokens for C++ TEXT mode
  /tmp/tokens_chat_io_{W}x{H}.bin    — image-only tokens for C++ IO mode
  /tmp/tokens_chat_mm_{W}x{H}.bin    — chat-templated MM tokens for C++ MM mode
  /tmp/tokens_mm_{W}x{H}.bin         — mirror of tokens_chat_mm for Python e2e
"""
import struct, os
from PIL import Image
from transformers import AutoProcessor

MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B"
BIN_DIR = "/tmp"

# ~500 token text
LONG_TEXT = (
    "Please provide a comprehensive analysis of the image. "
    "Identify all visible objects, people, animals, and their spatial relationships, "
    "including foreground and background elements, lighting conditions, color palette, "
    "mood, and any notable artistic style or composition techniques. "
) * 14  # ~486 raw text tokens; with chat template ~506

RESOLUTIONS = [(416, 672), (720, 1280), (1080, 1920), (1440, 2560)]
GRID_MAP = {
    (416, 672):   (1, 42, 26),
    (720, 1280):  (1, 80, 44),
    (1080, 1920): (1, 94, 52),
    (1440, 2560): (1, 94, 52),
}
TEXT_SEQ_LENS = [100, 512, 1024, 2048, 4096]
IMAGE_TOKEN_ID = 151655


def save_token_ids(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(ids)))
        for tid in ids:
            f.write(struct.pack("<q", int(tid)))
    return len(ids)


print("[1/4] Loading processor ...")
processor = AutoProcessor.from_pretrained(MODEL_DIR, padding_side="right")
print(f"       image_token_id={processor.image_token_id}")
print(f"       merge_size={processor.image_processor.merge_size}")

# ── Raw text tokens ──────────────────────────────────────
print("[2/4] Generating text tokens ...")
text_tokens = processor.tokenizer(LONG_TEXT, add_special_tokens=False)['input_ids']
n_text = save_token_ids(f"{BIN_DIR}/tokens_text_only.bin", text_tokens)
print(f"       /tmp/tokens_text_only.bin: {n_text} tokens")

# ── Padded text tokens for C++ ──
for S in TEXT_SEQ_LENS:
    pad = (S // n_text + 1)
    padded = (text_tokens * pad)[:S]
    save_token_ids(f"{BIN_DIR}/tokens_chat_text_only_{S}.bin", padded)
    print(f"       tokens_chat_text_only_{S}.bin: {len(padded)} tokens")

# ── IO tokens for C++ (pure image_token_id) ──
print("[3/4] Generating IO tokens ...")
for W, H in RESOLUTIONS:
    grid = GRID_MAP[(W, H)]
    t, h, w = grid
    merge_size = processor.image_processor.merge_size
    vis_tokens = t * h * w // (merge_size ** 2)
    io_ids = [IMAGE_TOKEN_ID] * vis_tokens
    save_token_ids(f"{BIN_DIR}/tokens_chat_io_{W}x{H}.bin", io_ids)
    print(f"       tokens_chat_io_{W}x{H}.bin: {len(io_ids)} tokens (grid={grid})")

# ── MM tokens (chat template + image + text) ──
print("[4/4] Generating MM tokens ...")
for W, H in RESOLUTIONS:
    grid = GRID_MAP[(W, H)]
    t, h, w = grid
    merge_size = processor.image_processor.merge_size
    vis_tokens = t * h * w // (merge_size ** 2)

    # Build conversation exactly like the real Qwen3VLEmbedder pipeline
    img = Image.new('RGB', (W, H), color='blue')
    conversation = [
        {"role": "system", "content": [
            {"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": LONG_TEXT}]},
    ]
    text_str = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)
    tokenized = processor.tokenizer(text_str)
    token_ids = tokenized["input_ids"]

    # Expand <|image_pad|> tokens
    expanded = []
    for tid in token_ids:
        if tid == IMAGE_TOKEN_ID:
            expanded.extend([IMAGE_TOKEN_ID] * vis_tokens)
        else:
            expanded.append(tid)

    # Save to both paths (C++ reads tokens_chat_mm_, Python e2e reads tokens_mm_)
    n_mm = save_token_ids(f"{BIN_DIR}/tokens_chat_mm_{W}x{H}.bin", expanded)
    save_token_ids(f"{BIN_DIR}/tokens_mm_{W}x{H}.bin", expanded)
    print(f"       tokens_chat_mm_{W}x{H}.bin: {n_mm} tokens")

print("\nDone. All token files regenerated.")
