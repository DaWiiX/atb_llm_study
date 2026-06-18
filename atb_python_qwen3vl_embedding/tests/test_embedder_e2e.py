"""End-to-end embedder comparison: ATB engine.encode() vs Transformers reference.

Tests the **pooled + L2-normalised** embedding output (``engine.encode()``)
against the transformers reference on three input modes: text-only,
image-only, and image+text.

Cosine >= 0.99 threshold for PASS.

Usage::

    python tests/test_embedder_e2e.py
    python tests/test_embedder_e2e.py --threshold 0.95
"""
# ── Buffer size MUST be set before any engine/graph import ──────────
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
set_atb_buffer_size(5 * 1024 * 1024 * 1024)  # 5 GB

# ── Standard imports ───────────────────────────────────────────────
import argparse
import sys
from typing import Optional

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR
from atb_python_qwen3vl_embedding.tests.data_utils import (
    cosine,
    load_tf_ref,
    pool_and_normalize,
)


# ═══════════════════════════════════════════════════════════════════
# Test cases (same 3 modes as test_e2e.py)
# ═══════════════════════════════════════════════════════════════════

def build_cases():
    """Return list of (name, image_or_None, text_or_None) test cases."""
    return [
        ("Text-Only",  None,
         "What is the capital of France?"),
        ("Image-Only", Image.new('RGB', (120, 200), color='red'),
         None),
        ("Image+Text", Image.new('RGB', (64, 64), color='blue'),
         "Describe."),
    ]


# ═══════════════════════════════════════════════════════════════════
# ATB embedder — engine.encode() path
# ═══════════════════════════════════════════════════════════════════

def run_atb_embed(model_dir: str, processor, cases) -> dict:
    """Run engine.encode() for each case, return dict[name] = pooled embedding."""
    print("[ATB] Loading Qwen3VLEngine ...")
    engine = Qwen3VLEngine(model_dir)
    print(f"[ATB] Loaded: {engine.n_layer} text layers, "
          f"{engine.v_depth} vision blocks")

    results = {}
    for name, img, txt in cases:
        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        if txt is not None:
            content.append({"type": "text", "text": txt})
        msgs = [{"role": "system", "content": [
            {"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content}]

        tf_in = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=True)
        input_ids = tf_in['input_ids']
        attn_mask = tf_in.get('attention_mask', torch.ones_like(input_ids))

        pv_raw = grid_thw = None
        if img is not None:
            img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            pv_raw, grid_thw = engine.preprocess_image(img_arr)

        emb = engine.encode(input_ids, pv_raw, grid_thw,
                            attention_mask=attn_mask, normalize=True)
        results[name] = emb.cpu().float().squeeze(0)
        print(f"[ATB] {name:<12} → {tuple(results[name].shape)}")

    del engine
    torch.npu.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# TF embedder — Qwen3VLModel + pool_and_normalize path
# ═══════════════════════════════════════════════════════════════════

def run_tf_embed(model_dir: str, processor, cases) -> dict:
    """Run TF model + pool_and_normalize for each case."""
    print("\n[TF] Loading transformers reference ...")
    model = load_tf_ref(model_dir)
    print("[TF] Loaded")

    results = {}
    for name, img, txt in cases:
        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        if txt is not None:
            content.append({"type": "text", "text": txt})
        msgs = [{"role": "system", "content": [
            {"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content}]

        tf_in = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=True)
        ids_t = tf_in['input_ids'].to(model.device)
        mask_t = tf_in.get('attention_mask', torch.ones_like(ids_t)).to(model.device)

        kwargs = dict(use_cache=False, input_ids=ids_t, attention_mask=mask_t)
        if img is not None:
            kwargs["pixel_values"] = model.place(tf_in.get('pixel_values'))
            kwargs["image_grid_thw"] = tf_in.get('image_grid_thw').to(model.device)

        with torch.no_grad():
            out = model(**kwargs)
        emb = pool_and_normalize(out.last_hidden_state.cpu(), mask_t.cpu())
        results[name] = emb
        print(f"[TF]  {name:<12} → {tuple(emb.shape)}")

    del model
    torch.npu.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════

def compare(atb_emb: dict, tf_emb: dict, threshold: float) -> bool:
    """Compare pooled embeddings by cosine similarity."""
    print(f"\n{'─' * 60}")
    print(f"{'Case':<14} {'shape':<18} {'cosine':>10} {'L2 diff':>10} {'st':>5}")
    print(f"{'─' * 60}")
    all_pass = True
    for name in atb_emb:
        a, b = atb_emb[name], tf_emb[name]
        if a.shape != b.shape:
            print(f"{name:<14} SHAPE MISMATCH atb={a.shape} tf={b.shape}")
            all_pass = False
            continue
        cs = cosine(a, b)
        l2 = (a - b).norm().item()
        ok = cs >= threshold
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"{name:<14} {str(tuple(a.shape)):<18} "
              f"{cs:>10.6f} {l2:>10.6f} {status:>5}")
    print(f"{'─' * 60}")
    return all_pass


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    # 0.99: moderate fp16 accumulation (E2E embedder pipeline) — see THRESHOLDS.md
    p.add_argument('--threshold', type=float, default=0.99,
                   help='Cosine similarity threshold for PASS (default 0.99)')
    p.add_argument('--model-dir', default=QWEN3VL_EMB_MODEL_DIR,
                   help='Override QWEN3VL_EMB_MODEL_DIR from .env')
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    print(f"Model dir: {args.model_dir}")
    print("Mode: ATB vs TF pooled-embedding comparison")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir,
                                              padding_side='right')

    cases = build_cases()
    atb_emb = run_atb_embed(args.model_dir, processor, cases)
    tf_emb = run_tf_embed(args.model_dir, processor, cases)

    ok = compare(atb_emb, tf_emb, args.threshold)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
