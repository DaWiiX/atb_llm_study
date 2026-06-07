"""End-to-end embedder comparison: ATB engine vs Qwen3VLForEmbedding wrapper.

This is the **user-facing** validation. It compares the final normalised
embedding vector produced by:
  - ``atb_python_qwen3vl_embedding.engine.Qwen3VLEngine.encode()``
  - ``Qwen3VLEmbedder.process()`` (reference wrapper at
    ``/workspace/gitCode/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py``)

Both produce L2-normalised vectors; cosine ≥ 0.99 means the two engines
will return the same top-k results in retrieval.

The reference path uses ``Qwen3VLEmbedder`` which is the actual deployment
class, so the inputs go through ``format_model_input`` → system prompt
injection → ``process_vision_info`` → processor — i.e. the production
pre-processing pipeline.  The ATB path mimics that by calling the same
processor on the same conversation, then driving ``engine.encode()`` with
``input_ids`` / engine-preprocessed image.

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
import torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR, QWEN3VL_EMB_SRC
sys.path.insert(0, QWEN3VL_EMB_SRC)
    

# ═══════════════════════════════════════════════════════════════════
# Test cases — same shape as test_e2e.py but expressed as embedder inputs
# ═══════════════════════════════════════════════════════════════════

def build_cases():
    """Return list of (name, embedder_input_dict)."""
    img_only = Image.new('RGB', (120, 200), color='red')
    img_mix = Image.new('RGB', (64, 64), color='blue')
    return [
        ("Text-Only",  {"text": "What is the capital of France?"}),
        ("Image-Only", {"image": img_only}),
        ("Image+Text", {"image": img_mix, "text": "Describe."}),
    ]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: ATB embedder
# ═══════════════════════════════════════════════════════════════════

def run_atb_phase(model_dir, processor, cases):
    """Use Qwen3VLEngine.encode() on the same conversation a Qwen3VLEmbedder
    would build. Returns dict[name] = normalised embedding (1, hidden) CPU.

    The conversation matches Qwen3VLEmbedder.format_model_input — system
    prompt + user content. This is what the deployment uses; mismatching
    it would invalidate the comparison.
    """
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: F401

    print("[ATB] Loading Qwen3VLEngine ...")
    engine = Qwen3VLEngine(model_dir)
    print(f"[ATB] Loaded: {engine.n_layer} text layers, "
          f"{engine.v_depth} vision blocks")

    results = {}
    for name, inp in cases:
        # 1. Build the same conversation Qwen3VLEmbedder builds
        content = []
        if "image" in inp:
            content.append({"type": "image", "image": inp["image"]})
        if "text" in inp:
            content.append({"type": "text", "text": inp["text"]})
        msgs = [
            {"role": "system", "content": [
                {"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content},
        ]

        # 2. Tokenise + (optional) preprocess image via the engine
        tf_in = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=True)
        input_ids = tf_in['input_ids']
        attn_mask = tf_in.get('attention_mask',
                              torch.ones_like(input_ids))

        pv_raw = grid_thw = None
        if "image" in inp:
            img_arr = torch.from_numpy(np.array(inp["image"])).permute(2, 0, 1)
            pv_raw, grid_thw = engine.preprocess_image(img_arr)

        # 3. Engine encode → L2-normalised pooled embedding
        emb = engine.encode(
            input_ids, pv_raw, grid_thw,
            attention_mask=attn_mask, normalize=True)
        results[name] = emb.cpu().float()
        print(f"[ATB] {name:<12} → {tuple(emb.shape)}")

    del engine
    torch.npu.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Qwen3VLEmbedder reference (deployment-grade)
# ═══════════════════════════════════════════════════════════════════

def run_ref_phase(model_dir, cases):
    """Use Qwen3VLEmbedder.process() — the actual deployment entrypoint.

    Loaded AFTER ATB is released so NPU memory does not collide.
    """
    from models.qwen3_vl_embedding import Qwen3VLEmbedder

    print("\n[REF] Loading Qwen3VLEmbedder (deployment wrapper) ...")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=model_dir,
        torch_dtype=torch.float16,
    )
    print("[REF] Loaded")

    results = {}
    for name, inp in cases:
        emb = embedder.process([inp], normalize=True)
        results[name] = emb.cpu().float()
        print(f"[REF] {name:<12} → {tuple(emb.shape)}")

    del embedder
    torch.npu.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# Compare & report
# ═══════════════════════════════════════════════════════════════════

def compare(atb, ref, threshold):
    print(f"\n{'─' * 55}")
    print(f"{'Case':<14} {'shape':<18} {'cosine':>10} {'L2 diff':>10} {'st':>5}")
    print(f"{'─' * 55}")
    all_pass = True
    for name in atb:
        a, r = atb[name], ref[name]
        if a.shape != r.shape:
            print(f"{name:<14} SHAPE MISMATCH atb={a.shape} ref={r.shape}")
            all_pass = False
            continue
        cs = F.cosine_similarity(a.flatten(), r.flatten(), dim=0).item()
        l2 = (a - r).norm().item()
        ok = cs > threshold
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"{name:<14} {str(tuple(a.shape)):<18} "
              f"{cs:>10.6f} {l2:>10.6f} {status:>5}")
    print(f"{'─' * 55}")
    return all_pass


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--threshold', type=float, default=0.99,
                   help='Cosine similarity threshold for PASS (default 0.99)')
    p.add_argument('--model-dir', default=QWEN3VL_EMB_MODEL_DIR)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    print(f"Model dir: {args.model_dir}")
    print(f"Reference: Qwen3VLEmbedder from {QWEN3VL_EMB_SRC}")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir,
                                              padding_side='right')

    cases = build_cases()
    atb_emb = run_atb_phase(args.model_dir, processor, cases)
    ref_emb = run_ref_phase(args.model_dir, cases)
    ok = compare(atb_emb, ref_emb, args.threshold)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
