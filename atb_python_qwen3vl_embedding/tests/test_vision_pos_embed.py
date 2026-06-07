"""Test for VisPosEmb ATB graph.

Validates that the ATB graph output (pos, cos, sin) matches the CPU reference
implementation (fast_pos_embed_interpolate + compute_rot_pos_emb) with cosine
similarity > 0.999 for each output tensor.

Usage::

    python tests/test_vision_pos_embed.py
    python tests/test_vision_pos_embed.py --threshold 0.998
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
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR
from atb_python_qwen3vl_embedding.engine_utils import (
    compute_posemb_indices, compute_rope_indices,
    compute_rot_pos_emb, fast_pos_embed_interpolate,
)
from atb_python_qwen3vl_embedding.vision_pos_embed import (
    build_vision_posemb_graph, run_posemb_npu,
)
from atb_python_qwen3vl_embedding.utils import to_npu_half


# ═══════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════

def build_test_cases():
    """Return list of (name, grid_thw) test cases."""
    # grid_thw format: [[t, h, w]]
    # Use realistic sizes from the Qwen3VL model
    return [
        ("Small (40×22)",  torch.tensor([[1, 40, 22]], dtype=torch.long)),
        ("Medium (80×44)", torch.tensor([[1, 80, 44]], dtype=torch.long)),
        ("Large (94×52)",  torch.tensor([[1, 94, 52]], dtype=torch.long)),
    ]


def run_test(engine, grid_thw, threshold):
    """Run a single test case and return (pos_cos, cos_cos, sin_cos, all_pass)."""
    merge_size = engine.merge_size
    num_grid = engine.num_grid

    # ── CPU reference ──────────────────────────────────────────────
    pos_ref = fast_pos_embed_interpolate(
        grid_thw, engine.v_pos_embed, num_grid, merge_size)

    rope_ref = compute_rot_pos_emb(grid_thw, engine.vis_rotary, merge_size)
    rope_ref = rope_ref.reshape(-1, rope_ref.shape[-1])
    emb_ref = torch.cat((rope_ref, rope_ref), dim=-1)
    cos_ref = emb_ref.cos()
    sin_ref = emb_ref.sin()

    # ── ATB graph ──────────────────────────────────────────────────
    idx_wt = compute_posemb_indices(grid_thw, num_grid, merge_size)
    rope_idx = compute_rope_indices(grid_thw, engine.vis_rotary, merge_size)
    freq_npu = to_npu_half(rope_idx['freq_table'])

    pos_atb, cos_atb, sin_atb = run_posemb_npu(
        engine.g_v_posemb, engine.v_pe_w_table,
        idx_wt, rope_idx, freq_npu)

    # Convert ATB outputs to CPU float32 for comparison
    pos_atb_cpu = pos_atb.cpu().float()
    cos_atb_cpu = cos_atb.cpu().float()
    sin_atb_cpu = sin_atb.cpu().float()

    # ── Compare ────────────────────────────────────────────────────
    pos_cs = F.cosine_similarity(pos_ref.flatten(), pos_atb_cpu.flatten(), dim=0).item()
    cos_cs = F.cosine_similarity(cos_ref.flatten(), cos_atb_cpu.flatten(), dim=0).item()
    sin_cs = F.cosine_similarity(sin_ref.flatten(), sin_atb_cpu.flatten(), dim=0).item()

    pos_ok = pos_cs > threshold
    cos_ok = cos_cs > threshold
    sin_ok = sin_cs > threshold
    all_pass = pos_ok and cos_ok and sin_ok

    return pos_cs, cos_cs, sin_cs, all_pass


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--threshold', type=float, default=0.999,
                   help='Cosine similarity threshold for PASS (default 0.999)')
    p.add_argument('--model-dir', default=QWEN3VL_EMB_MODEL_DIR,
                   help='Override QWEN3VL_EMB_MODEL_DIR from .env')
    args = p.parse_args(argv)

    print(f"Model dir : {args.model_dir}")
    print(f"Threshold : {args.threshold}")

    # Load engine (builds all ATB graphs including g_v_posemb)
    print("\n[Setup] Loading Qwen3VLEngine ...")
    engine = Qwen3VLEngine(args.model_dir)
    print(f"[Setup] Loaded: {engine.n_layer} text layers, "
          f"{engine.v_depth} vision blocks")

    cases = build_test_cases()

    print(f"\n{'─' * 65}")
    print(f"{'Case':<22} {'pos_cos':>10} {'cos_cos':>10} {'sin_cos':>10} {'status':>8}")
    print(f"{'─' * 65}")

    all_pass = True
    for name, grid_thw in cases:
        pos_cs, cos_cs, sin_cs, passed = run_test(engine, grid_thw, args.threshold)
        status = "PASS" if passed else "FAIL"
        all_pass &= passed
        print(f"{name:<22} {pos_cs:>10.6f} {cos_cs:>10.6f} {sin_cs:>10.6f} {status:>8}")

    print(f"{'─' * 65}")

    # Cleanup
    del engine
    torch.npu.empty_cache()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
