"""310P SelfAttention parameter combination sweep (BSND layout only).

Tests many ATB SelfAttention parameter combinations to find which work
on 310P (Atlas 推理系列产品). BNSD layout is excluded (all combos fail
on 910B as well).

The failing combo on 310P is:
    BSND + PA_ENCODER + MASK_TYPE_NORM + head_dim=128

This script tests BSND alternatives: is_triu_mask, kernel_type,
seqlen alignment, mask variations, etc. Run on 310P to see which pass.

Usage:
    ASCEND_PLATFORM=310P python tests/test_310p_combinations.py
"""

import os
import sys
import math
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
import torch_atb
import torch_npu  # noqa: F401

from atb_python_qwen3vl_embedding import data_utils, utils
from atb_python_qwen3vl_embedding.text_attention import build_attention
from atb_python_qwen3vl_embedding.text_model import make_causal_mask
from atb_python_qwen3vl_embedding.transformers_runner import run_attention
from atb_python_qwen3vl_embedding.utils import make_self_attention


def _cosine(a, b):
    """Compute cosine similarity between two tensors."""
    return F.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _run_full_attention(name, num_heads, num_kv_heads, head_dim,
                        use_mask, B, S, **sa_kwargs):
    """Build full Qwen3VLTextAttention graph, compare to transformers ref.

    Returns (ok, error_msg, cosine).
    """
    print(f"\n  [{name}] nh={num_heads} kv_nh={num_kv_heads} hd={head_dim} "
          f"S={S} mask={use_mask} sa_kwargs={sa_kwargs}")
    try:
        hidden_size = num_heads * head_dim
        config = data_utils.make_config(
            hidden_size=hidden_size, num_heads=num_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim)
        gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=42)
        gen_data["config"] = config

        if use_mask:
            m2d = make_causal_mask(S)
            gen_data["attention_mask"] = m2d.unsqueeze(0).unsqueeze(0).float()

        ref_out, weights = run_attention(gen_data, seed=42)

        _, graph_op, _ = build_attention(
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            B=B, S=S, use_mask=use_mask, **sa_kwargs,
        )

        ntoken = B * S
        inputs = [
            gen_data["hidden_states"].half().npu(),
            weights["q_proj.weight"].half().npu(),
            weights["k_proj.weight"].half().npu(),
            weights["v_proj.weight"].half().npu(),
            weights["o_proj.weight"].half().npu(),
            weights["q_norm.weight"].half().npu(),
            weights["k_norm.weight"].half().npu(),
            gen_data["cos"].reshape(ntoken, head_dim).half().npu(),
            gen_data["sin"].reshape(ntoken, head_dim).half().npu(),
        ]
        if use_mask:
            mask = make_causal_mask(S).half().npu()
            if utils.is_310p():
                mask = utils.nd_to_nz_fp16(mask)
            inputs.append(mask)
        inputs.append(torch.tensor([ntoken], dtype=torch.int32))

        atb_out = graph_op.forward(inputs)[0].cpu().float()
        cos = _cosine(ref_out, atb_out)
        ok = cos > 0.99
        status = "PASS" if ok else f"FAIL (cos={cos:.6f})"
        print(f"    {status}")
        return ok, None, cos
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"    ERROR: {err}")
        return False, err, 0.0


def main():
    platform = utils.get_platform()
    device = torch.npu.get_device_name(0)
    print(f"=== 310P SelfAttention Parameter Combination Sweep ===")
    print(f"Platform: {platform}")
    print(f"Device: {device}")

    utils.set_atb_buffer_size(500 * 1024 * 1024)

    nh = 16
    kv_nh = 16
    hd = 128
    B = 1

    results = {}

    # ── Section A: Baseline (current code) ───────────────────────
    print("\n" + "=" * 60)
    print("SECTION A: Baseline (BSND + PA_ENCODER + MASK_NORM)")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "A1: S=16", nh, kv_nh, hd, use_mask=True, B=B, S=16)
    results["A1"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "A2: S=4 (text-only typical)", nh, kv_nh, hd, use_mask=True, B=B, S=4)
    results["A2"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "A3: S=32", nh, kv_nh, hd, use_mask=True, B=B, S=32)
    results["A3"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "A4: S=64", nh, kv_nh, hd, use_mask=True, B=B, S=64)
    results["A4"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "A5: S=256", nh, kv_nh, hd, use_mask=True, B=B, S=256)
    results["A5"] = (ok, err, cos)

    # ── Section B: is_triu_mask variations ───────────────────────
    print("\n" + "=" * 60)
    print("SECTION B: is_triu_mask (internal causal optimization)")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "B1: is_triu=1 + MASK_NORM", nh, kv_nh, hd,
        use_mask=True, B=B, S=16, is_triu_mask=1)
    results["B1"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "B2: is_triu=1 + NO_MASK", nh, kv_nh, hd,
        use_mask=False, B=B, S=16, is_triu_mask=1)
    results["B2"] = (ok, err, cos)

    # ── Section C: Kernel type variations ────────────────────────
    print("\n" + "=" * 60)
    print("SECTION C: Kernel type")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "C1: HIGH_PRECISION", nh, kv_nh, hd,
        use_mask=True, B=B, S=16,
        kernel_type=torch_atb.SelfAttentionParam.KernelType.KERNELTYPE_HIGH_PRECISION)
    results["C1"] = (ok, err, cos)

    # ── Section D: Mask type variations ──────────────────────────
    print("\n" + "=" * 60)
    print("SECTION D: Mask type variations")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "D1: MASK_TYPE_UNDEFINED (no mask)", nh, kv_nh, hd,
        use_mask=False, B=B, S=16)
    results["D1"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "D2: MASK_TYPE_NORM_COMPRESS", nh, kv_nh, hd,
        use_mask=False, B=B, S=16,
        mask_type=torch_atb.SelfAttentionParam.MaskType.MASK_TYPE_NORM_COMPRESS)
    results["D2"] = (ok, err, cos)

    # ── Section E: calc_type variations ──────────────────────────
    print("\n" + "=" * 60)
    print("SECTION E: calc_type (non-PA_ENCODER)")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "E1: UNDEFINED", nh, kv_nh, hd,
        use_mask=True, B=B, S=16,
        calc_type=torch_atb.SelfAttentionParam.CalcType.UNDEFINED)
    results["E1"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "E2: ENCODER", nh, kv_nh, hd,
        use_mask=True, B=B, S=16,
        calc_type=torch_atb.SelfAttentionParam.CalcType.ENCODER)
    results["E2"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "E3: DECODER", nh, kv_nh, hd,
        use_mask=True, B=B, S=16,
        calc_type=torch_atb.SelfAttentionParam.CalcType.DECODER)
    results["E3"] = (ok, err, cos)

    # ── Section F: GQA mode (not expanded) ───────────────────────
    print("\n" + "=" * 60)
    print("SECTION F: GQA mode (kv_head_num < head_num)")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "F1: GQA nh=16 kv=8 hd=128", 16, 8, 128,
        use_mask=True, B=B, S=16)
    results["F1"] = (ok, err, cos)

    # ── Section G: Small head_dim (vision path config) ───────────
    print("\n" + "=" * 60)
    print("SECTION G: head_dim=64 (vision path)")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "G1: hd=64 nh=16 kv=16", 16, 16, 64,
        use_mask=True, B=B, S=16)
    results["G1"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "G2: hd=64 nh=16 kv=8 (GQA)", 16, 8, 64,
        use_mask=True, B=B, S=16)
    results["G2"] = (ok, err, cos)

    # ── Section H: Real model params, large seqlen ───────────────
    print("\n" + "=" * 60)
    print("SECTION H: Large seqlen (image-only typical)")
    print("=" * 60)

    ok, err, cos = _run_full_attention(
        "H1: S=880 hd=128", nh, kv_nh, hd,
        use_mask=True, B=B, S=880)
    results["H1"] = (ok, err, cos)

    ok, err, cos = _run_full_attention(
        "H2: S=880 hd=128 is_triu=1", nh, kv_nh, hd,
        use_mask=True, B=B, S=880, is_triu_mask=1)
    results["H2"] = (ok, err, cos)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v[0])
    total = len(results)
    print(f"Results: {passed}/{total} passed\n")

    for key, (ok, err, cos) in results.items():
        if ok:
            print(f"  PASS  — {key}  (cos={cos:.6f})")
        elif err:
            print(f"  ERROR — {key}: {err}")
        else:
            print(f"  FAIL  — {key}  (cos={cos:.6f})")

    if platform == "310P":
        print("\n310P Analysis:")
        baseline_ok = results.get("A1", (False, None, 0))[0]
        if not baseline_ok:
            print("  → Baseline (BSND+PA_ENCODER+MASK_NORM) fails on 310P")

        triu_ok = results.get("B1", (False, None, 0))[0]
        if triu_ok:
            print("  → is_triu_mask=1 works! Use MASK_TYPE_NORM + is_triu_mask=1")
        no_mask_ok = results.get("D1", (False, None, 0))[0]
        if no_mask_ok:
            print("  → MASK_TYPE_UNDEFINED (no mask) works!")
        compress_ok = results.get("D2", (False, None, 0))[0]
        if compress_ok:
            print("  → MASK_TYPE_NORM_COMPRESS works!")
        calc_ok = any(results.get(k, (False, None, 0))[0]
                      for k in ["E1", "E2", "E3"])
        if calc_ok:
            print("  → Non-PA_ENCODER calc_type works!")
        hd64_ok = results.get("G1", (False, None, 0))[0]
        if hd64_ok:
            print("  → head_dim=64 works (confirms hd=128 is the issue)")
        gqa_ok = results.get("F1", (False, None, 0))[0]
        if gqa_ok:
            print("  → GQA mode works (confirms MHA expansion not needed)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
