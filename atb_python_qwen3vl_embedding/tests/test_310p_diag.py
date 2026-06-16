"""310P platform diagnostic: isolate which ATB operator/param combo fails.

On 310P (Atlas推理系列产品), text-path tests fail while vision-path tests pass.
The key differences between vision and text SelfAttention:
  - Vision: MHA (num_kv_heads == num_heads), no mask, Linear+bias, Split QKV
  - Text:   GQA (num_kv_heads < num_heads), MASK_TYPE_NORM, Linear no-bias

This script tests each combination to isolate the exact failure point.
Run on 310P to see which tests fail; on 910B all should pass.

Usage:
    ASCEND_PLATFORM=310P python atb_python_qwen3vl_embedding/tests/test_310p_diag.py
"""

import os
import sys
import math
import torch
import torch_npu  # noqa: F401 — required for NPU ops
import torch.nn.functional as F

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from atb_python_qwen3vl_embedding import data_utils, utils
from atb_python_qwen3vl_embedding.text_attention import build_attention
from atb_python_qwen3vl_embedding.text_model import make_causal_mask
from atb_python_qwen3vl_embedding.utils import is_310p, make_causal_mask_nz_npu
from atb_python_qwen3vl_embedding.transformers_runner import run_attention


def _cosine(a, b):
    """Compute cosine similarity between two tensors."""
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _run_attention_test(name, num_heads, num_kv_heads, head_dim, use_mask, B=1, S=16):
    """Build and run a single attention graph, return (ok, error_msg, cosine)."""
    print(f"\n  [{name}] nh={num_heads} kv_nh={num_kv_heads} hd={head_dim} mask={use_mask}")
    try:
        hidden_size = num_heads * head_dim
        config = data_utils.make_config(
            hidden_size=hidden_size, num_heads=num_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim)
        gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=42)
        gen_data["config"] = config  # required by run_attention

        if use_mask:
            m2d = make_causal_mask(S)
            gen_data["attention_mask"] = m2d.unsqueeze(0).unsqueeze(0).float()

        # Reference (transformers eager)
        ref_out, weights = run_attention(gen_data, seed=42)

        # ATB graph
        _, graph_op, _ = build_attention(
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            B=B, S=S, use_mask=use_mask,
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
            if is_310p():
                mask = make_causal_mask_nz_npu(S)
            else:
                mask = make_causal_mask(S).half().npu()
            inputs.append(mask)
        inputs.append(torch.tensor([ntoken], dtype=torch.int32))

        atb_out = graph_op.forward(inputs)[0].cpu().float()
        cos = _cosine(ref_out, atb_out)
        mse = float(torch.mean((ref_out.float() - atb_out.float()) ** 2))
        max_diff = float(torch.max(torch.abs(ref_out.float() - atb_out.float())))
        # 0.99: moderate fp16 accumulation (single attention, cross-framework) — see THRESHOLDS.md
        ok = cos > 0.99
        status = "PASS" if ok else f"FAIL (cos={cos:.6f})"
        print(f"    {status}  mse={mse:.8f}  max_diff={max_diff:.8f}")
        return ok, None, cos
    except RuntimeError as e:
        print(f"    ERROR: {e}")
        return False, str(e), 0.0


def main():
    platform = utils.get_platform()
    print(f"=== 310P Diagnostic Test Suite ===")
    print(f"Platform: {platform}")
    print(f"Device: {torch.npu.get_device_name(0)}")

    utils.set_atb_buffer_size(300 * 1024 * 1024)

    head_dim = 64
    results = {}

    # Test 1: Baseline — MHA + no mask (should always pass)
    ok, err, cos = _run_attention_test("T1: MHA+nomask", 4, 4, head_dim, use_mask=False)
    results["T1: MHA+nomask"] = ok

    # Test 2: MHA + mask (isolates mask)
    ok, err, cos = _run_attention_test("T2: MHA+mask", 4, 4, head_dim, use_mask=True)
    results["T2: MHA+mask"] = ok

    # Test 3: GQA + no mask (isolates GQA)
    ok, err, cos = _run_attention_test("T3: GQA+nomask", 4, 2, head_dim, use_mask=False)
    results["T3: GQA+nomask"] = ok

    # Test 4: GQA + mask (real text attention — the failing case)
    ok, err, cos = _run_attention_test("T4: GQA+mask", 4, 2, head_dim, use_mask=True)
    results["T4: GQA+mask"] = ok

    # Test 5: Real model GQA (nh=32, kv_nh=4 — Qwen3VL-Embedding-2B)
    ok, err, cos = _run_attention_test("T5: Real GQA+mask", 32, 4, 64, use_mask=True)
    results["T5: Real GQA+mask"] = ok

    # Test 6: Real model MHA (nh=32, kv_nh=32)
    ok, err, cos = _run_attention_test("T6: Real MHA+mask", 32, 32, 64, use_mask=True)
    results["T6: Real MHA+mask"] = ok

    # ── hd=128 tests (isolates head_dim effect — text path uses hd=128) ──

    # Test 7: MHA + mask + hd=128 (isolates head_dim + mask interaction)
    ok, err, cos = _run_attention_test("T7: MHA+mask hd=128", 4, 4, 128, use_mask=True)
    results["T7: MHA+mask hd=128"] = ok

    # Test 8: MHA + nomask + hd=128 (isolates head_dim alone)
    ok, err, cos = _run_attention_test("T8: MHA+nomask hd=128", 4, 4, 128, use_mask=False)
    results["T8: MHA+nomask hd=128"] = ok

    # Test 9: Real model MHA + mask + hd=128 (nh=16, kv_nh=16 — actual 310P config)
    ok, err, cos = _run_attention_test("T9: Real MHA+mask hd=128", 16, 16, 128, use_mask=True)
    results["T9: Real MHA+mask hd=128"] = ok

    # Test 10: Real model MHA + mask + hd=128, S=4 (text-only typical seqlen)
    ok, err, cos = _run_attention_test("T10: Real MHA+mask hd=128 S=4", 16, 16, 128, use_mask=True, S=4)
    results["T10: Real MHA+mask hd=128 S=4"] = ok

    # Test 11: Real model MHA + mask + hd=128, S=880 (image-only typical seqlen)
    ok, err, cos = _run_attention_test("T11: Real MHA+mask hd=128 S=880", 16, 16, 128, use_mask=True, S=880)
    results["T11: Real MHA+mask hd=128 S=880"] = ok

    # ── Summary ──
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} — {name}")

    if platform == "310P":
        print("\n310P Analysis:")
        if not results["T1: MHA+nomask"]:
            print("  → SelfAttention itself is broken on 310P (even MHA+no mask)")

        # hd=64 analysis
        hd64_mask_fail = results["T1: MHA+nomask"] and not results["T2: MHA+mask"]
        if hd64_mask_fail:
            print("  → hd=64: MASK_TYPE_NORM is the problem (MHA passes without mask)")
        if results["T1: MHA+nomask"] and not results["T3: GQA+nomask"]:
            print("  → hd=64: GQA is the problem (MHA passes, GQA fails)")

        # hd=128 analysis
        if results.get("T8: MHA+nomask hd=128", False):
            print("  → hd=128: SelfAttention works WITHOUT mask")
        else:
            print("  → hd=128: SelfAttention fails even WITHOUT mask (head_dim too large)")
        if results.get("T8: MHA+nomask hd=128", False) and not results.get("T7: MHA+mask hd=128", False):
            print("  → hd=128: mask is the trigger (works without mask, fails with mask)")
        if not results.get("T7: MHA+mask hd=128", False):
            print("  → hd=128 + mask combination FAILS — this is the text path failure")

        # Check S dependency
        if not results.get("T9: Real MHA+mask hd=128", False):
            print("  → T9 (hd=128, mask, S=16): FAIL")
        if not results.get("T10: Real MHA+mask hd=128 S=4", False):
            print("  → T10 (hd=128, mask, S=4): FAIL — small seqlen also fails")
        if not results.get("T11: Real MHA+mask hd=128 S=880", False):
            print("  → T11 (hd=128, mask, S=880): FAIL — large seqlen also fails")
    else:
        print("\nAll tests should pass on 910B — validating test infrastructure.")
        if passed < total:
            print(f"  WARNING: {total - passed} tests failed unexpectedly on 910B!")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
