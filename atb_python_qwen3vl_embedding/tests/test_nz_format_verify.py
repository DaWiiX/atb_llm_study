"""Verify that torch_npu.empty_with_format creates correct FRACTAL_NZ tensors.

This script validates:
1. empty_with_format with FRACTAL_NZ creates a valid tensor
2. Data copy works correctly (NZ-layout CPU data → NZ-layout NPU tensor)
3. npu_format_cast can be used to change format
4. ATB SelfAttention can accept it (optional, requires graph build)

Usage:
    ASCEND_PLATFORM=310P python tests/test_nz_format_verify.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

from atb_python_qwen3vl_embedding.env import ASCEND_PLATFORM
from atb_python_qwen3vl_embedding.utils import make_causal_mask_nz, set_atb_buffer_size

# ACL format constants
FRACTAL_NZ = 29  # ACL_FORMAT_FRACTAL_NZ
ND_FORMAT = 2    # ACL_FORMAT_ND

set_atb_buffer_size(300 * 1024 * 1024)  # 300 MB for NPU memory pool (small diagnostic SA graph)


def test_01_create_nz_tensor():
    """Test 1: Can we create a FRACTAL_NZ format tensor?"""
    print("\n=== Test 1: Create FRACTAL_NZ tensor ===")
    for S in [3, 4, 8, 16, 32]:
        n1 = (S + 15) // 16
        s_pad = n1 * 16
        shape = (1, n1, s_pad, 16)
        try:
            t = torch_npu.empty_with_format(
                shape, dtype=torch.float16, device='npu:0',
                acl_format=FRACTAL_NZ)
            fmt = torch_npu.get_npu_format(t)
            print(f"  S={S}: shape={shape}, format={fmt}, "
                  f"dtype={t.dtype}, device={t.device}  ✅")
        except RuntimeError as e:
            print(f"  S={S}: shape={shape}, FAILED: {e}  ❌")


def test_02_copy_data():
    """Test 2: Copy CPU NZ-layout data into FRACTAL_NZ NPU tensor."""
    print("\n=== Test 2: Copy NZ data into FRACTAL_NZ tensor ===")
    for S in [3, 4, 8, 16]:
        n1 = (S + 15) // 16
        s_pad = n1 * 16
        shape = (1, n1, s_pad, 16)

        # Generate NZ-layout data on CPU
        mask_cpu = make_causal_mask_nz(S, device="cpu").half()

        # Create NPU tensor with NZ format
        mask_npu = torch_npu.empty_with_format(
            shape, dtype=torch.float16, device='npu:0',
            acl_format=FRACTAL_NZ)

        # Copy data from CPU to NPU
        mask_npu.copy_(mask_cpu)

        # Read back and compare
        mask_back = mask_npu.cpu()
        diff = (mask_cpu - mask_back).abs().max().item()
        fmt = torch_npu.get_npu_format(mask_npu)

        status = "✅" if diff == 0.0 else f"❌ (max_diff={diff})"
        print(f"  S={S}: shape={shape}, format={fmt}, roundtrip max_diff={diff:.6f} {status}")


def test_03_mask_values():
    """Test 3: Verify causal mask values after roundtrip."""
    print("\n=== Test 3: Verify causal mask values ===")
    for S in [4, 8]:
        n1 = (S + 15) // 16
        s_pad = n1 * 16
        shape = (1, n1, s_pad, 16)

        mask_cpu = make_causal_mask_nz(S, device="cpu").half()
        mask_npu = torch_npu.empty_with_format(
            shape, dtype=torch.float16, device='npu:0',
            acl_format=FRACTAL_NZ)
        mask_npu.copy_(mask_cpu)
        mask_back = mask_npu.cpu()

        # Reconstruct ND mask from NZ layout: ND[row][col] = NZ[0][col//16][row][col%16]
        nd = torch.zeros(S, S)
        for r in range(S):
            for c in range(S):
                nd[r, c] = mask_back[0, c // 16, r, c % 16].item()

        # Check causal structure: upper triangle should be -65504, else 0
        errors = 0
        for r in range(S):
            for c in range(S):
                expected = -65504.0 if c > r else 0.0
                got = nd[r, c].item()
                if abs(got - expected) > 0.01:
                    errors += 1
                    if errors <= 3:
                        print(f"  MISMATCH at [{r},{c}]: expected={expected}, got={got}")
        status = "✅" if errors == 0 else f"❌ ({errors} errors)"
        print(f"  S={S}: causal structure {status}")


def test_04_shape_broadcast():
    """Test 4: Verify shapes for typical model S values."""
    print("\n=== Test 4: Typical model S values ===")
    for S, label in [(3, "text-only"), (880, "image-only"), (883, "text+image")]:
        n1 = (S + 15) // 16
        s_pad = n1 * 16
        shape = (1, n1, s_pad, 16)
        elems = n1 * s_pad * 16
        try:
            mask_cpu = make_causal_mask_nz(S, device="cpu").half()
            mask_npu = torch_npu.empty_with_format(
                shape, dtype=torch.float16, device='npu:0',
                acl_format=FRACTAL_NZ)
            mask_npu.copy_(mask_cpu)
            fmt = torch_npu.get_npu_format(mask_npu)
            print(f"  {label} (S={S}): shape={shape}, elems={elems}, format={fmt} ✅")
        except RuntimeError as e:
            print(f"  {label} (S={S}): FAILED: {e} ❌")


def test_03b_format_cast():
    """Test 3b: Alternative approach — create ND tensor, then npu_format_cast."""
    print("\n=== Test 3b: npu_format_cast approach ===")
    for S in [4, 8, 16]:
        n1 = (S + 15) // 16
        s_pad = n1 * 16
        shape = (1, n1, s_pad, 16)

        # Create NZ-layout data on CPU
        mask_cpu = make_causal_mask_nz(S, device="cpu").half()

        # Approach B: create ND tensor, copy data, then cast format
        mask_npu_nd = mask_cpu.clone().npu()  # ND format, NZ data
        fmt_before = torch_npu.get_npu_format(mask_npu_nd)

        try:
            mask_npu_nz = torch_npu.npu_format_cast(mask_npu_nd, FRACTAL_NZ)
            fmt_after = torch_npu.get_npu_format(mask_npu_nz)

            # Check if data is preserved after format cast
            mask_back = mask_npu_nz.cpu()
            diff = (mask_cpu - mask_back).abs().max().item()

            status = "✅" if diff == 0.0 else f"❌ (data changed, max_diff={diff})"
            print(f"  S={S}: format {fmt_before}→{fmt_after}, roundtrip diff={diff:.6f} {status}")
        except RuntimeError as e:
            print(f"  S={S}: npu_format_cast FAILED: {e} ❌")


def test_05_atb_acceptance():
    """Test 5: Can ATB SelfAttention accept this format tensor?"""
    print("\n=== Test 5: ATB SelfAttention with FRACTAL_NZ mask ===")
    try:
        import torch_atb
    except ImportError:
        print("  SKIP: torch_atb not available")
        return

    S = 8
    nh, kvh, hd = 4, 4, 64
    B = 1
    ntoken = B * S

    n1 = (S + 15) // 16
    s_pad = n1 * 16
    shape = (1, n1, s_pad, 16)

    # Approach A: empty_with_format
    mask_cpu = make_causal_mask_nz(S, device="cpu").half()

    for approach, create_fn in [
        ("empty_with_format(FN_Z)", lambda: _empty_nz(shape, mask_cpu)),
        ("npu_format_cast", lambda: _cast_nz(mask_cpu)),
        ("ND format (current bug)", lambda: mask_cpu.npu()),
    ]:
        try:
            mask = create_fn()
            fmt = torch_npu.get_npu_format(mask)
            p = torch_atb.SelfAttentionParam(
                head_num=nh, kv_head_num=kvh,
                qk_scale=1.0 / (hd ** 0.5),
            )
            p.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
            p.input_layout = torch_atb.InputLayout.TYPE_BSND
            p.mask_type = torch_atb.SelfAttentionParam.MaskType.MASK_TYPE_NORM

            builder = torch_atb.Builder(f"test_{approach}")
            q_in = builder.add_input("q")
            k_in = builder.add_input("k")
            v_in = builder.add_input("v")
            m_in = builder.add_input("mask")
            sl_in = builder.add_input("seqlen")
            sa = builder.add_node([q_in, k_in, v_in, m_in, sl_in], p)
            builder.mark_output(sa.get_output(0))
            graph = builder.build()

            q = torch.randn(ntoken, nh, hd, dtype=torch.float16, device='npu:0')
            k = torch.randn(ntoken, kvh, hd, dtype=torch.float16, device='npu:0')
            v = torch.randn(ntoken, kvh, hd, dtype=torch.float16, device='npu:0')
            seqlen = torch.tensor([ntoken], dtype=torch.int32)
            out = graph.forward([q, k, v, mask, seqlen])[0]
            print(f"  {approach}: format={fmt}, output shape={out.shape} ✅")
        except RuntimeError as e:
            print(f"  {approach}: FAILED — {e} ❌")


def _empty_nz(shape, cpu_data):
    """Create FRACTAL_NZ tensor via empty_with_format + copy."""
    t = torch_npu.empty_with_format(
        shape, dtype=torch.float16, device='npu:0',
        acl_format=FRACTAL_NZ)
    t.copy_(cpu_data)
    return t


def _cast_nz(cpu_data):
    """Create tensor via .npu() then npu_format_cast."""
    t = cpu_data.npu()
    return torch_npu.npu_format_cast(t, FRACTAL_NZ)


def main():
    print("=" * 60)
    print("FRACTAL_NZ Format Verification")
    print(f"Platform: {ASCEND_PLATFORM}")
    print(f"Device: {torch.npu.get_device_name(0)}")
    print("=" * 60)

    test_01_create_nz_tensor()
    test_02_copy_data()
    test_03_mask_values()
    test_03b_format_cast()
    test_04_shape_broadcast()

    # Note: on 910B, SelfAttention expects ND format mask, so
    # ATB acceptance test will FAIL with NZ format. Only valid on 310P.
    if ASCEND_PLATFORM == '310P':
        test_05_atb_acceptance()
    else:
        print("\n=== Test 5: ATB SelfAttention — SKIP (only valid on 310P) ===")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
