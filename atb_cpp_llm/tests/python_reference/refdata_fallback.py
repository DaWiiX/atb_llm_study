"""Three-tier fallback for computing SDPA reference outputs.

When ATB/NPU is unavailable (e.g. 310P with unsupported op combos), this
module provides deterministic reference computation via alternative backends.

Tiers (tried in order):
  1. torch_atb (NPU, fastest — may fail on 310P for some ops)
  2. torch_npu + F.scaled_dot_product_attention (NPU, no ATB dependency)
  3. torch + F.scaled_dot_product_attention (CPU, always available)

Usage:
    from refdata_fallback import compute_sdpa_reference
    out = compute_sdpa_reference(q, k, v, mask, nh, kvh)
"""

import warnings
from typing import Optional

import torch
import torch.nn.functional as F


def _sdpa_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                nh: int, kvh: int,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reference SDPA using vanilla PyTorch (CPU, always available).

    Inputs: (S, n_heads, hd) — returns (S, nh, hd).
    GQA: replicate K/V heads to match Q head count.
    """
    S, _, hd = q.shape
    if kvh != nh:
        assert nh % kvh == 0
        repeat = nh // kvh
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    q_b = q.unsqueeze(0).transpose(1, 2)  # (1, nh, S, hd)
    k_b = k.unsqueeze(0).transpose(1, 2)
    v_b = v.unsqueeze(0).transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q_b, k_b, v_b, attn_mask=attn_mask, is_causal=False)
    return out.transpose(1, 2).squeeze(0)


def compute_sdpa_reference(q_fp16: torch.Tensor,
                           k_fp16: torch.Tensor,
                           v_fp16: torch.Tensor,
                           nh: int, kvh: int,
                           attn_mask: Optional[torch.Tensor] = None,
                           ) -> torch.Tensor:
    """Compute SDPA reference with automatic backend fallback.

    Args:
        q_fp16:   (S, nh,  hd) fp16 query
        k_fp16:   (S, kvh, hd) fp16 key
        v_fp16:   (S, kvh, hd) fp16 value
        nh:       number of query heads
        kvh:      number of KV heads
        attn_mask: optional (S, S) fp32 additive mask

    Returns:
        (S, nh, hd) fp16 reference output

    Tries torch_atb first (fastest), then torch_npu, then CPU torch.
    Prints which tier was used.
    """
    # Tier 1: torch_atb (NPU, fastest)
    try:
        import torch_atb
        p = torch_atb.SelfAttentionParam(
            head_num=nh, kv_head_num=kvh,
            qk_scale=1.0 / (k_fp16.shape[-1] ** 0.5),
        )
        p.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
        p.input_layout = torch_atb.InputLayout.TYPE_BSND
        if attn_mask is not None:
            p.mask_type = torch_atb.SelfAttentionParam.MaskType.MASK_TYPE_NORM

        op = torch_atb.Operation(p)
        seqlen = torch.tensor([q_fp16.shape[0]], dtype=torch.int32)
        inputs = [q_fp16.npu(), k_fp16.npu(), v_fp16.npu()]
        if attn_mask is not None:
            inputs.append(attn_mask.half().npu())
        inputs.append(seqlen)
        out = op.forward(inputs)[0].cpu()
        print("[refdata] ✓ torch_atb (NPU)")
        return out
    except Exception as e:
        warnings.warn(f"[refdata] torch_atb failed: {e}")

    # Tier 2: torch_npu SDPA (NPU, no ATB dependency)
    try:
        import torch_npu  # noqa: F401
        q_npu = q_fp16.float().npu()
        k_npu = k_fp16.float().npu()
        v_npu = v_fp16.float().npu()
        mask_npu = attn_mask.npu() if attn_mask is not None else None
        out = _sdpa_torch(q_npu, k_npu, v_npu, nh, kvh, mask_npu)
        out_fp16 = out.cpu().half()
        print("[refdata] ✓ torch_npu SDPA (NPU, no ATB)")
        return out_fp16
    except Exception as e:
        warnings.warn(f"[refdata] torch_npu failed: {e}")

    # Tier 3: CPU torch (always available)
    q_cpu = q_fp16.float()
    k_cpu = k_fp16.float()
    v_cpu = v_fp16.float()
    out = _sdpa_torch(q_cpu, k_cpu, v_cpu, nh, kvh, attn_mask)
    print("[refdata] ✓ torch CPU SDPA (always available)")
    return out.half()
