"""
ATB parameter factories and utility functions.

Serves as the foundation layer for all ATB graph builders in this package.
Provides factories for every ATB operation used: Linear, RMSNorm, LayerNorm,
SelfAttention, RopeOperation, Split, Elementwise ops, and Activations.

Also provides:
    - Buffer size configuration (must be called once before any graph build)
    - Builder instantiation helper
    - Cosine similarity comparison utility for testing
    - Platform detection (910B vs 310P) for operator compatibility
"""
import os
import torch
import torch_atb
import torch.nn.functional as F


# ── Platform detection ────────────────────────────────────────────────

def get_platform() -> str:
    """Return the current Ascend NPU platform: '910B' or '310P'."""
    return os.getenv("ASCEND_PLATFORM", "910B")


def is_310p() -> bool:
    """Return True when running on Ascend 310P (Atlas推理系列产品)."""
    return get_platform() == "310P"


# ── Buffer management ───────────────────────────────────────────────

def set_atb_buffer_size(size_bytes: int):
    """Set ATB internal buffer size. Must be called once before any graph build.

    Calling this more than once may corrupt graph outputs.
    """
    torch_atb.set_buffer_size(size_bytes)


def get_atb_builder(name: str):
    """Return a new ATB graph builder with the given name."""
    return torch_atb.Builder(name)


# ── Tensor placement helpers ────────────────────────────────────────

def to_npu_half(t: torch.Tensor) -> torch.Tensor:
    """Return tensor as float16 on NPU.

    If the tensor is already NPU float16 this is a no-op. Callers keep
    sequence-length tensors separate because ATB attention expects them on CPU.
    """
    if t.device.type == "npu" and t.dtype == torch.float16:
        return t
    return t.half().npu()


def to_cpu_float(t: torch.Tensor) -> torch.Tensor:
    """Return tensor as float32 on CPU."""
    if t.device.type == "cpu" and t.dtype == torch.float32:
        return t
    return t.cpu().float()


def prepare_npu_weights(weights):
    """Convert a weight list to NPU float16, preserving order."""
    return [to_npu_half(w) for w in weights]


def make_seqlen_tensor(ntokens: int) -> torch.Tensor:
    """Create the CPU int32 sequence-length tensor required by ATB attention."""
    return torch.tensor([ntokens], dtype=torch.int32)


# ── ATB parameter factories ─────────────────────────────────────────

def make_linear(has_bias: bool = False):
    """ATB LinearParam (torch.nn.Linear equivalent, no bias by default)."""
    return torch_atb.LinearParam(has_bias=has_bias)


def make_rms_norm(eps: float = 1e-6):
    """ATB RmsNormParam for RMS Normalization."""
    p = torch_atb.RmsNormParam()
    p.layer_type = torch_atb.RmsNormParam.RmsNormType.RMS_NORM_NORM
    p.norm_param.epsilon = eps
    return p


def make_layer_norm(eps: float = 1e-6):
    """ATB LayerNormParam for standard Layer Normalization."""
    p = torch_atb.LayerNormParam()
    p.layer_type = torch_atb.LayerNormParam.LayerNormType.LAYER_NORM_NORM
    p.norm_param.epsilon = eps
    p.norm_param.begin_norm_axis = 1
    p.norm_param.begin_params_axis = 1
    return p


def make_elewise_add():
    """ATB ElewiseParam for element-wise addition (a + b)."""
    p = torch_atb.ElewiseParam()
    p.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    return p


def make_elewise_mul():
    """ATB ElewiseParam for element-wise multiplication (a * b)."""
    p = torch_atb.ElewiseParam()
    p.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_MUL
    return p


def make_silu():
    """ATB ActivationParam for SiLU/Swish activation."""
    p = torch_atb.ActivationParam()
    p.activation_type = torch_atb.ActivationType.ACTIVATION_SWISH
    return p


def make_split(split_dim: int, split_num: int):
    """ATB SplitParam.

    Splits a tensor along split_dim into split_num equal parts.
    Preserves tensor rank (split dim is divided, not squeezed).
    """
    return torch_atb.SplitParam(split_dim=split_dim, split_num=split_num)


def make_self_attention(num_heads: int, num_kv_heads: int, head_dim: int,
                        mask_type=None, use_mask=False):
    """ATB SelfAttentionParam.

    Args:
        num_heads:   number of query heads
        num_kv_heads: number of key/value heads (supports GQA when < num_heads)
        head_dim:    dimension per attention head
        mask_type:   explicit mask type (if use_mask is False)
        use_mask:    if True, sets MASK_TYPE_NORM (additive causal mask, 0=attend)

    Layout is TYPE_BSND (batch × seq × num_heads × head_dim), 3D input.
    Uses PA_ENCODER calc type with scaling factor 1/sqrt(head_dim).
    """
    p = torch_atb.SelfAttentionParam(
        head_num=num_heads,
        kv_head_num=num_kv_heads,
        qk_scale=1.0 / (head_dim ** 0.5),
        calc_type=torch_atb.SelfAttentionParam.CalcType.PA_ENCODER,
        input_layout=torch_atb.InputLayout.TYPE_BSND,
    )
    if use_mask:
        p.mask_type = torch_atb.SelfAttentionParam.MaskType.MASK_TYPE_NORM
    elif mask_type is not None:
        p.mask_type = mask_type
    return p


def make_rope_operation():
    """Native ATB RopeOperation with half-rotation (rotary_coeff=2).

    Inputs (5):  [q, k, cos, sin, seqlen]
        q: (ntokens, num_heads * head_dim) float16
        k: (ntokens, num_kv_heads * head_dim) float16  (supports GQA)
        cos, sin: (ntokens, head_dim) float16
        seqlen: (batch,) int32
    Outputs (2): [ropeQ, ropeK]

    Uses the LLAMA-style contiguous-half rotation: pairs (i, i+hd/2).
    Replaces the deprecated build_rope_graph() which suffered from
    subgraph state contamination when reused across graph invocations.
    """
    return torch_atb.Operation(torch_atb.RopeParam(rotary_coeff=2))


# ── Comparison utility ──────────────────────────────────────────────

def compare_tensors(ref: torch.Tensor, atb: torch.Tensor,
                    threshold: float = 0.999, label: str = "") -> bool:
    """Compare two tensors using cosine similarity, MSE, and max diff.

    Prints shape, cosine similarity, MSE, max absolute difference.
    Returns True if cosine > threshold.
    """
    cos_sim = F.cosine_similarity(
        ref.float().flatten(), atb.float().flatten(), dim=0).item()
    mse = F.mse_loss(ref.float(), atb.float()).item()
    max_diff = (ref.float() - atb.float()).abs().max().item()

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}shape: {ref.shape}  cosine: {cos_sim:.6f}  "
          f"MSE: {mse:.2e}  max_diff: {max_diff:.2e}")

    passed = cos_sim > threshold
    print(f"{prefix}{'PASS' if passed else 'FAIL'} (>{threshold})")
    return passed
