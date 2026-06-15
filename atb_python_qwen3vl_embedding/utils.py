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
import torch
import torch_atb
import torch.nn.functional as F

from .env import ASCEND_PLATFORM


# ── Platform detection ────────────────────────────────────────────────

def get_platform() -> str:
    """Return the current Ascend NPU platform: '910B' or '310P'."""
    return ASCEND_PLATFORM


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
                        mask_type=None, use_mask=False,
                        calc_type=None, input_layout=None,
                        is_triu_mask: int = 0,
                        kernel_type=None,
                        kvcache_cfg=None):
    """ATB SelfAttentionParam with flexible configuration.

    Args:
        num_heads:    number of query heads
        num_kv_heads: number of key/value heads (supports GQA when < num_heads)
        head_dim:     dimension per attention head
        mask_type:    explicit mask type (if use_mask is False)
        use_mask:     if True, sets MASK_TYPE_NORM
        calc_type:    calc type enum (default PA_ENCODER)
        input_layout: input layout enum (default TYPE_BSND)
        is_triu_mask: 0/1, enable causal mask internal optimization
        kernel_type:  kernel precision type (default KERNELTYPE_DEFAULT)
    """
    p = torch_atb.SelfAttentionParam(
        head_num=num_heads,
        kv_head_num=num_kv_heads,
        qk_scale=1.0 / (head_dim ** 0.5),
    )
    p.calc_type = (calc_type
                   if calc_type is not None
                   else torch_atb.SelfAttentionParam.CalcType.PA_ENCODER)
    p.input_layout = (input_layout
                      if input_layout is not None
                      else torch_atb.InputLayout.TYPE_BSND)
    p.is_triu_mask = is_triu_mask
    if kernel_type is not None:
        p.kernel_type = kernel_type
    if kvcache_cfg is not None:
        p.kvcache_cfg = kvcache_cfg

    if use_mask:
        p.mask_type = torch_atb.SelfAttentionParam.MaskType.MASK_TYPE_NORM
    elif mask_type is not None:
        if isinstance(mask_type, int):
            # Support MASK_TYPE_CAUSAL_MASK (value=9) which is not exposed
            # in the Python MaskType enum but IS supported by C++ ATB.
            p.mask_type = torch_atb.SelfAttentionParam.MaskType(mask_type)
        else:
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


# ── 310P NZ mask conversion ──────────────────────────────────────────


def make_causal_mask_nz(S: int, device: str = "cpu") -> torch.Tensor:
    """Generate causal mask directly in NZ (FRACTAL_NZ) layout for 310P.

    Combines causal mask generation and NZ conversion in one step:
    - Upper triangle (col > row): -65504 (masked)
    - Lower triangle + diagonal: 0 (attend)
    - Padding: 0

    Args:
        S:      sequence length
        device: target device ('cpu' or 'npu')
    Returns:
        (1, n1, s_pad, 16) fp32 tensor in NZ layout
    """
    n1 = (S + 15) // 16
    s_pad = n1 * 16
    mask_nz = torch.zeros(1, n1, s_pad, 16, device=device)
    for r in range(S):
        for c in range(r + 1, S):     # col > row = masked (upper triangle)
            block_col = c // 16
            col_in_block = c % 16
            mask_nz[0, block_col, r, col_in_block] = -65504.0
    return mask_nz


def make_causal_mask_nz_npu(S: int) -> torch.Tensor:
    """Generate causal mask and place on NPU with FRACTAL_NZ format.

    This is the 310P production path — matches C++ qwen3vl_model.cpp:569-581:
      1. Generate causal mask in NZ layout on CPU (fp16)
      2. Allocate NPU tensor with ACL_FORMAT_FRACTAL_NZ (format=29)
      3. Copy CPU data to NPU

    The format tag is CRITICAL: without it, ATB SelfAttention sees ND format,
    tries internal ND→NZ TransdataOperation, and fails on 310P with
    "call operation setup fail".

    Args:
        S: sequence length
    Returns:
        (1, n1, s_pad, 16) fp16 tensor on NPU, format=FRACTAL_NZ
    """
    import torch_npu  # noqa: F401 — required for empty_with_format
    n1 = (S + 15) // 16
    s_pad = n1 * 16
    shape = (1, n1, s_pad, 16)
    # 1. Generate NZ-layout data on CPU (fp16)
    cpu_data = make_causal_mask_nz(S, device="cpu").half()
    # 2. Allocate NPU tensor with FRACTAL_NZ format
    nz_tensor = torch_npu.empty_with_format(
        shape, dtype=torch.float16, device=torch.device("npu:0"),
        acl_format=29)  # 29 = ACL_FORMAT_FRACTAL_NZ
    # 3. Copy data — memory layouts must match (both NZ)
    nz_tensor.copy_(cpu_data)
    return nz_tensor


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
