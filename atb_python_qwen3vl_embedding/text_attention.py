"""Qwen3VLTextAttention ATB graph builder.

Pipeline:
    hidden_states ──→ q_proj → reshape → q_norm → reshape ─┐
                   │→ k_proj → reshape → k_norm → reshape ─┼→ RopeOp → SelfAttention → o_proj
                   │→ v_proj → reshape ─────────────────────┘
    cos, sin, seqlen ────────────────────────────────────────┘

Q/K receive RMSNorm on head_dim before RoPE (Qwen3VLTextRMSNorm).
SelfAttention uses BSND layout with optional MASK_TYPE_NORM causal mask.
Uses native ATB RopeOperation (rotary_coeff=2) for LLAMA-style half-rotation.
"""
from .utils import (
    make_linear, make_rms_norm, make_self_attention,
    get_atb_builder, make_rope_operation,
)


def add_attention_graph(builder, inp_hidden,
                        inp_q_w, inp_k_w, inp_v_w, inp_o_w,
                        inp_qn_w, inp_kn_w,
                        inp_cos, inp_sin, inp_seqlen,
                        B, S, num_heads, num_kv_heads, head_dim,
                        name_prefix="attn_", eps=1e-6, use_mask=True,
                        inp_mask=None):
    """Add Qwen3VLTextAttention subgraph into an existing builder.

    Args:
        builder:      ATB builder instance.
        inp_hidden:   hidden_states input (B, S, hidden_size).
        inp_q/k/v/o_w: projection weight inputs.
        inp_qn/kn_w:  Q/K RMSNorm weight inputs (shape: head_dim,).
        inp_cos/sin:  rotary position embedding inputs (B*S, head_dim).
        inp_seqlen:   sequence length int32 tensor [B*S].
        B, S:         batch size, sequence length.
        num_heads:    number of query heads.
        num_kv_heads: number of key/value heads (GQA when < num_heads).
        head_dim:     dimension per attention head.
        use_mask:     if True, MASK_TYPE_NORM additive mask is used.
        inp_mask:     2D (S, S) float16 mask tensor (required if use_mask=True).

    Returns the output tensor reference from o_proj.
    """
    pfx = name_prefix

    # Q: Linear → reshape to (B*S, nh, hd) → RMSNorm → flatten back
    q_lin = builder.add_node([inp_hidden, inp_q_w], make_linear())
    builder.reshape(q_lin.get_output(0),
                    lambda s: [s[0] * s[1], num_heads, head_dim], pfx + "q2d")
    q_n = builder.add_node([pfx + "q2d", inp_qn_w], make_rms_norm(eps))
    builder.reshape(q_n.get_output(0),
                    lambda s: [s[0], s[1] * s[2]], pfx + "q4r")

    # K: Linear → reshape to (B*S, kv_nh, hd) → RMSNorm → flatten back
    k_lin = builder.add_node([inp_hidden, inp_k_w], make_linear())
    builder.reshape(k_lin.get_output(0),
                    lambda s: [s[0] * s[1], num_kv_heads, head_dim], pfx + "k2d")
    k_n = builder.add_node([pfx + "k2d", inp_kn_w], make_rms_norm(eps))
    builder.reshape(k_n.get_output(0),
                    lambda s: [s[0], s[1] * s[2]], pfx + "k4r")

    # V: Linear → reshape to 3D (B*S, kv_nh, hd)
    v_lin = builder.add_node([inp_hidden, inp_v_w], make_linear())
    builder.reshape(v_lin.get_output(0),
                    lambda s: [s[0] * s[1], num_kv_heads, head_dim], pfx + "v2d")

    # Native RopeOperation: Q and K together, GQA-compatible
    rope_op = make_rope_operation()
    rope_node = builder.add_node(
        [pfx + "q4r", pfx + "k4r", inp_cos, inp_sin, inp_seqlen], rope_op)

    builder.reshape(rope_node.get_output(0),
                    lambda s: [s[0], num_heads, head_dim], pfx + "q_rope")
    builder.reshape(rope_node.get_output(1),
                    lambda s: [s[0], num_kv_heads, head_dim], pfx + "k_rope")

    # SelfAttention: 5 inputs when masked, 4 otherwise
    sa_inputs = [pfx + "q_rope", pfx + "k_rope", pfx + "v2d"]
    if use_mask:
        sa_inputs.append(inp_mask)
    sa_inputs.append(inp_seqlen)

    sa = builder.add_node(
        sa_inputs,
        make_self_attention(num_heads, num_kv_heads, head_dim, use_mask=use_mask),
    )
    builder.reshape(sa.get_output(0),
                    lambda s: [B, S, s[1] * s[2]], pfx + "attn_flat")

    o_lin = builder.add_node([pfx + "attn_flat", inp_o_w], make_linear())
    return o_lin.get_output(0)


def build_attention(num_heads, num_kv_heads, head_dim, B=1, S=16,
                    eps=1e-6, use_mask=False):
    """Build standalone Qwen3VLTextAttention ATB graph for testing.

    Returns (builder, graph_op, input_names).
    """
    builder = get_atb_builder("Qwen3VLTextAttention")

    inp_hs = builder.add_input("hidden_states")
    inp_q_w = builder.add_input("q_weight")
    inp_k_w = builder.add_input("k_weight")
    inp_v_w = builder.add_input("v_weight")
    inp_o_w = builder.add_input("o_weight")
    inp_qn_w = builder.add_input("q_norm_weight")
    inp_kn_w = builder.add_input("k_norm_weight")
    inp_cos = builder.add_input("cos")
    inp_sin = builder.add_input("sin")
    inp_mask = builder.add_input("mask") if use_mask else None
    inp_seqlen = builder.add_input("seqlen")

    out_name = add_attention_graph(
        builder, inp_hs, inp_q_w, inp_k_w, inp_v_w, inp_o_w,
        inp_qn_w, inp_kn_w, inp_cos, inp_sin, inp_seqlen,
        B, S, num_heads, num_kv_heads, head_dim,
        eps=eps, use_mask=use_mask, inp_mask=inp_mask,
    )
    builder.mark_output(out_name)

    graph_op = builder.build()
    input_names = [
        "hidden_states", "q_weight", "k_weight", "v_weight", "o_weight",
        "q_norm_weight", "k_norm_weight", "cos", "sin", "seqlen",
    ]
    if use_mask:
        input_names.insert(-1, "mask")  # mask goes before seqlen
    return builder, graph_op, input_names
