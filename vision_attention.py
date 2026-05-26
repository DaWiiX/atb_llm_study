"""Qwen3VLVisionAttention ATB graph builder.

Pipeline:
    hidden → qkv Linear → reshape(4D) → Split(3-way) → Q, K, V
    Q ──→ one-step-flatten ─┐
    K ──→ one-step-flatten ─┼→ RopeOperation → SelfAttention → proj Linear
    V ──→ reshape(3D) ──────┘

Uses native ATB RopeOperation (rotary_coeff=2) for half-rotation.
SelfAttention: PA_ENCODER with TYPE_BSND layout (3D: N×nh×hd).

Note: Split preserves tensor rank (4D → 4D with split dim divided).
Q/K are flattened directly from Split output in one step to avoid
named-reshape shape propagation issues in ATB.
"""
from .utils import make_rope_operation, make_linear, make_self_attention, \
    get_atb_builder, make_split


def add_vision_attention_graph(builder, inp_hidden,
                               inp_qkv_w, inp_qkv_b,
                               inp_proj_w, inp_proj_b,
                               inp_cos, inp_sin, inp_seqlen,
                               num_heads, head_dim, name_prefix="va_"):
    """Add Qwen3VLVisionAttention subgraph into an existing builder.

    Args:
        builder:     ATB builder instance.
        inp_hidden:  (N, hidden_size) float16.
        inp_qkv_w/b: combined QKV Linear weight and bias.
        inp_proj_w/b: output projection weight and bias.
        inp_cos/sin: (N, head_dim) rotary embeddings.
        inp_seqlen:  int32 [N].
        num_heads:   number of attention heads.
        head_dim:    dimension per head (hidden_size // num_heads).

    Returns the output tensor reference.
    """
    pfx = name_prefix

    qkv = builder.add_node([inp_hidden, inp_qkv_w, inp_qkv_b],
                           make_linear(has_bias=True))
    builder.reshape(qkv.get_output(0),
                    lambda s: [s[0], 3, num_heads, head_dim], pfx + "qkv4")
    sp = builder.add_node([pfx + "qkv4"], make_split(1, 3))

    # Q path: Split output (N, 1, nh, hd) → flat (N, nh*hd) in one step
    builder.reshape(sp.get_output(0),
                    lambda s: [s[0], s[2] * s[3]], pfx + "q_flat")

    # K path: same one-step flatten
    builder.reshape(sp.get_output(1),
                    lambda s: [s[0], s[2] * s[3]], pfx + "k_flat")

    # Native RopeOperation: Q and K together
    rope_op = make_rope_operation()
    rope_node = builder.add_node(
        [pfx + "q_flat", pfx + "k_flat", inp_cos, inp_sin, inp_seqlen], rope_op)

    builder.reshape(rope_node.get_output(0),
                    lambda s: [s[0], num_heads, head_dim], pfx + "q_rope")
    builder.reshape(rope_node.get_output(1),
                    lambda s: [s[0], num_heads, head_dim], pfx + "k_rope")

    # V path: Split output (N, 1, nh, hd) → 3D (N, nh, hd), no RoPE
    builder.reshape(sp.get_output(2),
                    lambda s: [s[0], s[2], s[3]], pfx + "v3d")

    sa = builder.add_node(
        [pfx + "q_rope", pfx + "k_rope", pfx + "v3d", inp_seqlen],
        make_self_attention(num_heads, num_heads, head_dim))
    builder.reshape(sa.get_output(0),
                    lambda s: [s[0], s[1] * s[2]], pfx + "sa_flat")

    proj = builder.add_node(
        [pfx + "sa_flat", inp_proj_w, inp_proj_b],
        make_linear(has_bias=True))
    return proj.get_output(0)


def build_vision_attention(num_heads, head_dim, name="VisionAttention"):
    """Build standalone Qwen3VLVisionAttention ATB graph for testing.

    Returns (builder, graph_op, input_names).
    """
    builder = get_atb_builder(name)

    x_in = builder.add_input("x")
    qkv_w = builder.add_input("qkv_w")
    qkv_b = builder.add_input("qkv_b")
    proj_w = builder.add_input("proj_w")
    proj_b = builder.add_input("proj_b")
    c_in = builder.add_input("c")
    s_in = builder.add_input("s")
    seq_in = builder.add_input("seq")

    out = add_vision_attention_graph(
        builder, x_in, qkv_w, qkv_b, proj_w, proj_b,
        c_in, s_in, seq_in, num_heads, head_dim)
    builder.mark_output(out)

    graph_op = builder.build()
    input_names = ["x", "qkv_w", "qkv_b", "proj_w", "proj_b", "c", "s", "seq"]
    return builder, graph_op, input_names
