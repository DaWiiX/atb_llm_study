"""Qwen3VLVisionBlock ATB graph builder.

Pipeline:
    hidden → LayerNorm → VisionAttention → +residual
           → LayerNorm → VisionMLP → +residual → output
"""
from .utils import make_elewise_add, make_layer_norm, get_atb_builder
from .vision_attention import add_vision_attention_graph
from .vision_mlp import add_vision_mlp_graph


def add_vision_block_graph(builder, inp_hidden,
                           inp_qkv_w, inp_qkv_b, inp_proj_w, inp_proj_b,
                           inp_fc1_w, inp_fc1_b, inp_fc2_w, inp_fc2_b,
                           inp_n1_w, inp_n1_b, inp_n2_w, inp_n2_b,
                           inp_cos, inp_sin, inp_seqlen,
                           num_heads, head_dim, name_prefix="vb_"):
    """Add Qwen3VLVisionBlock subgraph into an existing builder.

    Args:
        (see vision_attention.add_vision_attention_graph for attention params)
        inp_fc1/fc2_w/b: MLP linear weights and biases.
        inp_n1/n2_w/b:   LayerNorm weights and biases.

    Returns the output tensor reference.
    """
    pfx = name_prefix

    ln1 = builder.add_node([inp_hidden, inp_n1_w, inp_n1_b], make_layer_norm())
    attn_out = add_vision_attention_graph(
        builder, ln1.get_output(0),
        inp_qkv_w, inp_qkv_b, inp_proj_w, inp_proj_b,
        inp_cos, inp_sin, inp_seqlen, num_heads, head_dim,
        name_prefix=pfx + "attn_")
    r1 = builder.add_node([inp_hidden, attn_out], make_elewise_add())

    ln2 = builder.add_node([r1.get_output(0), inp_n2_w, inp_n2_b],
                           make_layer_norm())
    mlp_out = add_vision_mlp_graph(
        builder, ln2.get_output(0),
        inp_fc1_w, inp_fc1_b, inp_fc2_w, inp_fc2_b,
        name_prefix=pfx + "mlp_")

    r2 = builder.add_node([r1.get_output(0), mlp_out], make_elewise_add())
    return r2.get_output(0)


def build_vision_block(num_heads, head_dim, name="VisionBlock"):
    """Build standalone Qwen3VLVisionBlock ATB graph for testing.

    Returns (builder, graph_op, input_names).

    Input names (16): x, qkv_w, qkv_b, proj_w, proj_b,
                      fc1_w, fc1_b, fc2_w, fc2_b,
                      n1_w, n1_b, n2_w, n2_b,
                      c, s, seq
    """
    builder = get_atb_builder(name)

    x_in = builder.add_input("x")
    qkv_w = builder.add_input("qkv_w")
    qkv_b = builder.add_input("qkv_b")
    proj_w = builder.add_input("proj_w")
    proj_b = builder.add_input("proj_b")
    fc1_w = builder.add_input("fc1_w")
    fc1_b = builder.add_input("fc1_b")
    fc2_w = builder.add_input("fc2_w")
    fc2_b = builder.add_input("fc2_b")
    n1_w = builder.add_input("n1_w")
    n1_b = builder.add_input("n1_b")
    n2_w = builder.add_input("n2_w")
    n2_b = builder.add_input("n2_b")
    c_in = builder.add_input("c")
    s_in = builder.add_input("s")
    seq_in = builder.add_input("seq")

    out = add_vision_block_graph(
        builder, x_in,
        qkv_w, qkv_b, proj_w, proj_b,
        fc1_w, fc1_b, fc2_w, fc2_b,
        n1_w, n1_b, n2_w, n2_b,
        c_in, s_in, seq_in, num_heads, head_dim)
    builder.mark_output(out)

    graph_op = builder.build()
    input_names = [
        "x", "qkv_w", "qkv_b", "proj_w", "proj_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        "n1_w", "n1_b", "n2_w", "n2_b",
        "c", "s", "seq",
    ]
    return builder, graph_op, input_names
