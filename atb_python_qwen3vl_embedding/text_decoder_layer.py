"""Qwen3VLTextDecoderLayer ATB graph builder.

Pipeline:
    hidden_states → input_layernorm → Attention → +residual
                  → post_attention_layernorm → MLP → +residual → output

Composes Qwen3VLTextAttention (with RoPE + causal mask) and Qwen3VLTextMLP.
"""
from .utils import make_rms_norm, make_elewise_add, get_atb_builder
from .text_attention import add_attention_graph
from .text_mlp import add_mlp_graph


def add_decoder_layer_graph(builder, inp_hidden,
                            inp_q_w, inp_k_w, inp_v_w, inp_o_w,
                            inp_qn_w, inp_kn_w,
                            inp_gate_w, inp_up_w, inp_down_w,
                            inp_iln_w, inp_pln_w,
                            inp_cos, inp_sin, inp_seqlen,
                            B, S, num_heads, num_kv_heads, head_dim,
                            name_prefix="layer_", eps=1e-6, use_mask=False,
                            inp_mask=None):
    """Add Qwen3VLTextDecoderLayer subgraph into an existing builder.

    Args:
        (see text_attention.add_attention_graph for attention params)
        inp_gate/up/down_w: MLP projection weights.
        inp_iln/pln_w:      input and post-attention layernorm weights.

    Returns the output tensor reference.
    """
    pfx = name_prefix

    iln = builder.add_node([inp_hidden, inp_iln_w], make_rms_norm(eps))
    normed = iln.get_output(0)

    attn_out = add_attention_graph(
        builder, normed,
        inp_q_w, inp_k_w, inp_v_w, inp_o_w,
        inp_qn_w, inp_kn_w,
        inp_cos, inp_sin, inp_seqlen,
        B, S, num_heads, num_kv_heads, head_dim,
        name_prefix=pfx + "attn_", eps=eps, use_mask=use_mask,
        inp_mask=inp_mask,
    )

    add1 = builder.add_node([inp_hidden, attn_out], make_elewise_add())
    h1 = add1.get_output(0)

    pln = builder.add_node([h1, inp_pln_w], make_rms_norm(eps))
    normed_h1 = pln.get_output(0)

    mlp_out = add_mlp_graph(
        builder, normed_h1,
        inp_gate_w, inp_up_w, inp_down_w,
        name_prefix=pfx + "mlp_",
    )

    add2 = builder.add_node([h1, mlp_out], make_elewise_add())
    return add2.get_output(0)


def build_decoder_layer(num_heads, num_kv_heads, head_dim,
                        intermediate_size, B=1, S=16, eps=1e-6,
                        use_mask=False):
    """Build standalone Qwen3VLTextDecoderLayer ATB graph for testing.

    Returns (builder, graph_op, input_names).
    """
    builder = get_atb_builder("Qwen3VLTextDecoderLayer")

    inp_hs = builder.add_input("hidden_states")
    inp_q_w = builder.add_input("q_weight")
    inp_k_w = builder.add_input("k_weight")
    inp_v_w = builder.add_input("v_weight")
    inp_o_w = builder.add_input("o_weight")
    inp_qn_w = builder.add_input("q_norm_weight")
    inp_kn_w = builder.add_input("k_norm_weight")
    inp_gate_w = builder.add_input("gate_weight")
    inp_up_w = builder.add_input("up_weight")
    inp_down_w = builder.add_input("down_weight")
    inp_iln_w = builder.add_input("input_ln_weight")
    inp_pln_w = builder.add_input("post_ln_weight")
    inp_cos = builder.add_input("cos")
    inp_sin = builder.add_input("sin")
    inp_mask = builder.add_input("mask") if use_mask else None
    inp_seqlen = builder.add_input("seqlen")

    out_name = add_decoder_layer_graph(
        builder, inp_hs,
        inp_q_w, inp_k_w, inp_v_w, inp_o_w,
        inp_qn_w, inp_kn_w,
        inp_gate_w, inp_up_w, inp_down_w,
        inp_iln_w, inp_pln_w,
        inp_cos, inp_sin, inp_seqlen,
        B, S, num_heads, num_kv_heads, head_dim,
        eps=eps, use_mask=use_mask, inp_mask=inp_mask,
    )
    builder.mark_output(out_name)

    graph_op = builder.build()
    input_names = [
        "hidden_states", "q_weight", "k_weight", "v_weight", "o_weight",
        "q_norm_weight", "k_norm_weight",
        "gate_weight", "up_weight", "down_weight",
        "input_ln_weight", "post_ln_weight",
        "cos", "sin", "seqlen",
    ]
    if use_mask:
        input_names.insert(-1, "mask")
    return builder, graph_op, input_names
