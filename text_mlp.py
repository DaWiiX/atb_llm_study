"""Qwen3VLTextMLP ATB graph builder (SwiGLU).

Pipeline:
    hidden_states → gate_proj → SiLU ─┐
                   → up_proj ────────┼→ ElewiseMul → down_proj → output
"""
from .utils import make_linear, make_elewise_mul, make_silu, get_atb_builder


def add_mlp_graph(builder, inp_hidden,
                  inp_gate_w, inp_up_w, inp_down_w,
                  name_prefix="mlp_"):
    """Add SwiGLU MLP subgraph into an existing builder.

    Args:
        builder:     ATB builder instance.
        inp_hidden:  hidden_states input.
        inp_gate_w:  gate_proj weight.
        inp_up_w:    up_proj weight.
        inp_down_w:  down_proj weight.
        name_prefix: prefix for intermediate tensor names.

    Returns the output tensor reference.
    """
    pfx = name_prefix

    gate = builder.add_node([inp_hidden, inp_gate_w], make_linear())
    act = builder.add_node([gate.get_output(0)], make_silu())
    up = builder.add_node([inp_hidden, inp_up_w], make_linear())
    mul = builder.add_node([act.get_output(0), up.get_output(0)], make_elewise_mul())
    down = builder.add_node([mul.get_output(0), inp_down_w], make_linear())
    return down.get_output(0)


def build_mlp(hidden_size, intermediate_size, name="Qwen3VLTextMLP"):
    """Build standalone Qwen3VLTextMLP ATB graph for testing.

    Returns (builder, graph_op, input_names).
    """
    builder = get_atb_builder(name)

    inp_hs = builder.add_input("hidden_states")
    inp_gw = builder.add_input("gate_weight")
    inp_uw = builder.add_input("up_weight")
    inp_dw = builder.add_input("down_weight")

    out_name = add_mlp_graph(builder, inp_hs, inp_gw, inp_uw, inp_dw)
    builder.mark_output(out_name)

    graph_op = builder.build()
    input_names = ["hidden_states", "gate_weight", "up_weight", "down_weight"]
    return builder, graph_op, input_names
