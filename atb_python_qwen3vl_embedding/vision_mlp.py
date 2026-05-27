"""Qwen3VLVisionMLP ATB graph builder.

Pipeline:
    hidden_states → linear_fc1 → GELU → linear_fc2 → output
"""
from .utils import make_linear, get_atb_builder
import torch_atb


def add_vision_mlp_graph(builder, inp_hidden,
                         inp_fc1_w, inp_fc1_b,
                         inp_fc2_w, inp_fc2_b,
                         name_prefix="vmlp_"):
    """Add Qwen3VLVisionMLP subgraph into an existing builder.

    Args:
        builder:    ATB builder instance.
        inp_hidden: hidden_states input.
        inp_fc1/fc2_w/b: linear_fc1 and linear_fc2 weights and biases.

    Returns the output tensor reference.
    """
    pfx = name_prefix

    fc1 = builder.add_node([inp_hidden, inp_fc1_w, inp_fc1_b],
                           make_linear(has_bias=True))
    act = builder.add_node([fc1.get_output(0)],
        torch_atb.ActivationParam(
            activation_type=torch_atb.ActivationType.ACTIVATION_GELU))
    fc2 = builder.add_node([act.get_output(0), inp_fc2_w, inp_fc2_b],
                           make_linear(has_bias=True))
    return fc2.get_output(0)


def build_vision_mlp(config, name="VisionMLP"):
    """Build standalone Qwen3VLVisionMLP ATB graph for testing.

    Returns (builder, graph_op, input_names).
    """
    builder = get_atb_builder(name)

    x_in = builder.add_input("x")
    fc1_w = builder.add_input("fc1_w")
    fc1_b = builder.add_input("fc1_b")
    fc2_w = builder.add_input("fc2_w")
    fc2_b = builder.add_input("fc2_b")

    out = add_vision_mlp_graph(builder, x_in, fc1_w, fc1_b, fc2_w, fc2_b)
    builder.mark_output(out)

    graph_op = builder.build()
    input_names = ["x", "fc1_w", "fc1_b", "fc2_w", "fc2_b"]
    return builder, graph_op, input_names
