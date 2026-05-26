"""Qwen3VLVisionPatchEmbed ATB graph builder.

Conv3d(kernel=stride=[tp, p, p]) is equivalent to:
    reshape input (N, C, tp, p, p) → (N, C*tp*p*p)
    Linear(out_dim, C*tp*p*p, bias=True)

Input pixels are already flattened: (N, C*tp*p*p) or (N*C*tp*p*p,).
"""
from .utils import make_linear, get_atb_builder


def add_patch_embed_graph(builder, inp_hidden, inp_weight, inp_bias,
                          in_channels, temporal_patch_size, patch_size,
                          embed_dim, name_prefix="pe_"):
    """Add patch embedding subgraph (reshape → Linear) into an existing builder.

    Args:
        inp_hidden: pixels input, (N*C*tp*p*p,) or (N, C*tp*p*p).
        inp_weight: (embed_dim, C*tp*p*p) Linear weight.
        inp_bias:   (embed_dim,) Linear bias.
        in_channels, temporal_patch_size, patch_size: conv kernel dims.
        embed_dim:  output embedding dimension (hidden_size).

    Returns the output tensor reference.
    """
    pfx = name_prefix
    kernel_size = in_channels * temporal_patch_size * patch_size * patch_size

    builder.reshape(inp_hidden, lambda s: [s[0] // kernel_size, kernel_size],
                    pfx + "flat")
    lin = builder.add_node([pfx + "flat", inp_weight, inp_bias],
                           make_linear(has_bias=True))
    return lin.get_output(0)


def build_patch_embed(in_channels, temporal_patch_size, patch_size, embed_dim,
                      name="PatchEmbed"):
    """Build standalone PatchEmbed ATB graph for testing.

    Returns (builder, graph_op, input_names).
    """
    builder = get_atb_builder(name)
    x_in = builder.add_input("x")
    w_in = builder.add_input("w")
    b_in = builder.add_input("b")

    out = add_patch_embed_graph(builder, x_in, w_in, b_in,
                                in_channels, temporal_patch_size, patch_size,
                                embed_dim)
    builder.mark_output(out)
    graph_op = builder.build()
    return builder, graph_op, ["x", "w", "b"]
