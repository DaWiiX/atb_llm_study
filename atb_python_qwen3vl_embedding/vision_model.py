"""Qwen3VLVisionModel ATB graph runner.

Split-graph strategy to avoid giant-graph OOM:
    1. first_layer:  patch_embed → +pos_embed → block 0      (built once)
    2. per_block:    single VisionBlock                        (built once, looped)
    3. merger:       LayerNorm → reshape → fc1 → GELU → fc2   (main output)
    4. deepstack:    reshape → LayerNorm → fc1 → GELU → fc2   (deepstack features)

Deepstack: vision block outputs at specific layers (e.g., [5, 11, 17]) are
fed through a deepstack merger and injected into early text decoder layers.
"""
import torch
import torch_atb
from .utils import (
    make_layer_norm, make_linear, make_elewise_add, get_atb_builder,
    make_seqlen_tensor, prepare_npu_weights, to_cpu_float, to_npu_half,
)
from .vision_block import add_vision_block_graph
from .vision_patch_embed import add_patch_embed_graph


# ═════════════════════════════════════════════════════════════════════
# Graph builders
# ═════════════════════════════════════════════════════════════════════

def build_vision_first_layer(config):
    """Build ATB graph: patch_embed → +pos_embed → VisionBlock(block 0).

    Input names (19 total):
        pixels, pe_w, pe_b, pos, c, s, seq,
        qkv_w, qkv_b, proj_w, proj_b,
        fc1_w, fc1_b, fc2_w, fc2_b,
        n1_w, n1_b, n2_w, n2_b
    """
    hs = config.hidden_size
    nh = config.num_heads
    c_in = config.in_channels
    tp = config.temporal_patch_size
    p = config.patch_size

    builder = get_atb_builder("VisFirstLayer")
    pixels = builder.add_input("pixels")
    pe_w = builder.add_input("pe_w")
    pe_b = builder.add_input("pe_b")
    pos = builder.add_input("pos")
    c = builder.add_input("c")
    s = builder.add_input("s")
    seq = builder.add_input("seq")

    bw = [builder.add_input(n) for n in [
        "qkv_w", "qkv_b", "proj_w", "proj_b",
        "fc1_w", "fc1_b", "fc2_w", "fc2_b",
        "n1_w", "n1_b", "n2_w", "n2_b"]]

    patched = add_patch_embed_graph(
        builder, pixels, pe_w, pe_b, c_in, tp, p, hs, "pe_")
    h0 = builder.add_node([patched, pos], make_elewise_add())

    out = add_vision_block_graph(builder, h0.get_output(0),
        bw[0], bw[1], bw[2], bw[3],
        bw[4], bw[5], bw[6], bw[7],
        bw[8], bw[9], bw[10], bw[11],
        c, s, seq, nh, hs // nh, name_prefix="L0_")
    builder.mark_output(out)

    return builder.build()


def build_vision_merger(config, is_deepstack=False):
    """Build PatchMerger graph (main or deepstack).

    Main merger (is_deepstack=False):
        LayerNorm(hidden) → reshape(group_4→1) → fc1 → GELU → fc2

    Deepstack merger (is_deepstack=True):
        reshape(group_4→1) → LayerNorm(merger_hidden) → fc1 → GELU → fc2

    The fc2 projects to text hidden_size (out_hidden_size from config).
    """
    hs = config.hidden_size
    merge = config.spatial_merge_size
    mer_hs = hs * merge * merge
    suffix = "DSM" if is_deepstack else "Merger"

    builder = get_atb_builder(f"Vis{suffix}")
    x_in = builder.add_input("x")
    n_w = builder.add_input("n_w")
    n_b = builder.add_input("n_b")
    f1_w = builder.add_input("f1_w")
    f1_b = builder.add_input("f1_b")
    f2_w = builder.add_input("f2_w")
    f2_b = builder.add_input("f2_b")

    if is_deepstack:
        builder.reshape(x_in,
                        lambda s: [s[0] * s[1] // mer_hs, mer_hs], "ds_flat")
        ln = builder.add_node(["ds_flat", n_w, n_b], make_layer_norm())
    else:
        ln = builder.add_node([x_in, n_w, n_b], make_layer_norm())
        builder.reshape(ln.get_output(0),
                        lambda s: [s[0] * s[1] // mer_hs, mer_hs], "mf")

    post_norm = ln.get_output(0) if is_deepstack else "mf"
    fc1 = builder.add_node([post_norm, f1_w, f1_b], make_linear(has_bias=True))
    act = builder.add_node([fc1.get_output(0)],
        torch_atb.ActivationParam(
            activation_type=torch_atb.ActivationType.ACTIVATION_GELU))
    fc2 = builder.add_node([act.get_output(0), f2_w, f2_b],
                           make_linear(has_bias=True))
    builder.mark_output(fc2.get_output(0))
    return builder.build()


def build_deepstack_merger(config):
    """Build deepstack merger: reshape → LayerNorm → fc1 → GELU → fc2."""
    return build_vision_merger(config, is_deepstack=True)


# ═════════════════════════════════════════════════════════════════════
# Weight collectors
# ═════════════════════════════════════════════════════════════════════

def collect_patch_embed_weight(vm):
    """Extract patch_embed weights as Linear (flatten Conv3d → Linear).

    Returns (weight, bias) for ATB Linear node.
    """
    cfg = vm.config
    ksize = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    w = vm.patch_embed.proj.weight.data.reshape(cfg.hidden_size, ksize).contiguous()
    b = vm.patch_embed.proj.bias.data
    return w, b


def collect_block_weights(blk):
    """Extract 12 weight tensors from a Qwen3VLVisionBlock.

    Order: [qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b,
            n1_w, n1_b, n2_w, n2_b]
    """
    return [
        blk.attn.qkv.weight.data, blk.attn.qkv.bias.data,
        blk.attn.proj.weight.data, blk.attn.proj.bias.data,
        blk.mlp.linear_fc1.weight.data, blk.mlp.linear_fc1.bias.data,
        blk.mlp.linear_fc2.weight.data, blk.mlp.linear_fc2.bias.data,
        blk.norm1.weight.data, blk.norm1.bias.data,
        blk.norm2.weight.data, blk.norm2.bias.data,
    ]


def collect_merger_weights(merger):
    """Extract 6 weight tensors from a PatchMerger.

    Order: [norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b]
    """
    return [
        merger.norm.weight.data, merger.norm.bias.data,
        merger.linear_fc1.weight.data, merger.linear_fc1.bias.data,
        merger.linear_fc2.weight.data, merger.linear_fc2.bias.data,
    ]


# ═════════════════════════════════════════════════════════════════════
# Runners
# ═════════════════════════════════════════════════════════════════════

def _first_layer_inputs(pixel_values, pe_w, pe_b, pos_embeds, cos, sin,
                        block0_weights):
    """Build first-layer graph inputs in ATB order."""
    pv = pixel_values.reshape(-1) if pixel_values.ndim == 2 else pixel_values
    npatches = pos_embeds.shape[0]
    inputs = [
        to_npu_half(pv),
        to_npu_half(pe_w), to_npu_half(pe_b),
        to_npu_half(pos_embeds),
        to_npu_half(cos), to_npu_half(sin),
        make_seqlen_tensor(npatches),
    ]
    inputs.extend(prepare_npu_weights(block0_weights))
    return inputs


def _block_inputs(hidden, block_weights, cos, sin):
    """Build VisionBlock graph inputs in ATB order."""
    npatches = hidden.shape[0]
    inputs = [to_npu_half(hidden)]
    inputs.extend(prepare_npu_weights(block_weights))
    inputs.extend([to_npu_half(cos), to_npu_half(sin),
                   make_seqlen_tensor(npatches)])
    return inputs


def _merger_inputs(hidden, merger_weights):
    """Build PatchMerger graph inputs in ATB order."""
    inputs = [to_npu_half(hidden)]
    inputs.extend(prepare_npu_weights(merger_weights))
    return inputs


def run_first_layer(graph, pixel_values, pe_w, pe_b, pos_embeds, cos, sin,
                    block0_weights):
    """Execute patch_embed + pos_embed + block 0 and return CPU float."""
    return to_cpu_float(graph.forward(_first_layer_inputs(
        pixel_values, pe_w, pe_b, pos_embeds, cos, sin, block0_weights))[0])


def run_first_layer_npu(graph, pixel_values, pe_w, pe_b, pos_embeds, cos, sin,
                        block0_weights):
    """Execute patch_embed + pos_embed + block 0 and keep output on NPU."""
    return graph.forward(_first_layer_inputs(
        pixel_values, pe_w, pe_b, pos_embeds, cos, sin, block0_weights))[0]


def run_block(graph, hidden, block_weights, cos, sin):
    """Execute one VisionBlock and return CPU float."""
    return to_cpu_float(graph.forward(_block_inputs(
        hidden, block_weights, cos, sin))[0])


def run_block_npu(graph, hidden_npu, block_weights, cos, sin):
    """Execute one VisionBlock and keep output on NPU."""
    return graph.forward(_block_inputs(hidden_npu, block_weights, cos, sin))[0]


def run_merger(graph, hidden, merger_weights):
    """Execute PatchMerger and return CPU float."""
    return to_cpu_float(graph.forward(_merger_inputs(hidden, merger_weights))[0])


def run_merger_npu(graph, hidden_npu, merger_weights):
    """Execute PatchMerger and keep output on NPU."""
    return graph.forward(_merger_inputs(hidden_npu, merger_weights))[0]


def run_vision_model(vision_model, pixel_values, pos_embeds, cos, sin,
                     graph_first, graph_block, graph_merger,
                     graph_deepstack=None):
    """Run full Qwen3VLVisionModel through ATB graphs.

    The public API is CPU-facing and returns CPU float tensors, while the block
    loop stays NPU-resident. Collected Transformers weights are CPU tensors and
    are converted once before graph execution.
    """
    vm = vision_model
    cfg = vm.config
    ds_indexes = cfg.deepstack_visual_indexes

    pe_w, pe_b = collect_patch_embed_weight(vm)
    pe_w, pe_b = to_npu_half(pe_w), to_npu_half(pe_b)
    block_weights = [
        prepare_npu_weights(collect_block_weights(vm.blocks[li]))
        for li in range(cfg.depth)
    ]
    merger_weights = prepare_npu_weights(collect_merger_weights(vm.merger))
    ds_weights = []
    if graph_deepstack is not None:
        ds_weights = [
            prepare_npu_weights(collect_merger_weights(merger))
            for merger in vm.deepstack_merger_list
        ]

    pos_npu = to_npu_half(pos_embeds)
    cos_npu = to_npu_half(cos)
    sin_npu = to_npu_half(sin)

    h = run_first_layer_npu(graph_first, pixel_values, pe_w, pe_b,
                            pos_npu, cos_npu, sin_npu, block_weights[0])

    deepstack_features = []
    for li in range(1, cfg.depth):
        h = run_block_npu(graph_block, h, block_weights[li], cos_npu, sin_npu)
        if li in ds_indexes and graph_deepstack is not None:
            ds_idx = ds_indexes.index(li)
            ds_out = run_merger_npu(graph_deepstack, h, ds_weights[ds_idx])
            deepstack_features.append(to_cpu_float(ds_out))

    merged = run_merger_npu(graph_merger, h, merger_weights)
    return to_cpu_float(merged), deepstack_features
