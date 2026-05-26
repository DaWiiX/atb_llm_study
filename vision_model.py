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

def run_first_layer(graph, pixel_values, pe_w, pe_b, pos_embeds, cos, sin,
                    block0_weights):
    """Execute patch_embed + pos_embed + block 0 on ATB NPU.

    Args:
        pixel_values: (N, C*tp*p*p) or (N*C*tp*p*p,) float.
        pe_w, pe_b:   patch_embed Linear weights.
        pos_embeds:   (N, hidden) position embeddings.
        cos, sin:     (N, nh*hd) rotary embeddings.
        block0_weights: 12 block 0 weight tensors.

    Returns (N, hidden) float on CPU.
    """
    pv = pixel_values.reshape(-1) if pixel_values.ndim == 2 else pixel_values
    npatches = pos_embeds.shape[0]
    inputs = [pv.half().npu(),
              pe_w.half().npu(), pe_b.half().npu(),
              pos_embeds.half().npu(),
              cos.half().npu(), sin.half().npu(),
              torch.tensor([npatches], dtype=torch.int32)]
    inputs.extend([w.half().npu() for w in block0_weights])
    return graph.forward(inputs)[0].cpu().float()


def run_block(graph, hidden, block_weights, cos, sin):
    """Execute one VisionBlock on ATB NPU.

    Args:
        hidden:       (N, hidden) float on CPU.
        block_weights: 12 weight tensors for this block.
        cos, sin:     (N, nh*hd) rotary embeddings.

    Returns (N, hidden) float on CPU.
    """
    npatches = hidden.shape[0]
    inputs = [hidden.half().npu()]
    inputs.extend([w.half().npu() for w in block_weights])
    inputs.extend([cos.half().npu(), sin.half().npu(),
                   torch.tensor([npatches], dtype=torch.int32)])
    return graph.forward(inputs)[0].cpu().float()


def run_merger(graph, hidden, merger_weights):
    """Execute PatchMerger on ATB NPU.

    Returns (N_out, out_hidden_size) float on CPU.
    """
    inputs = [hidden.half().npu()]
    inputs.extend([w.half().npu() for w in merger_weights])
    return graph.forward(inputs)[0].cpu().float()


def run_vision_model(vision_model, pixel_values, pos_embeds, cos, sin,
                     graph_first, graph_block, graph_merger,
                     graph_deepstack=None):
    """Run full Qwen3VLVisionModel through ATB graphs.

    Args:
        vision_model:   Qwen3VLVisionModel instance (for weights).
        pixel_values:   (N, C*tp*p*p) from preprocess_image.
        pos_embeds:     (N, hidden) position embeddings.
        cos, sin:       (N, nh*hd) rotary embeddings.
        graph_first/graph_block/graph_merger: pre-built ATB graphs.
        graph_deepstack: optional deepstack merger graph.

    Returns (merged_output, deepstack_feature_list).
    """
    vm = vision_model
    cfg = vm.config
    ds_indexes = cfg.deepstack_visual_indexes
    pe_w, pe_b = collect_patch_embed_weight(vm)

    h = run_first_layer(graph_first, pixel_values, pe_w, pe_b,
                        pos_embeds, cos, sin,
                        collect_block_weights(vm.blocks[0]))

    deepstack_features = []
    for li in range(1, cfg.depth):
        h = run_block(graph_block, h, collect_block_weights(vm.blocks[li]),
                      cos, sin)
        if li in ds_indexes and graph_deepstack is not None:
            ds_weights = collect_merger_weights(
                vm.deepstack_merger_list[ds_indexes.index(li)])
            ds_out = run_merger(graph_deepstack, h, ds_weights)
            deepstack_features.append(ds_out)

    merged = run_merger(graph_merger, h, collect_merger_weights(vm.merger))
    return merged, deepstack_features
