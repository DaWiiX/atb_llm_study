"""Test Qwen3VLVisionModel: transformers vs ATB graph (split-graph strategy).

Uses 3 subgraphs (first_layer, per_block, merger) and loops blocks at runtime.

Provides both a fast small-dimension smoke test and a slow full-dimension
validation using real Qwen3VL-Embedding-2B vision model dimensions.
"""
import os, torch, torch.nn.functional as F

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from atb_python_qwen3vl_embedding.vision_model import (
    build_vision_first_layer, build_vision_merger, run_vision_model,
)
from atb_python_qwen3vl_embedding.vision_block import build_vision_block

# 1 GB for NPU memory pool (full-dimension vision model with 24 blocks and large weights)
set_atb_buffer_size(1024 * 1024 * 1024)


def test_vision_model(depth=1, seed=42):
    """Fast smoke test: small dimensions for quick ATB graph verification.

    Uses minimal model dimensions (hs=128, nh=4, interm=256, patch_size=4)
    for rapid iteration during development.
    """
    print(f"\n=== Qwen3VLVisionModel (depth={depth}) ===")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig

    cfg = Qwen3VLVisionConfig(
        hidden_size=128, num_heads=4, intermediate_size=256,
        depth=depth, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=2, in_channels=3, out_hidden_size=256,
        num_position_embeddings=64, deepstack_visual_indexes=[],
        hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = 'eager'

    torch.manual_seed(seed); vm = Qwen3VLVisionModel(cfg); vm.eval()
    kernel = 3 * 2 * 4 * 4
    npatches = 16
    torch.manual_seed(99); pv = torch.randn(npatches, kernel)
    gth = torch.tensor([[1, 4, 4]])

    with torch.no_grad():
        ref, _ = vm(pv, gth)
        pos = vm.fast_pos_embed_interpolate(gth)
        rope = vm.rot_pos_emb(gth)
        seq = pv.shape[0]
        rope = rope.reshape(seq, -1)
        emb = torch.cat((rope, rope), dim=-1)

    g_first = build_vision_first_layer(cfg)
    nh_v = cfg.num_heads; hd_v = cfg.hidden_size // nh_v
    _, g_block, _ = build_vision_block(nh_v, hd_v, "VisBlockLoop")
    g_merger = build_vision_merger(cfg)

    atb, _ = run_vision_model(vm, pv, pos, emb.cos(), emb.sin(),
                              g_first, g_block, g_merger)
    compare_tensors(ref, atb, label=f"VisionModel-D{depth}")
    cs = F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item()
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return cs > 0.999


def test_vision_model_full_dim(depth=24, seed=42):
    """Slow test, full model dimensions — validates split-graph multi-block
    looping with real Qwen3VL-Embedding-2B vision sizes.

    Real dimensions:
      hidden_size=1024, num_heads=16, intermediate_size=4096,
      patch_size=16, out_hidden_size=2048, depth=24.

    Threshold 0.95 accounts for fp16 cumulative error across 24 vision blocks
    (each block has attention + MLP, ~48 fp16 ops total). This is comparable
    to the 28-layer text model threshold documented in THRESHOLDS.md.
    If cosine < 0.95, diagnose per-block precision.
    """
    print(f"\n=== Qwen3VLVisionModel (full dim, depth={depth}) ===")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig

    # Real Qwen3VL-Embedding-2B vision config dimensions
    cfg = Qwen3VLVisionConfig(
        hidden_size=1024, num_heads=16, intermediate_size=4096,
        depth=depth, patch_size=16, temporal_patch_size=2,
        spatial_merge_size=2, in_channels=3, out_hidden_size=2048,
        num_position_embeddings=64, deepstack_visual_indexes=[],
        hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = 'eager'

    torch.manual_seed(seed); vm = Qwen3VLVisionModel(cfg); vm.eval()
    kernel = 3 * 2 * 16 * 16
    npatches = 16
    torch.manual_seed(99); pv = torch.randn(npatches, kernel)
    gth = torch.tensor([[1, 4, 4]])

    with torch.no_grad():
        ref, _ = vm(pv, gth)
        pos = vm.fast_pos_embed_interpolate(gth)
        rope = vm.rot_pos_emb(gth)
        seq = pv.shape[0]
        rope = rope.reshape(seq, -1)
        emb = torch.cat((rope, rope), dim=-1)

    g_first = build_vision_first_layer(cfg)
    nh_v = cfg.num_heads; hd_v = cfg.hidden_size // nh_v
    _, g_block, _ = build_vision_block(nh_v, hd_v, "VisBlockLoopFull")
    g_merger = build_vision_merger(cfg)

    atb, _ = run_vision_model(vm, pv, pos, emb.cos(), emb.sin(),
                              g_first, g_block, g_merger)
    compare_tensors(ref, atb, label=f"VisionModel-full_dim-D{depth}")
    cs = F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item()
    # 0.95: full 24-block fp16 accumulation threshold — see THRESHOLDS.md.
    # If cosine < 0.95, diagnose per-block precision to rule out code bugs.
    threshold = 0.95
    print(f"[VisionModel-{depth}L] cosine: {cs:.6f}  "
          f"threshold: {threshold}")
    print(f"[VisionModel-{depth}L] PASS (>{threshold})" if cs > threshold
          else f"FAIL (<={threshold})")
    return cs > threshold


if __name__ == "__main__":
    # Default runs the fast smoke test only.
    # Import and call test_vision_model_full_dim() explicitly for CI / full validation.
    ok = test_vision_model(depth=2)
    exit(0 if ok else 1)
