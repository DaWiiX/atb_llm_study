"""Test Qwen3VLVisionModel: transformers vs ATB graph (split-graph strategy).

Uses 3 subgraphs (first_layer, per_block, merger) and loops blocks at runtime.
"""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from atb_python_qwen3vl_embedding.vision_model import (
    build_vision_first_layer, build_vision_merger, run_vision_model,
)
from atb_python_qwen3vl_embedding.vision_block import build_vision_block
def test_vision_model(depth=1, seed=42):
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

    set_atb_buffer_size(500 * 1024 * 1024)
    g_first = build_vision_first_layer(cfg)
    nh_v = cfg.num_heads; hd_v = cfg.hidden_size // nh_v
    _, g_block, _ = build_vision_block(nh_v, hd_v, "VisBlockLoop")
    g_merger = build_vision_merger(cfg)

    atb, _ = run_vision_model(vm, pv, pos, emb.cos(), emb.sin(),
                              g_first, g_block, g_merger)
    compare_tensors(ref, atb, label=f"VisionModel-D{depth}")
    cs = F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item()
    return cs > 0.999


if __name__ == "__main__":
    ok = test_vision_model(depth=2)
    exit(0 if ok else 1)
