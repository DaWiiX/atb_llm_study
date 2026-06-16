"""Test Qwen3VLVisionBlock: transformers vs ATB graph.

Provides both a fast small-dimension smoke test and a slow full-dimension
validation using real Qwen3VL-Embedding-2B vision model dimensions.
"""
import os, torch, torch_npu, torch.nn.functional as F
import warnings; warnings.filterwarnings('ignore')
from atb_python_qwen3vl_embedding.vision_block import build_vision_block
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionRotaryEmbedding

# 500 MB for NPU memory pool (medium-sized graph with attention + MLP + weights)
set_atb_buffer_size(500 * 1024 * 1024)


def test_vision_block(hs=128, nh=4, seqlen=24, seed=42):
    """Fast smoke test: small dimensions for quick ATB graph verification.

    Uses minimal model dimensions (hs=128, nh=4) for rapid iteration during
    development.
    """
    print("\n=== Qwen3VLVisionBlock ===")
    hd = hs // nh
    config = Qwen3VLVisionConfig(hidden_size=hs, num_heads=nh, intermediate_size=256, depth=1,
        patch_size=16, temporal_patch_size=2, out_hidden_size=256)
    config._attn_implementation = 'eager'

    rot = Qwen3VLVisionRotaryEmbedding(hd // 2)
    gh, gw = 4, 6; mh = max(gh, gw); ft = rot(mh)
    ri = torch.arange(gh).unsqueeze(1).expand(gh, gw).reshape(-1)
    ci = torch.arange(gw).unsqueeze(0).expand(gh, gw).reshape(-1)
    pi = torch.stack([ri, ci], dim=-1).long()
    rope = ft[pi].flatten(1); emb = torch.cat((rope, rope), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    torch.manual_seed(seed); model = Qwen3VLVisionBlock(config); model.eval()
    torch.manual_seed(99); x = torch.randn(seqlen, hs)
    cu = torch.tensor([0, seqlen], dtype=torch.int32)
    with torch.no_grad(): ref = model(x, cu_seqlens=cu, position_embeddings=(cos, sin))

    _, g, _ = build_vision_block(nh, hd)
    atb = g.forward([
        x.half().npu(),
        model.attn.qkv.weight.data.half().npu(), model.attn.qkv.bias.data.half().npu(),
        model.attn.proj.weight.data.half().npu(), model.attn.proj.bias.data.half().npu(),
        model.mlp.linear_fc1.weight.data.half().npu(), model.mlp.linear_fc1.bias.data.half().npu(),
        model.mlp.linear_fc2.weight.data.half().npu(), model.mlp.linear_fc2.bias.data.half().npu(),
        model.norm1.weight.data.half().npu(), model.norm1.bias.data.half().npu(),
        model.norm2.weight.data.half().npu(), model.norm2.bias.data.half().npu(),
        cos.half().npu(), sin.half().npu(),
        torch.tensor([seqlen], dtype=torch.int32),
    ])[0].cpu().float()
    compare_tensors(ref, atb, label="VisionBlock")
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item() > 0.999


def test_vision_block_full_dim(hs=1024, nh=16, seqlen=24, seed=42):
    """Slow test, full model dimensions — validates memory alignment, matmul
    tiling, and fp16 precision with real Qwen3VL-Embedding-2B vision sizes.

    Real dimensions: hidden_size=1024, num_heads=16, head_dim=64.
    Tests the combined attention + MLP + residual path in a single vision block.
    Compared to the smoke test this is ~10x larger per matrix dimension.
    0.999: single fp16 operator threshold — see THRESHOLDS.md.
    """
    print("\n=== Qwen3VLVisionBlock (full dim) ===")
    hd = hs // nh  # 64
    # Real Qwen3VL-Embedding-2B vision config dimensions
    config = Qwen3VLVisionConfig(hidden_size=hs, num_heads=nh, intermediate_size=4096,
        depth=1, patch_size=16, temporal_patch_size=2, out_hidden_size=2048)
    config._attn_implementation = 'eager'

    rot = Qwen3VLVisionRotaryEmbedding(hd // 2)
    gh, gw = 4, 6; mh = max(gh, gw); ft = rot(mh)
    ri = torch.arange(gh).unsqueeze(1).expand(gh, gw).reshape(-1)
    ci = torch.arange(gw).unsqueeze(0).expand(gh, gw).reshape(-1)
    pi = torch.stack([ri, ci], dim=-1).long()
    rope = ft[pi].flatten(1); emb = torch.cat((rope, rope), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    torch.manual_seed(seed); model = Qwen3VLVisionBlock(config); model.eval()
    torch.manual_seed(99); x = torch.randn(seqlen, hs)
    cu = torch.tensor([0, seqlen], dtype=torch.int32)
    with torch.no_grad(): ref = model(x, cu_seqlens=cu, position_embeddings=(cos, sin))

    _, g, _ = build_vision_block(nh, hd)
    atb = g.forward([
        x.half().npu(),
        model.attn.qkv.weight.data.half().npu(), model.attn.qkv.bias.data.half().npu(),
        model.attn.proj.weight.data.half().npu(), model.attn.proj.bias.data.half().npu(),
        model.mlp.linear_fc1.weight.data.half().npu(), model.mlp.linear_fc1.bias.data.half().npu(),
        model.mlp.linear_fc2.weight.data.half().npu(), model.mlp.linear_fc2.bias.data.half().npu(),
        model.norm1.weight.data.half().npu(), model.norm1.bias.data.half().npu(),
        model.norm2.weight.data.half().npu(), model.norm2.bias.data.half().npu(),
        cos.half().npu(), sin.half().npu(),
        torch.tensor([seqlen], dtype=torch.int32),
    ])[0].cpu().float()
    compare_tensors(ref, atb, label="VisionBlock-full_dim")
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item() > 0.999


if __name__ == "__main__":
    # Default runs the fast smoke test only.
    # Import and call test_vision_block_full_dim() explicitly for CI / full validation.
    ok = test_vision_block()
    exit(0 if ok else 1)
