"""Test Qwen3VLVisionPatchEmbed: ATB vs transformers.

Provides both a fast small-dimension smoke test and a slow full-dimension
validation using real Qwen3VL-Embedding-2B vision model dimensions.
"""
import os, torch, torch_npu, torch_atb, torch.nn.functional as F, warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'; warnings.filterwarnings('ignore')
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from atb_python_qwen3vl_embedding.vision_patch_embed import build_patch_embed

# 500 MB for NPU memory pool (sized for full-dimension patch embedding test)
set_atb_buffer_size(500 * 1024 * 1024)


def test_patch_embed():
    """Fast smoke test: small dimensions for quick ATB graph verification.

    Uses patch_size=4 with embed_dim=128 for rapid iteration during development.
    """
    print("\n=== Qwen3VLVisionPatchEmbed ===")
    from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionPatchEmbed, Qwen3VLVisionConfig
    embed_dim = 128
    config = Qwen3VLVisionConfig(hidden_size=embed_dim, patch_size=4,
        temporal_patch_size=2, in_channels=3)
    config._attn_implementation = 'eager'

    torch.manual_seed(42); model = Qwen3VLVisionPatchEmbed(config); model.eval()
    Nframes = 2; total = Nframes * 3 * 2 * 4 * 4
    torch.manual_seed(99)
    x = torch.randn(total)

    with torch.no_grad():
        ref = model(x)

    # ATB
    kernel_size = 3 * 2 * 4 * 4
    weight_reshaped = model.proj.weight.data.reshape(embed_dim, kernel_size).contiguous()
    _, g, _ = build_patch_embed(3, 2, 4, embed_dim)

    atb_out = g.forward([
        x.half().npu(),
        weight_reshaped.half().npu(),
        model.proj.bias.data.half().npu(),
    ])[0].cpu().float()

    compare_tensors(ref, atb_out, label="VisionPatchEmbed")
    cs = F.cosine_similarity(ref.flatten(), atb_out.flatten(), dim=0).item()
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return cs > 0.999


def test_patch_embed_full_dim():
    """Slow test, full model dimensions — validates memory alignment and fp16
    precision with real Qwen3VL-Embedding-2B vision patch embedding sizes.

    Real dimensions: embed_dim=1024 (hidden_size), patch_size=16.
    Compared to the smoke test (embed_dim=128, patch_size=4) this is ~4x
    larger patch size and ~8x larger embed dimension.
    0.999: single fp16 operator threshold — see THRESHOLDS.md.
    """
    print("\n=== Qwen3VLVisionPatchEmbed (full dim) ===")
    from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionPatchEmbed, Qwen3VLVisionConfig
    # Real Qwen3VL-Embedding-2B vision config dimensions
    embed_dim = 1024  # hidden_size
    config = Qwen3VLVisionConfig(hidden_size=embed_dim, patch_size=16,
        temporal_patch_size=2, in_channels=3)
    config._attn_implementation = 'eager'

    torch.manual_seed(42); model = Qwen3VLVisionPatchEmbed(config); model.eval()
    Nframes = 2; total = Nframes * 3 * 2 * 16 * 16
    torch.manual_seed(99)
    x = torch.randn(total)

    with torch.no_grad():
        ref = model(x)

    # ATB
    kernel_size = 3 * 2 * 16 * 16
    weight_reshaped = model.proj.weight.data.reshape(embed_dim, kernel_size).contiguous()
    _, g, _ = build_patch_embed(3, 2, 16, embed_dim)

    atb_out = g.forward([
        x.half().npu(),
        weight_reshaped.half().npu(),
        model.proj.bias.data.half().npu(),
    ])[0].cpu().float()

    compare_tensors(ref, atb_out, label="VisionPatchEmbed-full_dim")
    cs = F.cosine_similarity(ref.flatten(), atb_out.flatten(), dim=0).item()
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return cs > 0.999


if __name__ == '__main__':
    # Default runs the fast smoke test only.
    # Import and call test_patch_embed_full_dim() explicitly for CI / full validation.
    ok = test_patch_embed()
    exit(0 if ok else 1)
