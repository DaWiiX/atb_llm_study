"""Test Qwen3VLVisionPatchEmbed: ATB vs transformers."""
import os, torch, torch_npu, torch_atb, torch.nn.functional as F, warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'; warnings.filterwarnings('ignore')
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from atb_python_qwen3vl_embedding.vision_patch_embed import build_patch_embed

def test_patch_embed():
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
    set_atb_buffer_size(100 * 1024 * 1024)
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

if __name__ == '__main__':
    ok = test_patch_embed()
    exit(0 if ok else 1)
