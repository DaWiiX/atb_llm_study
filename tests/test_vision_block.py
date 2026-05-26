"""Test Qwen3VLVisionBlock: transformers vs ATB graph."""
import sys, os, torch, torch_npu, torch.nn.functional as F
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path: sys.path.insert(0, _pkg_dir)
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')
import warnings; warnings.filterwarnings('ignore')
from atb_python_model.vision_block import build_vision_block
from atb_python_model.utils import set_atb_buffer_size, compare_tensors
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionRotaryEmbedding


def test_vision_block(hs=128, nh=4, seqlen=24, seed=42):
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
    set_atb_buffer_size(500 * 1024 * 1024)
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
    return F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item() > 0.999


if __name__ == "__main__":
    ok = test_vision_block()
    exit(0 if ok else 1)
