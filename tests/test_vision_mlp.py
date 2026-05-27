"""Test Qwen3VLVisionMLP: transformers vs ATB graph."""
import sys, os, torch, torch_npu, torch.nn.functional as F
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')
from atb_python_qwen3vl_embedding.vision_mlp import build_vision_mlp
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionMLP
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig


def test_vision_mlp(hidden_size=128, intermediate_size=256, seqlen=24, seed=42):
    print("\n=== Qwen3VLVisionMLP ===")
    config = Qwen3VLVisionConfig(hidden_size=hidden_size, intermediate_size=intermediate_size,
        hidden_act="gelu_pytorch_tanh")
    torch.manual_seed(seed); model = Qwen3VLVisionMLP(config); model.eval()
    torch.manual_seed(99); x = torch.randn(seqlen, hidden_size)
    with torch.no_grad(): ref = model(x)

    _, g, _ = build_vision_mlp(hidden_size, intermediate_size)
    set_atb_buffer_size(100 * 1024 * 1024)
    atb = g.forward([
        x.half().npu(),
        model.linear_fc1.weight.data.half().npu(), model.linear_fc1.bias.data.half().npu(),
        model.linear_fc2.weight.data.half().npu(), model.linear_fc2.bias.data.half().npu(),
    ])[0].cpu().float()
    compare_tensors(ref, atb, label="VisionMLP")
    return F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item() > 0.999


if __name__ == "__main__":
    ok = test_vision_mlp()
    exit(0 if ok else 1)
