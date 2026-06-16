"""Test Qwen3VLVisionMLP: transformers vs ATB graph.

Provides both a fast small-dimension smoke test and a slow full-dimension
validation using real Qwen3VL-Embedding-2B vision model dimensions.
"""
import os, torch, torch_npu, torch.nn.functional as F
from atb_python_qwen3vl_embedding.vision_mlp import build_vision_mlp
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionMLP
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig

# 500 MB for NPU memory pool (sized for full-dimension vision MLP test)
set_atb_buffer_size(500 * 1024 * 1024)


def test_vision_mlp(hidden_size=128, intermediate_size=256, seqlen=24, seed=42):
    """Fast smoke test: small dimensions for quick ATB graph verification.

    Uses minimal model dimensions (hs=128, interm=256) for rapid iteration
    during development.
    """
    print("\n=== Qwen3VLVisionMLP ===")
    config = Qwen3VLVisionConfig(hidden_size=hidden_size, intermediate_size=intermediate_size,
        hidden_act="gelu_pytorch_tanh")
    torch.manual_seed(seed); model = Qwen3VLVisionMLP(config); model.eval()
    torch.manual_seed(99); x = torch.randn(seqlen, hidden_size)
    with torch.no_grad(): ref = model(x)

    _, g, _ = build_vision_mlp(hidden_size, intermediate_size)
    atb = g.forward([
        x.half().npu(),
        model.linear_fc1.weight.data.half().npu(), model.linear_fc1.bias.data.half().npu(),
        model.linear_fc2.weight.data.half().npu(), model.linear_fc2.bias.data.half().npu(),
    ])[0].cpu().float()
    compare_tensors(ref, atb, label="VisionMLP")
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item() > 0.999


def test_vision_mlp_full_dim(hidden_size=1024, intermediate_size=4096, seqlen=24, seed=42):
    """Slow test, full model dimensions — validates memory alignment, matmul
    tiling, and fp16 precision with real Qwen3VL-Embedding-2B vision sizes.

    Real dimensions: hidden_size=1024, intermediate_size=4096.
    Compared to the smoke test this is ~10-20x larger per matrix dimension.
    0.999: single fp16 operator threshold — see THRESHOLDS.md.
    """
    print("\n=== Qwen3VLVisionMLP (full dim) ===")
    # Real Qwen3VL-Embedding-2B vision config dimensions
    config = Qwen3VLVisionConfig(hidden_size=hidden_size, intermediate_size=intermediate_size,
        hidden_act="gelu_pytorch_tanh")
    torch.manual_seed(seed); model = Qwen3VLVisionMLP(config); model.eval()
    torch.manual_seed(99); x = torch.randn(seqlen, hidden_size)
    with torch.no_grad(): ref = model(x)

    _, g, _ = build_vision_mlp(hidden_size, intermediate_size)
    atb = g.forward([
        x.half().npu(),
        model.linear_fc1.weight.data.half().npu(), model.linear_fc1.bias.data.half().npu(),
        model.linear_fc2.weight.data.half().npu(), model.linear_fc2.bias.data.half().npu(),
    ])[0].cpu().float()
    compare_tensors(ref, atb, label="VisionMLP-full_dim")
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item() > 0.999


if __name__ == "__main__":
    # Default runs the fast smoke test only.
    # Import and call test_vision_mlp_full_dim() explicitly for CI / full validation.
    ok = test_vision_mlp()
    exit(0 if ok else 1)
