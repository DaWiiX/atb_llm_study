"""Test Qwen3VLTextMLP: transformers vs ATB graph.

Provides both a fast small-dimension smoke test and a slow full-dimension
validation using real Qwen3VL-Embedding-2B model dimensions.
"""
import os
import torch
import torch_npu  # noqa: needed for .npu()

from atb_python_qwen3vl_embedding import data_utils, utils
from atb_python_qwen3vl_embedding.text_mlp import build_mlp
from atb_python_qwen3vl_embedding.transformers_runner import run_mlp

# 500 MB for NPU memory pool (sized for full-dimension test with large weights)
utils.set_atb_buffer_size(500 * 1024 * 1024)


def test_mlp(B=1, S=16, seed=42):
    """Fast smoke test: small dimensions for quick ATB graph verification.

    Uses minimal model dimensions (hidden_size=256, intermediate_size=512)
    for rapid iteration during development.
    """
    print("\n=== Qwen3VLTextMLP ===")
    config = data_utils.make_config(intermediate_size=512)

    gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen_data["config"] = config
    ref_out, weights = run_mlp(gen_data, seed=seed)

    _, graph_op, _ = build_mlp(config.hidden_size, config.intermediate_size)

    atb_out = graph_op.forward([
        gen_data["hidden_states"].half().npu(),
        weights["gate_proj.weight"].half().npu(),
        weights["up_proj.weight"].half().npu(),
        weights["down_proj.weight"].half().npu(),
    ])[0].cpu().float()
    return utils.compare_tensors(ref_out, atb_out, label="MLP")


def test_mlp_full_dim(B=1, S=16, seed=42):
    """Slow test, full model dimensions — validates memory alignment, matmul
    tiling, and fp16 precision with real Qwen3VL-Embedding-2B sizes.

    Real dimensions: hidden_size=2048, intermediate_size=6144.
    Compared to the smoke test this is ~12x larger per matrix dimension.
    0.999: single fp16 operator threshold — see THRESHOLDS.md.
    """
    print("\n=== Qwen3VLTextMLP (full dim) ===")
    # Real Qwen3VL-Embedding-2B text config dimensions
    config = data_utils.make_config(
        hidden_size=2048, num_heads=16, num_kv_heads=8, head_dim=128,
        intermediate_size=6144)

    gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen_data["config"] = config
    ref_out, weights = run_mlp(gen_data, seed=seed)

    _, graph_op, _ = build_mlp(config.hidden_size, config.intermediate_size)

    atb_out = graph_op.forward([
        gen_data["hidden_states"].half().npu(),
        weights["gate_proj.weight"].half().npu(),
        weights["up_proj.weight"].half().npu(),
        weights["down_proj.weight"].half().npu(),
    ])[0].cpu().float()
    # 0.999: single fp16 operator threshold — see THRESHOLDS.md
    return utils.compare_tensors(ref_out, atb_out, label="MLP-full_dim")


if __name__ == "__main__":
    # Default runs the fast smoke test only.
    # Import and call test_mlp_full_dim() explicitly for CI / full validation.
    ok = test_mlp()
    exit(0 if ok else 1)
