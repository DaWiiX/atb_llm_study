"""
Test each Qwen3VL component: transformers vs ATB graph.

Usage:
  python tests/test_attention.py
  python tests/test_mlp.py
  python tests/test_decoder_layer.py
  python tests/test_text_model.py
"""
import os
import torch
import torch_npu  # noqa: needed for .npu()

from atb_python_qwen3vl_embedding import data_utils, utils
from atb_python_qwen3vl_embedding.text_attention import build_attention
from atb_python_qwen3vl_embedding.transformers_runner import run_attention


def test_attention(B=1, S=16, seed=42):
    print("\n=== Qwen3VLTextAttention ===")
    config = data_utils.make_config(num_heads=4, num_kv_heads=4, head_dim=64)
    d = config.head_dim

    gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen_data["config"] = config
    ref_out, weights = run_attention(gen_data, seed=seed)

    _, graph_op, _ = build_attention(
        num_heads=4, num_kv_heads=4, head_dim=d, B=B, S=S, use_mask=False,
    )

    ntoken = B * S
    utils.set_atb_buffer_size(300 * 1024 * 1024)
    atb_out = graph_op.forward([
        gen_data["hidden_states"].half().npu(),
        weights["q_proj.weight"].half().npu(), weights["k_proj.weight"].half().npu(),
        weights["v_proj.weight"].half().npu(), weights["o_proj.weight"].half().npu(),
        weights["q_norm.weight"].half().npu(), weights["k_norm.weight"].half().npu(),
        gen_data["cos"].reshape(ntoken, d).half().npu(),
        gen_data["sin"].reshape(ntoken, d).half().npu(),
        torch.tensor([ntoken], dtype=torch.int32),  # CPU, required by SelfAttention
    ])[0].cpu().float()
    return utils.compare_tensors(ref_out, atb_out, label="Attention")


if __name__ == "__main__":
    ok = test_attention()
    exit(0 if ok else 1)
