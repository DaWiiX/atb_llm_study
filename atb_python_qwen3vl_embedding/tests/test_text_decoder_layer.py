"""Test Qwen3VLTextDecoderLayer: transformers vs ATB graph.

Covers both ``use_mask=False`` and ``use_mask=True`` paths to validate that
mask plumbing through the composed (Attention + MLP + residual) subgraph is
correct. The real engine always runs with ``use_mask=True``.
"""
import os
import torch
import torch_npu  # noqa: needed for .npu()

from atb_python_qwen3vl_embedding import data_utils, utils
from atb_python_qwen3vl_embedding.text_decoder_layer import build_decoder_layer
from atb_python_qwen3vl_embedding.text_model import make_causal_mask
from atb_python_qwen3vl_embedding.transformers_runner import run_decoder_layer


def _run_once(B, S, use_mask, seed=42):
    label = f"DecoderLayer[mask={use_mask},S={S}]"
    print(f"\n=== Qwen3VLTextDecoderLayer ({label}) ===")
    config = data_utils.make_config(num_heads=4, num_kv_heads=4, head_dim=64,
                                    intermediate_size=512)
    d = config.head_dim

    gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen_data["config"] = config

    # Provide the SAME additive causal mask to both backends when use_mask=True.
    if use_mask:
        m2d = make_causal_mask(S)                       # (S, S) float
        gen_data["attention_mask"] = m2d.unsqueeze(0).unsqueeze(0).float()  # (1,1,S,S)
    ref_out, weights = run_decoder_layer(gen_data, seed=seed)

    _, graph_op, _ = build_decoder_layer(
        num_heads=4, num_kv_heads=4, head_dim=d,
        intermediate_size=512, B=B, S=S, use_mask=use_mask,
    )

    ntoken = B * S
    inputs = [
        gen_data["hidden_states"].half().npu(),
        weights["q_proj.weight"].half().npu(), weights["k_proj.weight"].half().npu(),
        weights["v_proj.weight"].half().npu(), weights["o_proj.weight"].half().npu(),
        weights["q_norm.weight"].half().npu(), weights["k_norm.weight"].half().npu(),
        weights["gate_proj.weight"].half().npu(), weights["up_proj.weight"].half().npu(),
        weights["down_proj.weight"].half().npu(),
        weights["input_ln.weight"].half().npu(), weights["post_ln.weight"].half().npu(),
        gen_data["cos"].reshape(ntoken, d).half().npu(),
        gen_data["sin"].reshape(ntoken, d).half().npu(),
    ]
    if use_mask:
        inputs.append(make_causal_mask(S).half().npu())
    inputs.append(torch.tensor([ntoken], dtype=torch.int32))  # CPU, required by SA

    atb_out = graph_op.forward(inputs)[0].cpu().float()
    return utils.compare_tensors(ref_out, atb_out, label=label)


def test_decoder_layer(B=1, S=16, seed=42):
    utils.set_atb_buffer_size(500 * 1024 * 1024)
    ok_nomask = _run_once(B, S, use_mask=False, seed=seed)
    ok_mask = _run_once(B, S, use_mask=True, seed=seed)
    return ok_nomask and ok_mask


if __name__ == "__main__":
    ok = test_decoder_layer()
    exit(0 if ok else 1)
