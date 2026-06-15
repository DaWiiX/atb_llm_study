"""Test each Qwen3VL component: transformers vs ATB graph.

Covers both ``use_mask=False`` (no causal mask) and ``use_mask=True``
(additive causal mask) paths. The masked path is what the real engine uses
in every layer, so leaving it untested previously hid a real bug.

Usage:
  python tests/test_text_attention.py
"""
import os
import torch
import torch_npu  # noqa: needed for .npu()

from atb_python_qwen3vl_embedding import data_utils, utils
from atb_python_qwen3vl_embedding.text_attention import build_attention
from atb_python_qwen3vl_embedding.text_model import make_causal_mask
from atb_python_qwen3vl_embedding.utils import is_310p, make_causal_mask_nz_npu
from atb_python_qwen3vl_embedding.transformers_runner import run_attention


def _run_once(B, S, use_mask, seed=42):
    label = f"Attention[mask={use_mask},S={S}]"
    print(f"\n=== Qwen3VLTextAttention ({label}) ===")
    config = data_utils.make_config(num_heads=4, num_kv_heads=4, head_dim=64)
    d = config.head_dim

    gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen_data["config"] = config

    # Provide the SAME additive causal mask to both backends when use_mask=True.
    if use_mask:
        m2d = make_causal_mask(S)                       # (S, S) float
        gen_data["attention_mask"] = m2d.unsqueeze(0).unsqueeze(0).float()  # (1,1,S,S) for TF eager
    ref_out, weights = run_attention(gen_data, seed=seed)

    _, graph_op, _ = build_attention(
        num_heads=4, num_kv_heads=4, head_dim=d, B=B, S=S, use_mask=use_mask,
    )

    ntoken = B * S
    inputs = [
        gen_data["hidden_states"].half().npu(),
        weights["q_proj.weight"].half().npu(), weights["k_proj.weight"].half().npu(),
        weights["v_proj.weight"].half().npu(), weights["o_proj.weight"].half().npu(),
        weights["q_norm.weight"].half().npu(), weights["k_norm.weight"].half().npu(),
        gen_data["cos"].reshape(ntoken, d).half().npu(),
        gen_data["sin"].reshape(ntoken, d).half().npu(),
    ]
    if use_mask:
        if is_310p():
            inputs.append(make_causal_mask_nz_npu(S))
        else:
            inputs.append(make_causal_mask(S).half().npu())
    inputs.append(torch.tensor([ntoken], dtype=torch.int32))  # CPU, required by SA

    atb_out = graph_op.forward(inputs)[0].cpu().float()
    return utils.compare_tensors(ref_out, atb_out, label=label)


def test_attention(B=1, S=16, seed=42):
    # set buffer size once, before any graph build
    utils.set_atb_buffer_size(300 * 1024 * 1024)
    ok_nomask = _run_once(B, S, use_mask=False, seed=seed)
    ok_mask = _run_once(B, S, use_mask=True, seed=seed)
    return ok_nomask and ok_mask


if __name__ == "__main__":
    ok = test_attention()
    exit(0 if ok else 1)
