"""Test Qwen3VLTextDecoderLayer: transformers vs ATB graph.

Covers both ``use_mask=False`` and ``use_mask=True`` paths to validate that
mask plumbing through the composed (Attention + MLP + residual) subgraph is
correct. The real engine always runs with ``use_mask=True``.

Provides both a fast small-dimension smoke test and a slow full-dimension
validation using real Qwen3VL-Embedding-2B text model dimensions.
"""
import os
import torch
import torch_npu  # noqa: needed for .npu()

from atb_python_qwen3vl_embedding import data_utils, utils
from atb_python_qwen3vl_embedding.text_decoder_layer import build_decoder_layer
from atb_python_qwen3vl_embedding.text_model import make_causal_mask
from atb_python_qwen3vl_embedding.utils import is_310p, make_causal_mask_nz_npu
from atb_python_qwen3vl_embedding.transformers_runner import run_decoder_layer

# 1 GB for NPU memory pool (sized for full-dimension decoder layer with GQA and large weights)
utils.set_atb_buffer_size(1024 * 1024 * 1024)


def _run_once(B, S, use_mask, num_heads=4, num_kv_heads=4, head_dim=64,
              intermediate_size=512, seed=42):
    label = f"DecoderLayer[mask={use_mask},S={S},nh={num_heads},hd={head_dim}]"
    print(f"\n=== Qwen3VLTextDecoderLayer ({label}) ===")
    config = data_utils.make_config(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        intermediate_size=intermediate_size)
    d = config.head_dim

    gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen_data["config"] = config

    # Provide the SAME additive causal mask to both backends when use_mask=True.
    if use_mask:
        m2d = make_causal_mask(S)                       # (S, S) float
        gen_data["attention_mask"] = m2d.unsqueeze(0).unsqueeze(0).float()  # (1,1,S,S)
    ref_out, weights = run_decoder_layer(gen_data, seed=seed)

    _, graph_op, _ = build_decoder_layer(
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=d,
        intermediate_size=intermediate_size, B=B, S=S, use_mask=use_mask,
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
        if is_310p():
            inputs.append(make_causal_mask_nz_npu(S))
        else:
            inputs.append(make_causal_mask(S).half().npu())
    inputs.append(torch.tensor([ntoken], dtype=torch.int32))  # CPU, required by SA

    atb_out = graph_op.forward(inputs)[0].cpu().float()
    return utils.compare_tensors(ref_out, atb_out, label=label)


def test_decoder_layer(B=1, S=16, seed=42):
    """Fast smoke test: small dimensions for quick ATB graph verification.

    Uses minimal model dimensions (nh=4, kv_nh=4, hd=64, interm=512)
    for rapid iteration during development.
    """
    ok_nomask = _run_once(B, S, use_mask=False, seed=seed)
    ok_mask = _run_once(B, S, use_mask=True, seed=seed)
    return ok_nomask and ok_mask


def test_decoder_layer_full_dim(B=1, S=16, seed=42):
    """Slow test, full model dimensions — validates memory alignment, GQA
    path (kv_nh=8 < nh=16), and fp16 precision with real Qwen3VL-Embedding-2B
    text dimensions.

    Real dimensions: hidden_size=2048, nh=16, kv_nh=8 (GQA), hd=128,
    intermediate_size=6144. Compared to the smoke test this is ~12x larger.

    Tests both use_mask=False and use_mask=True paths.
    0.999: single fp16 operator threshold — see THRESHOLDS.md.
    """
    # Real Qwen3VL-Embedding-2B text config dimensions
    nh, kv_nh, hd, interm = 16, 8, 128, 6144
    ok_nomask = _run_once(B, S, use_mask=False,
                          num_heads=nh, num_kv_heads=kv_nh, head_dim=hd,
                          intermediate_size=interm, seed=seed)
    ok_mask = _run_once(B, S, use_mask=True,
                        num_heads=nh, num_kv_heads=kv_nh, head_dim=hd,
                        intermediate_size=interm, seed=seed)
    return ok_nomask and ok_mask


if __name__ == "__main__":
    # Default runs the fast smoke test only.
    # Import and call test_decoder_layer_full_dim() explicitly for CI / full validation.
    ok = test_decoder_layer()
    exit(0 if ok else 1)
