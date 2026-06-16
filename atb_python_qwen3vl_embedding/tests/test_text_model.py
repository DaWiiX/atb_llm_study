"""Test Qwen3VLTextModel: transformers vs ATB (split-graph loop).

Precision thresholds by depth (fp16 accumulation):
  - 1 layer:     0.999  (single op precision, minimal accumulation)
  - 2+ layers:   0.99   (moderate accumulation, used for smoke tests)
  - 6-27 layers: 0.98   (significant accumulation)
  - 28 layers:   0.95   (full depth; if < 0.95 diagnose per-layer precision)

These thresholds account for float16 cumulative error amplification across
multiple decoder layers. Each layer introduces ~1e-3 to ~2e-3 cosine drift
due to matmul rounding in fp16. Over 28 layers, the drift can compound to
~0.03-0.05 reduction in cosine similarity. If the 28-layer cosine falls
below 0.95, per-layer precision should be diagnosed to determine whether
the drop is genuine fp16 accumulation or a code bug.
"""
import os, torch, torch.nn.functional as F

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from atb_python_qwen3vl_embedding.text_model import (
    build_text_layer_graph, build_text_norm_graph, run_text_model,
)

# 500 MB for NPU memory pool (medium-sized graph with GQA attention + MLP + weights)
set_atb_buffer_size(500 * 1024 * 1024)


def test_text_model(B=1, S=16, num_layers=2, seed=42):
    """Fast smoke test: 2 layers with small dimensions.

    Uses minimal model dimensions (hidden=256, nh=4, hd=64) for quick
    ATB graph verification. Threshold 0.99 is appropriate for 2 layers
    of fp16 accumulation.

    For full-depth precision validation, see test_text_model_28_layers().
    """
    # S=16 chosen so the (S, S) causal mask's last dim (16*2=32 bytes) meets
    # the ATB 32-byte last-dim alignment requirement. The real engine always
    # runs with S well above this threshold.
    print(f"\n=== Qwen3VLTextModel ({num_layers} layers, split-graph) ===")
    from atb_python_qwen3vl_embedding import data_utils
    from atb_python_qwen3vl_embedding.transformers_runner import run_text_model as run_tf

    config = data_utils.make_config(
        num_heads=4, num_kv_heads=4, head_dim=64,
        intermediate_size=512, num_hidden_layers=num_layers)
    d = config.head_dim

    gen = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen['config'] = config
    ref_out, weights = run_tf(gen, seed=seed)

    # Build ATB graphs
    g_layer = build_text_layer_graph(4, 4, d, 512, B=B, S=S, use_mask=True)
    g_norm = build_text_norm_graph(config.hidden_size)

    # Run: we need the actual Qwen3VLTextModel for weight extraction
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    torch.manual_seed(seed)
    tm = Qwen3VLTextModel(config); tm.eval()
    # Copy weights from the TF runner
    for i in range(num_layers):
        pfx = f"layer{i}."
        layer = tm.layers[i]
        layer.self_attn.q_proj.weight.data.copy_(weights[pfx + 'q_proj.weight'])
        layer.self_attn.k_proj.weight.data.copy_(weights[pfx + 'k_proj.weight'])
        layer.self_attn.v_proj.weight.data.copy_(weights[pfx + 'v_proj.weight'])
        layer.self_attn.o_proj.weight.data.copy_(weights[pfx + 'o_proj.weight'])
        layer.self_attn.q_norm.weight.data.copy_(weights[pfx + 'q_norm.weight'])
        layer.self_attn.k_norm.weight.data.copy_(weights[pfx + 'k_norm.weight'])
        layer.mlp.gate_proj.weight.data.copy_(weights[pfx + 'gate_proj.weight'])
        layer.mlp.up_proj.weight.data.copy_(weights[pfx + 'up_proj.weight'])
        layer.mlp.down_proj.weight.data.copy_(weights[pfx + 'down_proj.weight'])
        layer.input_layernorm.weight.data.copy_(weights[pfx + 'input_ln.weight'])
        layer.post_attention_layernorm.weight.data.copy_(weights[pfx + 'post_ln.weight'])
    tm.norm.weight.data.copy_(weights['norm.weight'])

    ntoken = B * S
    atb_out = run_text_model(
        tm, gen['hidden_states'],
        gen['cos'].reshape(ntoken, d),
        gen['sin'].reshape(ntoken, d),
        ntoken, g_layer, g_norm,
    )

    cs = F.cosine_similarity(ref_out.flatten(), atb_out.flatten(), dim=0).item()
    # 0.99 (2+ layers): moderate fp16 accumulation; 0.999 (1 layer): single op — see THRESHOLDS.md
    threshold = 0.99 if num_layers >= 2 else 0.999
    print(f"[TextModel-{num_layers}L] shape: {ref_out.shape}  cosine: {cs:.6f}")
    print(f"[TextModel-{num_layers}L] PASS (>{threshold})" if cs > threshold else f"FAIL (<={threshold})")
    return cs > threshold


def test_text_model_28_layers(B=1, S=16, seed=42):
    """Full-depth precision validation: 28 layers with real model dimensions.

    Uses Qwen3VL-Embedding-2B dimensions:
      hidden_size=2048, num_heads=32, num_kv_heads=4 (GQA), head_dim=128,
      intermediate_size=6144, num_hidden_layers=28.

    Threshold 0.95 accounts for fp16 cumulative error across 28 layers.
    If cosine < 0.95, per-layer precision should be diagnosed to determine
    whether the drop is genuine fp16 accumulation or a code bug.

    Uses the same split-graph strategy as the 2-layer smoke test: one
    DecoderLayer graph is built and reused across all 28 layers.
    """
    num_layers = 28
    # S=16 chosen so the (S, S) causal mask's last dim (16*2=32 bytes) meets
    # the ATB 32-byte last-dim alignment requirement.
    print(f"\n=== Qwen3VLTextModel ({num_layers} layers, full-depth, split-graph) ===")
    from atb_python_qwen3vl_embedding import data_utils
    from atb_python_qwen3vl_embedding.transformers_runner import run_text_model as run_tf

    config = data_utils.make_config(
        hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=128,
        intermediate_size=6144, num_hidden_layers=num_layers)
    d = config.head_dim

    gen = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen['config'] = config
    ref_out, weights = run_tf(gen, seed=seed)

    # Build ATB graphs (split-graph: one layer graph reused for all 28 layers)
    g_layer = build_text_layer_graph(32, 4, d, 6144, B=B, S=S, use_mask=True)
    g_norm = build_text_norm_graph(config.hidden_size)

    # Create Qwen3VLTextModel for weight extraction
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    torch.manual_seed(seed)
    tm = Qwen3VLTextModel(config); tm.eval()
    # Copy weights from the TF runner
    for i in range(num_layers):
        pfx = f"layer{i}."
        layer = tm.layers[i]
        layer.self_attn.q_proj.weight.data.copy_(weights[pfx + 'q_proj.weight'])
        layer.self_attn.k_proj.weight.data.copy_(weights[pfx + 'k_proj.weight'])
        layer.self_attn.v_proj.weight.data.copy_(weights[pfx + 'v_proj.weight'])
        layer.self_attn.o_proj.weight.data.copy_(weights[pfx + 'o_proj.weight'])
        layer.self_attn.q_norm.weight.data.copy_(weights[pfx + 'q_norm.weight'])
        layer.self_attn.k_norm.weight.data.copy_(weights[pfx + 'k_norm.weight'])
        layer.mlp.gate_proj.weight.data.copy_(weights[pfx + 'gate_proj.weight'])
        layer.mlp.up_proj.weight.data.copy_(weights[pfx + 'up_proj.weight'])
        layer.mlp.down_proj.weight.data.copy_(weights[pfx + 'down_proj.weight'])
        layer.input_layernorm.weight.data.copy_(weights[pfx + 'input_ln.weight'])
        layer.post_attention_layernorm.weight.data.copy_(weights[pfx + 'post_ln.weight'])
    tm.norm.weight.data.copy_(weights['norm.weight'])

    ntoken = B * S
    atb_out = run_text_model(
        tm, gen['hidden_states'],
        gen['cos'].reshape(ntoken, d),
        gen['sin'].reshape(ntoken, d),
        ntoken, g_layer, g_norm,
    )

    cs = F.cosine_similarity(ref_out.flatten(), atb_out.flatten(), dim=0).item()
    mse = F.mse_loss(ref_out.float(), atb_out.float()).item()
    max_diff = (ref_out.float() - atb_out.float()).abs().max().item()
    # 0.95: full 28-layer fp16 accumulation threshold — see THRESHOLDS.md.
    # If cosine < 0.95, diagnose per-layer precision to rule out code bugs.
    threshold = 0.95
    print(f"[TextModel-{num_layers}L] shape: {ref_out.shape}  cosine: {cs:.6f}  "
          f"MSE: {mse:.2e}  max_diff: {max_diff:.2e}")
    print(f"[TextModel-{num_layers}L] PASS (>{threshold})" if cs > threshold else f"FAIL (<={threshold})")
    return cs > threshold


if __name__ == "__main__":
    ok1 = test_text_model(num_layers=2)
    ok2 = test_text_model_28_layers()
    exit(0 if (ok1 and ok2) else 1)
