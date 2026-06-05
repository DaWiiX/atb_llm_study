"""Test Qwen3VLTextModel: transformers vs ATB (split-graph loop)."""
import os, torch, torch.nn.functional as F

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from atb_python_qwen3vl_embedding.text_model import (
    build_text_layer_graph, build_text_norm_graph, run_text_model,
)


def test_text_model(B=1, S=16, num_layers=2, seed=42):
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
    set_atb_buffer_size(500 * 1024 * 1024)
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
    # Float16 accumulation through multi-layer: allow 0.99
    threshold = 0.99 if num_layers >= 2 else 0.999
    print(f"[TextModel-{num_layers}L] shape: {ref_out.shape}  cosine: {cs:.6f}")
    print(f"[TextModel-{num_layers}L] PASS (>{threshold})" if cs > threshold else f"FAIL (<={threshold})")
    return cs > threshold


if __name__ == "__main__":
    ok = test_text_model(num_layers=2)
    exit(0 if ok else 1)
