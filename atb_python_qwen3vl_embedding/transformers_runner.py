"""
Transformers reference runner for ATB comparison testing.

Runs the real transformers implementations (Qwen3VLTextAttention, MLP,
DecoderLayer, TextModel) with random weights to produce reference outputs
and weight dictionaries for comparison with ATB graph implementations.
"""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
)


def run_attention(data_dict, seed=42):
    """Run Qwen3VLTextAttention, return (output, weights_dict)."""
    config = data_dict["config"]
    torch.manual_seed(seed)
    attn = Qwen3VLTextAttention(config, layer_idx=0)
    attn.eval()

    with torch.no_grad():
        output, _ = attn(
            hidden_states=data_dict["hidden_states"],
            position_embeddings=(data_dict["cos"], data_dict["sin"]),
            attention_mask=data_dict.get("attention_mask"),
        )

    weights = {
        "q_proj.weight": attn.q_proj.weight.data,
        "k_proj.weight": attn.k_proj.weight.data,
        "v_proj.weight": attn.v_proj.weight.data,
        "o_proj.weight": attn.o_proj.weight.data,
        "q_norm.weight": attn.q_norm.weight.data,
        "k_norm.weight": attn.k_norm.weight.data,
    }
    return output, weights


def run_mlp(data_dict, seed=42):
    """Run Qwen3VLTextMLP, return (output, weights_dict)."""
    config = data_dict["config"]
    torch.manual_seed(seed)
    mlp = Qwen3VLTextMLP(config)
    mlp.eval()

    with torch.no_grad():
        output = mlp(data_dict["hidden_states"])

    weights = {
        "gate_proj.weight": mlp.gate_proj.weight.data,
        "up_proj.weight": mlp.up_proj.weight.data,
        "down_proj.weight": mlp.down_proj.weight.data,
    }
    return output, weights


def run_decoder_layer(data_dict, seed=42):
    """Run Qwen3VLTextDecoderLayer, return (output, weights_dict)."""
    config = data_dict["config"]
    torch.manual_seed(seed)
    layer = Qwen3VLTextDecoderLayer(config, layer_idx=0)
    layer.eval()

    with torch.no_grad():
        output = layer(
            hidden_states=data_dict["hidden_states"],
            position_embeddings=(data_dict["cos"], data_dict["sin"]),
            attention_mask=data_dict.get("attention_mask"),
        )

    weights = {}
    for w in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        weights[f"{w}.weight"] = getattr(layer.self_attn, w).weight.data
    weights["q_norm.weight"] = layer.self_attn.q_norm.weight.data
    weights["k_norm.weight"] = layer.self_attn.k_norm.weight.data
    for w in ["gate_proj", "up_proj", "down_proj"]:
        weights[f"{w}.weight"] = getattr(layer.mlp, w).weight.data
    weights["input_ln.weight"] = layer.input_layernorm.weight.data
    weights["post_ln.weight"] = layer.post_attention_layernorm.weight.data
    return output, weights


def run_text_model(data_dict, seed=42):
    """Run 2-layer Qwen3VLTextModel, return (output, weights_dict)."""
    config = data_dict["config"]
    torch.manual_seed(seed)
    model = Qwen3VLTextModel(config)
    model.eval()

    with torch.no_grad():
        result = model(
            inputs_embeds=data_dict["hidden_states"],
            attention_mask=None,
            position_ids=data_dict["position_ids"],
            use_cache=False,
        )

    weights = {}
    weights["norm.weight"] = model.norm.weight.data
    weights["embed_tokens.weight"] = model.embed_tokens.weight.data
    for i, layer in enumerate(model.layers):
        pfx = f"layer{i}."
        for w in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weights[pfx + f"{w}.weight"] = getattr(layer.self_attn, w).weight.data
        weights[pfx + "q_norm.weight"] = layer.self_attn.q_norm.weight.data
        weights[pfx + "k_norm.weight"] = layer.self_attn.k_norm.weight.data
        for w in ["gate_proj", "up_proj", "down_proj"]:
            weights[pfx + f"{w}.weight"] = getattr(layer.mlp, w).weight.data
        weights[pfx + "input_ln.weight"] = layer.input_layernorm.weight.data
        weights[pfx + "post_ln.weight"] = layer.post_attention_layernorm.weight.data

    return result.last_hidden_state, weights
