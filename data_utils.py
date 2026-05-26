"""
Test data generation utilities.

Creates small Qwen3VLTextConfig instances and generates random hidden_states
with corresponding rotary position embeddings for ATB vs transformers comparison.
"""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLTextRotaryEmbedding


def make_config(hidden_size=256, num_heads=4, num_kv_heads=4, head_dim=64,
                intermediate_size=512, num_hidden_layers=2, **extra):
    """Create a minimal Qwen3VLTextConfig for unit testing.

    Defaults produce a small model suitable for quick ATB graph verification.
    """
    cfg = Qwen3VLTextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        hidden_act="silu",
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
        pad_token_id=0,
        **extra,
    )
    cfg._attn_implementation = "eager"
    return cfg


def generate_base_data(config, B=1, S=16, seed=42):
    """Generate random hidden_states and rotary embeddings for a config.

    Returns dict with keys: hidden_states, position_ids, cos, sin, attention_mask.
    """
    torch.manual_seed(seed)
    hidden_states = torch.randn(B, S, config.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(0, S, dtype=torch.long).unsqueeze(0)
    rotary_emb = Qwen3VLTextRotaryEmbedding(config)
    cos, sin = rotary_emb(hidden_states, position_ids)
    return {
        "hidden_states": hidden_states,
        "position_ids": position_ids,
        "cos": cos.float(),
        "sin": sin.float(),
        "attention_mask": None,
    }
