"""Shared test utilities for Qwen3VLEngine tests.

Provides timing helpers, similarity metrics, and TF reference model loading.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Timing utilities
# ═══════════════════════════════════════════════════════════════════

def now() -> float:
    return time.perf_counter()


def sync() -> None:
    if torch.npu.is_available():
        torch.npu.synchronize()


def ms(seconds_list) -> np.ndarray:
    return np.asarray(seconds_list) * 1000.0


# ═══════════════════════════════════════════════════════════════════
# Similarity / pooling
# ═══════════════════════════════════════════════════════════════════

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def pool_and_normalize(last_hidden: torch.Tensor,
                       attention_mask=None) -> torch.Tensor:
    """Last-token pool + L2 normalize — matches engine.encode() output."""
    if attention_mask is not None:
        seq_lens = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[0, seq_lens[0], :]
    else:
        pooled = last_hidden[0, -1, :]
    return F.normalize(pooled.float(), p=2, dim=-1)


# ═══════════════════════════════════════════════════════════════════
# TF reference model loading
# ═══════════════════════════════════════════════════════════════════

def load_tf_ref(model_dir: str):
    """Load Qwen3VLModel on NPU with weights from safetensors."""
    import safetensors.torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"

    model = Qwen3VLModel(cfg).eval().half().npu()
    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors",
                                     device="cpu")
    sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, \
        f"TF weight load failed: missing={missing}, unexpected={unexpected}"
    return model
