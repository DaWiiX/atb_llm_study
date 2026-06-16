"""Text model diagnostics — isolate where ATB text path diverges from TF.

After test_vision_diagnostics confirmed vision encoder is correct,
this script tests each component of the text pipeline:

  1. test_embed_weights
       Compare engine.embed_tokens weight vs TF embed_tokens weight
       → should be identical (same safetensors)

  2. test_position_ids
       Compare ATB get_rope_index() vs TF get_rope_index() with same inputs
       → isolates MRoPE position calculation

  3. test_rope_cos_sin
       Compare ATB text_rope(pid) vs TF rotary_emb(pid) with SAME position_ids
       → isolates RoPE implementation

  4. test_causal_mask
       Compare ATB make_causal_mask vs TF's eager attention mask

  5. test_text_layer
       Run ONE text decoder layer with identical inputs
       → isolates ATB graph vs TF layer

Usage::

    python tests/test_text_diagnostics.py
"""
# ── Buffer size MUST be set before any engine/graph import ──────────
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size

set_atb_buffer_size(5 * 1024 * 1024 * 1024)  # 5 GB

# ── Standard imports ───────────────────────────────────────────────
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image

from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def report(label, cs, threshold=0.999):
    """Report PASS/FAIL. Default 0.999: single fp16 operator threshold — see THRESHOLDS.md."""
    status = "PASS" if cs >= threshold else "FAIL"
    print(f"  [{status}] {label:<35} cosine={cs:.6f}")
    return cs >= threshold


def load_tf_ref(model_dir: str):
    """Load Qwen3VLModel on NPU (half precision) from safetensors."""
    import safetensors.torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    ref = Qwen3VLModel(cfg).eval().half().npu()
    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors", device="cpu")
    sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (
        f"weight mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    return ref


# ═══════════════════════════════════════════════════════════════════
# Test 1: embedding weights
# ═══════════════════════════════════════════════════════════════════

def test_embed_weights(engine, model_dir):
    """Compare ATB embed_tokens weight vs TF embed_tokens weight (both from safetensors)."""
    print("\n── Test 1: embed_tokens weights ──")

    import safetensors.torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    ref = Qwen3VLModel(cfg).eval()

    # Load safetensors into TF model (same as load_tf_ref but CPU float32)
    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors", device="cpu")
    sd_f32 = {k.removeprefix("model."): v.float() for k, v in sd.items()}
    missing, unexpected = ref.load_state_dict(sd_f32, strict=False)
    assert not missing and not unexpected, (
        f"weight mismatch: missing={len(missing)} unexpected={len(unexpected)}")

    # ATB weight (from engine.embed_w)
    atb_w = engine.embed_w  # CPU float32

    # TF weight (now loaded from safetensors)
    tf_w = ref.get_input_embeddings().weight.data.cpu().float()

    print(f"  ATB embed_w: {atb_w.shape} dtype={atb_w.dtype}")
    print(f"  TF  embed_w: {tf_w.shape} dtype={tf_w.dtype}")

    cs = cosine(atb_w, tf_w)
    exact_match = torch.equal(atb_w, tf_w)
    print(f"  Exact match: {exact_match}")
    print(f"  Max diff: {(atb_w - tf_w).abs().max().item():.8f}")

    # 0.9999: identity check — same safetensors weights, any deviation is a loading bug
    return report("embed_tokens weight", cs, threshold=0.9999)


# ═══════════════════════════════════════════════════════════════════
# Test 2: position_ids (MRoPE)
# ═══════════════════════════════════════════════════════════════════

def test_position_ids(proc, engine, model_dir):
    """Compare ATB get_rope_index() vs TF get_rope_index()."""
    print("\n── Test 2: position_ids (MRoPE) ──")

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    ref = Qwen3VLModel(cfg).eval()

    from atb_python_qwen3vl_embedding.engine_utils import get_rope_index

    all_ok = True
    for name, img in [("120x200-red", Image.new('RGB', (120, 200), color='red')),
                      ("64x64-blue",  Image.new('RGB', (64, 64), color='blue'))]:

        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': 'Describe.'}]}]
        tf_in = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt')
        input_ids = tf_in['input_ids']
        gth = tf_in['image_grid_thw']

        # ATB position_ids
        atb_pid, _ = get_rope_index(
            input_ids, gth, None, None,
            image_token_id=engine.img_tok,
            spatial_merge_size=engine.spatial_merge)

        # TF position_ids
        with torch.no_grad():
            tf_pid, _ = ref.get_rope_index(input_ids, gth, None, None)

        print(f"\n  [{name}]  atb_pid: {atb_pid.shape}  tf_pid: {tf_pid.shape}")

        # Compare
        exact = torch.equal(atb_pid, tf_pid)
        cs = cosine(atb_pid.float(), tf_pid.float())
        print(f"  Exact match: {exact}")
        if not exact:
            diff_mask = (atb_pid != tf_pid)
            print(f"  Mismatch count: {diff_mask.sum().item()}/{atb_pid.numel()}")
            # Show first few mismatches
            idxs = diff_mask.nonzero(as_tuple=False)[:5]
            for idx in idxs:
                b, s, d = idx.tolist()
                print(f"    [{b},{s},{d}]: atb={atb_pid[b,s,d].item()} tf={tf_pid[b,s,d].item()}")

        # 0.9999: identity check — same position_id computation, any deviation is a logic bug
        all_ok &= report(f"position_ids ({name})", cs, threshold=0.9999)

    return all_ok


# ═══════════════════════════════════════════════════════════════════
# Test 3: RoPE cos/sin
# ═══════════════════════════════════════════════════════════════════

def test_rope_cos_sin(proc, engine, model_dir):
    """Compare ATB text_rope(pid) vs TF rotary_emb(pid) with SAME position_ids."""
    print("\n── Test 3: RoPE cos/sin ──")

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    ref = Qwen3VLModel(cfg).eval()

    from atb_python_qwen3vl_embedding.engine_utils import get_rope_index

    all_ok = True
    for name, img in [("120x200-red", Image.new('RGB', (120, 200), color='red')),
                      ("64x64-blue",  Image.new('RGB', (64, 64), color='blue'))]:

        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': 'Describe.'}]}]
        tf_in = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt')
        input_ids = tf_in['input_ids']
        gth = tf_in['image_grid_thw']
        S = input_ids.shape[1]

        # Use TF position_ids (same for both)
        with torch.no_grad():
            tf_pid, _ = ref.get_rope_index(input_ids, gth, None, None)

        # ATB RoPE (with TF position_ids)
        atb_cos, atb_sin = engine.text_rope(tf_pid)
        atb_cos = atb_cos.reshape(-1, engine.hd_t)
        atb_sin = atb_sin.reshape(-1, engine.hd_t)

        # TF RoPE (with same position_ids)
        # rotary_emb expects (batch, seq_len, ...) input
        dummy_input = torch.zeros(1, S, engine.hidden_t)
        with torch.no_grad():
            tf_cos, tf_sin = ref.language_model.rotary_emb(dummy_input, tf_pid.float())

        # Reshape to match ATB: (B*S, head_dim)
        tf_cos = tf_cos.reshape(-1, engine.hd_t)
        tf_sin = tf_sin.reshape(-1, engine.hd_t)

        print(f"\n  [{name}]  atb_cos: {atb_cos.shape}  tf_cos: {tf_cos.shape}")

        cs_cos = cosine(atb_cos, tf_cos)
        cs_sin = cosine(atb_sin, tf_sin)
        # 0.9999: identity check — same RoPE computation with same position_ids, any deviation is a numerical bug
        all_ok &= report(f"cos ({name})", cs_cos, threshold=0.9999)
        all_ok &= report(f"sin ({name})", cs_sin, threshold=0.9999)

    return all_ok


# ═══════════════════════════════════════════════════════════════════
# Test 4: causal mask
# ═══════════════════════════════════════════════════════════════════

def test_causal_mask():
    """Compare ATB make_causal_mask vs TF's eager attention mask."""
    print("\n── Test 4: causal mask ──")

    from atb_python_qwen3vl_embedding.text_model import make_causal_mask

    for S in [4, 16, 32]:
        atb_mask = make_causal_mask(S)  # (S, S) float16/float32

        # TF eager attention mask: 0 for attend, -inf for masked
        tf_mask = torch.zeros(S, S)
        tf_mask[torch.triu(torch.ones(S, S), diagonal=1).bool()] = -65504.0

        print(f"\n  S={S}  atb: {atb_mask.shape} dtype={atb_mask.dtype}"
              f"  tf: {tf_mask.shape} dtype={tf_mask.dtype}")
        print(f"  atb range: [{atb_mask.min().item():.1f}, {atb_mask.max().item():.1f}]")
        print(f"  tf  range: [{tf_mask.min().item():.1f}, {tf_mask.max().item():.1f}]")

        cs = cosine(atb_mask, tf_mask)
        exact = torch.equal(atb_mask.float(), tf_mask.float())
        print(f"  Exact match: {exact}")
        if not exact:
            diff = (atb_mask.float() - tf_mask.float()).abs()
            print(f"  Max diff: {diff.max().item():.4f}")
            diff_mask = diff > 0.01
            if diff_mask.any():
                print(f"  Diff positions: {diff_mask.nonzero(as_tuple=False).tolist()[:5]}")

        # 0.9999: identity check — deterministic mask construction, any deviation is a logic bug
        report(f"causal mask S={S}", cs, threshold=0.9999)

    return True  # always pass (just print)


# ═══════════════════════════════════════════════════════════════════
# Test 5: single text layer
# ═══════════════════════════════════════════════════════════════════

def test_text_layer(proc, engine, model_dir):
    """Run ONE text decoder layer with identical inputs, compare outputs."""
    print("\n── Test 5: single text decoder layer ──")

    ref = load_tf_ref(model_dir)

    from atb_python_qwen3vl_embedding.text_model import run_text_layer_npu, make_causal_mask
    from atb_python_qwen3vl_embedding.utils import make_seqlen_tensor, is_310p, make_causal_mask_nz_npu
    from atb_python_qwen3vl_embedding.engine_utils import get_rope_index

    # Use simple text-only input to avoid deepstack complications
    msgs = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'What is the capital of France?'}]}]
    tf_in = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt')
    input_ids = tf_in['input_ids']
    S = input_ids.shape[1]

    # Shared inputs — use ATB embed_w
    inputs_embeds = F.embedding(input_ids, engine.embed_w).half()

    # Use TF position_ids (known correct from test 2)
    with torch.no_grad():
        tf_pid, _ = ref.get_rope_index(input_ids, None, None, None)
        # TF RoPE — all inputs must be on NPU
        dummy = torch.zeros(1, S, engine.hidden_t, device='npu')
        tf_cos, tf_sin = ref.language_model.rotary_emb(dummy, tf_pid.float().npu())

    # ATB RoPE (with same position_ids) — ATB expects (B*S, head_dim)
    atb_cos, atb_sin = engine.text_rope(tf_pid)
    cos_npu = atb_cos.reshape(-1, engine.hd_t).half().npu()
    sin_npu = atb_sin.reshape(-1, engine.hd_t).half().npu()

    # TF RoPE — keep (batch, seq_len, head_dim) shape for TF layer
    tf_cos_npu = tf_cos.half().npu()  # (1, S, head_dim)
    tf_sin_npu = tf_sin.half().npu()  # (1, S, head_dim)

    # ATB text layer graph
    engine._ensure_text_graph(S)
    causal_mask = make_causal_mask(S).half().npu()
    if is_310p():
        causal_mask_atb = make_causal_mask_nz_npu(S)
    else:
        causal_mask_atb = causal_mask

    # Run first text layer only — ATB
    hidden_atb = inputs_embeds.half().npu()
    seqlen_t = make_seqlen_tensor(S)
    atb_out = run_text_layer_npu(
        engine.g_t_layer, hidden_atb,
        engine.t_layer_weights[0],
        cos_npu, sin_npu, seqlen_t,
        causal_mask=causal_mask_atb).cpu().float()

    # Run first text layer only — TF (same inputs_embeds, same position_embeddings)
    hidden_tf = inputs_embeds.half().npu()
    with torch.no_grad():
        tf_layer_out = ref.language_model.layers[0](
            hidden_tf,
            position_embeddings=(tf_cos_npu, tf_sin_npu),
            attention_mask=causal_mask.unsqueeze(0).unsqueeze(0).float())
    tf_out = tf_layer_out.cpu().float()

    cs = cosine(atb_out, tf_out)
    print(f"\n  S={S}  atb_out: {atb_out.shape}  tf_out: {tf_out.shape}")
    print(f"  cosine: {cs:.6f}")

    # 0.99: moderate fp16 accumulation threshold (single text layer + GQA) — see THRESHOLDS.md
    if cs < 0.99:
        print("\n  DEBUG: checking individual layer weights...")
        # Compare layer weights
        for i, name in enumerate(["q_w", "k_w", "v_w", "o_w", "qn_w", "kn_w",
                                   "gate_w", "up_w", "down_w", "input_ln_w", "post_ln_w"]):
            atb_w = engine.t_layer_weights[0][i].cpu().float()
            tf_name_map = {
                "q_w": "self_attn.q_proj.weight",
                "k_w": "self_attn.k_proj.weight",
                "v_w": "self_attn.v_proj.weight",
                "o_w": "self_attn.o_proj.weight",
                "qn_w": "self_attn.q_norm.weight",
                "kn_w": "self_attn.k_norm.weight",
                "gate_w": "mlp.gate_proj.weight",
                "up_w": "mlp.up_proj.weight",
                "down_w": "mlp.down_proj.weight",
                "input_ln_w": "input_layernorm.weight",
                "post_ln_w": "post_attention_layernorm.weight",
            }
            tf_w = ref.language_model.layers[0].get_parameter(tf_name_map[name]).cpu().float()
            w_cs = cosine(atb_w, tf_w)
            # 0.999: single fp16 operator threshold — weight comparison for debugging per-layer precision
            status = "OK" if w_cs > 0.999 else "MISMATCH"
            print(f"    {name:<12} cosine={w_cs:.6f} [{status}]")

    del ref
    torch.npu.empty_cache()
    return report("single text layer", cs)


# ═══════════════════════════════════════════════════════════════════
# Test 6: safetensors key audit
# ═══════════════════════════════════════════════════════════════════

def test_safetensors_keys(model_dir):
    """List all keys containing 'embed_tokens' in the safetensors file."""
    print("\n── Test 6: safetensors key audit ──")
    import safetensors.torch

    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors", device="cpu")

    # Find all embed_tokens keys
    embed_keys = [k for k in sd.keys() if 'embed_tokens' in k]
    print(f"  Keys containing 'embed_tokens': {embed_keys}")
    for k in embed_keys:
        v = sd[k]
        print(f"    {k}: shape={v.shape} dtype={v.dtype}")

    # Also check what ATB engine loads
    from atb_python_qwen3vl_embedding.engine_utils import load_weights, get_embed_weight
    weights = load_weights(model_dir)
    atb_key = "model.language_model.embed_tokens.weight"
    print(f"\n  ATB looks up key: '{atb_key}'")
    if atb_key in weights:
        atb_w = weights[atb_key]
        print(f"    Found: shape={atb_w.shape} dtype={atb_w.dtype}")
        if atb_key in sd:
            sd_w = sd[atb_key].float()
            print(f"    Exact match with safetensors: {torch.equal(atb_w, sd_w)}")
    else:
        print(f"    NOT FOUND! Available keys with 'embed': "
              f"{[k for k in weights.keys() if 'embed' in k.lower()]}")

    return True


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    model_dir = QWEN3VL_EMB_MODEL_DIR
    print(f"Model dir: {model_dir}")

    from transformers import AutoProcessor
    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine

    proc = AutoProcessor.from_pretrained(model_dir)
    engine = Qwen3VLEngine(model_dir)
    print(f"Engine: {engine.n_layer} text layers, {engine.v_depth} vision blocks")

    results = {}
    results['embed_weights'] = test_embed_weights(engine, model_dir)
    results['position_ids'] = test_position_ids(proc, engine, model_dir)
    results['rope_cos_sin'] = test_rope_cos_sin(proc, engine, model_dir)
    test_causal_mask()
    test_safetensors_keys(model_dir)
    results['text_layer'] = test_text_layer(proc, engine, model_dir)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")
    print(f"{'='*60}")

    print("""
Interpretation:
  - If embed_weights FAIL → weight loading bug
  - If position_ids FAIL → MRoPE get_rope_index mismatch
  - If rope_cos_sin FAIL → RoPE implementation mismatch
  - If text_layer FAIL with weight mismatch → check safetensors key audit
  - If text_layer FAIL but weights OK → ATB graph execution issue
""")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
