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
    """Cosine similarity of two flattened tensors."""
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def report(label, cs, threshold=0.99):
    """Print a PASS/FAIL line."""
    status = "PASS" if cs >= threshold else "FAIL"
    print(f"  [{status}] {label:<30} cosine={cs:.6f}")
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
# Test images — same as test_e2e.py
# ═══════════════════════════════════════════════════════════════════

TEST_IMAGES = [
    ("120x200-red", Image.new('RGB', (120, 200), color='red')),
    ("64x64-blue",  Image.new('RGB', (64, 64), color='blue')),
]


# ═══════════════════════════════════════════════════════════════════
# Test 1: preprocess match
# ═══════════════════════════════════════════════════════════════════

def test_preprocess_match(proc, engine):
    """ATB preprocess_image vs TF processor pixel_values."""
    print("\n── Test 1: preprocess match ──")
    all_ok = True
    for name, img in TEST_IMAGES:
        img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)

        # ATB path
        pv_atb, gth_atb = engine.preprocess_image(img_arr)

        # TF path — processor returns {pixel_values, image_grid_thw}
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': 'x'}]}]
        tf_out = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt')
        pv_tf = tf_out['pixel_values']       # (N, C*tp*p*p)
        gth_tf = tf_out['image_grid_thw']    # (1, 3)

        print(f"\n  [{name}]  atb: {pv_atb.shape} grid={gth_atb.tolist()}"
              f"   tf: {pv_tf.shape} grid={gth_tf.tolist()}")

        # grid_thw must match exactly
        grid_ok = gth_atb.tolist() == gth_tf.tolist()
        all_ok &= grid_ok
        print(f"    grid_thw match: {grid_ok}")

        # pixel_values cosine
        cs = cosine(pv_atb, pv_tf.float())
        all_ok &= report(f"pixel_values cosine ({name})", cs, threshold=0.999)

        # max diff
        md = (pv_atb - pv_tf.float()).abs().max().item()
        print(f"    max_diff={md:.4f}")

    return all_ok


# ═══════════════════════════════════════════════════════════════════
# Test 2: vision encoder only
# ═══════════════════════════════════════════════════════════════════

def test_vision_only(proc, engine, model_dir):
    """Feed SAME pixel_values to both ATB _run_vision and TF visual().

    Uses TF processor's pixel_values to avoid any preprocess difference.
    """
    print("\n── Test 2: vision encoder only (same pixel_values) ──")

    ref = load_tf_ref(model_dir)

    all_ok = True
    for name, img in TEST_IMAGES:
        # Get TF pixel_values
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': 'x'}]}]
        tf_out = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt')
        pv_tf = tf_out['pixel_values']       # (N, C*tp*p*p) float32
        gth_tf = tf_out['image_grid_thw']    # (1, 3)

        print(f"\n  [{name}]  pv={pv_tf.shape}  grid={gth_tf.tolist()}")

        # ── TF vision forward ──
        # Qwen3VLVisionModel.forward(pixel_values, grid_thw=...)
        with torch.no_grad():
            vis_tf_out = ref.visual(pv_tf.half().npu(), grid_thw=gth_tf.npu())
            # vis_tf_out = (hidden_states, deepstack_feature_lists)
            vis_tf = vis_tf_out[0].cpu().float()  # merged vision embeds
            ds_tf = [d.cpu().float() for d in vis_tf_out[1]] if vis_tf_out[1] else []

        # ── ATB vision forward (with TF pixel_values) ──
        vis_atb, ds_atb = engine._run_vision(pv_tf.float(), gth_tf)
        vis_atb = vis_atb.cpu().float()

        # Compare vision embeds
        cs_vis = cosine(vis_atb, vis_tf)
        all_ok &= report(f"vision merged ({name})", cs_vis)

        # Compare deepstack features
        for i, (da, dt) in enumerate(zip(ds_atb, ds_tf)):
            da_cpu = da.cpu().float() if da.is_npu else da.float()
            cs_ds = cosine(da_cpu, dt)
            all_ok &= report(f"deepstack[{i}] ({name})", cs_ds)

    del ref
    torch.npu.empty_cache()
    return all_ok


# ═══════════════════════════════════════════════════════════════════
# Test 3: text model with TF vision embeds
# ═══════════════════════════════════════════════════════════════════

def test_text_with_tf_vision(proc, engine, model_dir):
    """Feed TF's image_embeds + deepstack into ATB _run_text, compare cosine.

    Isolates text model + deepstack injection from vision encoder.
    """
    print("\n── Test 3: text model with TF vision embeds (isolate deepstack) ──")

    ref = load_tf_ref(model_dir)

    name, img = TEST_IMAGES[0]  # use 120x200-red
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    pv_atb, gth_atb = engine.preprocess_image(img_arr)

    # TF input_ids + pixel_values
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img}, {'type': 'text', 'text': 'Describe.'}]}]
    tf_in = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt')
    input_ids = tf_in['input_ids']        # (1, S)
    S = input_ids.shape[1]
    pv_tf = tf_in['pixel_values']         # (N, C*tp*p*p)
    gth_tf = tf_in['image_grid_thw']      # (1, 3)

    # ── TF vision → image_embeds + deepstack (on NPU, fp16) ──
    with torch.no_grad():
        vis_tf_out = ref.visual(pv_tf.half().npu(), grid_thw=gth_tf.npu())
        # vis_tf_out[0] = merged hidden_states (N_vis, hidden_size)
        # vis_tf_out[1] = deepstack_feature_lists (list of tensors)
        vis_tf_npu = vis_tf_out[0].half().npu()   # image_embeds for injection
        ds_tf_npu = [d.half().npu() for d in vis_tf_out[1]] if vis_tf_out[1] else []

    print(f"  vis_tf_npu={vis_tf_npu.shape}  deepstack={len(ds_tf_npu)} features")

    # ── Build ATB text inputs using TF's vision embeds ──
    inputs_embeds = F.embedding(input_ids, engine.embed_w).half().npu()
    vis_mask = input_ids.squeeze(0) == engine.img_tok   # (S,) bool
    inputs_embeds[0, vis_mask.npu(), :] = vis_tf_npu

    from atb_python_qwen3vl_embedding.engine_utils import get_rope_index
    pid, _ = get_rope_index(
        input_ids, gth_tf, None, None,
        image_token_id=engine.img_tok,
        spatial_merge_size=engine.spatial_merge)

    engine._ensure_text_graph(S)

    # ATB _run_text with TF vision embeds + TF deepstack
    atb_out = engine._run_text(
        inputs_embeds, pid, vis_mask,
        ds_tf_npu if ds_tf_npu else None).cpu().float()

    # TF full forward with same pixel_values
    with torch.no_grad():
        tf_out = ref(
            input_ids=input_ids.npu(),
            pixel_values=pv_tf.half().npu(),
            image_grid_thw=gth_tf.npu()).last_hidden_state.cpu().float()

    cs = cosine(atb_out, tf_out)
    print(f"\n  ATB _run_text vs TF full forward (same TF vision inputs)")
    del ref
    torch.npu.empty_cache()
    return report("text with TF vision embeds", cs)


# ═══════════════════════════════════════════════════════════════════
# Test 4: e2e with TF preprocess
# ═══════════════════════════════════════════════════════════════════

def test_e2e_with_tf_preprocess(proc, engine, model_dir):
    """Quick check: run engine.forward() with TF's pixel_values.

    If this passes but test_e2e fails, the problem is 100% in preprocessing.
    """
    print("\n── Test 4: e2e with TF preprocess (fast check) ──")

    ref = load_tf_ref(model_dir)

    all_ok = True
    for name, img in TEST_IMAGES:
        # TF input_ids + pixel_values
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': 'Describe.'}]}]
        tf_in = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt')
        input_ids = tf_in['input_ids']
        pv_tf = tf_in['pixel_values']
        gth_tf = tf_in['image_grid_thw']

        # ATB forward with TF pixel_values
        atb_out = engine.forward(input_ids, pv_tf.float(), gth_tf)

        # TF forward
        with torch.no_grad():
            tf_out = ref(
                input_ids=input_ids.npu(),
                pixel_values=pv_tf.half().npu(),
                image_grid_thw=gth_tf.npu()).last_hidden_state.cpu().float()

        cs = cosine(atb_out, tf_out)
        all_ok &= report(f"e2e with TF preprocess ({name})", cs)

    del ref
    torch.npu.empty_cache()
    return all_ok


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
    results['preprocess'] = test_preprocess_match(proc, engine)
    results['vision_only'] = test_vision_only(proc, engine, model_dir)
    results['text_with_tf_vision'] = test_text_with_tf_vision(proc, engine, model_dir)
    results['e2e_with_tf_preprocess'] = test_e2e_with_tf_preprocess(proc, engine, model_dir)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")
    print(f"{'='*60}")

    # Interpretation guide
    print("""
Interpretation:
  - If preprocess FAIL but all others PASS → preprocess is the sole issue
  - If vision_only FAIL → vision encoder internals (blocks, merger, RoPE)
  - If text_with_tf_vision FAIL → deepstack injection or text layer
  - If e2e_with_tf_preprocess PASS → 100% preprocess problem
""")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())