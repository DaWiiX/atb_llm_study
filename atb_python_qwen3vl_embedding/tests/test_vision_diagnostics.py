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
from atb_python_qwen3vl_embedding.tests.data_utils import load_tf_ref


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def cosine(a, b):
    """Cosine similarity of two flattened tensors."""
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def report(label, cs, threshold=0.99):
    """Print a PASS/FAIL line.
    Default threshold 0.99: moderate fp16 accumulation — see THRESHOLDS.md.
    """
    status = "PASS" if cs >= threshold else "FAIL"
    print(f"  [{status}] {label:<30} cosine={cs:.6f}")
    return cs >= threshold


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
    """ATB preprocess_image vs TF processor pixel_values.

    Uses the shared compare_preprocess_with_tf() from test_preprocess.py
    and additionally validates that engine.preprocess_image (thin wrapper)
    returns the same result as the standalone preprocess_image.
    """
    print("\n── Test 1: preprocess match ──")
    from atb_python_qwen3vl_embedding.tests.test_preprocess import (
        compare_preprocess_with_tf,
    )

    ip = proc.image_processor
    all_ok = True
    for name, img in TEST_IMAGES:
        result = compare_preprocess_with_tf(
            img, proc, patch_size=ip.patch_size,
            temporal_patch_size=ip.temporal_patch_size,
            merge_size=ip.merge_size,
            min_pixels=ip.min_pixels, max_pixels=ip.max_pixels)

        print(f"\n  [{name}]  atb: {result['atb_pv'].shape}"
              f" grid={result['atb_gth'].tolist()}"
              f"   tf: {result['tf_pv'].shape}"
              f" grid={result['tf_gth'].tolist()}")

        # grid_thw must match exactly
        print(f"    grid_thw match: {result['grid_match']}")
        all_ok &= result['grid_match']

        # pixel_values cosine
        all_ok &= report(f"pixel_values cosine ({name})",
                         result['cosine'], threshold=0.999)

        # max diff
        print(f"    max_diff={result['max_diff']:.4f}")

        # Verify engine.preprocess_image (thin wrapper) matches standalone
        img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        eng_pv, eng_gth = engine.preprocess_image(img_arr)
        if not torch.equal(eng_gth, result['atb_gth']):
            print("    [WARN] engine.preprocess_image grid_thw differs from standalone")
            all_ok = False
        eng_cs = F.cosine_similarity(
            result['atb_pv'].float().flatten(),
            eng_pv.float().flatten(), dim=0).item()
        if eng_cs < 0.9999:
            print(f"    [WARN] engine.preprocess_image pv cosine vs standalone: {eng_cs:.6f}")
            all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════════════════
# Test 2: vision encoder only
# ═══════════════════════════════════════════════════════════════════

def test_vision_only(proc, engine, model_dir):
    """Feed SAME pixel_values to both ATB _run_vision and TF visual().

    Uses TF processor's pixel_values to avoid any preprocess difference.
    """
    print("\n── Test 2: vision encoder only (same pixel_values) ──")

    ref = load_tf_ref(model_dir, precision="half")

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
            vis_tf_out = ref.visual(ref.place(pv_tf), grid_thw=gth_tf.to(ref.device))
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

    ref = load_tf_ref(model_dir, precision="half")

    name, img = TEST_IMAGES[0]  # use 120x200-red

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
        vis_tf_out = ref.visual(ref.place(pv_tf), grid_thw=gth_tf.to(ref.device))
        # vis_tf_out[0] = merged hidden_states (N_vis, hidden_size)
        # vis_tf_out[1] = deepstack_feature_lists (list of tensors)
        vis_tf_npu = vis_tf_out[0].half().npu()   # image_embeds for injection
        ds_tf_npu = [d.half().npu() for d in vis_tf_out[1]] if vis_tf_out[1] else []

    print(f"  vis_tf_npu={vis_tf_npu.shape}  deepstack={len(ds_tf_npu)} features")

    # ── Build TF inputs_embeds using TF embed weights (masked_scatter) ──
    with torch.no_grad():
        ie_tf = ref.get_input_embeddings()(input_ids.to(ref.device)).to(ref.dtype)
        image_embeds_cat = torch.cat([vis_tf_npu], dim=0).to(ie_tf.device, ie_tf.dtype)
        tok_emb = ref.get_input_embeddings()(
            torch.tensor(engine.img_tok, dtype=torch.long, device=ie_tf.device))
        special_image_mask = (ie_tf == tok_emb)
        image_mask = special_image_mask.all(-1, keepdim=True).expand_as(ie_tf)
        ie_tf = ie_tf.masked_scatter(image_mask, image_embeds_cat)

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

    # TF language_model with same inputs (apples-to-apples comparison)
    vis_pos_masks = vis_mask.unsqueeze(0).to(ref.device)  # (1, S) bool
    with torch.no_grad():
        tf_out = ref.language_model(
            inputs_embeds=ie_tf,
            position_ids=pid.to(ref.device),
            visual_pos_masks=vis_pos_masks if ds_tf_npu else None,
            deepstack_visual_embeds=[d.to(ref.device, ref.dtype) for d in ds_tf_npu] if ds_tf_npu else None,
        ).last_hidden_state.cpu().float()

    cs = cosine(atb_out, tf_out)
    print(f"\n  ATB _run_text vs TF language_model (same TF vision inputs)")
    del ref
    torch.npu.empty_cache()
    return report("text with TF vision embeds", cs)


# ═══════════════════════════════════════════════════════════════════
# Test 4: e2e with TF preprocess
# ═══════════════════════════════════════════════════════════════════

def test_e2e_with_tf_preprocess(proc, engine, model_dir):
    """Isolate text model with TF vision embeds (loop over test images).

    Feeds TF vision embeddings into both ATB _run_text and TF
    language_model.  Apples-to-apples comparison at the text-model level.
    """
    print("\n── Test 4: text model with TF vision embeds (both images) ──")

    ref = load_tf_ref(model_dir, precision="half")
    from atb_python_qwen3vl_embedding.engine_utils import get_rope_index

    all_ok = True
    for name, img in TEST_IMAGES:
        # TF input_ids + pixel_values
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img}, {'type': 'text', 'text': 'Describe.'}]}]
        tf_in = proc.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt')
        input_ids = tf_in['input_ids']
        S = input_ids.shape[1]
        pv_tf = tf_in['pixel_values']
        gth_tf = tf_in['image_grid_thw']

        print(f"\n  [{name}]  S={S}  pv={pv_tf.shape}  grid={gth_tf.tolist()}")

        # ── TF vision → image_embeds + deepstack ──
        with torch.no_grad():
            vis_tf_out = ref.visual(ref.place(pv_tf), grid_thw=gth_tf.to(ref.device))
            vis_tf_npu = vis_tf_out[0].half().npu()
            ds_tf_npu = [d.half().npu() for d in vis_tf_out[1]] if vis_tf_out[1] else []

        # ── Build TF inputs_embeds with TF vision embeds injected ──
        with torch.no_grad():
            ie_tf = ref.get_input_embeddings()(input_ids.to(ref.device)).to(ref.dtype)
            image_embeds_cat = torch.cat([vis_tf_npu], dim=0).to(ie_tf.device, ie_tf.dtype)
            tok_emb = ref.get_input_embeddings()(
                torch.tensor(engine.img_tok, dtype=torch.long, device=ie_tf.device))
            special_image_mask = (ie_tf == tok_emb)
            image_mask = special_image_mask.all(-1, keepdim=True).expand_as(ie_tf)
            ie_tf = ie_tf.masked_scatter(image_mask, image_embeds_cat)

        # ── Build ATB inputs_embeds with TF vision embeds injected ──
        inputs_embeds_atb = F.embedding(input_ids, engine.embed_w).half().npu()
        vis_mask = input_ids.squeeze(0) == engine.img_tok
        inputs_embeds_atb[0, vis_mask.npu(), :] = vis_tf_npu

        # ── Position IDs ──
        pid, _ = get_rope_index(
            input_ids, gth_tf, None, None,
            image_token_id=engine.img_tok,
            spatial_merge_size=engine.spatial_merge)

        engine._ensure_text_graph(S)

        # ATB _run_text with TF vision embeds + TF deepstack
        atb_out = engine._run_text(
            inputs_embeds_atb, pid, vis_mask,
            ds_tf_npu if ds_tf_npu else None).cpu().float()

        # TF language_model with same inputs
        vis_pos_masks = vis_mask.unsqueeze(0).to(ref.device)
        with torch.no_grad():
            tf_out = ref.language_model(
                inputs_embeds=ie_tf,
                position_ids=pid.to(ref.device),
                visual_pos_masks=vis_pos_masks if ds_tf_npu else None,
                deepstack_visual_embeds=[d.to(ref.device, ref.dtype) for d in ds_tf_npu] if ds_tf_npu else None,
            ).last_hidden_state.cpu().float()

        cs = cosine(atb_out, tf_out)
        all_ok &= report(f"text with TF vision ({name})", cs)

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
  - If text_with_tf_vision FAIL → text model or deepstack injection (single image)
  - If e2e_with_tf_preprocess FAIL → text model or deepstack injection (both images)
""")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())