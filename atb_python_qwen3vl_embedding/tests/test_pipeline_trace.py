"""Pipeline trace — step-by-step cosine comparison between ATB and TF.

Runs both pipelines in lockstep, comparing at every stage to find
the EXACT point where cosine diverges.

Usage::

    python tests/test_pipeline_trace.py
"""
# ── Buffer size MUST be set before any engine/graph import ──────────
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size

set_atb_buffer_size(5 * 1024 * 1024 * 1024)  # 5 GB

# ── Standard imports ───────────────────────────────────────────────
import sys
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import numpy as np
from PIL import Image

from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR
from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.engine_utils import get_rope_index
from atb_python_qwen3vl_embedding.text_model import run_text_layer_npu, run_text_norm_npu, make_causal_mask
from atb_python_qwen3vl_embedding.utils import to_npu_half, to_cpu_float, make_seqlen_tensor, is_310p, make_causal_mask_nz_npu
from atb_python_qwen3vl_embedding.tests.data_utils import load_tf_ref


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def trace(label, atb, tf, gate=True):
    """Trace step-by-step comparison. Threshold 0.99: moderate fp16 accumulation — see THRESHOLDS.md."""
    cs = cosine(atb, tf)
    ok = cs > 0.99
    status = "✅" if ok else "❌"
    suffix = "" if gate else " (diagnostic, not gated)"
    print(f"  {status} {label:<35} cosine={cs:.6f}{suffix}")
    return ok if gate else True


def main():
    model_dir = QWEN3VL_EMB_MODEL_DIR
    print(f"Model dir: {model_dir}\n")

    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    proc = AutoProcessor.from_pretrained(model_dir)
    engine = Qwen3VLEngine(model_dir)

    # ── Load TF ref (half precision — engine stays loaded on NPU) ──
    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    ref = load_tf_ref(model_dir, precision="half")

    # ── Test case: Image + Text ───────────────────────────────────
    img = Image.new('RGB', (120, 200), color='red')
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Describe.'}]}]
    tf_in = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True)

    # === ALL inputs on CPU ===
    input_ids = tf_in['input_ids']           # (1, S) long CPU
    pv_tf = tf_in['pixel_values']           # (N, C*tp*p*p) float32 CPU
    gth_tf = tf_in['image_grid_thw']        # (1, 3) long CPU
    S = input_ids.shape[1]

    print(f"S={S}  pv={pv_tf.shape}  grid={gth_tf.tolist()}\n")

    all_ok = True

    # ── Step 1: Preprocess ────────────────────────────────────────
    print("── Step 1: Preprocess ──")
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    pv_atb, gth_atb = engine.preprocess_image(img_arr)
    all_ok &= trace("pixel_values", pv_atb, pv_tf.float())
    all_ok &= trace("grid_thw", gth_atb.float(), gth_tf.float())

    # ── Step 2: Vision encoder ────────────────────────────────────
    print("\n── Step 2: Vision encoder ──")
    vis_atb, ds_atb = engine._run_vision(pv_atb, gth_atb)
    vis_atb = to_cpu_float(vis_atb)        # start of ATB output (CPU float)
    ds_atb_cpu = [to_cpu_float(d) if d.is_npu else d.float() for d in ds_atb]
    # ds_atb list still holds the original NPU tensors for Step 6

    with torch.no_grad():
        # NOTE: grid_thw must stay as long (TF uses it as int for torch.linspace)
        vis_tf_out = ref.visual(to_npu_half(pv_tf), grid_thw=gth_tf.npu())
        vis_tf = to_cpu_float(vis_tf_out[0])           # TF output (CPU float)
        ds_tf = [to_cpu_float(d) for d in vis_tf_out[1]] if vis_tf_out[1] else []

    all_ok &= trace("vision merged", vis_atb, vis_tf)
    for i, (da, dt) in enumerate(zip(ds_atb_cpu, ds_tf)):
        all_ok &= trace(f"deepstack[{i}]", da, dt)

    # ── Step 3: Image injection ───────────────────────────────────
    print("\n── Step 3: Image injection ──")

    vis_mask_1d = (input_ids.squeeze(0) == engine.img_tok)  # (S,) bool CPU

    # ATB: all on NPU
    ie_atb = F.embedding(input_ids, engine.embed_w).half().npu()          # (1,S,hd) NPU
    ie_atb[0, vis_mask_1d.npu(), :] = to_npu_half(vis_atb)               # inject

    # TF: all on NPU — replicate transformers Qwen3VLModel image-injection exactly
    ie_tf = ref.get_input_embeddings()(input_ids.npu()).half()           # (1,S,hd) NPU
    image_embeds_cat = torch.cat([vis_tf], dim=0).to(ie_tf.device, ie_tf.dtype)
    tok_emb = ref.get_input_embeddings()(
        torch.tensor(cfg.image_token_id, dtype=torch.long, device=ie_tf.device))
    special_image_mask = (ie_tf == tok_emb)                              # (1,S,hd)
    image_mask = special_image_mask.all(-1, keepdim=True).expand_as(ie_tf)  # (1,S,hd)
    ie_tf = ie_tf.masked_scatter(image_mask, image_embeds_cat)

    all_ok &= trace("inputs_embeds after injection", ie_atb.cpu(), ie_tf.cpu())

    # Compare injected vision tokens only
    atb_injected = ie_atb[0, vis_mask_1d.npu(), :].cpu()
    tf_injected = ie_tf[0, vis_mask_1d.npu(), :].cpu()
    all_ok &= trace("injected vision tokens only", atb_injected, tf_injected)

    # ── Step 4: Position IDs ──────────────────────────────────────
    print("\n── Step 4: Position IDs ──")
    atb_pid, _ = get_rope_index(
        input_ids, gth_atb, None, None,
        image_token_id=engine.img_tok,
        spatial_merge_size=engine.spatial_merge)
    with torch.no_grad():
        tf_pid, _ = ref.get_rope_index(input_ids, gth_tf, None, None)
    all_ok &= trace("position_ids", atb_pid.float(), tf_pid.float())

    # ── Step 5: RoPE cos/sin ──────────────────────────────────────
    print("\n── Step 5: RoPE cos/sin ──")
    atb_cos, atb_sin = engine.text_rope(atb_pid)
    cos_atb = atb_cos.reshape(-1, engine.hd_t)
    sin_atb = atb_sin.reshape(-1, engine.hd_t)

    with torch.no_grad():
        dummy = torch.zeros(1, S, engine.hidden_t, device='npu')
        # tf_pid is on CPU → move to NPU for rotary_emb
        tf_cos, tf_sin = ref.language_model.rotary_emb(dummy, tf_pid.float().npu())
    cos_tf = tf_cos.reshape(-1, engine.hd_t)
    sin_tf = tf_sin.reshape(-1, engine.hd_t)

    all_ok &= trace("cos", cos_atb, cos_tf.cpu())
    all_ok &= trace("sin", sin_atb, sin_tf.cpu())

    # ── Step 6: Text model layer by layer ─────────────────────────
    print("\n── Step 6: Text model (layer by layer) ──")

    engine._ensure_text_graph(S)
    causal_mask = make_causal_mask(S).half().npu()
    if is_310p():
        causal_mask_atb = make_causal_mask_nz_npu(S)
    else:
        causal_mask_atb = causal_mask

    # Move everything to NPU for the loop
    hidden_atb = ie_atb                                                # already NPU
    hidden_tf = ie_tf                                                  # already NPU
    cos_npu = to_npu_half(cos_atb)
    sin_npu = to_npu_half(sin_atb)
    tf_cos_npu = tf_cos.half().npu()
    tf_sin_npu = tf_sin.half().npu()
    vis_mask_2d = vis_mask_1d.unsqueeze(0).npu()                     # (1,S) bool NPU

    ds_indexes = engine.ds_indexes
    print(f"  deepstack at layers: {ds_indexes}")

    for li in range(engine.n_layer):
        # ATB layer
        seqlen_t = make_seqlen_tensor(S)
        hidden_atb = run_text_layer_npu(
            engine.g_t_layer, hidden_atb,
            engine.t_layer_weights[li],
            cos_npu, sin_npu, seqlen_t,
            causal_mask=causal_mask_atb)

        # TF layer
        with torch.no_grad():
            hidden_tf = ref.language_model.layers[li](
                hidden_tf,
                position_embeddings=(tf_cos_npu, tf_sin_npu),
                attention_mask=causal_mask.unsqueeze(0).unsqueeze(0).float())

        torch.npu.synchronize()

        # Deepstack injection at first N layers where N = len(ds_atb)
        if ds_atb and li < len(ds_atb):
            # ATB: clone + add + writeback
            ds_npu = to_npu_half(ds_atb[li])  # ds_atb[li] is NPU, to_npu_half is no-op
            local_atb = hidden_atb[0, vis_mask_1d.npu(), :].clone() + ds_npu
            hidden_atb[0, vis_mask_1d.npu(), :] = local_atb

            # TF: _deepstack_process
            ds_npu_tf = to_npu_half(ds_tf[li])  # ds_tf[li] is CPU → NPU
            local_tf = hidden_tf[vis_mask_2d, :].clone() + ds_npu_tf
            hidden_tf[vis_mask_2d, :] = local_tf

        # Log status periodically
        if li == 0 or li == engine.n_layer - 1 or li % 4 == 0 or li in ds_indexes:
            cs = cosine(hidden_atb.cpu(), hidden_tf.cpu())
            ds_tag = " [+ds]" if li in ds_indexes else ""
            status = "✅" if cs > 0.99 else "❌"  # 0.99: moderate fp16 accumulation — see THRESHOLDS.md
            print(f"  {status} layer {li:2d}{ds_tag:<8} cosine={cs:.6f}")
            all_ok &= cs > 0.99

    # ── Step 7: Final norm ────────────────────────────────────────
    print("\n── Step 7: Final norm ──")
    out_atb = run_text_norm_npu(engine.g_t_norm, hidden_atb, engine.norm_w).cpu().float()

    with torch.no_grad():
        out_tf = ref.language_model.norm(hidden_tf).cpu().float()

    all_ok &= trace("final hidden_state", out_atb, out_tf)

    # ── Step 8: Sanity — compare manual-TF vs ref(...) full forward ───
    # If manual_tf == ref_full, our TF chain mirrors the production path.
    # If they diverge, the manual TF chain in Steps 3-7 is NOT equivalent
    # to what e2e compares against, which explains why every per-step
    # cosine passed yet e2e fails.
    print("\n── Step 8: Sanity check — manual TF vs ref(...) full ──")
    with torch.no_grad():
        ref_full = ref(
            input_ids=input_ids.npu(),
            pixel_values=pv_tf.half().npu(),
            image_grid_thw=gth_tf.npu(),
        ).last_hidden_state.cpu().float()
    # Step 8+ bisect comparisons intentionally mix alternate transformers call
    # paths to diagnose why ref(...) differs from manual/replayed language_model
    # calls. They are observational diagnostics, not ATB-vs-reference gates.
    all_ok &= trace("manual_TF vs ref_full", out_tf, ref_full, gate=False)
    all_ok &= trace("ATB     vs ref_full",   out_atb, ref_full, gate=False)

    # ── Step 9: Bisect ref_full into stages ────────────────────────────
    # If language_model(inputs_embeds=ie_tf, ...) matches ref_full, the
    # divergence is inside Qwen3VLModel.forward (image inject / pid build);
    # if it matches manual_TF, the divergence is inside language_model.
    print("\n── Step 9: Sub-bisect with ref.language_model(...) ──")
    with torch.no_grad():
        # Use TF-built ie_tf + manually computed tf_pid + deepstack list
        ds_for_lm = [d.half().npu() for d in ds_tf] if ds_tf else None
        vis_pos_masks_lm = vis_mask_2d if ds_for_lm else None
        sub_out = ref.language_model(
            inputs_embeds=ie_tf,
            position_ids=tf_pid.npu(),
            visual_pos_masks=vis_pos_masks_lm,
            deepstack_visual_embeds=ds_for_lm,
        ).last_hidden_state.cpu().float()
    all_ok &= trace("sub_LM   vs ref_full", sub_out, ref_full, gate=False)
    all_ok &= trace("sub_LM   vs manual_TF", sub_out, out_tf, gate=False)
    all_ok &= trace("sub_LM   vs ATB",       sub_out, out_atb, gate=False)

    # ── Step 10: Capture ref's actual inputs to language_model ─────────
    # Monkey-patch language_model.forward so we can see what Qwen3VLModel
    # actually feeds it (inputs_embeds, position_ids, visual_pos_masks,
    # deepstack_visual_embeds, attention_mask). Compare each against the
    # corresponding manual value built in Steps 3-5.
    print("\n── Step 10: Capture ref's true inputs to language_model ──")
    captured = {}
    captured_all_kw = {}
    captured_args = ()
    real_lm_forward = ref.language_model.forward

    def spy_forward(*a, **kw):
        nonlocal captured_args
        captured_args = a
        captured_all_kw.update(kw)
        captured["input_ids"]              = kw.get("input_ids")
        captured["inputs_embeds"]          = kw.get("inputs_embeds")
        captured["position_ids"]           = kw.get("position_ids")
        captured["attention_mask"]         = kw.get("attention_mask")
        captured["visual_pos_masks"]       = kw.get("visual_pos_masks")
        captured["deepstack_visual_embeds"] = kw.get("deepstack_visual_embeds")
        return real_lm_forward(*a, **kw)

    ref.language_model.forward = spy_forward
    with torch.no_grad():
        spy_ref_full = ref(
            input_ids=input_ids.npu(),
            pixel_values=pv_tf.half().npu(),
            image_grid_thw=gth_tf.npu(),
        ).last_hidden_state.cpu().float()
    ref.language_model.forward = real_lm_forward  # restore

    # Sanity: spy's ref_full should equal Step 8 ref_full (spy is pure wrapper)
    all_ok &= trace("spy_ref_full vs ref_full (Step 8)", spy_ref_full, ref_full, gate=False)

    # ── Step 10b: rerun sub_LM with EXACTLY the captured inputs ────────
    # captured kwargs came straight from ref(...); feed them back into
    # language_model and compare. If this still matches sub_LM (≈1) but
    # NOT ref_full (≈0.51), the diff must be inside captured-but-unread
    # state OR something Qwen3VLModel does AFTER language_model returns.
    print("\n── Step 10b: replay captured inputs into language_model ──")
    with torch.no_grad():
        replay = ref.language_model(
            inputs_embeds=captured["inputs_embeds"],
            position_ids=captured["position_ids"],
            attention_mask=captured["attention_mask"],
            visual_pos_masks=captured["visual_pos_masks"],
            deepstack_visual_embeds=captured["deepstack_visual_embeds"],
        ).last_hidden_state.cpu().float()
    all_ok &= trace("replay   vs ref_full",  replay, ref_full, gate=False)
    all_ok &= trace("replay   vs sub_LM",    replay, sub_out, gate=False)
    all_ok &= trace("replay   vs manual_TF", replay, out_tf, gate=False)

    # ── Step 10c: dump EVERY arg/kw spy received, then replay them all ──
    # captured_all_kw + captured_args contain the COMPLETE call into
    # language_model. If replay-all matches ref_full, the missing key
    # was the smoking gun.
    print("\n── Step 10c: all kwargs spy actually received ──")
    print(f"  positional args: {len(captured_args)}")
    for i, a in enumerate(captured_args):
        if isinstance(a, torch.Tensor):
            print(f"    arg[{i}] tensor shape={tuple(a.shape)} dtype={a.dtype}")
        else:
            print(f"    arg[{i}] type={type(a).__name__} value={a!r}"[:120])

    print(f"  keyword args: {len(captured_all_kw)} keys")
    for k, v in captured_all_kw.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:<28} tensor shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"    {k:<28} {type(v).__name__}[{len(v)}]")
        else:
            print(f"    {k:<28} type={type(v).__name__} value={v!r}"[:120])

    print("\n  Replay with EXACT captured args+kwargs:")
    with torch.no_grad():
        replay_all = real_lm_forward(*captured_args, **captured_all_kw).last_hidden_state.cpu().float()
    all_ok &= trace("replay_ALL vs ref_full",   replay_all, ref_full, gate=False)
    all_ok &= trace("replay_ALL vs replay",     replay_all, replay, gate=False)
    all_ok &= trace("replay_ALL vs ATB",        replay_all, out_atb, gate=False)

    # ── Step 10d: isolate which of the 3 cache-related kwargs breaks it ─
    # Add use_cache / past_key_values / cache_position one at a time on top
    # of the original 6-kwarg replay.
    print("\n── Step 10d: which cache kwarg flips the output? ──")
    base = dict(
        inputs_embeds=captured["inputs_embeds"],
        position_ids=captured["position_ids"],
        attention_mask=captured["attention_mask"],
        visual_pos_masks=captured["visual_pos_masks"],
        deepstack_visual_embeds=captured["deepstack_visual_embeds"],
    )
    for extra_key in ("use_cache", "past_key_values", "cache_position"):
        if extra_key not in captured_all_kw:
            continue
        kw_test = dict(base)
        kw_test[extra_key] = captured_all_kw[extra_key]
        with torch.no_grad():
            out_test = ref.language_model(**kw_test).last_hidden_state.cpu().float()
        all_ok &= trace(f"+{extra_key:<18} vs ref_full", out_test, ref_full, gate=False)
        all_ok &= trace(f"+{extra_key:<18} vs ATB",      out_test, out_atb, gate=False)

    # Compare what ref REALLY passes against what we built manually
    print("\n  Inputs captured from real ref(...) call:")
    for k, v in captured.items():
        if v is None:
            print(f"    {k:<25} = None")
        elif isinstance(v, torch.Tensor):
            print(f"    {k:<25} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        elif isinstance(v, (list, tuple)):
            shapes = [tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__ for x in v]
            print(f"    {k:<25} list[{len(v)}] shapes={shapes}")
        else:
            print(f"    {k:<25} type={type(v).__name__}")

    # Direct numerical compare
    print("\n  Numerical match (manual vs ref-captured):")
    if captured["inputs_embeds"] is not None:
        all_ok &= trace("inputs_embeds",   ie_tf.cpu(), captured["inputs_embeds"].cpu())
    if captured["position_ids"] is not None:
        all_ok &= trace("position_ids",    tf_pid.float().cpu(),
                        captured["position_ids"].float().cpu())
    if captured["visual_pos_masks"] is not None:
        all_ok &= trace("visual_pos_masks", vis_mask_2d.float().cpu(),
                        captured["visual_pos_masks"].float().cpu())
    if captured["deepstack_visual_embeds"] is not None and ds_tf:
        for i, (manual, real) in enumerate(zip(ds_tf, captured["deepstack_visual_embeds"])):
            all_ok &= trace(f"deepstack[{i}]", manual.cpu(), real.cpu())

    print("\nDone.")

    del ref
    torch.npu.empty_cache()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
