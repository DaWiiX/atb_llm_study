"""Level 3: Deepstack integration test — full vision pipeline with deepstack extraction.

Validates that deepstack features extracted at vision blocks [5, 11, 17] through
the ATB deepstack merger match the TF reference. Uses the real model config and
weights from safetensors.

Also validates deepstack injection into text layers: runs a few text decoder layers
with deepstack features added to visual token positions, comparing ATB vs TF.
"""
import os, sys, torch, torch.nn.functional as F
import numpy as np
from PIL import Image

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_qwen3vl_embedding.utils import (
    set_atb_buffer_size, to_npu_half, to_cpu_float, make_seqlen_tensor, is_310p,
    make_causal_mask_nz_npu,
)
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR
from atb_python_qwen3vl_embedding.tests.data_utils import load_tf_ref

set_atb_buffer_size(5 * 1024 * 1024 * 1024)  # 5 GB

from atb_python_qwen3vl_embedding.engine_utils import (
    load_config, load_weights, get_rope_index, TextRotaryEmbedding,
)
from atb_python_qwen3vl_embedding.text_model import (
    build_text_layer_graph, build_text_norm_graph,
    run_text_layer_npu, run_text_norm_npu, make_causal_mask,
)
from atb_python_qwen3vl_embedding.vision_model import (
    build_vision_first_layer, build_vision_merger, build_deepstack_merger,
    run_first_layer_npu, run_block_npu, run_merger_npu,
)
from atb_python_qwen3vl_embedding.vision_block import build_vision_block
from atb_python_qwen3vl_embedding.vision_pos_embed import (
    build_vision_posemb_graph, run_posemb_npu,
)
from atb_python_qwen3vl_embedding.engine_utils import (
    get_vision_block_weights, get_patch_embed_weights, get_vision_pos_embed,
    get_merger_weights, get_text_layer_weights, get_text_norm_weight,
    get_embed_weight, VisionRotaryEmbedding, compute_posemb_indices,
    compute_rope_indices,
)


def cosine(a, b):
    """Cosine similarity between two tensors (handles NPU and CPU)."""
    if hasattr(a, 'is_npu') and a.is_npu:
        a = a.cpu()
    if hasattr(b, 'is_npu') and b.is_npu:
        b = b.cpu()
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_deepstack_extraction(model_dir=None):
    """Test deepstack feature extraction from vision blocks.

    Runs the full vision pipeline on both ATB and TF, extracts deepstack
    features at blocks [5, 11, 17], and compares them.
    """
    if model_dir is None:
        model_dir = QWEN3VL_EMB_MODEL_DIR

    import torch_npu  # noqa: F401

    print(f"\n{'='*60}")
    print("Deepstack Integration: Vision Feature Extraction")
    print(f"{'='*60}")
    print(f"Model dir: {model_dir}")

    # ── Load config ───────────────────────────────────────────────
    cfg = load_config(model_dir)
    v_cfg = cfg["vision_config"]
    t_cfg = cfg["text_config"]
    ds_indexes = v_cfg["deepstack_visual_indexes"]

    hs = v_cfg["hidden_size"]       # 1024
    nh_v = v_cfg["num_heads"]       # 16
    hd_v = hs // nh_v               # 64
    merge_size = v_cfg["spatial_merge_size"]  # 2
    v_depth = v_cfg["depth"]        # 24
    num_grid = int(v_cfg["num_position_embeddings"] ** 0.5)  # 48

    nh_t = t_cfg["num_attention_heads"]   # 32
    nkv_t = t_cfg["num_key_value_heads"]  # 4
    hd_t = t_cfg["head_dim"]              # 128
    hidden_t = t_cfg["hidden_size"]       # 2048
    interm_t = t_cfg["intermediate_size"] # 6144

    print(f"  Vision: hs={hs}  nh={nh_v}  hd={hd_v}  depth={v_depth}  "
          f"merge={merge_size}")
    print(f"  Deepstack indexes: {ds_indexes}")
    print(f"  Text:   hs={hidden_t}  nh={nh_t}  nkv={nkv_t}  hd={hd_t}")

    # ── Load weights ─────────────────────────────────────────────
    weights = load_weights(model_dir)

    v_block_weights = [
        [to_npu_half(w) for w in get_vision_block_weights(weights, i)]
        for i in range(v_depth)
    ]
    v_pe_w, v_pe_b = get_patch_embed_weights(weights, hs)
    v_pe_w = to_npu_half(v_pe_w)
    v_pe_b = to_npu_half(v_pe_b)
    v_pos_embed = get_vision_pos_embed(weights)
    v_pe_w_table = to_npu_half(v_pos_embed)

    v_merger_w = [to_npu_half(w) for w in
                  get_merger_weights(weights, is_deepstack=False)]
    v_ds_w = [
        [to_npu_half(w) for w in
         get_merger_weights(weights, is_deepstack=True, ds_idx=i)]
        for i in range(len(ds_indexes))
    ]

    # ── Build ATB vision graphs ──────────────────────────────────
    vision_config = type('VisionConfig', (), {
        'hidden_size': hs,
        'num_heads': nh_v,
        'intermediate_size': v_cfg["intermediate_size"],
        'depth': v_depth,
        'patch_size': v_cfg["patch_size"],
        'temporal_patch_size': v_cfg["temporal_patch_size"],
        'spatial_merge_size': merge_size,
        'in_channels': v_cfg["in_channels"],
        'out_hidden_size': v_cfg["out_hidden_size"],
        'num_position_embeddings': v_cfg["num_position_embeddings"],
        'deepstack_visual_indexes': ds_indexes,
        'hidden_act': v_cfg.get("hidden_act", "gelu_pytorch_tanh"),
    })()

    g_v_first = build_vision_first_layer(vision_config)
    _, g_v_block, _ = build_vision_block(nh_v, hd_v, "VisBlock")
    g_v_merger = build_vision_merger(vision_config)
    g_v_ds = build_deepstack_merger(vision_config)
    g_v_posemb = build_vision_posemb_graph()

    # ── Load TF reference (half precision — ATB graphs stay loaded) ──
    ref = load_tf_ref(model_dir, precision="half")

    # ── Test image ───────────────────────────────────────────────
    img = Image.new('RGB', (120, 200), color='red')
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)

    from atb_python_qwen3vl_embedding.preprocess import preprocess_image
    pv_atb, gth_atb = preprocess_image(
        img_arr, patch_size=v_cfg["patch_size"],
        temporal_patch_size=v_cfg["temporal_patch_size"],
        merge_size=merge_size)

    # TF pixel_values (from processor)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_dir)
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}]}]
    tf_in = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True)
    pv_tf = tf_in['pixel_values']
    gth_tf = tf_in['image_grid_thw']

    # ── ATB: run vision pipeline ─────────────────────────────────
    print("\n[ATB] Running vision pipeline ...")

    vis_rotary = VisionRotaryEmbedding(dim=hd_v // 2)
    idx_wt = compute_posemb_indices(gth_atb, num_grid, merge_size)
    rope_idx = compute_rope_indices(gth_atb, vis_rotary, merge_size)
    freq_npu = to_npu_half(rope_idx['freq_table'])

    pos_npu, cos_npu, sin_npu = run_posemb_npu(
        g_v_posemb, v_pe_w_table, idx_wt, rope_idx, freq_npu)

    pv_npu = to_npu_half(pv_atb.reshape(-1)
                         if pv_atb.ndim == 2 else pv_atb)
    npatches = idx_wt['idx00'].shape[0]
    seqlen_v = make_seqlen_tensor(npatches)

    torch.npu.synchronize()
    h = run_first_layer_npu(g_v_first, pv_npu, v_pe_w, v_pe_b,
                            pos_npu, cos_npu, sin_npu, v_block_weights[0],
                            seqlen_v)

    ds_atb = []
    for li in range(1, v_depth):
        torch.npu.synchronize()
        h = run_block_npu(g_v_block, h, v_block_weights[li],
                          cos_npu, sin_npu, seqlen_v)
        if li in ds_indexes:
            ds_idx = ds_indexes.index(li)
            ds_out = run_merger_npu(g_v_ds, h, v_ds_w[ds_idx])
            ds_atb.append(to_cpu_float(ds_out))
            print(f"  Deepstack[{li}] ATB shape: {tuple(ds_out.shape)}")

    # ── TF: run vision pipeline ──────────────────────────────────
    print("\n[TF] Running vision pipeline ...")
    with torch.no_grad():
        vis_tf_out = ref.visual(ref.place(pv_tf), grid_thw=gth_tf.to(ref.device))
        ds_tf = [to_cpu_float(d) for d in vis_tf_out[1]] if vis_tf_out[1] else []
        for i, d in enumerate(ds_tf):
            print(f"  Deepstack[{ds_indexes[i]}] TF  shape: {tuple(d.shape)}")

    # ── Compare deepstack features ───────────────────────────────
    print(f"\n{'─'*55}")
    print(f"{'Index':<10} {'ATB shape':<22} {'cosine':>10} {'status':>8}")
    print(f"{'─'*55}")
    all_ok = True
    for i in range(len(ds_atb)):
        da, dt = ds_atb[i], ds_tf[i]
        cs = cosine(da, dt)
        ok = cs >= 0.99  # 0.99: moderate fp16 accumulation (deepstack extraction) — see THRESHOLDS.md
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"  ds[{ds_indexes[i]}]    {str(tuple(da.shape)):<22} "
              f"{cs:>10.6f} {status:>8}")

        # Also print max_diff for diagnostics
        mse = F.mse_loss(da.float(), dt.float()).item()
        maxd = (da.float() - dt.float()).abs().max().item()
        print(f"          MSE={mse:.8f}  max_diff={maxd:.8f}")

    print(f"{'─'*55}")
    return all_ok


def test_deepstack_injection(model_dir=None):
    """Test deepstack injection into text decoder layers.

    Runs a few text decoder layers on both ATB and TF, with deepstack features
    injected at visual token positions. Compares hidden states after injection.
    """
    if model_dir is None:
        model_dir = QWEN3VL_EMB_MODEL_DIR

    import torch_npu  # noqa: F401

    print(f"\n{'='*60}")
    print("Deepstack Integration: Text Injection")
    print(f"{'='*60}")

    # ── Load config and weights ──────────────────────────────────
    cfg = load_config(model_dir)
    v_cfg = cfg["vision_config"]
    t_cfg = cfg["text_config"]
    ds_indexes = v_cfg["deepstack_visual_indexes"]
    img_tok = cfg["image_token_id"]
    merge_size = v_cfg["spatial_merge_size"]

    nh_t = t_cfg["num_attention_heads"]
    nkv_t = t_cfg["num_key_value_heads"]
    hd_t = t_cfg["head_dim"]
    hidden_t = t_cfg["hidden_size"]
    interm_t = t_cfg["intermediate_size"]
    n_layer = t_cfg["num_hidden_layers"]

    weights = load_weights(model_dir)

    embed_w = get_embed_weight(weights)
    norm_w = get_text_norm_weight(weights).half().npu()
    t_layer_weights = [
        [to_npu_half(w) for w in get_text_layer_weights(weights, i)]
        for i in range(n_layer)
    ]

    # ── Load TF reference (half precision — ATB graphs stay loaded) ──
    ref = load_tf_ref(model_dir, precision="half")

    # ── Generate deepstack features from TF ──────────────────────
    # (We use TF deepstack features so the comparison isolates text injection logic)
    img = Image.new('RGB', (120, 200), color='red')
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_dir)
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Describe.'}]}]
    tf_in = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True)

    input_ids = tf_in['input_ids']
    pv_tf = tf_in['pixel_values']
    gth_tf = tf_in['image_grid_thw']
    S = input_ids.shape[1]

    # Get deepstack features from TF vision
    with torch.no_grad():
        vis_tf_out = ref.visual(ref.place(pv_tf), grid_thw=gth_tf.to(ref.device))
        ds_tf = vis_tf_out[1] if vis_tf_out[1] else []
    print(f"  TF deepstack features: {len(ds_tf)}")

    # ── Build inputs_embeds with image injection ─────────────────
    vis_mask_1d = (input_ids.squeeze(0) == img_tok)

    # Get TF vision merged output
    with torch.no_grad():
        vis_tf_merged = vis_tf_out[0]
        # Use TF's own image injection (same as pipeline_trace)
        ie_tf = ref.get_input_embeddings()(input_ids.to(ref.device)).to(ref.dtype)
        tok_emb = ref.get_input_embeddings()(
            torch.tensor(img_tok, dtype=torch.long, device=ie_tf.device))
        special_image_mask = (ie_tf == tok_emb)
        image_mask = special_image_mask.all(-1, keepdim=True).expand_as(ie_tf)
        ie_tf = ie_tf.masked_scatter(image_mask, vis_tf_merged.to(ie_tf.device, ie_tf.dtype))

    # ATB inputs_embeds (same as TF for fair comparison)
    ie_atb = F.embedding(input_ids, embed_w).half().npu()
    vis_atb_emb = to_npu_half(to_cpu_float(vis_tf_merged))
    ie_atb[0, vis_mask_1d.npu(), :] = vis_atb_emb

    # ── Position IDs and RoPE ────────────────────────────────────
    atb_pid, _ = get_rope_index(
        input_ids, gth_tf, None, None,
        image_token_id=img_tok,
        spatial_merge_size=merge_size)

    text_rope = TextRotaryEmbedding(
        head_dim=hd_t,
        rope_theta=t_cfg.get("rope_theta", 5000000.0),
        mrope_section=tuple(t_cfg["rope_scaling"]["mrope_section"]),
    )
    atb_cos, atb_sin = text_rope(atb_pid)
    cos_npu = to_npu_half(atb_cos.reshape(-1, hd_t))
    sin_npu = to_npu_half(atb_sin.reshape(-1, hd_t))

    with torch.no_grad():
        dummy = torch.zeros(1, S, hidden_t, device=ref.device, dtype=ref.dtype)
        tf_pid, _ = ref.get_rope_index(input_ids, gth_tf, None, None)
        tf_cos, tf_sin = ref.language_model.rotary_emb(dummy, tf_pid.float().to(ref.device))
    tf_cos_npu = tf_cos.to(ref.device, ref.dtype)
    tf_sin_npu = tf_sin.to(ref.device, ref.dtype)

    # ── Build ATB text graphs ────────────────────────────────────
    g_t_layer = build_text_layer_graph(
        nh_t, nkv_t, hd_t, interm_t, B=1, S=S, use_mask=True)
    g_t_norm = build_text_norm_graph(hidden_t)

    causal_mask = make_causal_mask(S).half().npu()
    if is_310p():
        causal_mask_atb = make_causal_mask_nz_npu(S)
    else:
        causal_mask_atb = causal_mask
    # TF attention mask follows ref's device/dtype (causal_mask itself is ATB-only, NPU).
    causal_mask_tf = make_causal_mask(S).to(ref.device, ref.dtype).unsqueeze(0).unsqueeze(0)

    # ── Run text layers with deepstack injection ─────────────────
    print(f"\n{'─'*70}")
    print(f"{'Layer':<8} {'cosine':>10} {'status':>8}  note")
    print(f"{'─'*70}")

    hidden_atb = ie_atb
    hidden_tf = ie_tf
    vis_mask_2d = vis_mask_1d.unsqueeze(0).to(ref.device)
    seqlen_t = make_seqlen_tensor(S)

    all_ok = True
    ds_idx = 0  # which deepstack feature to inject

    for li in range(n_layer):
        # ATB layer
        hidden_atb = run_text_layer_npu(
            g_t_layer, hidden_atb, t_layer_weights[li],
            cos_npu, sin_npu, seqlen_t, causal_mask=causal_mask_atb)

        # TF layer
        with torch.no_grad():
            hidden_tf = ref.language_model.layers[li](
                hidden_tf,
                position_embeddings=(tf_cos_npu, tf_sin_npu),
                attention_mask=causal_mask_tf)

        # Deepstack injection
        if ds_idx < len(ds_tf):
            # ATB injection
            ds_npu_atb = to_npu_half(ds_tf[ds_idx])
            local_atb = hidden_atb[0, vis_mask_1d.npu(), :].clone() + ds_npu_atb
            hidden_atb[0, vis_mask_1d.npu(), :] = local_atb

            # TF injection (replicate _deepstack_process)
            ds_npu_tf = ds_tf[ds_idx].to(ref.device, ref.dtype)
            local_tf = hidden_tf[vis_mask_2d, :].clone() + ds_npu_tf
            hidden_tf[vis_mask_2d, :] = local_tf

            ds_idx += 1

        torch.npu.synchronize()

        # Log periodically and at deepstack injection points
        if li == 0 or li == n_layer - 1 or li % 8 == 0 or (li < len(ds_tf)):
            cs = cosine(hidden_atb.cpu().float(), hidden_tf.cpu().float())
            is_ds = " [+ds]" if li < len(ds_tf) else ""
            ok = cs >= 0.99  # 0.99: moderate fp16 accumulation (multi-layer injection) — see THRESHOLDS.md
            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            print(f"  layer {li:2d}{is_ds:<8} {cs:>10.6f} {status:>8}")

    # Final norm comparison
    out_atb = run_text_norm_npu(g_t_norm, hidden_atb, norm_w).cpu().float()
    with torch.no_grad():
        out_tf = ref.language_model.norm(hidden_tf).cpu().float()
    cs_final = cosine(out_atb, out_tf)
    # 0.99: moderate fp16 accumulation (final norm after multi-layer injection) — see THRESHOLDS.md
    ok_final = cs_final >= 0.99
    all_ok &= ok_final
    print(f"  {'─'*50}")
    print(f"  {'final':<8} {cs_final:>10.6f} {'PASS' if ok_final else 'FAIL':>8}  (norm)")

    print(f"{'─'*70}")
    return all_ok


if __name__ == "__main__":
    model_dir = QWEN3VL_EMB_MODEL_DIR
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]

    ok1 = test_deepstack_extraction(model_dir)
    ok2 = test_deepstack_injection(model_dir)
    exit(0 if (ok1 and ok2) else 1)
