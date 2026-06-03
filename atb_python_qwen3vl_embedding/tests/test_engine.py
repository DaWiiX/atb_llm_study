# ============================================================
# Pure ATB Engine E2E Test — zero transformers in inference path
# ============================================================
import os; os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import torch, torch_npu, torch_atb
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
set_atb_buffer_size(15000 * 1024 * 1024)

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR
from atb_python_qwen3vl_embedding.preprocess import preprocess_image
from atb_python_qwen3vl_embedding.engine_utils import (
    TextRotaryEmbedding, get_rope_index, get_embed_weight,
)
from atb_python_qwen3vl_embedding.text_model import (
    run_text_layer, run_text_norm, make_causal_mask,
)
from PIL import Image
import numpy as np

# ── Load transformers reference (comparison only, not inference) ───
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers import AutoProcessor
rp = QWEN3VL_EMB_MODEL_DIR
# Load transformers reference — must NOT use dtype override, or weights won't load from checkpoint
# Load transformers reference — manually load state dict from safetensors (model has 'model.' prefix)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
cfg = Qwen3VLConfig.from_pretrained(rp, trust_remote_code=True)
model_ref = Qwen3VLModel(cfg).eval().float()
import safetensors.torch
state_dict = safetensors.torch.load_file(f"{rp}/model.safetensors", device="cpu")
state_dict = {k.removeprefix("model."): v.float() for k, v in state_dict.items()}
miss, unexp = model_ref.load_state_dict(state_dict, strict=False)
print(f"TF model loaded: {len(miss)} missing, {len(unexp)} unexpected")
model_ref.config._attn_implementation = "eager"
model_ref.language_model.config._attn_implementation = "eager"
tm = model_ref.language_model
proc = AutoProcessor.from_pretrained(rp)
im_tok = model_ref.config.image_token_id

# ── Init engine ─────────────────────────────────────────────────────
engine = Qwen3VLEngine(rp)
print(f"Engine loaded: {engine.n_layer} text layers, {engine.v_depth} vision blocks")


def test_case(name, input_ids, pv_raw, grid_thw):
    """Run ATB engine and compare with TF manual (same inputs, same mask)."""
    S = input_ids.shape[1]
    engine._ensure_text_graph(S)

    # ── Shared inputs: embeddings + vision ─────────────────────────
    embed_w = engine.embed_w
    ie = torch.nn.functional.embedding(input_ids, embed_w).float()  # (1, S, hidden)

    vis_mask = None
    if pv_raw is not None and grid_thw is not None:
        vis, _ = engine._run_vision(pv_raw, grid_thw)
        vis_mask = input_ids.squeeze(0) == im_tok
        ie[0, vis_mask, :] = vis.to(ie)

    # ── Shared position embeddings ─────────────────────────────────
    pid, _ = get_rope_index(input_ids, grid_thw, None, None,
                             image_token_id=im_tok,
                             spatial_merge_size=engine.spatial_merge)
    cos, sin = engine.text_rope(pid)
    cos_f = cos.reshape(-1, engine.hd_t)

    # ── Shared mask ────────────────────────────────────────────────
    cm = make_causal_mask(S)
    mask_tf = cm.unsqueeze(0).unsqueeze(0).float()
    cm_npu = cm.half().npu()  # ATB needs float16 on NPU

    # ── ATB forward ────────────────────────────────────────────────
    h_atb = ie.clone()
    for li in range(engine.n_layer):
        sin_f = sin.reshape(-1, engine.hd_t)
        h_atb = run_text_layer(engine.g_t_layer, h_atb,
                               engine.t_layer_weights[li],
                               cos_f, sin_f, S, causal_mask=cm_npu)
    h_atb = run_text_norm(engine.g_t_norm, h_atb, engine.norm_w)

    # ── TF manual forward (same inputs as ATB) ─────────────────────
    h_tf = ie.clone()
    for li in range(engine.n_layer):
        layer = tm.layers[li]
        h_tf = layer(h_tf, position_embeddings=(cos, sin),
                     attention_mask=mask_tf)
    h_tf = tm.norm(h_tf).float()

    cs = torch.nn.functional.cosine_similarity(
        h_atb.flatten(), h_tf.flatten(), dim=0).item()
    status = "PASS" if cs > 0.99 else "FAIL"
    print(f" [{status}] {name} (S={S}): cos={cs:.6f}")
    return cs


# ═══════════════════════════════════════════════════════════════════
# Test 1: Text-Only
# ═══════════════════════════════════════════════════════════════════
msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is the capital of France?'}]}]
tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors='pt',
                                  add_generation_prompt=True)
test_case("Text-Only", tf_in['input_ids'], None, None)

# ═══════════════════════════════════════════════════════════════════
# Test 2: Image-Only
# ═══════════════════════════════════════════════════════════════════
img = Image.new('RGB', (120, 200), color='red')
msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}]}]
tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors='pt',
                                  add_generation_prompt=True)
pv_raw, grid_thw = preprocess_image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
test_case("Image-Only", tf_in['input_ids'], pv_raw, grid_thw)

# ═══════════════════════════════════════════════════════════════════
# Test 3: Image+Text
# ═══════════════════════════════════════════════════════════════════
img3 = Image.new('RGB', (64, 64), color='blue')
msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img3},
                                      {'type': 'text', 'text': 'Describe.'}]}]
tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors='pt',
                                  add_generation_prompt=True)
pv_raw3, grid_thw3 = preprocess_image(torch.from_numpy(np.array(img3)).permute(2, 0, 1))
test_case("Image+Text", tf_in['input_ids'], pv_raw3, grid_thw3)

print("\nDone.")
