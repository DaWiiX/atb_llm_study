# ============================================================
# E2E: ATB Qwen3VL-Embedding vs TF manual (same inputs)
# 三种输入：纯文本 / 纯图片 / 图片+文本
# ============================================================
import sys, os; os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, '/mnt/workspace/gitCode/atb_python_model')
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')
import torch, torch_npu, torch_atb
import torch.nn.functional as F
from atb_python_model.utils import set_atb_buffer_size
set_atb_buffer_size(5000 * 1024 * 1024)

from transformers import AutoModel, AutoProcessor
from PIL import Image; import numpy as np

rp = '/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B'
model = AutoModel.from_pretrained(rp, trust_remote_code=True, torch_dtype=torch.float32)
model.eval(); model.config._attn_implementation = "eager"
tm = model.language_model; tm.config._attn_implementation = "eager"; vm = model.visual
proc = AutoProcessor.from_pretrained(rp)
cfg_t = tm.config
nh_t, kv_t, hd_t, hidden_t, interm_t = (cfg_t.num_attention_heads, cfg_t.num_key_value_heads,
    cfg_t.head_dim, cfg_t.hidden_size, cfg_t.intermediate_size)

# ── Build all ATB graphs once ─────────────────────────────────────
from atb_python_model.vision_model import (build_vision_first_layer, build_vision_merger,
    build_deepstack_merger, run_vision_model)
from atb_python_model.vision_block import build_vision_block
from atb_python_model.preprocess import preprocess_image
from atb_python_model.text_decoder_layer import build_decoder_layer
from atb_python_model.text_model import (build_text_norm_graph, collect_text_layer_weights,
    make_causal_mask, run_text_layer, run_text_norm)

cfg_v = vm.config; nh_v = cfg_v.num_heads; hd_v = cfg_v.hidden_size // nh_v
g_v_first = build_vision_first_layer(cfg_v)
_, g_v_block, _ = build_vision_block(nh_v, hd_v, "VisBlock")
g_v_merger = build_vision_merger(cfg_v)
g_v_deepstack = build_deepstack_merger(cfg_v)
g_t_norm = build_text_norm_graph(hidden_t)

# ── Helper: run vision + inject ───────────────────────────────────
def get_vision_and_inject(input_ids, pv_raw, grid_thw):
    with torch.no_grad():
        pos = vm.fast_pos_embed_interpolate(grid_thw)
        rope = vm.rot_pos_emb(grid_thw).reshape(pv_raw.shape[0], -1)
        emb = torch.cat((rope, rope), dim=-1); cos_v, sin_v = emb.cos(), emb.sin()
    vis, ds = run_vision_model(vm, pv_raw, pos, cos_v, sin_v,
        g_v_first, g_v_block, g_v_merger, g_v_deepstack)
    ie = tm.embed_tokens(input_ids).float()
    im = input_ids.squeeze(0) == model.config.image_token_id
    ie[0, im, :] = vis.to(ie)
    return ie, ds, im

# ── Test one case ──────────────────────────────────────────────────
def test_case(name, input_ids, pv_raw, grid_thw):
    S = input_ids.shape[1]
    print(f"\n{'='*50}")
    print(f" {name} (S={S}" + (f", grid={grid_thw.tolist()})" if grid_thw is not None else ""))
    
    # Build text layer graph
    _, g_tl, _ = build_decoder_layer(nh_t, kv_t, hd_t, interm_t, B=1, S=S, use_mask=True)
    
    # Prepare inputs
    if pv_raw is not None and grid_thw is not None:
        ie, ds_feats, im_mask = get_vision_and_inject(input_ids, pv_raw, grid_thw)
    else:
        ie = tm.embed_tokens(input_ids).float()
        ds_feats, im_mask = [], None
    
    # Position embeddings
    pid, _ = model.get_rope_index(input_ids, grid_thw, None, None)
    cos, sin = tm.rotary_emb(ie, pid.float())
    cos_f = cos.reshape(-1, hd_t).float(); sin_f = sin.reshape(-1, hd_t).float()
    
    # Mask (same for ATB and TF)
    atb_mask = make_causal_mask(S)
    tf_mask = atb_mask.unsqueeze(0).unsqueeze(0).float()
    
    # ATB uses (S,) bool mask for deepstack: hidden[0, mask, :] += ds
    # TF needs (B, S) for _deepstack_process
    if im_mask is not None:
        im_mask_2d = im_mask.unsqueeze(0)  # (1, S) for TF
    else:
        im_mask_2d = None
    
    # ── ATB Forward ────────────────────────────────────────────
    h_atb = ie.clone()
    for li, layer in enumerate(tm.layers):
        w = collect_text_layer_weights(layer)
        h_atb = run_text_layer(g_tl, h_atb, w, cos_f, sin_f, S, causal_mask=atb_mask)
        if ds_feats and li < len(ds_feats):
            h_atb[0, im_mask, :] += ds_feats[li].to(h_atb)
    h_atb = run_text_norm(g_t_norm, h_atb, tm.norm.weight.data)
    
    # ── TF Manual Forward (same inputs as ATB) ─────────────────
    h_tf = ie.clone()
    for li, layer in enumerate(tm.layers):
        h_tf = layer(h_tf, position_embeddings=(cos, sin), attention_mask=tf_mask)
        if ds_feats and li < len(ds_feats):
            h_tf = tm._deepstack_process(h_tf, im_mask_2d, ds_feats[li])
    h_tf = tm.norm(h_tf)
    
    cs = F.cosine_similarity(h_atb.flatten(), h_tf.flatten(), dim=0).item()
    status = "PASS" if cs > 0.99 else "FAIL"
    print(f" [{status}] cos={cs:.6f}")
    return cs

# ══════════════════════════════════════════════════════════════════
# Test Case 1: 纯文本
# ══════════════════════════════════════════════════════════════════
msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is the capital of France?'}]}]
tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors='pt',
                                  add_generation_prompt=True)
test_case("Text-Only", tf_in['input_ids'], None, None)

# ══════════════════════════════════════════════════════════════════
# Test Case 2: 纯图片
# ══════════════════════════════════════════════════════════════════
img = Image.new('RGB', (120, 200), color='red')
msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}]}]
tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors='pt',
                                  add_generation_prompt=True)
pv_raw, grid_thw = preprocess_image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
test_case("Image-Only", tf_in['input_ids'], pv_raw, grid_thw)

# ══════════════════════════════════════════════════════════════════
# Test Case 3: 图片+文本
# ══════════════════════════════════════════════════════════════════
img3 = Image.new('RGB', (64, 64), color='blue')
msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img3}, {'type': 'text', 'text': 'Describe.'}]}]
tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors='pt',
                                  add_generation_prompt=True)
pv_raw3, grid_thw3 = preprocess_image(torch.from_numpy(np.array(img3)).permute(2, 0, 1))
test_case("Image+Text", tf_in['input_ids'], pv_raw3, grid_thw3)

print(f"\n{'='*50}")
print(" Done.")
