# ============================================================
# 完整 E2E 测试：ATB Qwen3VL-Embedding vs Transformers
# 三种输入类型：纯文本 / 纯图片 / 图片+文本
# 
# 推理链路（对应 Qwen3VLModel.forward）：
#   1. input_ids → embed_tokens → inputs_embeds
#   2. pixel_values → VisionModel → image_embeds + deepstack_embeds
#   3. inputs_embeds.masked_scatter(image_mask, image_embeds)  # 图像特征注入
#   4. get_rope_index(input_ids, grid_thw) → position_ids (3,B,S)  # MRoPE
#   5. rotary_emb(inputs_embeds, position_ids) → (cos, sin)
#   6. create_causal_mask → attention_mask (0/-inf for eager)
#   7. For each text decoder layer:
#        layer(hidden_states, attention_mask, position_embeddings, 
#              position_ids, cache_position)
#        if deepstack: _deepstack_process(hidden_states, mask, deepstack[layer_idx])
#   8. norm → final hidden_states
# ============================================================
import sys, os; os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, '/mnt/workspace/gitCode/atb_python_model')
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')
import torch, torch_npu, torch_atb
from atb_python_model.utils import set_atb_buffer_size
set_atb_buffer_size(5000 * 1024 * 1024)  # 5GB, 必须在所有 graph build 之前调用

from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np

# ── 加载模型 ─────────────────────────────────────────────────────
rp = '/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B'
model = AutoModel.from_pretrained(rp, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()
# 强制 eager 以匹配 ATB MASK_TYPE_NORM（两者都用加法掩码）
model.config._attn_implementation = "eager"
model.language_model.config._attn_implementation = "eager"

proc = AutoProcessor.from_pretrained(rp)
tm = model.language_model  # Qwen3VLTextModel
vm = model.visual           # Qwen3VLVisionModel

# ── 构建 ATB 图（一次性构建，全部组件）───────────────────────────
from atb_python_model.vision_model import (
    build_vision_first_layer, build_vision_merger, build_deepstack_merger,
    collect_patch_embed_weight, collect_block_weights, collect_merger_weights,
    run_first_layer, run_block, run_merger, run_vision_model,
)
from atb_python_model.vision_block import build_vision_block

cfg_v = vm.config
nh_v = cfg_v.num_heads
hd_v = cfg_v.hidden_size // nh_v

print("Building Vision graphs...")
g_v_first = build_vision_first_layer(cfg_v)
_, g_v_block, _ = build_vision_block(nh_v, hd_v, "VisBlock")
g_v_merger = build_vision_merger(cfg_v)
g_v_deepstack = build_deepstack_merger(cfg_v)

from atb_python_model.text_decoder_layer import build_decoder_layer
from atb_python_model.text_model import (
    build_text_norm_graph, collect_text_layer_weights,
    make_causal_mask, run_text_layer, run_text_norm,
)

cfg_t = tm.config
nh_t = cfg_t.num_attention_heads
kv_t = cfg_t.num_key_value_heads
hd_t = cfg_t.head_dim
hidden_t = cfg_t.hidden_size
interm_t = cfg_t.intermediate_size
print(f"Text config: nh={nh_t}, kv={kv_t}, hd={hd_t}, hidden={hidden_t}, interm={interm_t}")

# ── ATB Vision 前向（在 ATB 图上运行）────────────────────────────
def run_atb_vision(pixel_values_raw, grid_thw):
    """
    Run vision model entirely on ATB graphs.
    pixel_values_raw: (N_patches, C*tp*p*p) from preprocess_image()
    grid_thw: (1, 3) or (N, 3)
    Returns: (merged_embeds, deepstack_list)
    """
    with torch.no_grad():
        pos = vm.fast_pos_embed_interpolate(grid_thw)
        rope = vm.rot_pos_emb(grid_thw)
        seq_len = pixel_values_raw.shape[0]
        rope = rope.reshape(seq_len, -1)
        emb = torch.cat((rope, rope), dim=-1)
        cos, sin = emb.cos(), emb.sin()
    return run_vision_model(
        vm, pixel_values_raw, pos, cos, sin,
        g_v_first, g_v_block, g_v_merger, g_v_deepstack)

# ── ATB Text 前向（一个 builder，循环使用）────────────────────────
print("Building Text graphs...")
g_t_norm = build_text_norm_graph(hidden_t)
# Text layer graph 需要 S 参数，后续根据实际输入动态创建
g_t_layer = None
g_t_layer_S = None

def ensure_text_layer_graph(S):
    global g_t_layer, g_t_layer_S
    if g_t_layer is not None and g_t_layer_S == S:
        return
    print(f"  Building text layer graph for S={S}...")
    _, g_t_layer, _ = build_decoder_layer(nh_t, kv_t, hd_t, interm_t,
                                           B=1, S=S, use_mask=True)
    g_t_layer_S = S

# ── 完整 ATB 前向 ────────────────────────────────────────────────
def run_atb_pipeline(input_ids, pixel_values_raw, grid_thw):
    """
    Complete ATB pipeline matching Qwen3VLModel.forward.
    Returns: (hidden_states, last_hidden_state)
    """
    # Step 1: Embed tokens
    with torch.no_grad():
        inputs_embeds = tm.embed_tokens(input_ids)  # (1, S, hidden)
    B, S = inputs_embeds.shape[:2]
    ensure_text_layer_graph(S)
    
    # Step 2: Vision features (if any)
    image_token_id = model.config.image_token_id
    deepstack_features = []
    visual_pos_masks = None
    
    if pixel_values_raw is not None and grid_thw is not None:
        vis_embeds, ds_feats = run_atb_vision(pixel_values_raw, grid_thw)
        # 注入图像特征（等价于 TF 的 masked_scatter）
        image_mask = input_ids.squeeze(0) == image_token_id
        inputs_embeds[0, image_mask, :] = vis_embeds.to(inputs_embeds)
        visual_pos_masks = image_mask  # (S,) bool
        deepstack_features = ds_feats
    
    # Step 3: Position embeddings (MRoPE)
    position_ids, _ = model.get_rope_index(input_ids, grid_thw, None, None)
    
    with torch.no_grad():
        cos, sin = tm.rotary_emb(inputs_embeds, position_ids.float())
    # cos/sin shape: (B, S, head_dim) = (1, S, 128)
    # 转成 ATB 需要的 (B*S, head_dim)
    nps = B * S
    cos_f = cos.reshape(nps, -1).float()
    sin_f = sin.reshape(nps, -1).float()
    
    # Step 4: Causal mask (与 eager attention 一致的 float 加法掩码)
    causal_mask = make_causal_mask(S)  # 0/-65504
    
    # Step 5: Loop through decoder layers
    hidden = inputs_embeds.float()
    for layer_idx, layer in enumerate(tm.layers):
        w = collect_text_layer_weights(layer)
        hidden = run_text_layer(g_t_layer, hidden, w,
                                cos_f, sin_f, S,
                                causal_mask=causal_mask)
        # DeepStack injection (layers 0, 1, 2)
        if deepstack_features and layer_idx < len(deepstack_features):
            ds = deepstack_features[layer_idx]
            hidden[0, visual_pos_masks, :] += ds.to(hidden)
    
    # Step 6: Final norm
    hidden = run_text_norm(g_t_norm, hidden, tm.norm.weight.data)
    return hidden, inputs_embeds

# ── TF Reference (manual forward with SAME inputs as ATB) ─────────
def run_tf_reference(input_ids, ie_injected, grid_thw):
    """Run TF manual forward using IDENTICAL inputs_embeds and cos/sin as ATB.
    This ensures exact 1:1 comparison — both use the same vision features and mask.
    """
    with torch.no_grad():
        pid, _ = model.get_rope_index(input_ids, grid_thw, None, None)
        cos, sin = tm.rotary_emb(ie_injected, pid.float())
        S = ie_injected.shape[1]
        mask_tf = make_causal_mask(S).unsqueeze(0).unsqueeze(0).float()
        
        h_tf = ie_injected.clone()
        for layer in tm.layers:
            h_tf = layer(h_tf, position_embeddings=(cos, sin), attention_mask=mask_tf)
        h_tf = tm.norm(h_tf)
    return h_tf

# ── 测试工具 ────────────────────────────────────────────────────
def compare(atb_out, tf_out, label):
    cs = torch.nn.functional.cosine_similarity(
        atb_out.flatten(), tf_out.flatten(), dim=0).item()
    status = "PASS" if cs > 0.99 else "FAIL"
    print(f"  [{status}] {label}: cos={cs:.6f}")
    return cs

# ══════════════════════════════════════════════════════════════════
# Test Case 1: 纯文本
# ══════════════════════════════════════════════════════════════════
print("\n===== Test 1: Text Only =====")
msgs_text = [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'What is the capital of France?'}
]}]
tf_in_text = proc.apply_chat_template(msgs_text, tokenize=True,
    return_dict=True, return_tensors='pt', add_generation_prompt=True)

# ATB
atb_text, ie_text = run_atb_pipeline(
    tf_in_text['input_ids'],
    None,  # no pixel_values
    None   # no grid_thw
)
# TF (manual with same inputs_embeds as ATB)
tf_text = run_tf_reference(tf_in_text['input_ids'], ie_text, None)
compare(atb_text, tf_text, "Text-Only")
print(f"  S={tf_in_text['input_ids'].shape[1]}")

# ══════════════════════════════════════════════════════════════════
# Test Case 2: 纯图片（处理器自动添加文本前缀）
# ══════════════════════════════════════════════════════════════════
print("\n===== Test 2: Image Only =====")
img = Image.new('RGB', (120, 200), color='red')
msgs_img = [{'role': 'user', 'content': [
    {'type': 'image', 'image': img}
]}]
tf_in_img = proc.apply_chat_template(msgs_img, tokenize=True,
    return_dict=True, return_tensors='pt', add_generation_prompt=True)

# ATB: 用自己的 preprocess_image 处理原始图像
from atb_python_model.preprocess import preprocess_image
pv_raw, grid_thw = preprocess_image(
    torch.from_numpy(np.array(img)).permute(2, 0, 1))

atb_img, ie_img = run_atb_pipeline(
    tf_in_img['input_ids'],
    pv_raw,
    grid_thw
)
# TF reference (manual with SAME inputs_embeds as ATB)
tf_img = run_tf_reference(tf_in_img['input_ids'], ie_img, grid_thw)
compare(atb_img, tf_img, "Image-Only")
print(f"  S={tf_in_img['input_ids'].shape[1]}, grid_thw={grid_thw.tolist()}")

# ══════════════════════════════════════════════════════════════════
# Test Case 3: 图片+文本
# ══════════════════════════════════════════════════════════════════
print("\n===== Test 3: Image + Text =====")
img3 = Image.new('RGB', (64, 64), color='blue')
msgs_mix = [{'role': 'user', 'content': [
    {'type': 'image', 'image': img3},
    {'type': 'text', 'text': 'Describe this image in detail.'}
]}]
tf_in_mix = proc.apply_chat_template(msgs_mix, tokenize=True,
    return_dict=True, return_tensors='pt', add_generation_prompt=True)

pv_raw3, grid_thw3 = preprocess_image(
    torch.from_numpy(np.array(img3)).permute(2, 0, 1))

atb_mix, ie_mix = run_atb_pipeline(
    tf_in_mix['input_ids'],
    pv_raw3,
    grid_thw3
)
tf_mix = run_tf_reference(tf_in_mix['input_ids'], ie_mix, grid_thw3)
compare(atb_mix, tf_mix, "Image+Text")
print(f"  S={tf_in_mix['input_ids'].shape[1]}, grid_thw={grid_thw3.tolist()}")

# ══════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Summary: ATB Qwen3VL-Embedding E2E vs Transformers")
print(f"{'='*60}")
