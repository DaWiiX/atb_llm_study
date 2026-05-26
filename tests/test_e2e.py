"""End-to-end test: full preprocessing + VisionModel with real weights.

Pipeline:
    1. preprocess_image(raw CxHxW uint8) → pixel_values, grid_thw
    2. preprocess_vision (CPU: pos_embed, cos/sin from vision_model)
    3. ATB VisionModel (first_layer → loop blocks → merger)
    4. Compare with transformers reference
"""
import sys, os, torch, torch.nn.functional as F, warnings
warnings.filterwarnings('ignore')

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_model.utils import set_atb_buffer_size, compare_tensors
from atb_python_model.vision_model import (
    build_vision_first_layer, build_vision_merger, run_vision_model,
)
from atb_python_model.vision_block import build_vision_block
from atb_python_model.preprocess import preprocess_image


def test_e2e():
    print("\n=== Full E2E: preprocess + VisionModel (24 layers, real weights) ===")

    from transformers import AutoModel
    from PIL import Image
    import numpy as np

    real_path = '/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B'
    rm = AutoModel.from_pretrained(real_path, trust_remote_code=True,
                                   torch_dtype=torch.float32)
    rm.eval()
    vm = rm.visual
    cfg = vm.config

    # Read preprocessor config for correct params
    import json
    with open(f'{real_path}/preprocessor_config.json') as f:
        pp = json.load(f)
    p = pp['patch_size']; tp = pp['temporal_patch_size']; m = pp['merge_size']
    min_px = pp['min_pixels']; max_px = pp['max_pixels']
    print(f"Vision: hs={cfg.hidden_size}, nh={cfg.num_heads}, depth={cfg.depth}")
    print(f"Preprocess: patch={p}, tp={tp}, merge={m}, min_px={min_px}, max_px={max_px}")

    # 1. Create raw image tensor (C, H, W) uint8
    img = Image.new('RGB', (64, 64), color='red')
    arr = np.array(img)
    img_tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # 2. Our preprocessing
    pv, gth = preprocess_image(img_tensor, patch_size=p, temporal_patch_size=tp,
                               merge_size=m, min_pixels=min_px, max_pixels=max_px)
    print(f"Preprocessed: pv={pv.shape}, grid_thw={gth}")

    # 3. Transformers reference (using processor for validation)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(real_path)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": "test"}]}]
    tf_inputs = proc.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors='pt')

    with torch.no_grad():
        ref, _ = vm(tf_inputs['pixel_values'], tf_inputs['image_grid_thw'])

    # 4. CPU pos_embed + cos/sin from vision model
    with torch.no_grad():
        pos = vm.fast_pos_embed_interpolate(gth)
        rope = vm.rot_pos_emb(gth)
        seq_len = pv.shape[0]
        rope = rope.reshape(seq_len, -1)
        emb = torch.cat((rope, rope), dim=-1)
        cos, sin = emb.cos(), emb.sin()

    # 5. Build ATB graphs
    set_atb_buffer_size(2048 * 1024 * 1024)
    g_first = build_vision_first_layer(cfg)
    nh_v = cfg.num_heads; hd_v = cfg.hidden_size // nh_v
    _, g_block, _ = build_vision_block(nh_v, hd_v, "VisBlockLoop")
    g_merger = build_vision_merger(cfg)

    # 6. Run VisionModel
    atb_out, _ = run_vision_model(vm, pv, pos, cos, sin,
                                   g_first, g_block, g_merger)

    compare_tensors(ref, atb_out, label="E2E-Vision-24L")
    cs = F.cosine_similarity(ref.flatten(), atb_out.flatten(), dim=0).item()
    return cs > 0.95


if __name__ == "__main__":
    ok = test_e2e()
    exit(0 if ok else 1)
