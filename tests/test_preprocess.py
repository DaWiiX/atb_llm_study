"""Test preprocessing: pure Python vs transformers Processor."""
import sys, os, torch, torch.nn.functional as F
_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg not in sys.path: sys.path.insert(0, _pkg)
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')
import warnings; warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_model.preprocess import preprocess_image

def test_preprocess():
    print("\n=== Image Preprocessing ===")
    from transformers import AutoProcessor
    from PIL import Image
    import numpy as np

    proc = AutoProcessor.from_pretrained('/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B')
    ip = proc.image_processor
    p = ip.patch_size; tp = ip.temporal_patch_size; m = ip.merge_size
    min_px = ip.min_pixels; max_px = ip.max_pixels

    img = Image.new('RGB', (64, 64), color='red')
    img_arr = np.array(img)  # (64, 64, 3) uint8

    # Transformers processor
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "t"}]}]
    tf_out = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors='pt')
    tf_pv = tf_out['pixel_values']
    tf_gth = tf_out['image_grid_thw']

    # Our pure-Python preprocessing
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # (3, 64, 64) uint8
    my_pv, my_gth = preprocess_image(img_tensor, patch_size=p, temporal_patch_size=tp,
                                     merge_size=m, min_pixels=min_px, max_pixels=max_px)

    print(f"transformers pv: {tf_pv.shape}, gth: {tf_gth}")
    print(f"ours       pv: {my_pv.shape}, gth: {my_gth}")

    cs = F.cosine_similarity(tf_pv.float().flatten(), my_pv.flatten(), dim=0)
    print(f"pixel_values cosine: {cs:.6f}")
    print(f"grid_thw match: {torch.equal(tf_gth, my_gth)}")
    print("PASS" if cs > 0.999 else "FAIL")

    return cs > 0.999

if __name__ == "__main__":
    ok = test_preprocess()
    exit(0 if ok else 1)
