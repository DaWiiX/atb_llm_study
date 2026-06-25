"""Test preprocessing: pure Python vs transformers Processor.

Provides the shared compare_preprocess_with_tf() function that
test_vision_diagnostics.py also uses, so the preprocessing comparison
logic is defined once.
"""
import os, torch, torch.nn.functional as F
import warnings; warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import numpy as np
from PIL import Image

from atb_python_qwen3vl_embedding.preprocess import preprocess_image
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR


def compare_preprocess_with_tf(image: Image.Image, proc,
                                patch_size: int = 16,
                                temporal_patch_size: int = 2,
                                merge_size: int = 2,
                                min_pixels: int = 4096,
                                max_pixels: int = 1800 * 32 * 32):
    """Run ATB preprocess_image and transformers processor, compare results.

    Shared by test_preprocess.py and test_vision_diagnostics.py so that
    the preprocessing comparison logic lives in one place.

    Args:
        image: PIL Image.
        proc: transformers AutoProcessor.
        patch_size, temporal_patch_size, merge_size: as in preprocessor_config.json.
        min_pixels, max_pixels: pixel budget for SmartResize.

    Returns:
        dict with keys:
            passed: bool         — True when cosine >= 0.999 and grid_thw matches exactly
            cosine: float        — cosine similarity of pixel_values
            grid_match: bool     — whether image_grid_thw tensors are equal
            max_diff: float      — max absolute difference between pixel_values
            atb_pv: Tensor       — ATB pixel_values (N, C*tp*p*p)
            tf_pv:  Tensor       — transformers pixel_values
            atb_gth: Tensor      — ATB image_grid_thw (1, 3)
            tf_gth:  Tensor      — transformers image_grid_thw
    """
    img_arr = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # (C, H, W) uint8

    # ATB path
    atb_pv, atb_gth = preprocess_image(
        img_arr, patch_size=patch_size, temporal_patch_size=temporal_patch_size,
        merge_size=merge_size, min_pixels=min_pixels, max_pixels=max_pixels)

    # Transformers path
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": "t"}]}]
    tf_out = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                       return_tensors='pt')
    tf_pv = tf_out['pixel_values']
    tf_gth = tf_out['image_grid_thw']

    cs = F.cosine_similarity(tf_pv.float().flatten(), atb_pv.flatten(), dim=0).item()
    grid_match = torch.equal(tf_gth, atb_gth)
    max_diff = (atb_pv - tf_pv.float()).abs().max().item()

    return {
        'passed': cs > 0.999 and grid_match,
        'cosine': cs,
        'grid_match': grid_match,
        'max_diff': max_diff,
        'atb_pv': atb_pv,
        'tf_pv': tf_pv,
        'atb_gth': atb_gth,
        'tf_gth': tf_gth,
    }


def test_preprocess():
    """Compare ATB preprocessing against transformers Processor for multiple sizes."""
    print("\n=== Image Preprocessing ===")
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(QWEN3VL_EMB_MODEL_DIR)
    ip = proc.image_processor
    p = ip.patch_size; tp = ip.temporal_patch_size; m = ip.merge_size
    min_px = ip.min_pixels; max_px = ip.max_pixels

    # Multiple resolutions + edge cases
    test_images = [
        ("64x64-red",          Image.new('RGB', (64, 64), color='red')),
        ("120x200-blue",       Image.new('RGB', (120, 200), color='blue')),
        ("200x200-green",      Image.new('RGB', (200, 200), color='green')),
        ("16x16-black",        Image.new('RGB', (16, 16), color='black')),
        ("800x600-gradient",   Image.linear_gradient('L').convert('RGB')),
    ]

    all_ok = True
    for name, img in test_images:
        result = compare_preprocess_with_tf(
            img, proc, patch_size=p, temporal_patch_size=tp,
            merge_size=m, min_pixels=min_px, max_pixels=max_px)

        print(f"\n  [{name}]")
        print(f"    atb: {result['atb_pv'].shape} grid={result['atb_gth'].tolist()}"
              f"   tf: {result['tf_pv'].shape} grid={result['tf_gth'].tolist()}")
        print(f"    pixel_values cosine: {result['cosine']:.6f}")
        print(f"    grid_thw match: {result['grid_match']}")
        print(f"    max_diff: {result['max_diff']:.4f}")
        print("    PASS" if result['passed'] else "    FAIL")
        all_ok &= result['passed']

    assert all_ok, ("preprocess_image vs transformers Processor mismatch "
                    "(cosine < 0.999 or grid_thw differs; see per-case output above)")


if __name__ == "__main__":
    test_preprocess()
