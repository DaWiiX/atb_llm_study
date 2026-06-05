"""End-to-end test for Qwen3VLEngine.

Validates the **pure ATB inference path** on three input modes:
  1. Text-only
  2. Image-only
  3. Image + Text

ATB side is completely self-contained — `pixel_values`, `cos/sin`,
`position_ids`, and the causal mask are all computed inside the engine
from raw inputs (image tensor + token IDs). No intermediate data is
borrowed from transformers.

When `--no-ref` is **not** passed (default), the transformers reference
model is loaded *after* the ATB engine is released to avoid NPU OOM,
and `last_hidden_state` is compared by cosine similarity.

Usage::

    python tests/test_e2e.py                  # ATB + TF cosine comparison
    python tests/test_e2e.py --no-ref         # ATB only, sanity check
    python tests/test_e2e.py --threshold 0.95
"""
# ── Buffer size MUST be set before any engine/graph import ──────────
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size

set_atb_buffer_size(5 * 1024 * 1024 * 1024)  # 5 GB

# ── Standard imports ───────────────────────────────────────────────
import argparse
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 — required for .npu() ops
from PIL import Image

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR


# ═══════════════════════════════════════════════════════════════════
# Test case definition
# ═══════════════════════════════════════════════════════════════════

def build_cases():
    """Return list of (name, messages, raw_image_or_None) test cases."""
    img_only = Image.new('RGB', (120, 200), color='red')
    img_mix = Image.new('RGB', (64, 64), color='blue')

    return [
        (
            "Text-Only",
            [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'What is the capital of France?'}]}],
            None,
        ),
        (
            "Image-Only",
            [{'role': 'user', 'content': [
                {'type': 'image', 'image': img_only}]}],
            img_only,
        ),
        (
            "Image+Text",
            [{'role': 'user', 'content': [
                {'type': 'image', 'image': img_mix},
                {'type': 'text', 'text': 'Describe.'}]}],
            img_mix,
        ),
    ]


def prepare_inputs(case, processor, engine, keep_tf_pv: bool):
    """Tokenise via processor (token IDs only) and preprocess image via engine.

    ``keep_tf_pv`` controls whether to also retain transformers'
    ``pixel_values`` / ``image_grid_thw``, which the TF reference forward
    expects.  When False (``--no-ref`` mode), those keys are omitted.
    """
    name, msgs, raw_img = case
    tf_in = processor.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True)

    pv_raw = grid_thw = None
    if raw_img is not None:
        img_arr = torch.from_numpy(np.array(raw_img)).permute(2, 0, 1)
        pv_raw, grid_thw = engine.preprocess_image(img_arr)

    out = {
        'name': name,
        'input_ids': tf_in['input_ids'],
        'attention_mask': tf_in.get('attention_mask',
                                    torch.ones_like(tf_in['input_ids'])),
        'pv_raw': pv_raw,
        'grid_thw': grid_thw,
    }
    if keep_tf_pv:
        out['tf_pixel_values'] = tf_in.get('pixel_values')
        out['tf_grid_thw'] = tf_in.get('image_grid_thw')
    return out


# ═══════════════════════════════════════════════════════════════════
# Phase 1: ATB engine forward
# ═══════════════════════════════════════════════════════════════════

def run_atb_phase(model_dir: str, processor, cases, keep_tf_pv: bool):
    """Load engine, prepare inputs, run engine.forward() per case, release."""
    print("[ATB] Loading Qwen3VLEngine ...")
    engine = Qwen3VLEngine(model_dir)
    print(f"[ATB] Loaded: {engine.n_layer} text layers, "
          f"{engine.v_depth} vision blocks")

    inputs_all = [prepare_inputs(c, processor, engine, keep_tf_pv)
                  for c in cases]

    results = {}
    for inputs in inputs_all:
        name = inputs['name']
        S = inputs['input_ids'].shape[1]
        out = engine.forward(inputs['input_ids'],
                             inputs['pv_raw'],
                             inputs['grid_thw'])
        results[name] = out  # (1, S, hidden) float32 on CPU
        print(f"[ATB] {name:<12} S={S:<5} → {tuple(out.shape)}")

    # Release engine + free NPU buffers before TF model is loaded.
    del engine
    torch.npu.empty_cache()

    return results, inputs_all


# ═══════════════════════════════════════════════════════════════════
# Phase 2: transformers reference
# ═══════════════════════════════════════════════════════════════════

def load_tf_ref(model_dir: str):
    """Load Qwen3VLModel on NPU with weights from safetensors."""
    import safetensors.torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"

    ref = Qwen3VLModel(cfg).eval().half().npu()
    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors",
                                     device="cpu")
    sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (
        f"weight mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    return ref


def run_tf_phase(model_dir: str, inputs_all):
    """Load TF ref and run forward per case, return dict[name] = hidden CPU fp32."""
    print("\n[TF] Loading transformers reference ...")
    ref = load_tf_ref(model_dir)
    print("[TF] Loaded")

    results = {}
    for inputs in inputs_all:
        name = inputs['name']
        input_ids = inputs['input_ids'].npu()
        attn_mask = inputs['attention_mask'].npu()
        pv = inputs.get('tf_pixel_values')
        gth = inputs.get('tf_grid_thw')
        kwargs = {'input_ids': input_ids, 'attention_mask': attn_mask}
        if pv is not None and gth is not None:
            kwargs['pixel_values'] = pv.half().npu()
            kwargs['image_grid_thw'] = gth.npu()

        with torch.no_grad():
            out = ref(use_cache=False, **kwargs).last_hidden_state.cpu().float()
        results[name] = out
        print(f"[TF]  {name:<12} → {tuple(out.shape)}")

    del ref
    torch.npu.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# Compare & report
# ═══════════════════════════════════════════════════════════════════

def compare(atb: dict, tf: dict, threshold: float) -> bool:
    print(f"\n{'─' * 55}")
    print(f"{'Case':<14} {'ATB shape':<22} {'cosine':>10} {'status':>8}")
    print(f"{'─' * 55}")
    all_pass = True
    for name in atb:
        a, t = atb[name], tf[name]
        if a.shape != t.shape:
            print(f"{name:<14} SHAPE MISMATCH "
                  f"atb={tuple(a.shape)} tf={tuple(t.shape)}")
            all_pass = False
            continue
        cs = F.cosine_similarity(a.flatten(), t.flatten(), dim=0).item()
        ok = cs > threshold
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"{name:<14} {str(tuple(a.shape)):<22} "
              f"{cs:>10.6f} {status:>8}")
    print(f"{'─' * 55}")
    return all_pass


def sanity_check(atb: dict) -> bool:
    """Verify outputs are well-formed when no TF reference is available."""
    print(f"\n{'─' * 55}")
    print(f"{'Case':<14} {'shape':<22} {'L2 norm':>12} {'status':>6}")
    print(f"{'─' * 55}")
    all_ok = True
    for name, h in atb.items():
        nan = torch.isnan(h).any().item()
        nrm = h.norm().item()
        ok = (h.dim() == 3 and nrm > 0 and not nan)
        all_ok &= ok
        status = "OK" if ok else "BAD"
        print(f"{name:<14} {str(tuple(h.shape)):<22} {nrm:>12.4f} {status:>6}")
    print(f"{'─' * 55}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    p.add_argument('--no-ref', dest='ref', action='store_false',
                   help='Skip transformers reference comparison')
    p.add_argument('--threshold', type=float, default=0.99,
                   help='Cosine similarity threshold for PASS (default 0.99)')
    p.add_argument('--model-dir', default=QWEN3VL_EMB_MODEL_DIR,
                   help='Override QWEN3VL_EMB_MODEL_DIR from .env')
    p.set_defaults(ref=True)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    print(f"Model dir: {args.model_dir}")
    print(f"Mode: {'ATB + TF reference' if args.ref else 'ATB only (no ref)'}")

    # Processor is used only for tokenization & (when --ref) TF pixel_values.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir)

    cases = build_cases()
    atb_results, inputs_all = run_atb_phase(
        args.model_dir, processor, cases, keep_tf_pv=args.ref)

    if not args.ref:
        ok = sanity_check(atb_results)
        return 0 if ok else 1

    tf_results = run_tf_phase(args.model_dir, inputs_all)
    ok = compare(atb_results, tf_results, args.threshold)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
