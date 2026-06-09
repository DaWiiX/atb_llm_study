"""End-to-end embedder comparison: ATB engine vs Transformers reference.

Supports three modes:
  - ``--mode atb``     ATB engine only (benchmark)
  - ``--mode tf``      Transformers reference only (benchmark)
  - ``--mode both``    Compare ATB vs TF cosine similarity (default)

Supports two test suites:
  - Quick (default): 3 simple cases (text-only, image-only, image+text)
  - Full (``--bench``): 13-combination benchmark matrix
    5 TEXT_ONLY (S=100,512,1024,2048,4096) +
    4 IMAGE_ONLY + 4 IMAGE_AND_TEXT (416×672, 720×1280, 1080×1920, 1440×2560)

Both paths produce last-token pooled + L2-normalised embeddings.
Cosine ≥ 0.99 threshold for PASS.

Usage::

    # Quick smoke test (ATB vs TF)
    python tests/test_embedder_e2e.py

    # Full 13/13 benchmark — ATB only
    python tests/test_embedder_e2e.py --mode atb --bench --iter 5 --warmup 3

    # Full 13/13 benchmark — TF only
    python tests/test_embedder_e2e.py --mode tf --bench --iter 5 --warmup 3

    # Full 13/13 benchmark — both, with comparison
    python tests/test_embedder_e2e.py --mode both --bench --iter 5 --warmup 3
"""
# ── Buffer size MUST be set before any engine/graph import ──────────
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
set_atb_buffer_size(5 * 1024 * 1024 * 1024)  # 5 GB

# ── Standard imports ───────────────────────────────────────────────
import argparse
import struct
import sys
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from PIL import Image

from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR, QWEN3VL_EMB_SRC
sys.path.insert(0, QWEN3VL_EMB_SRC)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
BIN_DIR = "/tmp"

RESOLUTIONS = [(416, 672), (720, 1280), (1080, 1920), (1440, 2560)]
TEXT_SEQ_LENS = [100, 512, 1024, 2048, 4096]
BASE_TEXT = (
    "Please provide a comprehensive analysis of the image. "
    "Identify all visible objects, people, animals, and their spatial relationships, "
    "including foreground and background elements, lighting conditions, color palette, "
    "mood, and any notable artistic style or composition techniques. "
) * 14  # ~645 raw text tokens; ~506 with chat template wrapping

# Known grid_thw for each resolution (from C++ SmartResize output)
GRID_MAP = {
    (416, 672):   (1, 42, 26),
    (720, 1280):  (1, 80, 44),
    (1080, 1920): (1, 94, 52),
    (1440, 2560): (1, 94, 52),
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def now() -> float:
    return time.perf_counter()


def sync() -> None:
    if torch.npu.is_available():
        torch.npu.synchronize()


def ms(arr) -> np.ndarray:
    return np.array(arr) * 1000


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def load_token_ids(path: str) -> List[int]:
    """Load int32-count-prefixed int64 token IDs from binary file."""
    with open(path, "rb") as f:
        (count,) = struct.unpack("<i", f.read(4))
        fmt = f"<{count}q"
        return list(struct.unpack(fmt, f.read(count * 8)))


def load_pixel_values(path: str) -> np.ndarray:
    """Load int32-count-prefixed fp16 pixel_values from binary file.

    C++ saves fp16 values as raw uint16 bytes.  We unpack as uint16 then
    reinterpret the bits as float16 — a CAST (astype) would divide the
    integer values, which is wrong.
    """
    with open(path, "rb") as f:
        (count,) = struct.unpack("<i", f.read(4))
        fmt = f"<{count}H"
        raw = np.array(struct.unpack(fmt, f.read(count * 2)), dtype=np.uint16)
    return raw.view(np.float16)


def pool_and_normalize(last_hidden: torch.Tensor, attention_mask=None) -> torch.Tensor:
    """Last-token pool + L2 normalize — matches engine.encode() output."""
    if attention_mask is not None:
        # Find the last non-padded token for each batch item
        seq_lens = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[0, seq_lens[0], :]
    else:
        pooled = last_hidden[0, -1, :]
    return F.normalize(pooled.float(), p=2, dim=-1)


# ═══════════════════════════════════════════════════════════════════
# ATB engine runner
# ═══════════════════════════════════════════════════════════════════

class ATBRunner:
    """Wrapper for Qwen3VLEngine — handles encode() with timing."""

    def __init__(self, model_dir: str):
        print("[ATB] Loading Qwen3VLEngine ...")
        self.engine = Qwen3VLEngine(model_dir)
        print(f"[ATB] Loaded: {self.engine.n_layer} text layers, "
              f"{self.engine.v_depth} vision blocks")

    def close(self):
        del self.engine
        torch.npu.empty_cache()

    def encode_text(self, input_ids: List[int],
                    normalize: bool = True) -> torch.Tensor:
        """Text-only encode. Returns pooled embedding."""
        ids_t = torch.tensor([input_ids], dtype=torch.long)
        mask_t = torch.ones_like(ids_t, dtype=torch.long)
        return self.engine.encode(ids_t, attention_mask=mask_t,
                                  normalize=normalize).cpu().float().squeeze(0)

    def encode_vision(self, input_ids: List[int],
                      pixel_values: np.ndarray,
                      grid_thw: Tuple[int, int, int]) -> torch.Tensor:
        """Vision encode with preprocessed pixel_values and grid_thw."""
        ids_t = torch.tensor([input_ids], dtype=torch.long)
        mask_t = torch.ones_like(ids_t, dtype=torch.long)

        t, h, w = grid_thw
        num_patches = t * h * w
        patch_dim = len(pixel_values) // num_patches
        pv_t = torch.from_numpy(pixel_values.reshape(num_patches, patch_dim))
        gth_t = torch.tensor([grid_thw], dtype=torch.long)

        return self.engine.encode(ids_t, pv_t, gth_t,
                                  attention_mask=mask_t,
                                  normalize=True).cpu().float().squeeze(0)

    def benchmark(self, fn, n_warmup: int, n_iter: int) -> Tuple[np.ndarray, torch.Tensor]:
        """Run warmup + benchmark iterations, return (times_ms_array, last_output)."""
        for _ in range(n_warmup):
            fn()
        sync()

        times = []
        out = None
        for _ in range(n_iter):
            sync(); t0 = now()
            out = fn()
            sync()
            times.append(now() - t0)

        return ms(times), out


# ═══════════════════════════════════════════════════════════════════
# Transformers reference runner
# ═══════════════════════════════════════════════════════════════════

class TFRunner:
    """Wrapper for raw Qwen3VLModel — apples-to-apples with ATB engine."""

    def __init__(self, model_dir: str):
        import safetensors.torch
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        print("[TF] Loading Qwen3VLModel ...")
        cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
        cfg._attn_implementation = "eager"
        cfg.text_config._attn_implementation = "eager"

        self.model = Qwen3VLModel(cfg).eval().half().npu()
        sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors",
                                         device="cpu")
        sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, \
            f"TF weight load failed: missing={missing}, unexpected={unexpected}"
        print("[TF] Loaded")

    def close(self):
        del self.model
        torch.npu.empty_cache()

    def forward_text(self, input_ids: List[int]) -> torch.Tensor:
        """Text-only forward → last-token pooled + L2-normalised."""
        ids_t = torch.tensor([input_ids], dtype=torch.long).npu()
        mask_t = torch.ones_like(ids_t, dtype=torch.long).npu()
        with torch.no_grad():
            out = self.model(use_cache=False, input_ids=ids_t,
                            attention_mask=mask_t)
        return pool_and_normalize(out.last_hidden_state.cpu(), mask_t.cpu())

    def forward_vision(self, input_ids: List[int],
                       pixel_values: np.ndarray,
                       grid_thw: Tuple[int, int, int]) -> torch.Tensor:
        """Vision forward with pixel_values → last-token pooled + L2-normalised."""
        ids_t = torch.tensor([input_ids], dtype=torch.long).npu()
        mask_t = torch.ones_like(ids_t, dtype=torch.long).npu()

        t, h, w = grid_thw
        num_patches = t * h * w
        patch_dim = len(pixel_values) // num_patches
        pv_t = torch.from_numpy(
            pixel_values.reshape(num_patches, patch_dim).astype(np.float16)
        ).npu()
        gth_t = torch.tensor([grid_thw], dtype=torch.long).npu()

        with torch.no_grad():
            out = self.model(use_cache=False, input_ids=ids_t,
                            attention_mask=mask_t,
                            pixel_values=pv_t, image_grid_thw=gth_t)
        return pool_and_normalize(out.last_hidden_state.cpu(), mask_t.cpu())

    def benchmark(self, fn, n_warmup: int, n_iter: int) -> Tuple[np.ndarray, torch.Tensor]:
        """Run warmup + benchmark iterations."""
        for _ in range(n_warmup):
            fn()
        sync()

        times = []
        out = None
        for _ in range(n_iter):
            sync(); t0 = now()
            out = fn()
            sync()
            times.append(now() - t0)

        return ms(times), out


# ═══════════════════════════════════════════════════════════════════
# Test case builders
# ═══════════════════════════════════════════════════════════════════

def build_quick_cases():
    """3 simple test cases for smoke testing."""
    return [
        ("Text-Only",  "text",  {"text": "What is the capital of France?"}),
        ("Image-Only", "image", {"image": Image.new('RGB', (120, 200), color='red')}),
        ("Image+Text", "mixed", {"image": Image.new('RGB', (64, 64), color='blue'),
                                  "text": "Describe."}),
    ]


def build_bench_cases():
    """13-combination benchmark matrix.

    Returns list of (label, kind, params) where kind ∈ {text, image, mixed}
    and params contains the precomputed inputs.
    """
    cases = []

    # Load base text tokens
    text_tokens = load_token_ids(f"{BIN_DIR}/tokens_text_only.bin")

    # ── TEXT_ONLY (5 sequence lengths) ──
    for s in TEXT_SEQ_LENS:
        ids = (text_tokens * (s // len(text_tokens) + 1))[:s]
        cases.append((f"TEXT S={s}", "text", {"input_ids": ids}))

    # ── IMAGE_ONLY + IMAGE_AND_TEXT (4 resolutions) ──
    for w, h in RESOLUTIONS:
        grid_thw = GRID_MAP[(w, h)]
        t_dim, g_h, g_w = grid_thw
        vis_tokens = t_dim * g_h * g_w // 4  # merge_size=2

        # Load shared pixel_values
        pv_path = f"{BIN_DIR}/cpp_pv_{w}x{h}.bin"
        pv = load_pixel_values(pv_path) if os.path.isfile(pv_path) else None

        # ── IMAGE_ONLY ──
        io_ids = [IMAGE_TOKEN_ID] * vis_tokens
        cases.append((f"IO {w}×{h}", "image",
                      {"input_ids": io_ids, "pixel_values": pv, "grid_thw": grid_thw}))

        # ── IMAGE_AND_TEXT ──
        tok_path = f"{BIN_DIR}/tokens_mm_{w}x{h}.bin"
        if os.path.isfile(tok_path):
            mm_ids = load_token_ids(tok_path)
        else:
            # Fallback: simple construction matching benchmark.py
            mm_ids = ([151643] + [IMAGE_TOKEN_ID] * vis_tokens +
                      text_tokens + [151645])
        cases.append((f"MM {w}×{h}", "mixed",
                      {"input_ids": mm_ids, "pixel_values": pv, "grid_thw": grid_thw}))

    return cases


# ═══════════════════════════════════════════════════════════════════
# Quick test runners (existing 3-case simple test)
# ═══════════════════════════════════════════════════════════════════

def run_quick_atb(model_dir: str, processor, cases,
                  n_warmup: int, n_iter: int) -> Dict[str, torch.Tensor]:
    """ATB quick test: use engine.encode() via processor pipeline."""
    print("[ATB] Loading Qwen3VLEngine ...")
    engine = Qwen3VLEngine(model_dir)
    print(f"[ATB] Loaded: {engine.n_layer} text layers, "
          f"{engine.v_depth} vision blocks")

    results = {}
    for name, kind, inp in cases:
        content = []
        if "image" in inp:
            content.append({"type": "image", "image": inp["image"]})
        if "text" in inp:
            content.append({"type": "text", "text": inp["text"]})
        msgs = [{"role": "system", "content": [
            {"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content}]

        tf_in = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=True)
        input_ids = tf_in['input_ids']
        attn_mask = tf_in.get('attention_mask', torch.ones_like(input_ids))

        pv_raw = grid_thw = None
        if "image" in inp:
            img_arr = torch.from_numpy(np.array(inp["image"])).permute(2, 0, 1)
            pv_raw, grid_thw = engine.preprocess_image(img_arr)

        emb = engine.encode(input_ids, pv_raw, grid_thw,
                            attention_mask=attn_mask, normalize=True)
        results[name] = emb.cpu().float().squeeze(0)
        print(f"[ATB] {name:<12} → {tuple(results[name].shape)}")

    del engine
    torch.npu.empty_cache()
    return results


def run_quick_tf(model_dir: str, processor, cases,
                 n_warmup: int, n_iter: int) -> Dict[str, torch.Tensor]:
    """TF quick test: use Qwen3VLModel via processor pipeline."""
    import safetensors.torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    print("[TF] Loading Qwen3VLModel ...")
    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    model = Qwen3VLModel(cfg).eval().half().npu()
    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors", device="cpu")
    sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print("[TF] Loaded")

    results = {}
    for name, kind, inp in cases:
        content = []
        if "image" in inp:
            content.append({"type": "image", "image": inp["image"]})
        if "text" in inp:
            content.append({"type": "text", "text": inp["text"]})
        msgs = [{"role": "system", "content": [
            {"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": content}]

        tf_in = processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors='pt',
            add_generation_prompt=True)
        ids_t = tf_in['input_ids'].npu()
        mask_t = tf_in.get('attention_mask', torch.ones_like(ids_t)).npu()

        kwargs = dict(use_cache=False, input_ids=ids_t, attention_mask=mask_t)
        if "image" in inp:
            kwargs["pixel_values"] = tf_in.get('pixel_values').half().npu()
            kwargs["image_grid_thw"] = tf_in.get('image_grid_thw').npu()

        with torch.no_grad():
            out = model(**kwargs)
        emb = pool_and_normalize(out.last_hidden_state.cpu(), mask_t.cpu())
        results[name] = emb
        print(f"[TF]  {name:<12} → {tuple(emb.shape)}")

    del model
    torch.npu.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# Full benchmark runner (13-combination)
# ═══════════════════════════════════════════════════════════════════

def run_bench_atb(cases, n_warmup: int, n_iter: int,
                  save_bin: bool = False) -> Dict[str, dict]:
    """Run full 13-case ATB benchmark.  When save_bin=True, also writes
    /tmp/py_<label>.bin with the pooled fp32 output for C++ comparison."""
    atb = ATBRunner(QWEN3VL_EMB_MODEL_DIR)
    results = {}
    try:
        for label, kind, params in cases:
            ids = params["input_ids"]
            pv = params.get("pixel_values")
            gth = params.get("grid_thw")
            has_vis = pv is not None and gth is not None

            if has_vis:
                times, emb = atb.benchmark(
                    lambda ids=ids, pv=pv, gth=gth: atb.encode_vision(ids, pv, gth),
                    n_warmup, n_iter)
            else:
                times, emb = atb.benchmark(
                    lambda ids=ids: atb.encode_text(ids), n_warmup, n_iter)

            results[label] = {"times": times, "emb": emb}
            print(f"[ATB] {label:<18} {times.mean():>8.2f} ± {times.std():.2f} ms")

            if save_bin:
                bin_name = label.lower().replace(" ", "_").replace("×", "x")
                path = f"/tmp/py_{bin_name}.bin"
                arr = emb.numpy().astype("float32")
                with open(path, "wb") as f:
                    f.write(struct.pack("<q", arr.size))
                    f.write(arr.tobytes())
    finally:
        atb.close()
    return results


def run_bench_tf(cases, n_warmup: int, n_iter: int) -> Dict[str, dict]:
    """Run full 13-case TF benchmark."""
    tf = TFRunner(QWEN3VL_EMB_MODEL_DIR)
    results = {}
    try:
        for label, kind, params in cases:
            ids = params["input_ids"]
            pv = params.get("pixel_values")
            gth = params.get("grid_thw")
            has_vis = pv is not None and gth is not None

            if has_vis:
                times, emb = tf.benchmark(
                    lambda ids=ids, pv=pv, gth=gth: tf.forward_vision(ids, pv, gth),
                    n_warmup, n_iter)
            else:
                times, emb = tf.benchmark(
                    lambda ids=ids: tf.forward_text(ids), n_warmup, n_iter)

            results[label] = {"times": times, "emb": emb}
            print(f"[TF]  {label:<18} {times.mean():>8.2f} ± {times.std():.2f} ms")
    finally:
        tf.close()
    return results


# ═══════════════════════════════════════════════════════════════════
# Comparison helpers
# ═══════════════════════════════════════════════════════════════════

def compare_quick(atb_emb, tf_emb, threshold: float) -> bool:
    """Compare quick-test embeddings."""
    print(f"\n{'─' * 60}")
    print(f"{'Case':<14} {'shape':<18} {'cosine':>10} {'L2 diff':>10} {'st':>5}")
    print(f"{'─' * 60}")
    all_pass = True
    for name in atb_emb:
        a, b = atb_emb[name], tf_emb[name]
        if a.shape != b.shape:
            print(f"{name:<14} SHAPE MISMATCH atb={a.shape} tf={b.shape}")
            all_pass = False
            continue
        cs = cosine(a, b)
        l2 = (a - b).norm().item()
        ok = cs > threshold
        all_pass &= ok
        status = "PASS" if ok else "FAIL"
        print(f"{name:<14} {str(tuple(a.shape)):<18} "
              f"{cs:>10.6f} {l2:>10.6f} {status:>5}")
    print(f"{'─' * 60}")
    return all_pass


def compare_bench(atb_results: dict, tf_results: dict,
                  threshold: float) -> bool:
    """Compare full-benchmark embeddings and print summary table."""
    w = 90
    print(f"\n{'=' * w}")
    print("ATB vs TF — Full 13/13 Benchmark Comparison")
    print(f"{'=' * w}")
    fmt = "{:<18s} {:>6s}  {:>12s}  {:>12s}  {:>7s}  {:>10s}  {:>5s}"
    print(fmt.format("Mode", "S", "ATB (ms)", "TF (ms)", "ATB/TF", "Cosine", "St"))
    print(f"{'─' * w}")

    all_pass = True
    for label, kind, params in build_bench_cases():
        atb_r = atb_results.get(label)
        tf_r = tf_results.get(label)
        if atb_r is None or tf_r is None:
            print(f"{label:<18s}  {'MISSING':>12s}")
            all_pass = False
            continue

        atb_t = atb_r["times"]
        tf_t = tf_r["times"]
        cs = cosine(atb_r["emb"], tf_r["emb"])
        ok = cs > threshold
        if not ok:
            all_pass = False

        seq = len(params["input_ids"])
        atb_str = f"{atb_t.mean():.1f}±{atb_t.std():.1f}"
        tf_str = f"{tf_t.mean():.1f}±{tf_t.std():.1f}"
        ratio = f"{atb_t.mean()/tf_t.mean():.2f}x" if tf_t.mean() > 0 else "—"

        print(fmt.format(label, str(seq), atb_str, tf_str, ratio,
                         f"{cs:.6f}", "PASS" if ok else "FAIL"))

    print(f"{'=' * w}")
    return all_pass


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[list] = None):
    p = argparse.ArgumentParser(
        description="ATB vs TF embedder E2E comparison & benchmark")
    p.add_argument('--mode', choices=['atb', 'tf', 'both'], default='both',
                   help='Which engine(s) to run (default: both)')
    p.add_argument('--bench', action='store_true',
                   help='Run full 13/13 benchmark matrix (default: 3-case quick test)')
    p.add_argument('--threshold', type=float, default=0.99,
                   help='Cosine similarity threshold for PASS (default 0.99)')
    p.add_argument('--model-dir', default=QWEN3VL_EMB_MODEL_DIR)
    p.add_argument('--iter', type=int, default=5,
                   help='Benchmark iterations (default: 5)')
    p.add_argument('--warmup', type=int, default=3,
                   help='Warmup iterations (default: 3)')
    p.add_argument('--save-bin', action='store_true',
                   help='Save ATB pooler outputs as /tmp/py_*.bin for C++ comparison')
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    run_atb = args.mode in ('atb', 'both')
    run_tf  = args.mode in ('tf', 'both')

    print(f"Model dir: {args.model_dir}")
    print(f"Mode: {args.mode} | Bench: {args.bench} | "
          f"Iter: {args.iter} | Warmup: {args.warmup}")

    if args.bench:
        # ── Full 13/13 benchmark ──────────────────────────────
        cases = build_bench_cases()
        atb_results = {}
        tf_results = {}

        if run_atb:
            atb_results = run_bench_atb(cases, args.warmup, args.iter,
                                        save_bin=args.save_bin)
        if run_tf:
            tf_results = run_bench_tf(cases, args.warmup, args.iter)

        if run_atb and run_tf:
            ok = compare_bench(atb_results, tf_results, args.threshold)
            return 0 if ok else 1
        return 0
    else:
        # ── Quick 3-case smoke test ───────────────────────────
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_dir,
                                                  padding_side='right')
        cases = build_quick_cases()

        atb_emb = {}
        tf_emb = {}

        if run_atb:
            atb_emb = run_quick_atb(args.model_dir, processor, cases,
                                    args.warmup, args.iter)
        if run_tf:
            tf_emb = run_quick_tf(args.model_dir, processor, cases,
                                  args.warmup, args.iter)

        if run_atb and run_tf:
            ok = compare_quick(atb_emb, tf_emb, args.threshold)
            return 0 if ok else 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
