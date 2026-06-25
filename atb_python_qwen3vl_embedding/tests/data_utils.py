"""Shared test utilities for Qwen3VLEngine tests.

Provides timing helpers, similarity metrics, and TF reference model loading.
"""

import time

import os

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Timing utilities
# ═══════════════════════════════════════════════════════════════════

def now() -> float:
    return time.perf_counter()


def sync() -> None:
    if torch.npu.is_available():
        torch.npu.synchronize()


def ms(seconds_list) -> np.ndarray:
    return np.asarray(seconds_list) * 1000.0


def empty_npu_cache_safe() -> None:
    """Best-effort NPU cache cleanup that is safe after CPU fallback."""
    try:
        if torch.npu.is_available():
            torch.npu.empty_cache()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Similarity / pooling
# ═══════════════════════════════════════════════════════════════════

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def pool_and_normalize(last_hidden: torch.Tensor,
                       attention_mask=None) -> torch.Tensor:
    """Last-token pool + L2 normalize — matches engine.encode() output."""
    if attention_mask is not None:
        seq_lens = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[0, seq_lens[0], :]
    else:
        pooled = last_hidden[0, -1, :]
    return F.normalize(pooled.float(), p=2, dim=-1)


# ═══════════════════════════════════════════════════════════════════
# TF reference model loading
# ═══════════════════════════════════════════════════════════════════

class _TFRef:
    """Thin transparent proxy over a transformers Qwen3VLModel.

    Exposes ``device``/``dtype`` so call sites can build inputs matching the
    model's placement (``ref.place(t)`` / ``.to(ref.device)``), and forwards
    every attribute — including sub-modules like ``language_model`` — to the
    underlying model **by reference**. This is critical: tests monkeypatch
    ``ref.language_model.forward`` (see test_pipeline_trace.py spy); a deep
    wrapper would break that. ``__getattr__`` returns the real submodule so
    the patch lands on the object ``ref(...)`` actually calls internally.
    """

    def __init__(self, model, device, dtype):
        # Bypass __setattr__/__getattr__: store via object.__setattr__ so the
        # getattr fallback never recurses on the private name.
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "dtype", dtype)

    def __call__(self, *args, **kwargs):
        # __getattr__ does NOT intercept dunder/special methods like __call__,
        # so forward explicitly.
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        # _model is read via object.__getattribute__ in __call__; if we ever
        # get here for it, it means init hasn't run — raise to avoid recursion.
        if name == "_model":
            raise AttributeError(name)
        return getattr(self._model, name)

    def place(self, tensor):
        """Move a tensor to the model's device (and dtype if floating-point).

        Float inputs (pixel_values, hidden states, cos/sin) get both device
        and dtype; integer inputs (input_ids, grid_thw, position_ids) keep
        their dtype and only move device.
        """
        if tensor.is_floating_point():
            return tensor.to(self.device, self.dtype)
        return tensor.to(self.device)


def _build_tf_ref(model_dir: str, precision: str, device: str):
    """Construct and load a Qwen3VLModel (no probing, no proxy).

    Pure extraction of the original ``load_tf_ref`` body. ``device='cpu'``
    with ``precision='float32'`` yields a clean fp32 CPU model (validated by
    test_text_diagnostics.py); CPU cannot run fp16 forward, so callers that
    fall back to CPU must pass precision='float32'.
    """
    import safetensors.torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"

    model = Qwen3VLModel(cfg).eval()

    if precision in ("half", "float16"):
        model = model.half()
    elif precision == "float32":
        # Force float32: safetensors may store BFloat16 weights, which CPU
        # cannot matmul against float32 inputs (and which silently differ
        # from a true fp32 model on NPU too). Normalize the whole model.
        model = model.float()
    else:
        raise ValueError(
            f"Unknown precision: {precision!r}, expected 'float32' or 'half'")

    if device == "npu":
        model = model.npu()
    elif device != "cpu":
        raise ValueError(
            f"Unknown device: {device!r}, expected 'npu' or 'cpu'")

    sd = safetensors.torch.load_file(f"{model_dir}/model.safetensors",
                                     device="cpu")
    sd = {k.removeprefix("model."): v for k, v in sd.items()}
    if precision in ("half", "float16"):
        sd = {k: v.half() for k, v in sd.items()}
    else:  # float32
        sd = {k: v.float() for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, \
        f"TF weight load failed: missing={missing}, unexpected={unexpected}"
    return model


def _probe_tf_ref_npu(model, model_dir: str) -> None:
    """Run a minimal multimodal forward to detect unsupported NPU ops.

    A single image+text forward exercises both vision patch-embed (Conv3d)
    and attention-mask mask computation (ArgMaxWithValue) — the two op
    classes that fail on 310P. These are capability gaps (not shape
    dependent), so a minimal input reliably predicts real-input failure.

    Raises whatever the model raises (RuntimeError on 310P); callers catch.
    """
    import torch  # noqa: F401  (ensure torch_npu side-effects loaded)
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_dir)
    img = Image.new('RGB', (120, 200), color='red')
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Describe.'}]}]
    tf_in = proc.apply_chat_template(
        msgs, tokenize=True, return_dict=True, return_tensors='pt',
        add_generation_prompt=True)

    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    kwargs = {
        'input_ids': tf_in['input_ids'].to(dev),
        'attention_mask': tf_in['attention_mask'].to(dev),
        'pixel_values': tf_in['pixel_values'].to(dev, dt),
        'image_grid_thw': tf_in['image_grid_thw'].to(dev),
    }
    with torch.no_grad():
        model(use_cache=False, **kwargs)


def load_tf_ref(model_dir: str, precision: str = "float32", device: str = "npu"):
    """Load Qwen3VLModel from safetensors, with NPU→CPU fallback.

    Unified TF reference model loader for all test files. Always checks for
    missing / unexpected keys so weight-loading bugs are caught immediately.

    When ``device='npu'`` (default), runs an eager minimal forward probe
    after loading. If the probe raises (310P unsupported ops such as
    ArgMaxWithValue / Conv3d), the NPU model is discarded and rebuilt on
    CPU in float32 — mirroring the ATB→transformers-CPU fallback already
    used by the C++ reference generators (test_stage_reference.py).
    Decision is made at load time so call sites can place inputs on
    ``ref.device`` before any forward.

    Args:
        model_dir: Path to model directory containing config.json and
            model.safetensors.
        precision: ``"float32"`` (default) or ``"half"`` / ``"float16"``.
            Ignored when falling back to CPU (always float32 there, since
            CPU cannot run fp16 forward).
        device: Target device — ``"npu"`` (default) or ``"cpu"``.

    Returns:
        ``_TFRef`` proxy (callable like the model, with ``.device`` /
        ``.dtype`` / ``.place()``) wrapping Qwen3VLModel in eval mode.
    """
    if device == "cpu":
        model = _build_tf_ref(model_dir, "float32", "cpu")
        return _TFRef(model, torch.device("cpu"), torch.float32)

    if device != "npu":
        raise ValueError(
            f"Unknown device: {device!r}, expected 'npu' or 'cpu'")

    # Debug escape hatch: force the CPU fallback path even on NPU-capable
    # hardware (e.g. verify fallback on 910B, where the probe would otherwise
    # succeed). Not for production use. Set TFREF_FORCE_CPU=1 to enable.
    if os.environ.get("TFREF_FORCE_CPU") == "1":
        print("[load_tf_ref] TFREF_FORCE_CPU=1 — forcing CPU float32 reference.")
        model = _build_tf_ref(model_dir, "float32", "cpu")
        return _TFRef(model, torch.device("cpu"), torch.float32)

    model = _build_tf_ref(model_dir, precision, "npu")
    npu_dev = next(model.parameters()).device
    npu_dt = next(model.parameters()).dtype
    try:
        _probe_tf_ref_npu(model, model_dir)
        return _TFRef(model, npu_dev, npu_dt)
    except (RuntimeError, OSError, ValueError) as e:
        print(f"[load_tf_ref] NPU probe failed ({type(e).__name__}: {e}); "
              f"falling back to CPU float32 reference.")
        del model
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
        model = _build_tf_ref(model_dir, "float32", "cpu")
        return _TFRef(model, torch.device("cpu"), torch.float32)
