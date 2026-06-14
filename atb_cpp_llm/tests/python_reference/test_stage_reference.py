"""
Stage-by-stage reference value generator for C++/Python precision comparison.

Computes intermediate values at each pipeline stage and saves them to
/tmp/stage_*.bin for comparison with the C++ test_stage_precision binary.

On 310P, when ATB graph execution fails ("call operation setup fail"),
falls back to transformers reference for TEXT_ONLY mode. IMAGE modes
are skipped if vision ATB fails (vision pipeline is too complex to
replicate purely on CPU).

Usage:
    python tests/test_stage_reference.py
"""

import os
import sys
import struct
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
from atb_python_qwen3vl_embedding.preprocess import preprocess_image


def save_float32(path: str, data: np.ndarray, shape: list[int] | None = None):
    """Save float32 array to binary file: [ndim, shape..., data...]"""
    if shape is None:
        shape = list(data.shape)
    with open(path, "wb") as f:
        f.write(struct.pack("q", len(shape)))
        for s in shape:
            f.write(struct.pack("q", s))
        f.write(data.astype(np.float32).tobytes())


def save_int64(path: str, data: np.ndarray, shape: list[int] | None = None):
    """Save int64 array to binary file: [ndim, shape..., data...]"""
    if shape is None:
        shape = list(data.shape)
    with open(path, "wb") as f:
        f.write(struct.pack("q", len(shape)))
        for s in shape:
            f.write(struct.pack("q", s))
        f.write(data.astype(np.int64).tobytes())


def create_gradient_image(channels: int, height: int, width: int) -> torch.Tensor:
    """Same gradient image as C++ side."""
    image = torch.zeros(channels, height, width, dtype=torch.uint8)
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                value = (h * 255 // height + w * 255 // width + c * 85) % 256
                image[c, h, w] = value
    return image


# ── Transformers fallback for TEXT_ONLY reference ─────────────────────

def _compute_text_only_transformers(model_dir: str, input_ids: torch.Tensor):
    """Compute TEXT_ONLY encoding via transformers (CPU, no ATB/NPU).

    This is the fallback when ATB engine.encode() fails on 310P.
    Uses the same Qwen3VLForConditionalGeneration as the E2E reference.

    Returns: (embedding: np.ndarray, method: str)
    """
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        # Last hidden state, pool last token
        hidden = outputs.hidden_states[-1]  # (B, S, D)
        # Pooling: last non-padding token (same as _pooling_last)
        # Assume no padding for these reference inputs
        emb = hidden[:, -1, :]  # (B, D)
        emb = F.normalize(emb, p=2, dim=-1)

    return emb.flatten().float().numpy(), "transformers (CPU)"


def _compute_text_only_atb(engine, input_ids):
    """Compute TEXT_ONLY encoding via ATB engine.

    Returns: (embedding: np.ndarray, method: str)
    Raises on failure.
    """
    result_t = engine.encode(input_ids, normalize=True)
    return result_t.flatten().float().numpy(), "ATB engine"


def compute_text_only_reference(model_dir, engine, input_ids):
    """Compute TEXT_ONLY reference with ATB → transformers fallback.

    Tries ATB engine first. On failure, falls back to transformers CPU.
    Returns: (data: np.ndarray, method: str, error: str | None)
    """
    errors = []

    # Tier 1: ATB engine (fastest, uses NPU)
    try:
        data, method = _compute_text_only_atb(engine, input_ids)
        print(f"  ✓ {method}")
        return data, method, None
    except Exception as e:
        err_msg = f"ATB engine failed: {e}"
        errors.append(err_msg)
        print(f"  ✗ Tier 1 (ATB): {e}")

    # Tier 2: transformers CPU (always available)
    try:
        data, method = _compute_text_only_transformers(model_dir, input_ids)
        print(f"  ✓ {method} (fallback)")
        return data, method, None
    except Exception as e:
        err_msg = f"Transformers fallback also failed: {e}"
        errors.append(err_msg)
        print(f"  ✗ Tier 2 (transformers): {e}")

    return None, "none", "; ".join(errors)


def main():
    print("=" * 60)
    print("Stage Reference Generator")
    print(f"Platform: {os.getenv('ASCEND_PLATFORM', '910B')}")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)
    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    print("Loading Python ATB engine...")
    try:
        engine = Qwen3VLEngine(MODEL_DIR)
        print(f"Engine loaded: {engine.n_layer} text layers, {engine.v_depth} vision blocks")
        engine_ok = True
    except Exception as e:
        print(f"  ✗ Engine init FAILED: {e}")
        engine_ok = False

    img_h, img_w = 672, 476
    failed_stages = []

    # ═══════════════════════════════════════════════════════════
    # Stage 1: Preprocessing (always works, no ATB needed)
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 1: Preprocessing ---")
    img = create_gradient_image(3, img_h, img_w)
    pixel_values, grid_thw = preprocess_image(img)
    print(f"  pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}")
    print(f"  grid_thw: {grid_thw}")
    save_float32("/tmp/stage_pixels.bin", pixel_values.numpy())
    save_int64("/tmp/stage_grid_thw.bin", grid_thw.numpy())
    print("  ✓ Stage 1 OK")

    # ═══════════════════════════════════════════════════════════
    # Stage 2: (placeholder — position IDs computed in stage 5)
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 2: Vision Position Embedding ---")
    print("  (deferred to stage 5)")

    # ═══════════════════════════════════════════════════════════
    # Stage 3: Vision Model Output (merged embeddings)
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 3: Vision Model (merged embeddings) ---")
    if engine_ok:
        try:
            pv_npu = pixel_values.reshape(-1).half().npu()
            vis_embeds, ds_feats = engine._run_vision(pv_npu, grid_thw)
            torch.npu.synchronize()
            vis_cpu = vis_embeds.float().cpu().numpy()
            save_float32("/tmp/stage_vis_embeds.bin", vis_cpu)
            print(f"  vis_embeds: shape={vis_embeds.shape}, dtype={vis_embeds.dtype}")
            print(f"  first 8: {vis_cpu[0, :8].tolist()}")
            print("  ✓ Stage 3 OK")
        except Exception as e:
            failed_stages.append("Stage 3 (vision model)")
            print(f"  ✗ Stage 3 FAILED: {e}")
            print(f"  ⚠  /tmp/stage_vis_embeds.bin NOT written")
            if os.getenv('ASCEND_PLATFORM', '') == '310P':
                print(f"  💡 Hint: on 310P, vision SelfAttention uses no mask → should work.")
                print(f"     Check ATB log: cat $(ls -rt ~/ascend/log/atb/ | tail -n 1)")
            engine_ok = False  # subsequent stages also likely to fail
    else:
        failed_stages.append("Stage 3 (vision model)")
        print("  ⊘ Stage 3 SKIPPED (engine init failed)")

    # ═══════════════════════════════════════════════════════════
    # Stage 4: Text Embedding Lookup (always works, CPU only)
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 4: Text Embedding Lookup ---")
    merged_tokens = pixel_values.shape[0] // (engine.spatial_merge ** 2)
    image_token_id = engine.img_tok

    input_ids_io = torch.tensor([[image_token_id] * merged_tokens], dtype=torch.long)
    embeds_io = F.embedding(input_ids_io, engine.embed_w).half()
    embeds_io_cpu = embeds_io.float().numpy()
    save_float32("/tmp/stage_embeds_image_only.bin", embeds_io_cpu.reshape(-1, embeds_io.shape[-1]))
    print(f"  IMAGE_ONLY embeds: shape={embeds_io.shape}")

    input_ids_mt = torch.tensor([[151643] + [image_token_id] * merged_tokens + [15339, 1879]],
                                 dtype=torch.long)
    embeds_mt = F.embedding(input_ids_mt, engine.embed_w).half()
    embeds_mt_cpu = embeds_mt.float().numpy()
    save_float32("/tmp/stage_embeds_image_text.bin", embeds_mt_cpu.reshape(-1, embeds_mt.shape[-1]))
    print(f"  IMAGE_AND_TEXT embeds: shape={embeds_mt.shape}")
    print("  ✓ Stage 4 OK")

    # ═══════════════════════════════════════════════════════════
    # Stage 5: Position Encoding (MRoPE) — always works, CPU only
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 5: Position Encoding (MRoPE) ---")
    from atb_python_qwen3vl_embedding.engine_utils import get_rope_index

    pos_ids_io, _ = get_rope_index(
        input_ids_io, grid_thw, None, None,
        image_token_id=image_token_id,
        spatial_merge_size=engine.spatial_merge)
    save_int64("/tmp/stage_pos_ids_image_only.bin", pos_ids_io.numpy())
    print(f"  IMAGE_ONLY position_ids: shape={pos_ids_io.shape}")

    pos_ids_mt, _ = get_rope_index(
        input_ids_mt, grid_thw, None, None,
        image_token_id=image_token_id,
        spatial_merge_size=engine.spatial_merge)
    save_int64("/tmp/stage_pos_ids_image_text.bin", pos_ids_mt.numpy())
    print(f"  IMAGE_AND_TEXT position_ids: shape={pos_ids_mt.shape}")
    print("  ✓ Stage 5 OK")

    # ═══════════════════════════════════════════════════════════
    # Stage 6: Full Pipeline Outputs
    # ═══════════════════════════════════════════════════════════
    print("\n--- Stage 6: Full Pipeline ---")

    # ── TEXT_ONLY (has fallback to transformers CPU) ────────────
    print("\n  [TEXT_ONLY]")
    input_ids_t = torch.tensor([[151643, 15339, 1879]], dtype=torch.long)
    text_only_ok = False
    if engine_ok:
        try:
            result_t = engine.encode(input_ids_t, normalize=True)
            emb_t = result_t.flatten().float().numpy()
            save_float32("/tmp/stage_final_text_only.bin", emb_t)
            print(f"  ✓ ATB engine: shape={emb_t.shape}, first 8: {emb_t[:8].tolist()}")
            text_only_ok = True
        except Exception as e:
            failed_stages.append("Stage 6 TEXT_ONLY (ATB)")
            print(f"  ✗ ATB engine FAILED: {e}")
            print(f"  → Falling back to transformers CPU reference...")

            # Fallback: transformers reference directly (don't retry ATB)
            try:
                emb_t, _ = _compute_text_only_transformers(MODEL_DIR, input_ids_t)
                save_float32("/tmp/stage_final_text_only.bin", emb_t)
                print(f"  ✓ transformers (CPU fallback): shape={emb_t.shape}, "
                      f"first 8: {emb_t[:8].tolist()}")
                text_only_ok = True
            except Exception as e2:
                failed_stages.append("Stage 6 TEXT_ONLY (fallback)")
                print(f"  ✗ Fallback also FAILED: {e2}")
    else:
        print("  ⊘ TEXT_ONLY SKIPPED (engine not available)")
        failed_stages.append("Stage 6 TEXT_ONLY")

    # ── IMAGE_ONLY (no fallback — too complex for CPU-only) ───
    print("\n  [IMAGE_ONLY]")
    if engine_ok and text_only_ok:
        try:
            result_io = engine.encode(input_ids_io, pixel_values=pixel_values,
                                       image_grid_thw=grid_thw, normalize=True)
            emb_io = result_io.flatten().float().numpy()
            save_float32("/tmp/stage_final_image_only.bin", emb_io)
            print(f"  ✓ ATB engine: shape={emb_io.shape}, first 8: {emb_io[:8].tolist()}")
        except Exception as e:
            failed_stages.append("Stage 6 IMAGE_ONLY")
            print(f"  ✗ ATB engine FAILED: {e}")
            print(f"  ⚠  No CPU fallback for IMAGE_ONLY (vision pipeline too complex)")
            print(f"     /tmp/stage_final_image_only.bin NOT written")
    else:
        failed_stages.append("Stage 6 IMAGE_ONLY")
        print("  ⊘ IMAGE_ONLY SKIPPED (TEXT_ONLY prerequisite failed)")

    # ── IMAGE_AND_TEXT (no fallback — too complex for CPU-only) ─
    print("\n  [IMAGE_AND_TEXT]")
    if engine_ok and text_only_ok:
        try:
            result_mt = engine.encode(input_ids_mt, pixel_values=pixel_values,
                                       image_grid_thw=grid_thw, normalize=True)
            emb_mt = result_mt.flatten().float().numpy()
            save_float32("/tmp/stage_final_image_text.bin", emb_mt)
            print(f"  ✓ ATB engine: shape={emb_mt.shape}, first 8: {emb_mt[:8].tolist()}")
        except Exception as e:
            failed_stages.append("Stage 6 IMAGE_AND_TEXT")
            print(f"  ✗ ATB engine FAILED: {e}")
            print(f"  ⚠  No CPU fallback for IMAGE_AND_TEXT (vision pipeline too complex)")
            print(f"     /tmp/stage_final_image_text.bin NOT written")
    else:
        failed_stages.append("Stage 6 IMAGE_AND_TEXT")
        print("  ⊘ IMAGE_AND_TEXT SKIPPED (TEXT_ONLY prerequisite failed)")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    if failed_stages:
        print(f"⚠  {len(failed_stages)} stage(s) failed:")
        for s in failed_stages:
            print(f"  - {s}")
        print(f"Reference data INCOMPLETE — some /tmp/stage_*.bin files missing.")
        print(f"C++ tests depending on these stages will be skipped by build_and_test.sh.")
        return 1
    else:
        print("Reference values saved to /tmp/stage_*.bin")
        print("Now run: ./test_stage_precision")
        return 0
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
