"""
Pipeline-stage precision diagnosis for IMAGE_ONLY mode.

Saves Python intermediate tensors at every pipeline stage so they can be
compared with C++ counterparts (from test_accuracy with ATB_DEBUG_VISION=1).

Stages:
  1. pixel_values (preprocessed)
  2. pos_embed (bilinear interpolation output)
  3. vision RoPE cos/sin
  4. vision first layer output
  5. vision merger output (vis_embeds)
  6. deepstack features (ds_feats[0..2])
  7. inputs_embeds (after embedding + scatter)
  8. position_ids (GetRopeIndex output)
  9. text RoPE cos/sin (MRoPE::Compute output)
  10. text hidden states after each decoder layer (0..27)
  11. text final norm output
  12. pooling + normalize output

Usage:
    python tests/diag_pipeline.py
"""
import sys
import struct
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size

# Large buffer for IMAGE_ONLY (880 tokens + vision model)
set_atb_buffer_size(15 * 1024 * 1024 * 1024)

IMG_H, IMG_W, IMG_C = 720, 1280, 3
IMAGE_TOKEN_ID = 151655


def save_fp16(path, tensor):
    """Save as: [int64 dim] + [fp16 bytes]"""
    flat = tensor.detach().cpu().flatten()
    dim = flat.shape[0]
    fp16 = flat.half()
    with open(path, 'wb') as f:
        f.write(struct.pack('q', dim))
        f.write(fp16.numpy().tobytes())
    print(f"  Saved {path}: dim={dim}")


def save_fp32(path, tensor):
    """Save as: [int64 dim] + [fp32 bytes]"""
    flat = tensor.detach().cpu().flatten().float()
    dim = flat.shape[0]
    with open(path, 'wb') as f:
        f.write(struct.pack('q', dim))
        f.write(flat.numpy().tobytes())
    print(f"  Saved {path}: dim={dim}")


def save_int64(path, tensor):
    """Save as: [int64 dim] + [int64 bytes]"""
    flat = tensor.detach().cpu().flatten().long()
    dim = flat.shape[0]
    with open(path, 'wb') as f:
        f.write(struct.pack('q', dim))
        f.write(flat.numpy().tobytes())
    print(f"  Saved {path}: dim={dim}")


def create_test_image(c, h, w):
    image = torch.zeros(c, h, w, dtype=torch.uint8)
    for ci in range(c):
        for hi in range(h):
            for wi in range(w):
                value = (hi * 255 // h + wi * 255 // w + ci * 85) % 256
                image[ci, hi, wi] = value
    return image


def npu_sync():
    """Explicit NPU stream synchronize."""
    torch.npu.synchronize()


def main():
    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    from atb_python_qwen3vl_embedding.engine_utils import (
        compute_posemb_indices, compute_rope_indices, get_rope_index,
    )
    from atb_python_qwen3vl_embedding.vision_model import (
        run_first_layer_npu, run_block_npu, run_merger_npu,
    )
    from atb_python_qwen3vl_embedding.vision_pos_embed import run_posemb_npu
    from atb_python_qwen3vl_embedding.text_model import (
        run_text_layer_npu, run_text_norm_npu, make_causal_mask,
    )
    from atb_python_qwen3vl_embedding.preprocess import preprocess_image
    from atb_python_qwen3vl_embedding.utils import to_cpu_float, to_npu_half, make_seqlen_tensor

    print("=" * 60)
    print("Pipeline-Stage Precision Diagnosis — IMAGE_ONLY Mode")
    print("=" * 60)

    # ── Create engine ────────────────────────────────────────
    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine: {engine.n_layer} text layers, {engine.v_depth} vision blocks")
    print(f"  text_hidden={engine.hidden_t}, text_head_dim={engine.hd_t}")
    print(f"  vis_hidden={engine.v_cfg['hidden_size']}, vis_head_dim={engine.hd_v}")
    print(f"  deepstack_indexes={engine.ds_indexes}")

    # ── Stage 1: Preprocessing ───────────────────────────────
    print("\n── Stage 1: Preprocessing ──")
    img = create_test_image(IMG_C, IMG_H, IMG_W)
    pv, grid_thw = preprocess_image(img)
    num_patches = pv.shape[0]
    merged_tokens = num_patches // (engine.spatial_merge ** 2)
    print(f"  pixel_values: {pv.shape}, grid_thw: {grid_thw}")
    print(f"  num_patches={num_patches}, merged_tokens={merged_tokens}")
    save_fp16("/tmp/diag_py_pixels.bin", pv)
    save_int64("/tmp/diag_py_grid_thw.bin", grid_thw)

    # ── Stage 2: Position Embedding (bilinear interp) ────────
    print("\n── Stage 2: Position Embedding ──")
    idx_wt = compute_posemb_indices(grid_thw, engine.num_grid, engine.merge_size)

    # Also save the raw pos_embed weight table
    save_fp16("/tmp/diag_py_pos_embed_table.bin", engine.v_pos_embed)

    # ── Stage 3: Vision RoPE ─────────────────────────────────
    print("\n── Stage 3: Vision RoPE ──")
    rope_idx = compute_rope_indices(grid_thw, engine.vis_rotary, engine.merge_size)
    freq_npu = to_npu_half(rope_idx['freq_table'])

    # Compute pos_embed and RoPE on NPU
    pos_npu, cos_v_npu, sin_v_npu = run_posemb_npu(
        engine.g_v_posemb, engine.v_pe_w_table, idx_wt, rope_idx, freq_npu)

    # Copy vision RoPE cos/sin to CPU for comparison
    npu_sync()
    cos_v_host = to_cpu_float(cos_v_npu)
    sin_v_host = to_cpu_float(sin_v_npu)
    print(f"  Vision RoPE cos: {cos_v_host.shape}, sin: {sin_v_host.shape}")
    save_fp16("/tmp/diag_py_rope_cos.bin", cos_v_host)
    save_fp16("/tmp/diag_py_rope_sin.bin", sin_v_host)
    # Also save f32 for precise comparison
    save_fp32("/tmp/diag_py_rope_cos_f32.bin", cos_v_host)
    save_fp32("/tmp/diag_py_rope_sin_f32.bin", sin_v_host)

    # Also save the position IDs used for RoPE
    save_int64("/tmp/diag_py_rope_row.bin", rope_idx['pid_row'])
    save_int64("/tmp/diag_py_rope_col.bin", rope_idx['pid_col'])
    save_fp32("/tmp/diag_py_freq_table.bin", rope_idx['freq_table'])

    # ── Stage 4: Vision First Layer ──────────────────────────
    print("\n── Stage 4: Vision First Layer ──")
    pv_npu = to_npu_half(pv.reshape(-1))
    npatches = idx_wt['idx00'].shape[0]
    seqlen_v = make_seqlen_tensor(npatches)
    npu_sync()

    h_v_npu = run_first_layer_npu(engine.g_v_first, pv_npu,
                                   engine.v_pe_w, engine.v_pe_b,
                                   pos_npu, cos_v_npu, sin_v_npu,
                                   engine.v_block_weights[0], seqlen_v)
    npu_sync()
    h_v_first = to_cpu_float(h_v_npu)
    print(f"  Vision first layer output: {h_v_first.shape}")
    save_fp16("/tmp/diag_py_first_layer.bin", h_v_first)

    # ── Stage 5: Vision Blocks 1..23 ─────────────────────────
    print("\n── Stage 5: Vision Blocks ──")
    ds_feats = []
    for li in range(1, engine.v_depth):
        npu_sync()
        h_v_npu = run_block_npu(engine.g_v_block, h_v_npu, engine.v_block_weights[li],
                                cos_v_npu, sin_v_npu, seqlen_v)
        if li in engine.ds_indexes:
            ds_idx = engine.ds_indexes.index(li)
            ds_npu = run_merger_npu(engine.g_v_ds, h_v_npu, engine.v_ds_w[ds_idx])
            npu_sync()
            ds_host = to_cpu_float(ds_npu)
            print(f"  Deepstack layer {li} (idx={ds_idx}): {ds_host.shape}")
            save_fp16(f"/tmp/diag_py_ds_{ds_idx}.bin", ds_host)
            ds_feats.append(ds_npu)  # keep on NPU for later

        # Save last block output
        if li == engine.v_depth - 1:
            npu_sync()
            last_block = to_cpu_float(h_v_npu)
            save_fp16("/tmp/diag_py_last_block.bin", last_block)
            print(f"  Last block ({li}) output: {last_block.shape}")

    # ── Stage 6: Vision Merger ───────────────────────────────
    print("\n── Stage 6: Vision Merger ──")
    npu_sync()
    vis_embeds_npu = run_merger_npu(engine.g_v_merger, h_v_npu, engine.v_merger_w)
    npu_sync()
    vis_embeds_host = to_cpu_float(vis_embeds_npu)
    print(f"  Vision merged output: {vis_embeds_host.shape}")
    save_fp16("/tmp/diag_py_merged.bin", vis_embeds_host)

    # ── Stage 7: Inputs Embeds (embedding + scatter) ─────────
    print("\n── Stage 7: Inputs Embeds ──")
    input_ids = torch.tensor([[IMAGE_TOKEN_ID] * merged_tokens], dtype=torch.long)
    inputs_embeds = F.embedding(input_ids, engine.embed_w).half()  # CPU fp16
    print(f"  After embedding: {inputs_embeds.shape}, {inputs_embeds.dtype}")
    save_fp16("/tmp/diag_py_embed_lookup.bin", inputs_embeds)

    # Scatter vision features (simulate what forward() does)
    vis_mask = input_ids.squeeze(0) == IMAGE_TOKEN_ID
    inputs_embeds_npu = inputs_embeds.npu()
    npu_sync()
    inputs_embeds_npu[0, vis_mask.npu(), :] = vis_embeds_npu
    npu_sync()
    inputs_embeds_host = to_cpu_float(inputs_embeds_npu)
    print(f"  After scatter: {inputs_embeds_host.shape}")
    save_fp16("/tmp/diag_py_inputs_embeds.bin", inputs_embeds_host)

    # ── Stage 8: Position IDs (GetRopeIndex) ─────────────────
    print("\n── Stage 8: Position IDs ──")
    position_ids, _ = get_rope_index(
        input_ids, image_grid_thw=grid_thw,
        image_token_id=IMAGE_TOKEN_ID,
        spatial_merge_size=engine.spatial_merge)
    print(f"  position_ids shape: {position_ids.shape}")
    print(f"  position_ids range: [{position_ids.min().item()}, {position_ids.max().item()}]")
    print(f"  position_ids T: {position_ids[0, 0, :10].tolist()}")
    print(f"  position_ids H: {position_ids[1, 0, :10].tolist()}")
    print(f"  position_ids W: {position_ids[2, 0, :10].tolist()}")
    save_int64("/tmp/diag_py_position_ids.bin", position_ids)

    # ── Stage 9: Text MRoPE cos/sin ──────────────────────────
    print("\n── Stage 9: Text MRoPE ──")
    cos, sin = engine.text_rope(position_ids)
    print(f"  cos: {cos.shape}, sin: {sin.shape}")
    print(f"  cos first row (8): {cos[0, 0, :8].tolist()}")
    print(f"  sin first row (8): {sin[0, 0, :8].tolist()}")
    save_fp32("/tmp/diag_py_text_cos.bin", cos)
    save_fp32("/tmp/diag_py_text_sin.bin", sin)
    cos_npu = to_npu_half(cos.reshape(-1, engine.hd_t))
    sin_npu = to_npu_half(sin.reshape(-1, engine.hd_t))
    save_fp16("/tmp/diag_py_text_cos_fp16.bin", cos.reshape(-1, engine.hd_t))
    save_fp16("/tmp/diag_py_text_sin_fp16.bin", sin.reshape(-1, engine.hd_t))

    # ── Stage 10: Causal Mask ────────────────────────────────
    S = merged_tokens
    mask = make_causal_mask(S)
    print(f"  mask: {mask.shape}")
    save_fp16("/tmp/diag_py_mask.bin", mask)
    mask_npu = to_npu_half(mask)

    # ── Stage 11: Text Decoder (layer by layer) ──────────────
    print(f"\n── Stage 11: Text Decoder Layers (S={S}) ──")
    engine._ensure_text_graph(S)
    seqlen_t = make_seqlen_tensor(S)
    npu_sync()

    hidden_npu = inputs_embeds_npu
    for li in range(engine.n_layer):
        npu_sync()
        hidden_npu = run_text_layer_npu(engine.g_t_layer, hidden_npu,
                                         engine.t_layer_weights[li],
                                         cos_npu, sin_npu, seqlen_t,
                                         causal_mask=engine._cached_mask)
        # Deepstack injection
        if ds_feats and li < len(ds_feats):
            local = hidden_npu[0, vis_mask.npu(), :].clone() + ds_feats[li]
            hidden_npu[0, vis_mask.npu(), :] = local

        # Debug: save hidden states at specific layers
        if li in (0, 1, 2, 27):
            npu_sync()
            hs = to_cpu_float(hidden_npu)
            print(f"  Layer {li}: hidden_states {hs.shape}, "
                  f"range=[{hs.min().item():.6f}, {hs.max().item():.6f}]")
            save_fp16(f"/tmp/diag_py_hidden_L{li}.bin", hs)

    # ── Stage 12: Final Norm ─────────────────────────────────
    print("\n── Stage 12: Final Norm ──")
    npu_sync()
    norm_out = run_text_norm_npu(engine.g_t_norm, hidden_npu, engine.norm_w)
    norm_host = norm_out.cpu().float()  # This already returns CPU float32
    print(f"  Norm output: {norm_host.shape}")
    save_fp32("/tmp/diag_py_norm_out.bin", norm_host)

    # ── Stage 13: Pooling + Normalize ────────────────────────
    print("\n── Stage 13: Pooling ──")
    # LAST_TOKEN pooling
    attn_mask = torch.ones_like(input_ids)
    flipped = attn_mask.flip(dims=[1])
    last_pos = flipped.argmax(dim=1)
    col = attn_mask.shape[1] - last_pos - 1
    row = torch.arange(norm_host.shape[0], device=norm_host.device)
    pooled = norm_host[row, col]
    print(f"  Pooled (last token): {pooled.shape}, first 8: {pooled[0, :8].tolist()}")

    pooled_norm = F.normalize(pooled.float(), p=2, dim=-1)
    print(f"  Normalized: first 8: {pooled_norm[0, :8].tolist()}")
    save_fp32("/tmp/diag_py_final.bin", pooled_norm)
    save_fp32("/tmp/diag_py_pooled.bin", pooled)

    print("\n" + "=" * 60)
    print("All intermediate tensors saved to /tmp/diag_py_*.bin")
    print("=" * 60)


if __name__ == "__main__":
    main()
