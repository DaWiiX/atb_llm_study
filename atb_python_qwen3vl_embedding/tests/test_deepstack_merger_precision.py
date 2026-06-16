"""Level 2: Deepstack merger MLP precision test — ATB vs TF reference.

Validates a single deepstack merger MLP (reshape → LayerNorm → fc1 → GELU → fc2)
in isolation. Uses small synthetic dimensions with random weights for fast
unit testing.

Deepstack merger architecture:
    input:  (N_patches, hidden_size)           — vision block output
    reshape: (N_patches * hs // (hs*4), hs*4)   — spatial flatten (merge=2)
    LayerNorm(hs*4) → Linear(hs*4, hs*4)+bias → GELU → Linear(hs*4, out_hs)+bias

The TF reference is Qwen3VLVisionPatchMerger with use_postshuffle_norm=True.
"""
import os, torch, torch.nn.functional as F

from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size, compare_tensors
from atb_python_qwen3vl_embedding.vision_model import (
    build_deepstack_merger, run_merger_npu, collect_merger_weights,
)
from atb_python_qwen3vl_embedding.utils import to_npu_half, to_cpu_float, prepare_npu_weights

set_atb_buffer_size(500 * 1024 * 1024)  # 500 MB


def test_deepstack_merger_single(seed=42):
    """Test a single deepstack merger MLP: ATB vs TF reference.

    Uses tiny dimensions (hs=32, merge=2, out_hs=64) and random weights.
    Expected cosine >= 0.999 for a single fp16 MLP.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger
    from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig

    hs = 32          # hidden_size (vision)
    nh = 4           # num_heads
    merge = 2        # spatial_merge_size
    out_hs = 64      # out_hidden_size (text)
    mer_hs = hs * merge * merge  # 128
    npatches = 16    # number of vision patches (must be divisible by merge^2=4)

    print(f"\n=== Deepstack Merger Precision (hs={hs}, mer_hs={mer_hs}, out_hs={out_hs}) ===")

    # ── Build config ──────────────────────────────────────────────
    cfg = Qwen3VLVisionConfig(
        hidden_size=hs, num_heads=nh, intermediate_size=mer_hs * 2,
        depth=1, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=merge, in_channels=3, out_hidden_size=out_hs,
        num_position_embeddings=64, deepstack_visual_indexes=[0],
        hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = 'eager'

    # ── TF reference: PatchMerger with use_postshuffle_norm=True ──
    torch.manual_seed(seed)
    tf_merger = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=True)
    tf_merger.eval()

    # ── Random input ─────────────────────────────────────────────
    torch.manual_seed(99)
    x = torch.randn(npatches, hs)

    with torch.no_grad():
        ref = tf_merger(x)  # (npatches//4, out_hs)

    # ── Extract TF weights ────────────────────────────────────────
    # collect_merger_weights extracts from a PatchMerger instance
    tf_weights = collect_merger_weights(tf_merger)
    # tf_weights: [norm_w, norm_b, fc1_w, fc1_b, fc2_w, fc2_b]

    # ── Build ATB graph ──────────────────────────────────────────
    g_ds = build_deepstack_merger(cfg)

    # ── Run ATB ──────────────────────────────────────────────────
    x_npu = to_npu_half(x)
    atb_npu_weights = prepare_npu_weights(tf_weights)
    atb_out = run_merger_npu(g_ds, x_npu, atb_npu_weights)
    torch.npu.synchronize()
    atb = to_cpu_float(atb_out)  # (npatches//4, out_hs)

    # ── Compare ──────────────────────────────────────────────────
    print(f"  TF  shape: {tuple(ref.shape)}")
    print(f"  ATB shape: {tuple(atb.shape)}")
    compare_tensors(ref, atb, label="DeepstackMerger-Single")
    cs = F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item()
    ok = cs >= 0.999
    print(f"  {'PASS' if ok else 'FAIL'}  cosine={cs:.6f}  (threshold=0.999)")
    return ok


def test_deepstack_merger_varied_shapes(seed=42):
    """Test deepstack merger with varying patch counts to verify reshape logic.

    The reshape operation (s[0] * s[1] // mer_hs, mer_hs) must handle
    different numbers of vision patches correctly.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger
    from transformers.models.qwen3_vl.modular_qwen3_vl import Qwen3VLVisionConfig

    hs = 32
    nh = 4
    merge = 2
    out_hs = 64
    mer_hs = hs * merge * merge  # 128

    print(f"\n=== Deepstack Merger — Varied Patch Counts ===")

    cfg = Qwen3VLVisionConfig(
        hidden_size=hs, num_heads=nh, intermediate_size=mer_hs * 2,
        depth=1, patch_size=4, temporal_patch_size=2,
        spatial_merge_size=merge, in_channels=3, out_hidden_size=out_hs,
        num_position_embeddings=64, deepstack_visual_indexes=[0],
        hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = 'eager'

    torch.manual_seed(seed)
    tf_merger = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=True)
    tf_merger.eval()

    tf_weights = collect_merger_weights(tf_merger)

    g_ds = build_deepstack_merger(cfg)
    atb_npu_weights = prepare_npu_weights(tf_weights)

    all_ok = True
    for npatches in [4, 8, 16, 32, 64]:  # all divisible by merge^2=4
        torch.manual_seed(npatches * 7)
        x = torch.randn(npatches, hs)

        with torch.no_grad():
            ref = tf_merger(x)

        x_npu = to_npu_half(x)
        atb_out = run_merger_npu(g_ds, x_npu, atb_npu_weights)
        torch.npu.synchronize()
        atb = to_cpu_float(atb_out)

        cs = F.cosine_similarity(ref.flatten(), atb.flatten(), dim=0).item()
        ok = cs >= 0.999
        all_ok &= ok
        print(f"  npatches={npatches:<4} → shape={tuple(atb.shape)}  "
              f"cosine={cs:.6f}  {'PASS' if ok else 'FAIL'}")

    return all_ok


if __name__ == "__main__":
    ok1 = test_deepstack_merger_single()
    ok2 = test_deepstack_merger_varied_shapes()
    exit(0 if (ok1 and ok2) else 1)
