"""
Pure ATB inference engine for Qwen3-VL-Embedding-2B.

Zero dependency on transformers. Works directly with safetensors checkpoint
files and JSON configs from a model directory.

Usage:
    from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
    set_atb_buffer_size(5 * 1024 * 1024 * 1024)

    engine = Qwen3VLEngine("/path/to/Qwen3-VL-Embedding-2B")
    output = engine.forward(input_ids, pixel_values, image_grid_thw)
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from .preprocess import preprocess_image
from .engine_utils import (
    load_config, load_preprocessor_config, load_weights,
    get_text_layer_weights, get_text_norm_weight, get_embed_weight,
    get_vision_block_weights, get_patch_embed_weights,
    get_vision_pos_embed, get_merger_weights,
    VisionRotaryEmbedding, compute_rot_pos_emb, fast_pos_embed_interpolate,
    compute_posemb_indices, compute_rope_indices,
    TextRotaryEmbedding, get_rope_index,
)
from .vision_model import (
    build_vision_first_layer, build_vision_merger, build_deepstack_merger,
    run_first_layer_npu, run_block_npu, run_merger_npu,
)
from .vision_block import build_vision_block
from .vision_pos_embed import build_vision_posemb_graph, run_posemb_npu
from .text_model import (
    build_text_layer_graph, build_text_norm_graph,
    run_text_layer_npu, run_text_norm_npu, make_causal_mask,
)

from .utils import (
    to_cpu_float, to_npu_half, make_seqlen_tensor, is_310p,
    make_causal_mask_nz,
)


class Qwen3VLEngine:
    """Pure ATB inference engine for Qwen3-VL-Embedding.

    All computation runs on NPU via ATB graphs. Weights are loaded
    directly from safetensors files — no transformers dependency.
    """

    def __init__(self, model_dir: str):
        import torch_npu  # noqa: F401 — required for ATB NPU ops

        self.model_dir = Path(model_dir)

        # ── Load config ──────────────────────────────────────────────
        cfg = load_config(model_dir)
        self.t_cfg = cfg["text_config"]
        self.v_cfg = cfg["vision_config"]
        self.img_tok = cfg["image_token_id"]
        self.spatial_merge = cfg["vision_config"]["spatial_merge_size"]

        # ── Load weights ─────────────────────────────────────────────
        self.weights = load_weights(model_dir)

        # ── Text model params ────────────────────────────────────────
        self.nh_t = self.t_cfg["num_attention_heads"]
        self.nkv_t = self.t_cfg["num_key_value_heads"]
        self.hd_t = self.t_cfg["head_dim"]
        self.hidden_t = self.t_cfg["hidden_size"]
        self.interm_t = self.t_cfg["intermediate_size"]
        self.n_layer = self.t_cfg["num_hidden_layers"]

        # ── Vision model params ──────────────────────────────────────
        self.nh_v = self.v_cfg["num_heads"]
        self.hd_v = self.v_cfg["hidden_size"] // self.nh_v
        self.v_depth = self.v_cfg["depth"]
        self.ds_indexes = self.v_cfg["deepstack_visual_indexes"]
        self.merge_size = self.v_cfg["spatial_merge_size"]

        # ── Position embeddings ──────────────────────────────────────
        self.text_rope = TextRotaryEmbedding(
            head_dim=self.hd_t,
            rope_theta=self.t_cfg.get("rope_theta", 5000000.0),
            mrope_section=tuple(self.t_cfg["rope_scaling"]["mrope_section"]),
        )
        self.vis_rotary = VisionRotaryEmbedding(dim=self.hd_v // 2)

        # ── Preprocessor params ──────────────────────────────────────
        pp = load_preprocessor_config(model_dir)
        self.patch_size = pp["patch_size"]
        self.tp = pp["temporal_patch_size"]
        self.pp_min_px = pp["min_pixels"]
        self.pp_max_px = pp["max_pixels"]

        # ── Cached weights (NPU-resident float16) ────────────────────
        self.embed_w = get_embed_weight(self.weights)
        self.norm_w = get_text_norm_weight(self.weights).half().npu()
        self.t_layer_weights = [
            [to_npu_half(w) for w in get_text_layer_weights(self.weights, i)]
            for i in range(self.n_layer)
        ]

        # ── 310P GQA→MHA weight expansion ──────────────────────────
        # 310P SelfAttention may not support GQA (kv_head_num < head_num).
        # Expand K/V/K-norm weights to MHA so the graph uses the same
        # MHA SelfAttention that is confirmed working for vision path.
        if is_310p() and self.nkv_t < self.nh_t:
            self._expand_kv_weights_to_mha()

        # Vision: block weights by layer (NPU-resident)
        self.v_block_weights = [
            [to_npu_half(w) for w in get_vision_block_weights(self.weights, i)]
            for i in range(self.v_depth)
        ]
        self.v_pe_w, self.v_pe_b = get_patch_embed_weights(
            self.weights, self.v_cfg["hidden_size"])
        self.v_pe_w = to_npu_half(self.v_pe_w)
        self.v_pe_b = to_npu_half(self.v_pe_b)
        self.v_pos_embed = get_vision_pos_embed(self.weights)
        self.v_pe_w_table = to_npu_half(self.v_pos_embed)  # NPU fp16 for ATB Gather
        self.num_grid = int(self.v_cfg["num_position_embeddings"] ** 0.5)

        # Vision: merger weights (NPU-resident)
        self.v_merger_w = [to_npu_half(w) for w in
                           get_merger_weights(self.weights, is_deepstack=False)]
        self.v_ds_w = [
            [to_npu_half(w) for w in
             get_merger_weights(self.weights, is_deepstack=True, ds_idx=i)]
            for i in range(len(self.ds_indexes))
        ]

        # ── Build ATB graphs ────────────────────────────────────────
        self._build_graphs()

    def _make_vision_config(self):
        """Create a config-like object for ATB vision graph builders."""
        return VisionConfigWrapper(self.v_cfg)

    def _expand_kv_weights_to_mha(self):
        """Expand K/V/K-norm weights from GQA to MHA for 310P compatibility.

        Qwen3VL-Embedding-2B: nh=32, kv_nh=4, ratio=8.
        K weight (kv_nh*hd, hidden) → (nh*hd, hidden) by repeating each
        KV head group 8 times. Same for V and K-norm weights.

        This is numerically EXACT: replicated KV heads in MHA produce
        identical attention output as GQA with shared heads.
        """
        ratio = self.nh_t // self.nkv_t
        print(f"[310P] Expanding GQA→MHA: nh={self.nh_t} kv_nh={self.nkv_t} "
              f"ratio={ratio} hd={self.hd_t}")

        # K weight: (kv_nh*hd, hidden) → (nh*hd, hidden)
        # V weight: (kv_nh*hd, hidden) → (nh*hd, hidden)
        # K-norm: may be (hd,) per-head or (kv_nh*hd,) per-kv-group
        for li in range(self.n_layer):
            w = self.t_layer_weights[li]
            # w[1] = k_proj, w[2] = v_proj, w[5] = k_norm
            k_w = w[1]  # (kv_nh*hd, hidden)
            v_w = w[2]  # (kv_nh*hd, hidden)
            kn_w = w[5]  # (kv_nh*hd,) or (hd,)

            # Reshape, repeat, flatten
            # K: (kv_nh, hd, hidden) → repeat → (nh, hd, hidden) → (nh*hd, hidden)
            k_w_exp = k_w.reshape(self.nkv_t, self.hd_t, -1) \
                .repeat_interleave(ratio, dim=0) \
                .reshape(self.nh_t * self.hd_t, -1)
            w[1] = to_npu_half(k_w_exp.contiguous())

            # V: same
            v_w_exp = v_w.reshape(self.nkv_t, self.hd_t, -1) \
                .repeat_interleave(ratio, dim=0) \
                .reshape(self.nh_t * self.hd_t, -1)
            w[2] = to_npu_half(v_w_exp.contiguous())

            # K-norm: (kv_nh*hd,) or (hd,) → (nh*hd,)
            if kn_w.numel() == self.nkv_t * self.hd_t:
                # Full per-kv-group: (kv_nh*hd,) → (kv_nh, hd) → repeat → (nh*hd,)
                kn_w_exp = kn_w.reshape(self.nkv_t, self.hd_t) \
                    .repeat_interleave(ratio, dim=0) \
                    .reshape(self.nh_t * self.hd_t)
            elif kn_w.numel() == self.hd_t:
                # Per-head (hd,): no expansion needed, already correct for MHA
                kn_w_exp = kn_w
            else:
                raise RuntimeError(
                    f"Unexpected k_norm weight shape: {kn_w.shape}, "
                    f"expected ({self.nkv_t * self.hd_t},) or ({self.hd_t},)")
            w[5] = to_npu_half(kn_w_exp.contiguous())

        # Update effective head count so graph uses MHA
        self.nkv_t = self.nh_t

    def _ensure_text_graph(self, S: int):
        """Lazily build text decoder layer graph and cached mask for sequence length S."""
        if self._text_S == S and self.g_t_layer is not None:
            return
        self._text_S = S
        self.g_t_layer = build_text_layer_graph(
            self.nh_t, self.nkv_t, self.hd_t, self.interm_t,
            B=1, S=S, use_mask=True)
        if is_310p():
            # 310P: generate causal mask directly in NZ (FRACTAL_NZ) layout
            mask = to_npu_half(make_causal_mask_nz(S))
        else:
            mask = to_npu_half(make_causal_mask(S))
        self._cached_mask = mask

    def _build_graphs(self):
        """Build all ATB graphs."""
        # Vision
        self.g_v_first = build_vision_first_layer(self._make_vision_config())
        _, self.g_v_block, _ = build_vision_block(self.nh_v, self.hd_v, "VisBlock")
        self.g_v_merger = build_vision_merger(self._make_vision_config())
        self.g_v_ds = build_deepstack_merger(self._make_vision_config())
        self.g_v_posemb = build_vision_posemb_graph()

        # Text
        self.g_t_norm = build_text_norm_graph(self.hidden_t)
        self._text_S = None
        self.g_t_layer = None
        self._cached_mask = None

    # ── Preprocessing ──────────────────────────────────────────────

    def preprocess_image(self, image: torch.Tensor):
        """Preprocess raw image (C,H,W uint8) → ATB-compatible format."""
        return preprocess_image(
            image, patch_size=self.patch_size,
            temporal_patch_size=self.tp, merge_size=self.merge_size,
            min_pixels=self.pp_min_px, max_pixels=self.pp_max_px)

    # ── Vision inference ───────────────────────────────────────────

    def _run_vision(self, pixel_values, grid_thw):
        """Run full VisionModel on ATB NPU — outputs stay on NPU.

        Returns (vis_npu, ds_feats_npu) — both NPU float16 tensors.
        """
        # ── Position embedding + RoPE via ATB graph (NPU) ────────────
        idx_wt = compute_posemb_indices(grid_thw, self.num_grid, self.merge_size)
        rope_idx = compute_rope_indices(grid_thw, self.vis_rotary, self.merge_size)
        freq_npu = to_npu_half(rope_idx['freq_table'])

        pos_npu, cos_npu, sin_npu = run_posemb_npu(
            self.g_v_posemb, self.v_pe_w_table, idx_wt, rope_idx, freq_npu)

        # Pre-convert pixel values to NPU float16 once.
        pv_npu = to_npu_half(pixel_values.reshape(-1)
                             if pixel_values.ndim == 2 else pixel_values)

        # Pre-create seqlen tensor once — all blocks share the same npatches
        npatches = idx_wt['idx00'].shape[0]
        seqlen_v = make_seqlen_tensor(npatches)

        # Sync: all CPU-derived data (idx_wt, rope_idx → via .npu() inside
        # run_posemb_npu) and H2D transfers (pv_npu, freq_npu) must land on
        # NPU before the vision ATB graphs read them.  ATB graph execution
        # may happen on a separate NPU stream that does not see pending
        # transfers on the default stream.
        torch.npu.synchronize()

        h = run_first_layer_npu(self.g_v_first, pv_npu,
                                self.v_pe_w, self.v_pe_b,
                                pos_npu, cos_npu, sin_npu,
                                self.v_block_weights[0], seqlen_v)

        ds_feats = []
        for li in range(1, self.v_depth):
            # Sync between different ATB graphs.  The C++ engine calls
            # aclrtSynchronizeStream after every ExecuteGraph; Python's
            # graph.forward() is purely async.  Without this sync, a
            # subsequent graph may read the previous graph's output before
            # it is ready when the graphs execute on different internal
            # streams or when NPU-side ordering is not guaranteed.
            torch.npu.synchronize()
            h = run_block_npu(self.g_v_block, h, self.v_block_weights[li],
                              cos_npu, sin_npu, seqlen_v)
            if li in self.ds_indexes:
                ds_idx = self.ds_indexes.index(li)
                ds_feats.append(run_merger_npu(self.g_v_ds, h, self.v_ds_w[ds_idx]))

        # torch.npu.synchronize()
        vis = run_merger_npu(self.g_v_merger, h, self.v_merger_w)
        return vis, ds_feats

    # ── Text inference ─────────────────────────────────────────────

    def _run_text(self, inputs_embeds, position_ids,
                  visual_mask=None, deepstack_features=None):
        """Run full TextModel on ATB NPU — NPU-resident hidden states."""
        S = inputs_embeds.shape[1]
        self._ensure_text_graph(S)

        cos, sin = self.text_rope(position_ids)
        cos_npu = to_npu_half(cos.reshape(-1, self.hd_t))
        sin_npu = to_npu_half(sin.reshape(-1, self.hd_t))

        # Pre-create seqlen tensor once — all 28 layers share the same S
        seqlen_t = make_seqlen_tensor(S)

        hidden = to_npu_half(inputs_embeds)

        if visual_mask is not None:
            visual_mask = visual_mask.npu()

        # Sync: CPU-derived cos/sin and visual_mask are transferred to NPU
        # via to_npu_half() / .npu() above.  ATB graph may execute on a
        # separate NPU stream — without this sync it could read those
        # tensors before the H2D transfers complete.
        torch.npu.synchronize()

        for li in range(self.n_layer):
            # Sync between ATB text layer graph executions (same reason as
            # vision block loop — C++ engine syncs after every ExecuteGraph).
            torch.npu.synchronize()
            hidden = run_text_layer_npu(self.g_t_layer, hidden,
                                        self.t_layer_weights[li],
                                        cos_npu, sin_npu, seqlen_t,
                                        causal_mask=self._cached_mask)
            if deepstack_features and li < len(deepstack_features):
                # clone + add + writeback (matches TF _deepstack_process)
                local = hidden[0, visual_mask, :].clone() + deepstack_features[li]
                hidden[0, visual_mask, :] = local

        return run_text_norm_npu(self.g_t_norm, hidden, self.norm_w).cpu().float()

    # ── Full pipeline ───────────────────────────────────────────────

    def forward(self, input_ids: torch.LongTensor,
                pixel_values: torch.Tensor | None = None,
                image_grid_thw: torch.LongTensor | None = None,
                ) -> torch.Tensor:
        """Complete inference pipeline.

        Args:
            input_ids:      (B, S) LongTensor.
            pixel_values:   (N,) float32 (from preprocess_image) or None.
            image_grid_thw: (N, 3) LongTensor or None.

        Returns (B, S, hidden_size) float32 on CPU.
        """
        # 1. Text embeddings → NPU
        inputs_embeds = F.embedding(input_ids, self.embed_w).half().npu()

        # 2. Vision features — all NPU-resident
        ds_feats = []
        vis_mask = None
        if pixel_values is not None and image_grid_thw is not None:
            vis_embeds, ds_feats = self._run_vision(pixel_values, image_grid_thw)
            # CPU: compute mask; async H2D: vis_mask.npu(); async NPU: scatter.
            # Sync so the scatter sees a fully-transferred mask AND valid
            # vis_embeds from the (async) ATB vision graph.
            torch.npu.synchronize()
            vis_mask = input_ids.squeeze(0) == self.img_tok
            inputs_embeds[0, vis_mask.npu(), :] = vis_embeds

        # 3. Position IDs (CPU)
        position_ids, _ = get_rope_index(
            input_ids, image_grid_thw, None, None,
            image_token_id=self.img_tok,
            spatial_merge_size=self.spatial_merge)

        # 4. Text model (ATB NPU)
        return self._run_text(inputs_embeds, position_ids,
                              vis_mask, ds_feats if ds_feats else None)

    # ── Pooling ─────────────────────────────────────────────────────

    @staticmethod
    def embedding_pooling(hidden_state: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract last non-padded token embedding (same as _pooling_last).

        Args:
            hidden_state:   (B, S, D).
            attention_mask: (B, S) int, 1=valid, 0=padding.

        Returns (B, D).
        """
        flipped = attention_mask.flip(dims=[1])
        last_pos = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def encode(self, input_ids: torch.LongTensor,
               pixel_values: torch.Tensor | None = None,
               image_grid_thw: torch.LongTensor | None = None,
               attention_mask: torch.Tensor | None = None,
               normalize: bool = True,
               ) -> torch.Tensor:
        """Full encode → pool → (optionally normalize).

        Returns (B, hidden_size) embedding vector.
        """
        output = self.forward(input_ids, pixel_values, image_grid_thw)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        emb = self.embedding_pooling(output, attention_mask)
        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


class VisionConfigWrapper:
    """Minimal config object matching what ATB vision graph builders expect."""
    def __init__(self, cfg: dict):
        self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_heads"]
        self.intermediate_size = cfg["intermediate_size"]
        self.depth = cfg["depth"]
        self.patch_size = cfg["patch_size"]
        self.temporal_patch_size = cfg["temporal_patch_size"]
        self.spatial_merge_size = cfg["spatial_merge_size"]
        self.in_channels = cfg["in_channels"]
        self.out_hidden_size = cfg["out_hidden_size"]
        self.num_position_embeddings = cfg["num_position_embeddings"]
        self.deepstack_visual_indexes = cfg["deepstack_visual_indexes"]
        self.hidden_act = cfg.get("hidden_act", "gelu_pytorch_tanh")
