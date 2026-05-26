"""
Pure ATB inference engine for Qwen3-VL-Embedding-2B.

Zero dependency on transformers. Works directly with safetensors checkpoint
files and JSON configs from a model directory.

Usage:
    from atb_python_model.utils import set_atb_buffer_size
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
    TextRotaryEmbedding, get_rope_index,
)
from .vision_model import (
    build_vision_first_layer, build_vision_merger, build_deepstack_merger,
    run_first_layer, run_block, run_merger,
)
from .vision_block import build_vision_block
from .text_model import (
    build_text_layer_graph, build_text_norm_graph,
    run_text_layer, run_text_norm, make_causal_mask,
)
from .utils import set_atb_buffer_size
from .text_model import make_causal_mask
from .vision_model import (
    build_vision_first_layer, build_vision_merger, build_deepstack_merger,
    run_first_layer, run_block, run_merger,
)
from .vision_block import build_vision_block
from .text_model import (
    build_text_layer_graph, build_text_norm_graph,
    run_text_layer, run_text_norm,
    make_causal_mask,
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

        # ── Cached weights ───────────────────────────────────────────
        self.embed_w = get_embed_weight(self.weights)
        self.norm_w = get_text_norm_weight(self.weights)
        self.t_layer_weights = [get_text_layer_weights(self.weights, i)
                                for i in range(self.n_layer)]

        # Vision: block weights by layer
        self.v_block_weights = [get_vision_block_weights(self.weights, i)
                               for i in range(self.v_depth)]
        self.v_pe_w, self.v_pe_b = get_patch_embed_weights(
            self.weights, self.v_cfg["hidden_size"])
        self.v_pos_embed = get_vision_pos_embed(self.weights)
        self.num_grid = int(self.v_cfg["num_position_embeddings"] ** 0.5)

        # Vision: merger weights
        self.v_merger_w = get_merger_weights(self.weights, is_deepstack=False)
        self.v_ds_w = [get_merger_weights(self.weights, is_deepstack=True, ds_idx=i)
                       for i in range(len(self.ds_indexes))]

        # ── Build ATB graphs ────────────────────────────────────────
        self._build_graphs()

    def _build_graphs(self):
        """Build all ATB graphs."""
        # Vision
        self.g_v_first = build_vision_first_layer(self._make_vision_config())
        _, self.g_v_block, _ = build_vision_block(self.nh_v, self.hd_v, "VisBlock")
        self.g_v_merger = build_vision_merger(self._make_vision_config())
        self.g_v_ds = build_deepstack_merger(self._make_vision_config())

        # Text
        self.g_t_norm = build_text_norm_graph(self.hidden_t)
        self._text_S = None
        self.g_t_layer = None

    def _make_vision_config(self):
        """Create a config-like object for ATB vision graph builders."""
        return VisionConfigWrapper(self.v_cfg)

    def _ensure_text_graph(self, S: int):
        """Lazily build text decoder layer graph for sequence length S."""
        if self._text_S == S and self.g_t_layer is not None:
            return
        self._text_S = S
        self.g_t_layer = build_text_layer_graph(
            self.nh_t, self.nkv_t, self.hd_t, self.interm_t,
            B=1, S=S, use_mask=True)

    # ── Preprocessing ──────────────────────────────────────────────

    def preprocess_image(self, image: torch.Tensor):
        """Preprocess raw image (C,H,W uint8) → ATB-compatible format."""
        return preprocess_image(
            image, patch_size=self.patch_size,
            temporal_patch_size=self.tp, merge_size=self.merge_size,
            min_pixels=self.pp_min_px, max_pixels=self.pp_max_px)

    # ── Vision inference ───────────────────────────────────────────

    def _run_vision(self, pixel_values, grid_thw):
        """Run full VisionModel on ATB NPU.

        Returns (merged_features, deepstack_features_read_by, deepstack_features).
        """
        # Position embed (CPU)
        pos = fast_pos_embed_interpolate(
            grid_thw, self.v_pos_embed, self.num_grid, self.merge_size)

        # Rotary embed (CPU)
        rope = compute_rot_pos_emb(grid_thw, self.vis_rotary, self.merge_size)
        rope = rope.reshape(pixel_values.shape[0], -1)
        emb = torch.cat((rope, rope), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        # Block 0 (ATB)
        h = run_first_layer(self.g_v_first, pixel_values,
                            self.v_pe_w, self.v_pe_b, pos, cos, sin,
                            self.v_block_weights[0])

        # Blocks 1..23 (ATB, looped)
        ds_feats = []
        for li in range(1, self.v_depth):
            h = run_block(self.g_v_block, h, self.v_block_weights[li], cos, sin)
            if li in self.ds_indexes:
                ds_idx = self.ds_indexes.index(li)
                ds = run_merger(self.g_v_ds, h, self.v_ds_w[ds_idx])
                ds_feats.append(ds)

        # Main merger (ATB)
        merged = run_merger(self.g_v_merger, h, self.v_merger_w)
        return merged, ds_feats

    # ── Text inference ─────────────────────────────────────────────

    def _run_text(self, inputs_embeds, position_ids,
                  visual_mask=None, deepstack_features=None):
        """Run full TextModel on ATB NPU."""
        S = inputs_embeds.shape[1]
        self._ensure_text_graph(S)
        cm = make_causal_mask(S)

        hidden = inputs_embeds
        cos, sin = self.text_rope(position_ids)
        cos_f = cos.reshape(-1, self.hd_t)
        sin_f = sin.reshape(-1, self.hd_t)

        for li in range(self.n_layer):
            hidden = run_text_layer(self.g_t_layer, hidden,
                                    self.t_layer_weights[li],
                                    cos_f, sin_f, S,
                                    causal_mask=cm)
            if deepstack_features and li < len(deepstack_features):
                ds = deepstack_features[li].to(hidden.dtype)
                hidden[0, visual_mask, :] += ds

        return run_text_norm(self.g_t_norm, hidden, self.norm_w)

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
        # 1. Text embeddings (CPU)
        inputs_embeds = F.embedding(input_ids, self.embed_w)

        # 2. Vision features (ATB + CPU injection)
        ds_feats = []
        vis_mask = None
        if pixel_values is not None and image_grid_thw is not None:
            vis_embeds, ds_feats = self._run_vision(pixel_values, image_grid_thw)
            vis_mask = input_ids.squeeze(0) == self.img_tok
            inputs_embeds[0, vis_mask, :] = vis_embeds.to(inputs_embeds)

        # 3. Position IDs (CPU)
        position_ids, _ = get_rope_index(
            input_ids, image_grid_thw, None, None,
            image_token_id=self.img_tok,
            spatial_merge_size=self.spatial_merge)

        # 4. Text model (ATB)
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
