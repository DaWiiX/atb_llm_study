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
import threading
import time
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
    make_causal_mask_nz_npu,
)


class Qwen3VLEngine:
    """Pure ATB inference engine for Qwen3-VL-Embedding.

    All computation runs on NPU via ATB graphs. Weights are loaded
    directly from safetensors files — no transformers dependency.
    """

    def __init__(self, model_dir: str):
        self._closed = False
        self._lock = threading.Lock()
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

        # ── Debug / test access ────────────────────────────────────
        self._last_ds_feats = None  # populated by _run_vision

    # ── Resource cleanup ─────────────────────────────────────────────

    def close(self):
        """Release all NPU resources (tensors, ATB graphs, cached buffers).

        Idempotent — safe to call multiple times.  Individual resource
        release failures are logged but do not prevent releasing the
        remaining resources.

        Thread-safe: the _closed flag is protected by _lock so that
        concurrent calls to close() / forward() / encode() cannot race
        and cause double-free.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

        resource_errors = []

        # ── Release ATB graphs ──────────────────────────────────────
        for attr in (
            "g_v_first", "g_v_block", "g_v_merger", "g_v_ds",
            "g_v_posemb", "g_t_norm", "g_t_layer",
        ):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    setattr(self, attr, None)
                    del obj
            except Exception as e:
                resource_errors.append(f"{attr}: {e}")

        # ── Release NPU weight tensors ──────────────────────────────
        # Large nested structures — use pop/clear where possible.
        weight_attrs = [
            "embed_w", "norm_w", "v_pe_w", "v_pe_b",
            "v_pe_w_table", "v_pos_embed",
        ]
        for attr in weight_attrs:
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    setattr(self, attr, None)
                    del obj
            except Exception as e:
                resource_errors.append(f"{attr}: {e}")

        # Nested weight lists / dicts
        for top_attr in (
            "t_layer_weights", "v_block_weights",
            "v_merger_w", "v_ds_w",
        ):
            try:
                container = getattr(self, top_attr, None)
                if container is None:
                    continue
                # Clear each sub-list/item
                if isinstance(container, (list, tuple)):
                    for item in container:
                        if isinstance(item, (list, tuple)):
                            item.clear()
                    container.clear()
                setattr(self, top_attr, None)
                del container
            except Exception as e:
                resource_errors.append(f"{top_attr}: {e}")

        # ── Release cached mask ─────────────────────────────────────
        try:
            mask = getattr(self, '_cached_mask', None)
            if mask is not None:
                self._cached_mask = None
                del mask
        except Exception as e:
            resource_errors.append(f"_cached_mask: {e}")

        # ── Release CPU weight dict ─────────────────────────────────
        try:
            weights = getattr(self, 'weights', None)
            if weights is not None:
                weights.clear()
                self.weights = None
                del weights
        except Exception as e:
            resource_errors.append(f"weights: {e}")

        # ── Release RoPE objects (CPU side) ─────────────────────────
        for attr in ("text_rope", "vis_rotary"):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    setattr(self, attr, None)
                    del obj
            except Exception as e:
                resource_errors.append(f"{attr}: {e}")

        # ── Reclaim NPU memory ──────────────────────────────────────
        try:
            torch.npu.empty_cache()
        except Exception as e:
            resource_errors.append(f"empty_cache: {e}")

        # Log any errors that occurred during cleanup.
        if resource_errors:
            try:
                import sys
                print(
                    f"[Qwen3VLEngine.close] {len(resource_errors)} resource "
                    f"release error(s): {resource_errors}", file=sys.stderr)
            except Exception:
                pass

    def __del__(self):
        """Destructor — release NPU resources on garbage collection."""
        try:
            self.close()
        except Exception:
            # Must not raise from __del__, and we cannot assume that
            # torch / torch_npu / ATB are still importable at this point.
            pass

    def __enter__(self):
        """Enter context manager — returns self for use in `with` block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager — always closes resources.

        Does not suppress exceptions; the caller must handle them.
        """
        self.close()
        return False  # do not suppress any exception

    def _make_vision_config(self):
        """Create a config-like object for ATB vision graph builders."""
        return VisionConfigWrapper(self.v_cfg)

    def _ensure_text_graph(self, S: int):
        """Lazily build text decoder layer graph and cached mask for sequence length S."""
        if self._text_S == S and self.g_t_layer is not None:
            return
        self._text_S = S
        self.g_t_layer = build_text_layer_graph(
            self.nh_t, self.nkv_t, self.hd_t, self.interm_t,
            B=1, S=S, use_mask=True)
        if is_310p():
            # 310P: causal mask in NZ layout with FRACTAL_NZ format tag.
            # The format tag is critical — without it, ATB SelfAttention
            # sees ND format, tries internal TransdataOperation, and
            # fails on 310P with "call operation setup fail".
            mask = make_causal_mask_nz_npu(S)
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

    def _run_vision(self, pixel_values, grid_thw, return_intermediates=False):
        """Run full VisionModel on ATB NPU — outputs stay on NPU.

        Args:
            pixel_values: Preprocessed pixel values (float32 CPU tensor).
            grid_thw: Grid dimensions tensor.
            return_intermediates: If True, also returns a dict with per-stage
                wall-clock timing ('vision_pos', 'vision_model') measured with
                torch.npu.synchronize() boundaries.

        Returns:
            (vis_npu, ds_feats_npu) when return_intermediates=False (default).
            (vis_npu, ds_feats_npu, intermediates_dict) when True.
        """
        if return_intermediates:
            torch.npu.synchronize()
            t0 = time.perf_counter()

        # ── Position embedding + RoPE via ATB graph (NPU) ────────────
        idx_wt = compute_posemb_indices(grid_thw, self.num_grid, self.merge_size)
        rope_idx = compute_rope_indices(grid_thw, self.vis_rotary, self.merge_size)
        freq_npu = to_npu_half(rope_idx['freq_table'])

        pos_npu, cos_npu, sin_npu = run_posemb_npu(
            self.g_v_posemb, self.v_pe_w_table, idx_wt, rope_idx, freq_npu)

        if return_intermediates:
            torch.npu.synchronize()
            t1 = time.perf_counter()
            intermediates = {'vision_pos': t1 - t0}
            t0 = t1

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
                # Sync: the deepstack merger graph may run on a different
                # internal NPU stream than the VisionBlock graph above.
                # Without this barrier it can read h before the block
                # output is fully written.
                torch.npu.synchronize()
                ds_feats.append(run_merger_npu(self.g_v_ds, h, self.v_ds_w[ds_idx]))

        # Sync: the last VisionBlock in the loop above is async (ATB
        # graph.Run() returns immediately).  Without this barrier the
        # subsequent merger graph may read h before the block output is
        # fully computed — silent data corruption when graphs execute on
        # different internal NPU streams.
        torch.npu.synchronize()
        vis = run_merger_npu(self.g_v_merger, h, self.v_merger_w)

        if return_intermediates:
            torch.npu.synchronize()
            intermediates['vision_model'] = time.perf_counter() - t0

        self._last_ds_feats = ds_feats  # store for test introspection
        if return_intermediates:
            return vis, ds_feats, intermediates
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
                # Sync: run_text_layer_npu returns immediately (ATB graph
                # executes on a separate NPU stream).  Without this barrier
                # the PyTorch clone/read below may see stale hidden data.
                torch.npu.synchronize()
                local = hidden[0, visual_mask, :].clone() + deepstack_features[li]
                hidden[0, visual_mask, :] = local

        # Sync: the last DecoderLayer execution in the loop above is async
        # (ATB graph.Run() returns immediately).  Without this barrier,
        # FinalNorm may read hidden states before the last layer's output
        # is fully available on the NPU.
        torch.npu.synchronize()
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

        Raises:
            RuntimeError: if the engine has been closed.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Engine is closed and cannot be used.")
        # 1. Text embeddings → NPU
        inputs_embeds = F.embedding(input_ids, self.embed_w).half().npu()

        # 2. Vision features — all NPU-resident
        ds_feats = []
        vis_mask = None
        if pixel_values is not None and image_grid_thw is not None:
            vis_embeds, ds_feats = self._run_vision(pixel_values, image_grid_thw)
            # Sync: vis_embeds was produced by async ATB vision graphs
            # (run_merger_npu returns immediately).  Without this barrier,
            # the scatter below may read stale/partial vis_embeds data.
            # The mask H2D (vis_mask.npu()) is on the same PyTorch NPU
            # stream as the scatter — stream ordering handles the mask.
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

        Raises:
            RuntimeError: if the engine has been closed.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Engine is closed and cannot be used.")
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
