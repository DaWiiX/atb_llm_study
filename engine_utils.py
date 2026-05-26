"""
Engine utilities — config loading, weight loading, and pure-Python helpers
for MRoPE rotary embedding, position ID computation, and vision embeddings.

Zero dependency on transformers. Works directly with safetensors checkpoint
files and JSON configs from the model directory.
"""
import json
import math
import torch
import safetensors.torch


# ═════════════════════════════════════════════════════════════════════
# Config loading
# ═════════════════════════════════════════════════════════════════════

def load_config(model_dir: str) -> dict:
    """Load model config from config.json."""
    with open(f"{model_dir}/config.json") as f:
        return json.load(f)


def load_preprocessor_config(model_dir: str) -> dict:
    """Load preprocessor config from preprocessor_config.json."""
    with open(f"{model_dir}/preprocessor_config.json") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════
# Weight loading
# ═════════════════════════════════════════════════════════════════════

def load_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """Load all model weights from safetensors checkpoint into CPU tensors.

    Keys follow the HuggingFace format: model.language_model.*, model.visual.*
    All tensors are stored as float32 (converted from bf16 if needed).
    """
    weights = safetensors.torch.load_file(
        f"{model_dir}/model.safetensors", device="cpu")
    return {k: v.float() for k, v in weights.items()}


def get_text_layer_weights(weights: dict, layer_idx: int) -> list[torch.Tensor]:
    """Extract 11 weight tensors for a text decoder layer in ATB order.

    Return order: [q_w, k_w, v_w, o_w, qn_w, kn_w, gate_w, up_w, down_w,
                   input_ln_w, post_ln_w]
    """
    pfx = f"model.language_model.layers.{layer_idx}."
    a = lambda name: weights[f"{pfx}self_attn.{name}.weight"]
    m = lambda name: weights[f"{pfx}mlp.{name}.weight"]
    n = lambda name: weights[f"{pfx}{name}.weight"]
    return [a("q_proj"), a("k_proj"), a("v_proj"), a("o_proj"),
            a("q_norm"), a("k_norm"),
            m("gate_proj"), m("up_proj"), m("down_proj"),
            n("input_layernorm"), n("post_attention_layernorm")]


def get_text_norm_weight(weights: dict) -> torch.Tensor:
    """Get final text model norm weight."""
    return weights["model.language_model.norm.weight"]


def get_embed_weight(weights: dict) -> torch.Tensor:
    """Get token embedding weight (vocab_size, hidden_size)."""
    return weights["model.language_model.embed_tokens.weight"]


def get_vision_block_weights(weights: dict, block_idx: int) -> list[torch.Tensor]:
    """Extract 12 weight tensors for a vision block in ATB order.

    Return order: [qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b,
                    n1_w, n1_b, n2_w, n2_b]
    """
    pfx = f"model.visual.blocks.{block_idx}."
    b = lambda name: weights[f"{pfx}{name}"]
    return [b("attn.qkv.weight"), b("attn.qkv.bias"),
            b("attn.proj.weight"), b("attn.proj.bias"),
            b("mlp.linear_fc1.weight"), b("mlp.linear_fc1.bias"),
            b("mlp.linear_fc2.weight"), b("mlp.linear_fc2.bias"),
            b("norm1.weight"), b("norm1.bias"),
            b("norm2.weight"), b("norm2.bias")]


def get_patch_embed_weights(weights: dict, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get patch embedding weights reshaped for ATB Linear (Conv3d→Linear).

    Returns (weight, bias) where weight is (hidden_size, C*tp*p*p).
    """
    w = weights["model.visual.patch_embed.proj.weight"]  # (hs, C, tp, p, p)
    b = weights["model.visual.patch_embed.proj.bias"]    # (hs,)
    ksize = w.shape[1] * w.shape[2] * w.shape[3] * w.shape[4]  # C*tp*p*p
    w = w.reshape(hidden_size, ksize).contiguous()
    return w, b


def get_vision_pos_embed(weights: dict) -> torch.Tensor:
    """Get learned vision position embedding table."""
    return weights["model.visual.pos_embed.weight"]  # (2304, hidden_size)


def get_merger_weights(weights: dict, is_deepstack: bool = False,
                       ds_idx: int = 0) -> list[torch.Tensor]:
    """Get vision merger weights.

    Main merger: model.visual.merger.*
    Deepstack: model.visual.deepstack_merger_list.{ds_idx}.*

    Return order: [n_w, n_b, f1_w, f1_b, f2_w, f2_b]
    """
    if is_deepstack:
        pfx = f"model.visual.deepstack_merger_list.{ds_idx}."
    else:
        pfx = "model.visual.merger."
    b = lambda name: weights[f"{pfx}{name}"]
    return [b("norm.weight"), b("norm.bias"),
            b("linear_fc1.weight"), b("linear_fc1.bias"),
            b("linear_fc2.weight"), b("linear_fc2.bias")]


# ═════════════════════════════════════════════════════════════════════
# Vision RoPE embeddings (fast_pos_embed_interpolate + rot_pos_emb)
# ═════════════════════════════════════════════════════════════════════

class VisionRotaryEmbedding:
    """Pre-computed 2D rotary frequency table for vision encoder (Qwen3VLVisionRotaryEmbedding).

    head_dim // 2 frequencies per row/col, looked up by (row, col) 2D positions.
    """

    def __init__(self, dim: int, max_grid_size: int = 48):
        """
        Args:
            dim:            head_dim // 2 = hidden_size // (2 * num_heads)
            max_grid_size:  maximum H or W for frequency table (matches pos_embed grid)
        """
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_grid_size, dtype=torch.float32)
        self.freq_table = torch.einsum("i,j->ij", t, inv_freq)  # (max_grid_size, dim)
        self.dim = dim

    def __call__(self, max_hw: int) -> torch.Tensor:
        """Return frequency table slice up to max_hw positions."""
        return self.freq_table[:max_hw]  # (max_hw, dim)


def compute_rot_pos_emb(grid_thw: torch.Tensor, rotary_emb: VisionRotaryEmbedding,
                         merge_size: int = 2) -> torch.Tensor:
    """Compute vision 2D rotary position embeddings.

    Equivalent to Qwen3VLVisionModel.rot_pos_emb.

    Args:
        grid_thw:   (N, 3) tensor [t, h, w] for each image.
        rotary_emb: VisionRotaryEmbedding instance.
        merge_size: spatial_merge_size (default 2).

    Returns:
        (total_tokens, dim*2) tensor of rotary embeddings (sin values concatenated).
    """
    max_hw = int(grid_thw[:, 1:].max().item())
    freq_table = rotary_emb(max_hw)  # (max_hw, dim)

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long)

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(merge_size)
        intra_col = torch.arange(merge_size)

        row_idx = (block_rows[:, None, None, None] * merge_size +
                   intra_row[None, None, :, None])
        col_idx = (block_cols[None, :, None, None] * merge_size +
                   intra_col[None, None, None, :])

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset:offset + num_tokens] = coords
        offset += num_tokens

    return freq_table[pos_ids].flatten(1)  # (total_tokens, dim*2)


def fast_pos_embed_interpolate(grid_thw: torch.Tensor, pos_embed_weight: torch.Tensor,
                                num_grid_per_side: int, merge_size: int = 2) -> torch.Tensor:
    """Bilinear interpolation from learned 2D position embedding grid.

    Equivalent to Qwen3VLVisionModel.fast_pos_embed_interpolate.

    Args:
        grid_thw:          (N, 3) tensor.
        pos_embed_weight:  (num_grid_per_side^2, hidden_size) embedding table.
        num_grid_per_side: sqrt(num_position_embeddings), typically 48.
        merge_size:        spatial_merge_size (default 2).

    Returns:
        (total_patches, hidden_size) interpolated position embeddings with
        spatial merge (shuffle) applied.
    """
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        w_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        base_h = h_floor * num_grid_per_side
        base_h_ceil = h_ceil * num_grid_per_side

        indices = [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
        weights_ = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights_[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
    weight_tensor = torch.tensor(weight_list, dtype=pos_embed_weight.dtype)
    pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    patch_pos_embeds = patch_pos_embeds.split(
        [(h * w).item() for h, w in zip(grid_hs, grid_ws)])

    result = []
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t.item(), 1)
        pos_embed = (pos_embed
                     .view(t.item(), h.item() // merge_size, merge_size,
                           w.item() // merge_size, merge_size, -1)
                     .permute(0, 1, 3, 2, 4, 5)
                     .flatten(0, 4))
        result.append(pos_embed)
    return torch.cat(result)


# ═════════════════════════════════════════════════════════════════════
# MRoPE text rotary embedding (pure Python)
# ═════════════════════════════════════════════════════════════════════

class TextRotaryEmbedding:
    """MRoPE (Multi-dimensional Rotary Position Embedding) for the text model.

    Takes 3D position_ids (T, H, W) and produces interleaved cos/sin values.
    Equivalent to Qwen3VLTextRotaryEmbedding.
    """

    def __init__(self, head_dim: int, rope_theta: float = 5000000.0,
                 mrope_section: tuple = (24, 20, 20)):
        """
        Args:
            head_dim:      dimension per attention head (128).
            rope_theta:    base frequency (5,000,000 for Qwen3VL-Embedding-2B).
            mrope_section: section sizes for T, H, W dims in interleaved RoPE.
        """
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        dim = head_dim  # standard rope: inv_freq spans full head_dim with step 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.inv_freq = inv_freq

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Reorganize chunked [T...H...W] → interleaved [THWTHW...TT].

        Args:
            freqs: (3, bs, seq_len, head_dim//2)

        Returns:
            (bs, seq_len, head_dim//2)
        """
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MRoPE cos/sin from 3D position IDs.

        Args:
            position_ids: (3, batch, seq_len) LongTensor for T, H, W positions.

        Returns:
            (cos, sin), each of shape (batch, seq_len, head_dim).
        """
        B = position_ids.shape[1]
        # Expand inv_freq for all 3 dims
        inv_freq = self.inv_freq[None, None, :, None].expand(3, B, -1, 1)
        pid = position_ids[:, :, None, :].float()  # (3, B, 1, seq_len)

        freqs = (inv_freq @ pid).transpose(2, 3)  # (3, B, seq_len, head_dim//2)
        freqs = self._apply_interleaved_mrope(freqs)  # (B, seq_len, head_dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, seq_len, head_dim)
        return emb.cos(), emb.sin()


# ═════════════════════════════════════════════════════════════════════
# get_rope_index (pure Python)
# ═════════════════════════════════════════════════════════════════════

def get_rope_index(input_ids: torch.LongTensor,
                   image_grid_thw: torch.LongTensor | None = None,
                   video_grid_thw: torch.LongTensor | None = None,
                   attention_mask: torch.Tensor | None = None,
                   image_token_id: int = 151655,
                   video_token_id: int = 151656,
                   vision_start_token_id: int = 151652,
                   spatial_merge_size: int = 2,
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 3D MRoPE position IDs for text + image/video inputs.

    Equivalent to Qwen3VLModel.get_rope_index.

    For text-only: returns sequential (0, 1, 2, ...) for all 3 dims.
    For image+text: visual tokens get 2D grid positions, text tokens sequential.

    Returns:
        (position_ids, mrope_position_deltas).
        position_ids: (3, B, S) LongTensor.
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1],
                                   dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        for i, inp_ids in enumerate(total_input_ids):
            inp_ids = inp_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                inp_ids == vision_start_token_id).squeeze(1)
            vision_tokens = inp_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = inp_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (image_grid_thw[image_index][0],
                                image_grid_thw[image_index][1],
                                image_grid_thw[image_index][2])
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (video_grid_thw[video_index][0],
                                video_grid_thw[video_index][1],
                                video_grid_thw[video_index][2])
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(), h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size)
                text_len = ed - st

                st_idx = (llm_pos_ids_list[-1].max() + 1
                          if len(llm_pos_ids_list) > 0 else 0)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                    -1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (llm_pos_ids_list[-1].max() + 1
                          if len(llm_pos_ids_list) > 0 else 0)
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device)
        mrope_position_deltas = torch.zeros([input_ids.shape[0], 1],
                                             dtype=input_ids.dtype,
                                             device=input_ids.device)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(
                3, -1, -1).to(attention_mask.device)
            max_pid = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_pid + 1 - attention_mask.shape[-1]
        else:
            position_ids = (torch.arange(input_ids.shape[1],
                                          device=input_ids.device)
                            .view(1, 1, -1)
                            .expand(3, input_ids.shape[0], -1))
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1], device=input_ids.device,
                dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas
