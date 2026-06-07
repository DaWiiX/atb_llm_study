"""Vision PosEmb + RoPE ATB graph builder.

Computes position embedding interpolation and rotary position embeddings
entirely on NPU using ATB graph operations:

  Pos Embed path:
    4× Gather(pos_embed_weight, idx, axis=0) → 4× ElewiseMul(wt) → 3× ElewiseAdd

  RoPE path:
    2× Gather(freq_table, pid, axis=0) → Concat → Concat(duplicate) → Cos/Sin

CPU only needs to compute lightweight index/weight tensors (~10KB),
while the heavy gather + weighted sum + cos/sin runs on NPU.
"""
import torch
import torch_atb
from .utils import (
    make_elewise_add, make_elewise_mul, get_atb_builder,
    to_npu_half, to_cpu_float,
)


# ── Param factories ──────────────────────────────────────────────────

def make_gather(axis=0):
    """ATB GatherParam for index-based slice gathering."""
    return torch_atb.GatherParam(axis=axis)


def make_concat(concat_dim=1):
    """ATB ConcatParam for concatenating two tensors along a dimension."""
    return torch_atb.ConcatParam(concat_dim=concat_dim)


def make_elewise_cos():
    """ATB ElewiseParam for element-wise cosine."""
    p = torch_atb.ElewiseParam()
    p.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_COS
    return p


def make_elewise_sin():
    """ATB ElewiseParam for element-wise sine."""
    p = torch_atb.ElewiseParam()
    p.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_SIN
    return p


# ── Graph builder ────────────────────────────────────────────────────

def build_vision_posemb_graph():
    """Build ATB graph for vision position embedding + RoPE computation.

    Inputs (12):
        pe_w:     (num_grid², hidden_size) float16 — position embedding table
        idx00-11: (npatches,) int64 — bilinear interpolation 4-corner indices
        wt00-11:  (npatches, 1) float16 — bilinear interpolation 4-corner weights
        freq:     (max_hw, dim) float16 — rotary frequency table
        pid_row:  (npatches,) int64 — rotary row position IDs
        pid_col:  (npatches,) int64 — rotary col position IDs

    Outputs (3):
        pos_output: (npatches, hidden_size) float16 — interpolated position embedding
        cos_output: (npatches, dim*4) float16 — rotary cos
        sin_output: (npatches, dim*4) float16 — rotary sin
    """
    builder = get_atb_builder("VisPosEmb")

    # ── Declare inputs ─────────────────────────────────────────────
    pe_w = builder.add_input("pe_w")
    idx00 = builder.add_input("idx00")
    idx01 = builder.add_input("idx01")
    idx10 = builder.add_input("idx10")
    idx11 = builder.add_input("idx11")
    wt00 = builder.add_input("wt00")
    wt01 = builder.add_input("wt01")
    wt10 = builder.add_input("wt10")
    wt11 = builder.add_input("wt11")
    freq = builder.add_input("freq")
    pid_row = builder.add_input("pid_row")
    pid_col = builder.add_input("pid_col")

    # ── Pos Embed path: 4× Gather → 4× Mul → 3× Add ──────────────
    g00 = builder.add_node([pe_w, idx00], make_gather(axis=0))
    g01 = builder.add_node([pe_w, idx01], make_gather(axis=0))
    g10 = builder.add_node([pe_w, idx10], make_gather(axis=0))
    g11 = builder.add_node([pe_w, idx11], make_gather(axis=0))

    w00 = builder.add_node([g00.get_output(0), wt00], make_elewise_mul())
    w01 = builder.add_node([g01.get_output(0), wt01], make_elewise_mul())
    w10 = builder.add_node([g10.get_output(0), wt10], make_elewise_mul())
    w11 = builder.add_node([g11.get_output(0), wt11], make_elewise_mul())

    s01 = builder.add_node([w00.get_output(0), w01.get_output(0)], make_elewise_add())
    s012 = builder.add_node([s01.get_output(0), w10.get_output(0)], make_elewise_add())
    pos_out = builder.add_node([s012.get_output(0), w11.get_output(0)], make_elewise_add())

    # ── RoPE path: 2× Gather → Concat → Concat(duplicate) → Cos/Sin ─
    row_freq = builder.add_node([freq, pid_row], make_gather(axis=0))
    col_freq = builder.add_node([freq, pid_col], make_gather(axis=0))

    # Concat row and col frequencies → (npatches, dim*2)
    rope = builder.add_node([row_freq.get_output(0), col_freq.get_output(0)],
                            make_concat(concat_dim=1))

    # Duplicate: concat rope with itself → (npatches, dim*4)
    emb = builder.add_node([rope.get_output(0), rope.get_output(0)],
                           make_concat(concat_dim=1))

    cos_out = builder.add_node([emb.get_output(0)], make_elewise_cos())
    sin_out = builder.add_node([emb.get_output(0)], make_elewise_sin())

    # ── Mark outputs ───────────────────────────────────────────────
    builder.mark_output(pos_out.get_output(0))
    builder.mark_output(cos_out.get_output(0))
    builder.mark_output(sin_out.get_output(0))

    return builder.build()


# ── Runner ────────────────────────────────────────────────────────────

def _posemb_inputs(pe_w_npu, idx_wt, rope_idx, freq_npu):
    """Build graph inputs in ATB order.

    All tensor arguments should already be NPU-resident with correct dtype:
    - pe_w, wt*, freq: NPU float16
    - idx*, pid_*: NPU int64
    """
    inputs = [
        pe_w_npu,
        idx_wt['idx00'].npu(),
        idx_wt['idx01'].npu(),
        idx_wt['idx10'].npu(),
        idx_wt['idx11'].npu(),
        idx_wt['wt00'].unsqueeze(1).half().npu(),   # (N,) → (N,1) for broadcast
        idx_wt['wt01'].unsqueeze(1).half().npu(),
        idx_wt['wt10'].unsqueeze(1).half().npu(),
        idx_wt['wt11'].unsqueeze(1).half().npu(),
        freq_npu,
        rope_idx['pid_row'].npu(),
        rope_idx['pid_col'].npu(),
    ]
    return inputs


def run_posemb_npu(graph, pe_w_npu, idx_wt, rope_idx, freq_npu):
    """Execute VisPosEmb graph and return (pos, cos, sin) on NPU.

    Args:
        graph:     compiled ATB graph from build_vision_posemb_graph()
        pe_w_npu:  (num_grid², hidden_size) float16 NPU — pos embed table
        idx_wt:    dict from compute_posemb_indices()
        rope_idx:  dict from compute_rope_indices()
        freq_npu:  (max_hw, dim) float16 NPU — rotary frequency table

    Returns:
        (pos, cos, sin) — all NPU float16 tensors
    """
    inputs = _posemb_inputs(pe_w_npu, idx_wt, rope_idx, freq_npu)
    outputs = graph.forward(inputs)
    return outputs[0], outputs[1], outputs[2]
