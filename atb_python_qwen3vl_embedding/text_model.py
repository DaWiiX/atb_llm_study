"""Qwen3VLTextModel ATB graph runner.

Split-graph strategy: a single DecoderLayer graph is built once and looped at
runtime with per-layer weights. A separate FinalNorm graph handles the output
normalization.

This avoids building a giant 28-layer graph that would cause OOM.
"""
import torch
from .utils import (
    make_rms_norm, get_atb_builder,
    make_seqlen_tensor, prepare_npu_weights, to_cpu_float, to_npu_half,
)
from .text_decoder_layer import build_decoder_layer


def build_text_layer_graph(num_heads, num_kv_heads, head_dim,
                           intermediate_size, B=1, S=8, eps=1e-6,
                           use_mask=True):
    """Build the shared DecoderLayer graph (built once, reused across layers).

    Returns the compiled ATB graph operation.
    """
    _, graph_op, _ = build_decoder_layer(
        num_heads, num_kv_heads, head_dim, intermediate_size,
        B=B, S=S, eps=eps, use_mask=use_mask)
    return graph_op


def build_text_norm_graph(hidden_size, eps=1e-6):
    """Build final RMSNorm graph for text model output."""
    builder = get_atb_builder("TextFinalNorm")
    x_in = builder.add_input("x")
    w_in = builder.add_input("w")
    ln = builder.add_node([x_in, w_in], make_rms_norm(eps))
    builder.mark_output(ln.get_output(0))
    return builder.build()


# ── Weight collectors ────────────────────────────────────────────────

def collect_text_layer_weights(layer):
    """Extract 11 weight tensors from a Qwen3VLTextDecoderLayer in ATB order.

    Order: [q_w, k_w, v_w, o_w, qn_w, kn_w, gate_w, up_w, down_w, iln_w, pln_w]
    """
    a = layer.self_attn
    m = layer.mlp
    return [
        a.q_proj.weight.data, a.k_proj.weight.data, a.v_proj.weight.data,
        a.o_proj.weight.data,
        a.q_norm.weight.data, a.k_norm.weight.data,
        m.gate_proj.weight.data, m.up_proj.weight.data, m.down_proj.weight.data,
        layer.input_layernorm.weight.data,
        layer.post_attention_layernorm.weight.data,
    ]


# ── Mask factory ─────────────────────────────────────────────────────

def make_causal_mask(S, device='cpu'):
    """Create additive causal mask for ATB MASK_TYPE_NORM.

    Returns (S, S) float32 tensor on CPU.
        0 = attend, -65504 (fp16 min) = mask.
    The caller must convert to float16 and move to NPU before inference.
    """
    mask = torch.zeros(S, S)
    mask[torch.triu(torch.ones(S, S), diagonal=1).bool()] = -65504.0
    return mask.to(device)


# ── Runners ──────────────────────────────────────────────────────────

def _text_layer_inputs(hidden, layer_weights, cos, sin, causal_mask=None):
    """Build DecoderLayer inputs in graph order.

    The final sequence-length tensor is intentionally kept on CPU int32 because
    ATB Rope/SelfAttention expect a host-side length tensor.
    """
    ntoken = hidden.shape[0] * hidden.shape[1]
    inputs = [to_npu_half(hidden)]
    inputs.extend(prepare_npu_weights(layer_weights))
    inputs.extend([to_npu_half(cos), to_npu_half(sin)])
    if causal_mask is not None:
        inputs.append(to_npu_half(causal_mask))
    inputs.append(make_seqlen_tensor(ntoken))
    return inputs


def run_text_layer(graph, hidden_states, layer_weights, cos, sin, seqlen,
                   causal_mask=None):
    """Execute one DecoderLayer and return CPU float output.

    This CPU-facing wrapper accepts ordinary CPU tensors (including weights and
    masks), prepares ATB inputs, and preserves the public signature. ``seqlen``
    is retained for compatibility; the graph length is derived from
    ``hidden_states`` and passed as a CPU int32 tensor.
    """
    inputs = _text_layer_inputs(hidden_states, layer_weights, cos, sin,
                                causal_mask=causal_mask)
    return to_cpu_float(graph.forward(inputs)[0])


def run_text_layer_fast(graph, hidden_states, layer_weights, cos_npu, sin_npu,
                         seqlen, causal_mask=None):
    """Compatibility wrapper for callers that pre-transfer cos/sin to NPU."""
    return run_text_layer(graph, hidden_states, layer_weights, cos_npu, sin_npu,
                          seqlen, causal_mask=causal_mask)


def run_text_layer_npu(graph, hidden_npu, layer_weights, cos_npu, sin_npu,
                        seqlen, causal_mask=None):
    """Execute one DecoderLayer and keep output on NPU.

    Inputs may already be NPU-resident; placement helpers are no-op-safe. The
    ``seqlen`` argument is kept for API compatibility and the actual CPU int32
    graph input is derived from ``hidden_npu``.
    """
    inputs = _text_layer_inputs(hidden_npu, layer_weights, cos_npu, sin_npu,
                                causal_mask=causal_mask)
    return graph.forward(inputs)[0]


def run_text_norm(graph, hidden_states, norm_weight):
    """Execute final RMSNorm and return CPU float output."""
    inputs = [to_npu_half(hidden_states), to_npu_half(norm_weight)]
    return to_cpu_float(graph.forward(inputs)[0])


def run_text_norm_npu(graph, hidden_npu, norm_weight):
    """Execute final RMSNorm and keep output on NPU."""
    inputs = [to_npu_half(hidden_npu), to_npu_half(norm_weight)]
    return graph.forward(inputs)[0]


def run_text_model(text_model, hidden_states, cos, sin, seqlen,
                    layer_graph, norm_graph, use_mask=True):
    """Run full Qwen3VLTextModel through ATB graphs.

    The public API is CPU-facing and returns CPU float output, but the layer
    loop stays NPU-resident to avoid repeated transfers. ``seqlen`` is retained
    for compatibility; per-layer graph lengths are derived from ``hidden``.
    """
    S = hidden_states.shape[1]
    hidden = to_npu_half(hidden_states)
    cos_npu = to_npu_half(cos)
    sin_npu = to_npu_half(sin)
    causal_mask = to_npu_half(make_causal_mask(S)) if use_mask else None

    for layer in text_model.layers:
        w = prepare_npu_weights(collect_text_layer_weights(layer))
        hidden = run_text_layer_npu(layer_graph, hidden, w, cos_npu, sin_npu,
                                    seqlen, causal_mask=causal_mask)

    normed = run_text_norm_npu(norm_graph, hidden, text_model.norm.weight.data)
    return to_cpu_float(normed)
