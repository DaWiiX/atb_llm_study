"""Qwen3VLTextModel ATB graph runner.

Split-graph strategy: a single DecoderLayer graph is built once and looped at
runtime with per-layer weights. A separate FinalNorm graph handles the output
normalization.

This avoids building a giant 28-layer graph that would cause OOM.
"""
import torch
from .utils import make_rms_norm, get_atb_builder
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

def run_text_layer(graph, hidden_states, layer_weights, cos, sin, seqlen,
                   causal_mask=None):
    """Execute one DecoderLayer through ATB on NPU.

    Args:
        graph:         the shared DecoderLayer ATB graph.
        hidden_states: (B, S, hidden_size) float tensor on CPU.
        layer_weights: list of 11 weight tensors for this layer (NPU float16).
        cos, sin:      (B*S, head_dim) float position embeddings on CPU.
        seqlen:        int, total tokens (B*S).
        causal_mask:   (S, S) float causal mask on NPU (optional).

    Returns (B, S, hidden_size) float tensor on CPU.
    """
    ntoken = hidden_states.shape[0] * hidden_states.shape[1]
    inputs = [hidden_states.half().npu()]
    inputs.extend(layer_weights)  # already NPU float16
    inputs.extend([cos.half().npu(), sin.half().npu()])
    if causal_mask is not None:
        inputs.append(causal_mask)
    inputs.append(torch.tensor([ntoken], dtype=torch.int32))
    return graph.forward(inputs)[0].cpu().float()


def run_text_layer_fast(graph, hidden_states, layer_weights, cos_npu, sin_npu,
                         seqlen, causal_mask=None):
    """Execute one DecoderLayer — cos/sin already NPU fp16.

    Args:
        graph:         the shared DecoderLayer ATB graph.
        hidden_states: (B, S, hidden_size) float tensor on CPU.
        layer_weights: list of 11 weight tensors (NPU float16).
        cos_npu, sin_npu: (B*S, head_dim) float16 on NPU (pre-transferred).
        seqlen:        int, total tokens (B*S).
        causal_mask:   (S, S) float causal mask on NPU (optional).

    Returns (B, S, hidden_size) float tensor on CPU.
    """
    ntoken = hidden_states.shape[0] * hidden_states.shape[1]
    inputs = [hidden_states.half().npu()]
    inputs.extend(layer_weights)
    inputs.extend([cos_npu, sin_npu])
    if causal_mask is not None:
        inputs.append(causal_mask)
    inputs.append(torch.tensor([ntoken], dtype=torch.int32))
    return graph.forward(inputs)[0].cpu().float()


def run_text_layer_npu(graph, hidden_npu, layer_weights, cos_npu, sin_npu,
                        seqlen, causal_mask=None):
    """Execute one DecoderLayer on NPU — hidden stays on NPU (no copy).

    Args:
        graph:         the shared DecoderLayer ATB graph.
        hidden_npu:    (B, S, hidden_size) float16 on NPU.
        layer_weights: list of 11 weight tensors (NPU float16).
        cos_npu, sin_npu: (B*S, head_dim) float16 on NPU.
        seqlen:        int, total tokens.
        causal_mask:   (S, S) float16 on NPU (optional).

    Returns (B, S, hidden_size) float16 on NPU.
    """
    ntoken = hidden_npu.shape[0] * hidden_npu.shape[1]
    inputs = [hidden_npu]
    inputs.extend(layer_weights)
    inputs.extend([cos_npu, sin_npu])
    if causal_mask is not None:
        inputs.append(causal_mask)
    inputs.append(torch.tensor([ntoken], dtype=torch.int32))
    return graph.forward(inputs)[0]


def run_text_norm(graph, hidden_states, norm_weight):
    """Execute final RMSNorm on ATB NPU.

    Returns (B, S, hidden_size) float tensor on CPU.
    """
    inputs = [hidden_states.half().npu(), norm_weight.half().npu()]
    return graph.forward(inputs)[0].cpu().float()


def run_text_norm_npu(graph, hidden_npu, norm_weight):
    """Execute final RMSNorm on ATB NPU — input and output stay on NPU."""
    inputs = [hidden_npu, norm_weight]
    return graph.forward(inputs)[0]


def run_text_model(text_model, hidden_states, cos, sin, seqlen,
                    layer_graph, norm_graph, use_mask=True):
    """Run full Qwen3VLTextModel through ATB graphs (loop over layers).

    Args:
        text_model:     Qwen3VLTextModel instance (for weight access).
        hidden_states:  (B, S, hidden_size) float tensor.
        cos, sin:       (B*S, head_dim) float position embeddings.
        seqlen:         int, total tokens.
        layer_graph:    pre-built DecoderLayer ATB graph.
        norm_graph:     pre-built FinalNorm ATB graph.
        use_mask:       whether layer_graph was built with MASK_TYPE_NORM.

    Returns (B, S, hidden_size) float tensor on CPU.
    """
    S = hidden_states.shape[1]
    causal_mask = make_causal_mask(S) if use_mask else None

    for layer in text_model.layers:
        w = collect_text_layer_weights(layer)
        hidden_states = run_text_layer(layer_graph, hidden_states, w,
                                       cos, sin, seqlen,
                                       causal_mask=causal_mask)

    return run_text_norm(norm_graph, hidden_states, text_model.norm.weight.data)
