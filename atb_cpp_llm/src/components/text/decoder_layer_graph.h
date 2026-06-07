#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {
namespace components {
namespace text {

/// Text Decoder Layer graph.
///
/// Composes SelfAttentionGraph + SwiGLU MLP.
/// Pipeline:
///   hidden_states -> input_layernorm -> SelfAttentionGraph -> +residual
///                 -> post_attention_layernorm -> SwiGLU MLP -> +residual -> output
///
/// Inputs (no mask, use_qk_norm=true):
///   [hidden_states,
///    q_weight, k_weight, v_weight, o_weight,
///    q_norm_weight, k_norm_weight,
///    gate_weight, up_weight, down_weight,
///    input_ln_weight, post_ln_weight,
///    cos, sin, seqlen]
///
/// Inputs (no mask, use_qk_norm=false):
///   [hidden_states,
///    q_weight, k_weight, v_weight, o_weight,
///    gate_weight, up_weight, down_weight,
///    input_ln_weight, post_ln_weight,
///    cos, sin, seqlen]
///
/// Inputs (with mask): same + [mask] inserted before seqlen
///
/// Output: [decoder_output]
class TextDecoderLayerGraph {
public:
    /// Build a TextDecoderLayer graph.
    /// @param name         Graph name
    /// @param num_heads    Number of query heads
    /// @param num_kv_heads Number of key/value heads
    /// @param head_dim     Dimension per head
    /// @param seq_len      Sequence length
    /// @param epsilon      RMSNorm epsilon
    /// @param use_mask     Whether to include mask input
    /// @param out          Output: RAII operation handle
    /// @param use_qk_norm  Whether to apply RMSNorm on Q/K projections (Qwen3VL default: true).
    ///                     When false, q_norm_weight and k_norm_weight inputs are omitted,
    ///                     and Q/K linear outputs feed directly into RoPE.
    /// @param rotary_dim   RoPE rotation coefficient (2=half, 4=quarter, headDim/2=full).
    ///                     Default 2 matches Qwen3VL LLAMA-style contiguous-half rotation.
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        int32_t num_heads,
                        int32_t num_kv_heads,
                        int32_t head_dim,
                        int32_t seq_len,
                        float epsilon,
                        bool use_mask,
                        OperationHandle& out,
                        bool use_qk_norm = true,
                        int32_t rotary_dim = 2);
};

}  // namespace text
}  // namespace components
}  // namespace atb_llm
