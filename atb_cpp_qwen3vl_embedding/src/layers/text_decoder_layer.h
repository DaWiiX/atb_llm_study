#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {
namespace layers {

/// Qwen3VL Text Decoder Layer graph.
///
/// Pipeline:
///   hidden_states -> input_layernorm -> Attention -> +residual
///                 -> post_attention_layernorm -> MLP -> +residual -> output
///
/// Inputs (no mask):
///   [hidden_states,
///    q_weight, k_weight, v_weight, o_weight,
///    q_norm_weight, k_norm_weight,
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
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        int32_t num_heads,
                        int32_t num_kv_heads,
                        int32_t head_dim,
                        int32_t seq_len,
                        float epsilon,
                        bool use_mask,
                        OperationHandle& out);
};

} // namespace layers
} // namespace atb_llm
