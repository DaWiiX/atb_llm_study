#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {
namespace components {

/// Text Attention graph component.
///
/// Pipeline:
///   hidden_states -> q_proj -> reshape(B*S,nh,hd) -> q_norm -> reshape(B*S,nh*hd) --|
///                   -> k_proj -> reshape(B*S,kv_nh,hd) -> k_norm -> reshape(B*S,*) --|-> RopeOp -> SelfAttention -> reshape(B,S,*) -> o_proj
///                   -> v_proj -> reshape(B*S,kv_nh,hd) --------------------------------|
///   cos, sin, seqlen -> RopeOp
///   mask (optional)  -> SelfAttention
///
/// When use_qk_norm=false, Q/K norms are skipped and Q/K linear outputs
/// feed directly into RoPE. This matches models without QK normalization
/// (e.g., standard LLaMA, Mistral).
///
/// Inputs (with qk_norm, no mask): [hidden_states, q_weight, k_weight, v_weight, o_weight,
///                                   q_norm_weight, k_norm_weight, cos, sin, seqlen]
/// Inputs (with qk_norm, with mask): same + [mask] inserted before seqlen
/// Inputs (no qk_norm, no mask): [hidden_states, q_weight, k_weight, v_weight, o_weight,
///                                 cos, sin, seqlen]
/// Inputs (no qk_norm, with mask): same + [mask] inserted before seqlen
/// Output: [attention_output]
class SelfAttentionGraph {
public:
    /// Build a SelfAttention graph.
    /// @param name         Graph name
    /// @param num_heads    Number of query heads
    /// @param num_kv_heads Number of key/value heads (GQA)
    /// @param head_dim     Dimension per head
    /// @param seq_len      Sequence length (for reshape, currently unused)
    /// @param epsilon      RMSNorm epsilon for Q/K normalization
    /// @param use_mask     Whether to include mask input
    /// @param out          Output: RAII operation handle
    /// @param use_qk_norm  Whether to apply RMSNorm on Q/K projections. Default true.
    /// @param rotary_dim   RoPE rotation coefficient (2=half, 4=quarter, headDim/2=full).
    ///                     Default 2 matches LLAMA-style contiguous-half rotation.
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

} // namespace components
} // namespace atb_llm
