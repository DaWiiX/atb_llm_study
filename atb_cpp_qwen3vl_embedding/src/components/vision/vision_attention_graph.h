#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>

namespace atb_llm {
namespace components {

/// Qwen3VL Vision Attention graph component.
///
/// Pipeline:
///   hidden -> QKV Linear -> Reshape(N, 3, nh, hd) -> Split(3-way, dim=1)
///     Q(N,1,nh,hd) -> Flatten(N, nh*hd) -+
///     K(N,1,nh,hd) -> Flatten(N, nh*hd) -+-> RoPE -> Reshape(N,nh,hd)
///     V(N,1,nh,hd) -> Reshape(N, nh, hd)
///   Q_rope, K_rope, V_3d, seqlen -> SelfAttention -> Flatten(N, nh*hd) -> proj Linear
///
/// Note: Vision uses combined QKV (single Linear), not separate Q/K/V like Text.
/// No Q/K RMSNorm in vision attention (unlike text attention).
///
/// Inputs (8): [hidden, qkv_w, qkv_b, proj_w, proj_b, cos, sin, seqlen]
/// Output: [attention_output]
class VisionAttentionGraph {
public:
    /// Build a Vision Attention graph.
    /// @param name      Graph name
    /// @param num_heads Number of attention heads
    /// @param head_dim  Dimension per head
    /// @param out       Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        int32_t num_heads,
                        int32_t head_dim,
                        OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
