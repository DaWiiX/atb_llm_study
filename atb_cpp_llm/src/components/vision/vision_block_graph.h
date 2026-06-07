#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>

namespace atb_llm {
namespace components {

/// Qwen3VL Vision Block graph component.
///
/// Pipeline:
///   hidden -> LayerNorm -> VisionAttention -> +residual
///          -> LayerNorm -> VisionMLP -> +residual -> output
///
/// Inputs (16): [hidden,
///               qkv_w, qkv_b, proj_w, proj_b,
///               fc1_w, fc1_b, fc2_w, fc2_b,
///               n1_w, n1_b, n2_w, n2_b,
///               cos, sin, seqlen]
/// Output: [block_output]
class VisionBlockGraph {
public:
    /// Build a Vision Block graph.
    /// @param name      Graph name
    /// @param num_heads Number of attention heads
    /// @param head_dim  Dimension per head
    /// @param epsilon   LayerNorm epsilon
    /// @param out       Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        int32_t num_heads,
                        int32_t head_dim,
                        float epsilon,
                        OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
