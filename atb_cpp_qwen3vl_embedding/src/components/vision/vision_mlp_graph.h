#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {
namespace components {

/// Qwen3VL Vision MLP graph component.
///
/// Pipeline:
///   hidden -> fc1 Linear -> GELU -> fc2 Linear -> output
///
/// Note: Vision uses GELU (not SiLU) and simple two-layer MLP (not SwiGLU).
///
/// Inputs (5): [hidden, fc1_w, fc1_b, fc2_w, fc2_b]
/// Output: [mlp_output]
class VisionMlpGraph {
public:
    /// Build a Vision MLP graph.
    /// @param name  Graph name
    /// @param out   Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name, OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
