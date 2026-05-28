#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {
namespace components {

/// SwiGLU MLP graph component.
///
/// Pipeline:
///   hidden_states -> gate_proj -> SiLU --|
///                    up_proj -----------|-> ElewiseMul -> down_proj -> output
///
/// Inputs: [hidden_states, gate_weight, up_weight, down_weight]
/// Output: [mlp_output]
class SwiGluMlpGraph {
public:
    /// Build a SwiGLU MLP graph.
    /// @param name  Graph name
    /// @param out   Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name, OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
