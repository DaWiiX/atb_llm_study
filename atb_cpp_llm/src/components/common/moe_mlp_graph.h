#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {
namespace components {

/// Mixture-of-Experts MLP graph component (stub / extension point).
///
/// Future implementation will provide:
///   hidden_states -> gate_logits -> TopK routing --|
///                                       |          |
///                                   Expert_0 ... Expert_N -> weight merge -> output
///
/// Expected inputs (future): [hidden_states, gate_weight, up_weight, down_weight,
///                            expert_weight_0 .. expert_weight_N]
/// Expected output (future): [mlp_output]
///
/// @note This is a placeholder for MoE support. The interface may change
///       once actual MoE model architectures are integrated.
class MoEMlpGraph {
public:
    /// Build a MoE MLP graph.
    /// @param name  Graph name
    /// @param out   Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name, OperationHandle& out);
};

}  // namespace components
}  // namespace atb_llm
