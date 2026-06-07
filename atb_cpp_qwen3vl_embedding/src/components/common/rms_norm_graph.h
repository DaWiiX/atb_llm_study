#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <string>

namespace atb_llm {
namespace components {

/// RMSNorm graph component.
/// Wraps a single RMSNorm ATB operation as a standalone graph.
/// Inputs: [input, weight]
/// Output: [normalized_output]
class RmsNormGraph {
public:
    /// Build an RMSNorm graph.
    /// @param name     Graph name
    /// @param epsilon  RMSNorm epsilon
    /// @param out      Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        float epsilon,
                        OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
