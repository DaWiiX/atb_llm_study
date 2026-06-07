#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>

namespace atb_llm {
namespace components {

/// Qwen3VL Vision Merger graph component.
///
/// Main merger pipeline:
///   LayerNorm(hidden) -> Reshape(group_4) -> fc1 Linear -> GELU -> fc2 Linear
///
/// Deepstack merger pipeline:
///   Reshape(group_4) -> LayerNorm(reshaped) -> fc1 Linear -> GELU -> fc2 Linear
///
/// The fc2 projects to text hidden_size.
///
/// Inputs (7): [x, n_w, n_b, f1_w, f1_b, f2_w, f2_b]
/// Output: [merged_output]
class VisionMergerGraph {
public:
    /// Build a Vision Merger graph.
    /// @param name          Graph name
    /// @param hidden_size   Vision hidden size
    /// @param merge_size    Spatial merge size (typically 2)
    /// @param is_deepstack  If true, use deepstack ordering (reshape before norm)
    /// @param epsilon       LayerNorm epsilon
    /// @param out           Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        int32_t hidden_size,
                        int32_t merge_size,
                        bool is_deepstack,
                        float epsilon,
                        OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
