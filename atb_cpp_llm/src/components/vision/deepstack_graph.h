#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>

namespace atb_llm {
namespace components {

/// Deepstack Merger graph — convenience wrapper around VisionMergerGraph.
///
/// Deepstack features from intermediate vision blocks are fed through a
/// merger MLP and injected into early text decoder layers.
///
/// Pipeline (is_deepstack=true in VisionMergerGraph):
///   reshape(group_4) -> LayerNorm -> fc1 -> GELU -> fc2
///
/// Inputs (7): [x, n_w, n_b, f1_w, f1_b, f2_w, f2_b]
/// Output: [deepstack_output]
class DeepstackGraph {
public:
    /// Build a Deepstack Merger graph.
    /// @param name          Graph name
    /// @param hidden_size   Vision hidden size
    /// @param merge_size    Spatial merge size
    /// @param epsilon       LayerNorm epsilon
    /// @param out           Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        int32_t hidden_size,
                        int32_t merge_size,
                        float epsilon,
                        OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
