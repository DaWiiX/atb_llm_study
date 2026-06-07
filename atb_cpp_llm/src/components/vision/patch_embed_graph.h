#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>

namespace atb_llm {
namespace components {

/// Qwen3VL Patch Embedding graph component.
///
/// Conv3d(kernel=stride=[tp, p, p]) is equivalent to:
///   reshape input (N*C*tp*p*p,) -> (N, C*tp*p*p)
///   Linear(hidden_size, C*tp*p*p, bias=True)
///
/// Inputs (3): [pixels, weight, bias]
/// Output: [patched]  shape (N, hidden_size)
class PatchEmbedGraph {
public:
    /// Build a Patch Embedding graph.
    /// @param name                Graph name
    /// @param in_channels         Number of input channels (3 for RGB)
    /// @param temporal_patch_size Temporal patch size
    /// @param patch_size          Spatial patch size
    /// @param embed_dim           Output embedding dimension (hidden_size)
    /// @param out                 Output: RAII operation handle
    /// @return STATUS_OK on success
    static Status Build(const std::string& name,
                        int32_t in_channels,
                        int32_t temporal_patch_size,
                        int32_t patch_size,
                        int32_t embed_dim,
                        OperationHandle& out);
};

} // namespace components
} // namespace atb_llm
