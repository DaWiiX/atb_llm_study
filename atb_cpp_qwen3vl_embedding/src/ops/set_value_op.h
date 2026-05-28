#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace ops {

/// ATB SetValue operation wrapper.
/// Copies source tensor into destination tensor at specified positions.
/// Used for injecting vision tokens into the text embedding.
class SetValueOp {
public:
    /// Create a SetValue operation.
    /// @param starts  Start indices for each dimension
    /// @param ends    End indices (exclusive) for each dimension
    /// @return RAII-managed operation handle
    static OperationHandle Create(const std::vector<int64_t>& starts,
                                  const std::vector<int64_t>& ends);
};

} // namespace ops
} // namespace atb_llm
