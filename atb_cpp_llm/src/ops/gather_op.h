#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"
#include <cstdint>

namespace atb_llm {
namespace ops {

/// ATB Gather operation wrapper.
/// Gathers slices from input tensor along an axis using index tensor.
class GatherOp {
public:
    /// Create a Gather operation.
    /// @param axis        Axis to gather along (default: 0)
    /// @param batch_dims  Number of batch dimensions (default: 0)
    /// @return RAII-managed operation handle
    static OperationHandle Create(int64_t axis = 0, int64_t batch_dims = 0);
};

} // namespace ops
} // namespace atb_llm
