#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB Linear operation wrapper (matmul with optional bias).
/// Corresponds to Python utils.py:make_linear().
class LinearOp {
public:
    /// Create an ATB Linear operation.
    /// @param has_bias  Whether to add bias after matmul (default: false)
    /// @param transpose_a  Whether to transpose A matrix (default: false)
    /// @param transpose_b  Whether to transpose B matrix (default: true, weight is [n,k])
    /// @return RAII-managed operation handle
    static OperationHandle Create(bool has_bias = false,
                                  bool transpose_a = false,
                                  bool transpose_b = true);
};

} // namespace ops
} // namespace atb_llm
