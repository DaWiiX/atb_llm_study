#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB Softmax operation wrapper.
class SoftmaxOp {
public:
    /// Create a Softmax operation.
    /// @param axes  Axes along which to compute softmax
    static OperationHandle Create(const std::vector<int64_t>& axes);
};

} // namespace ops
} // namespace atb_llm
