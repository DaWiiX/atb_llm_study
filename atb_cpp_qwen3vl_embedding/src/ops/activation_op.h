#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB activation function wrapper.
/// Corresponds to Python utils.py:make_silu().
class ActivationOp {
public:
    /// SiLU/Swish activation: x * sigmoid(x)
    static OperationHandle MakeSiLU();

    /// GELU activation (tanh approximation)
    static OperationHandle MakeGELU();

    /// Fast GELU activation
    static OperationHandle MakeFastGELU();
};

} // namespace ops
} // namespace atb_llm
