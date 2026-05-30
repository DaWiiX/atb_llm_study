#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB element-wise operation wrapper.
/// Corresponds to Python utils.py:make_elewise_add(), make_elewise_mul(), etc.
class ElewiseOp {
public:
    /// Element-wise addition: a + b
    static OperationHandle MakeAdd();

    /// Element-wise multiplication: a * b
    static OperationHandle MakeMul();

    /// Data type cast
    /// @param out_dtype  Output data type (aclDataType)
    static OperationHandle MakeCast(aclDataType out_dtype);

    /// Element-wise multiply by scalar: a * value
    /// @param value  Scalar multiplier
    static OperationHandle MakeMuls(float value);

    /// Element-wise subtraction: a - b
    static OperationHandle MakeSub();
};

} // namespace ops
} // namespace atb_llm
