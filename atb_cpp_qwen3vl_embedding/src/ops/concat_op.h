#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB Concat operation wrapper.
/// Concatenates two tensors along a specified dimension.
class ConcatOp {
public:
    /// Create a Concat operation.
    /// @param concat_dim  Dimension to concatenate along
    static OperationHandle Create(int concat_dim);
};

} // namespace ops
} // namespace atb_llm
