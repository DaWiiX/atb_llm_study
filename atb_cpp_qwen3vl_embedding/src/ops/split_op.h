#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB Split operation wrapper.
/// Splits a tensor along split_dim into split_num equal parts.
/// Corresponds to Python utils.py:make_split().
class SplitOp {
public:
    /// Create a Split operation.
    /// @param split_dim  Dimension to split along
    /// @param split_num  Number of equal splits (2 or 3)
    static OperationHandle Create(int32_t split_dim, int32_t split_num);
};

} // namespace ops
} // namespace atb_llm
