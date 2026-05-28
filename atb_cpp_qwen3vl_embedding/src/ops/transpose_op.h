#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB Transpose operation wrapper.
/// Permutes tensor dimensions according to the given permutation.
class TransposeOp {
public:
    /// Create a Transpose operation.
    /// @param perm  Dimension permutation, e.g. {0, 2, 1, 3}
    static OperationHandle Create(const std::vector<int32_t>& perm);
};

} // namespace ops
} // namespace atb_llm
