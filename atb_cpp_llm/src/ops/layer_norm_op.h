#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB LayerNorm operation wrapper.
/// Used by Vision encoder (LayerNorm, not RMSNorm).
class LayerNormOp {
public:
    /// Create an ATB LayerNorm operation.
    /// @param epsilon            Small constant for numerical stability (default: 1e-5)
    /// @param begin_norm_axis    Axis to start normalization from (default: 1, matches Python)
    /// @param begin_params_axis  Axis to start gamma/beta from (default: 1, matches Python)
    /// @return RAII-managed operation handle
    static OperationHandle Create(float epsilon = 1e-5f,
                                  int32_t begin_norm_axis = 1,
                                  int32_t begin_params_axis = 1);
};

} // namespace ops
} // namespace atb_llm
