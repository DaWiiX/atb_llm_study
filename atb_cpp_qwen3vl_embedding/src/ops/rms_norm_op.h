#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB RMSNorm operation wrapper.
/// Corresponds to Python utils.py:make_rms_norm().
class RmsNormOp {
public:
    enum class LayerType {
        NORM = atb::infer::RmsNormParam::RMS_NORM_NORM,
        PRENORM = atb::infer::RmsNormParam::RMS_NORM_PRENORM,
        POSTNORM = atb::infer::RmsNormParam::RMS_NORM_POSTNORM,
    };

    /// Create an ATB RMSNorm operation.
    /// @param epsilon  Small constant for numerical stability (default: 1e-6)
    /// @param layer_type  Norm type: NORM, PRENORM, or POSTNORM
    /// @return RAII-managed operation handle
    static OperationHandle Create(float epsilon = 1e-6f,
                                  LayerType layer_type = LayerType::NORM);
};

} // namespace ops
} // namespace atb_llm
