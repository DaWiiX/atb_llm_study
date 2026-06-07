#include "ops/layer_norm_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle LayerNormOp::Create(float epsilon, int32_t begin_norm_axis,
                                     int32_t begin_params_axis) {
    atb::infer::LayerNormParam param;
    param.layerType = atb::infer::LayerNormParam::LAYER_NORM_NORM;
    param.normParam.epsilon = epsilon;
    param.normParam.beginNormAxis = begin_norm_axis;
    param.normParam.beginParamsAxis = begin_params_axis;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(LayerNormParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
