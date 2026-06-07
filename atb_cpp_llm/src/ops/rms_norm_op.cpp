#include "ops/rms_norm_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle RmsNormOp::Create(float epsilon, LayerType layer_type) {
    atb::infer::RmsNormParam param;
    param.layerType = static_cast<atb::infer::RmsNormParam::RmsNormType>(layer_type);
    param.normParam.epsilon = epsilon;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(RmsNormParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
