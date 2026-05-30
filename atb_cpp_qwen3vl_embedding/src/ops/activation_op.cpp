#include "ops/activation_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

namespace {
OperationHandle CreateActivation(atb::infer::ActivationType type) {
    atb::infer::ActivationParam param;
    param.activationType = type;
    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(ActivationParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}
} // anonymous namespace

OperationHandle ActivationOp::MakeSiLU() {
    return CreateActivation(atb::infer::ACTIVATION_SWISH);
}

OperationHandle ActivationOp::MakeGELU() {
    return CreateActivation(atb::infer::ACTIVATION_GELU);
}

OperationHandle ActivationOp::MakeFastGELU() {
    return CreateActivation(atb::infer::ACTIVATION_FAST_GELU);
}

} // namespace ops
} // namespace atb_llm
