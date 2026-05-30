#include "ops/rope_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle RopeOp::Create(int32_t rotaryCoeff) {
    atb::infer::RopeParam param;
    param.rotaryCoeff = rotaryCoeff;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(RopeParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

}  // namespace ops
}  // namespace atb_llm
