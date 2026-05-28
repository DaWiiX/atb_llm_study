#include "ops/concat_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle ConcatOp::Create(int concat_dim) {
    atb::infer::ConcatParam param;
    param.concatDim = concat_dim;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(ConcatParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
