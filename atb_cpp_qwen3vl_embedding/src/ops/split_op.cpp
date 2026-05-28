#include "ops/split_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle SplitOp::Create(int32_t split_dim, int32_t split_num) {
    atb::infer::SplitParam param;
    param.splitDim = split_dim;
    param.splitNum = split_num;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(SplitParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
