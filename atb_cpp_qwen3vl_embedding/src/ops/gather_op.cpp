#include "ops/gather_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle GatherOp::Create(int64_t axis, int64_t batch_dims) {
    atb::infer::GatherParam param;
    param.axis = axis;
    param.batchDims = batch_dims;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(GatherParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
