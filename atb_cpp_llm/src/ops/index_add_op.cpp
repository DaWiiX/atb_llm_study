#include "ops/index_add_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle IndexAddOp::Create(int64_t axis) {
    atb::infer::IndexAddParam param;
    param.indexType = atb::infer::IndexAddParam::INDEX_ADD;
    param.axis = axis;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(IndexAddParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
