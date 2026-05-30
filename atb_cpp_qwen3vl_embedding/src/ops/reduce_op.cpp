#include "ops/reduce_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle ReduceOp::Create(ReduceType reduce_type,
                                  const std::vector<int64_t>& axis) {
    atb::infer::ReduceParam param;
    param.reduceType = static_cast<atb::infer::ReduceParam::ReduceType>(reduce_type);
    for (auto a : axis) param.axis.push_back(a);

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(ReduceParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
