#include "ops/softmax_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle SoftmaxOp::Create(const std::vector<int64_t>& axes) {
    atb::infer::SoftmaxParam param;
    for (auto a : axes) {
        param.axes.push_back(a);
    }

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(SoftmaxParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
