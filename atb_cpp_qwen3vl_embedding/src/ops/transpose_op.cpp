#include "ops/transpose_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle TransposeOp::Create(const std::vector<int32_t>& perm) {
    atb::infer::TransposeParam param;
    for (auto p : perm) {
        param.perm.push_back(p);
    }

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(TransposeParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
