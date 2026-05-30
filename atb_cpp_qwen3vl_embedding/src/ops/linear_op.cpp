#include "ops/linear_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle LinearOp::Create(bool has_bias, bool transpose_a, bool transpose_b) {
    atb::infer::LinearParam param;
    param.hasBias = has_bias;
    param.transposeA = transpose_a;
    param.transposeB = transpose_b;

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(LinearParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
