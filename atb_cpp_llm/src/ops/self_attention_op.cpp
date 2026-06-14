#include "ops/self_attention_op.h"
#include "log/logger.h"
#include "utils/cpp11_compat.h"
#include <cmath>

namespace atb_llm {
namespace ops {

OperationHandle SelfAttentionOp::Create(int32_t num_heads,
                                         int32_t num_kv_heads,
                                         int32_t head_dim,
                                         bool use_mask) {
    atb::infer::SelfAttentionParam param;
    param.headNum = num_heads;
    param.kvHeadNum = num_kv_heads;
    param.qkScale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    param.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
    param.inputLayout = atb::infer::TYPE_BSND;

    if (use_mask) {
        param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    }

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(SelfAttentionParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
