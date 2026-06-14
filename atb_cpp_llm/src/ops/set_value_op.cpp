#include "ops/set_value_op.h"
#include "log/logger.h"

namespace atb_llm {
namespace ops {

OperationHandle SetValueOp::Create(const std::vector<int64_t>& starts,
                                    const std::vector<int64_t>& ends) {
    if (starts.size() != ends.size()) {
        LOG_ERROR("SetValueOp::Create: starts.size()=%zu != ends.size()=%zu",
                  starts.size(), ends.size());
        return OperationHandle(nullptr);
    }

    atb::infer::SetValueParam param;
    for (auto v : starts) param.starts.push_back(v);
    for (auto v : ends)   param.ends.push_back(v);
    // strides default to all-1 in ATB; just fill same size
    for (size_t i = 0; i < starts.size(); i++) param.strides.push_back(1);

    atb::Operation* op = nullptr;
    atb::Status s = atb::CreateOperation(param, &op);
    if (s != atb::NO_ERROR) {
        LOG_ERROR("CreateOperation(SetValueParam) failed: %d", static_cast<int>(s));
        return OperationHandle(nullptr);
    }
    return OperationHandle(op);
}

} // namespace ops
} // namespace atb_llm
