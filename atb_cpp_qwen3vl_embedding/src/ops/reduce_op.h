#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace ops {

/// ATB Reduce operation wrapper.
/// Reduces a tensor along specified axes (MAX, MIN, SUM).
class ReduceOp {
public:
    enum class ReduceType {
        MAX = atb::infer::ReduceParam::REDUCE_MAX,
        MIN = atb::infer::ReduceParam::REDUCE_MIN,
        SUM = atb::infer::ReduceParam::REDUCE_SUM,
    };

    /// Create a Reduce operation.
    /// @param reduce_type  Type of reduction (MAX, MIN, SUM)
    /// @param axis         Axes to reduce along
    /// @return RAII-managed operation handle
    static OperationHandle Create(ReduceType reduce_type,
                                  const std::vector<int64_t>& axis);
};

} // namespace ops
} // namespace atb_llm
