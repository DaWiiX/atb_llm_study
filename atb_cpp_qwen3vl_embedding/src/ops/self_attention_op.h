#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB SelfAttention operation wrapper.
/// Corresponds to Python utils.py:make_self_attention().
///
/// Uses BSND layout with PA_ENCODER calc type.
/// Supports GQA (num_kv_heads < num_heads).
class SelfAttentionOp {
public:
    /// Create a SelfAttention operation.
    /// @param num_heads    Number of query heads
    /// @param num_kv_heads Number of key/value heads (GQA when < num_heads)
    /// @param head_dim     Dimension per attention head
    /// @param use_mask     Whether to use MASK_TYPE_NORM causal mask
    static OperationHandle Create(int32_t num_heads,
                                  int32_t num_kv_heads,
                                  int32_t head_dim,
                                  bool use_mask = false);
};

} // namespace ops
} // namespace atb_llm
