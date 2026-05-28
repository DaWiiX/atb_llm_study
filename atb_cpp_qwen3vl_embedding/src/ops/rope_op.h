#pragma once
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "core/raii.h"

namespace atb_llm {
namespace ops {

/// ATB RoPE (Rotary Position Embedding) operation wrapper.
/// Corresponds to Python utils.py:make_rope_operation().
///
/// Inputs (5): [q, k, cos, sin, seqlen]
///   q:     (ntokens, num_heads * head_dim) float16
///   k:     (ntokens, num_kv_heads * head_dim) float16 (GQA-compatible)
///   cos:   (ntokens, head_dim) float16
///   sin:   (ntokens, head_dim) float16
///   seqlen: (batch,) int32
/// Outputs (2): [ropeQ, ropeK]
///
/// Uses LLAMA-style contiguous-half rotation: pairs (i, i+hd/2).
class RopeOp {
public:
    /// Create a RoPE operation with half-rotation (rotary_coeff=2).
    static OperationHandle Create();
};

} // namespace ops
} // namespace atb_llm
