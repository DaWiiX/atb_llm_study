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
/// Rotation pattern controlled by rotaryCoeff:
///   2       — contiguous-half (LLAMA-style): pairs (i, i+hd/2). Qwen3VL default.
///   4       — quarter rotation
///   headDim/2 — full rotation
class RopeOp {
public:
    /// Create a RoPE operation.
    /// @param rotaryCoeff  Rotation coefficient (2=half, 4=quarter, headDim/2=full).
    ///                     Default 2 matches Qwen3VL behavior.
    static OperationHandle Create(int32_t rotaryCoeff = 2);
};

}  // namespace ops
}  // namespace atb_llm
