#pragma once
#include "atb_llm/types.h"
#include <acl/acl.h>

namespace atb_llm {

class IRuntime;

/// 310P-targeted PIL AA bicubic resize, implemented by stacking ATB small ops
/// (Linear x2 + Transpose x2) instead of calling aclnnUpsampleBicubic2dAA
/// (the AA op is not supported on Atlas Inference Series; aclnnStatus=561103).
///
/// Algorithm: separable resampling, axis-by-axis, using PIL's exact
/// precompute_coeffs path (Resample.c precompute_coeffs:181-267 with
/// bicubic filter a=-0.5 and antialias support `2.0 * max(1, in/out)`).
/// Coefficients stay in fp32 (no 8bpc quantization / clip8) and are
/// densified into [out_size, in_size] fp16 weight matrices on the host;
/// each axis pass is a single fp16 MatMul on NPU.
///
/// Pipeline (need_h, need_v computed per axis):
///   H pass (if out_w != in_w):   view input as [C*H, in_w],
///                                Linear weight [out_w, in_w]^T (transposeB=true),
///                                output [C*H, out_w] viewed back to [1,C,H,out_w].
///   Transpose [1,C,H,W'] -> [1,C,W',H]  (perm {0,1,3,2}).
///   V pass (if out_h != in_h):   view as [C*W', H], Linear weight
///                                [out_h, H]^T, output [C*W', out_h] viewed
///                                back to [1,C,W',out_h].
///   Transpose [1,C,W',out_h] -> [1,C,out_h,W']  (perm {0,1,3,2}).
/// When both axes are identical (out_h==in_h && out_w==in_w), the call
/// degenerates to an async device-to-device memcpy. When only one axis
/// resizes, only that axis runs (the inactive transpose pair is skipped).
///
/// Input/output are device-side ACL_FLOAT16, contiguous NCHW [1,C,H,W].
/// The wrapper synchronizes the runtime ONCE before freeing intermediates
/// (mirrors qwen3vl_preprocess.cpp:580/:703 sync-before-free discipline):
/// aclrtFree reclaims memory immediately rather than stream-ordered, so the
/// queue must drain before any cleanup Free. That sync also makes
/// @p output_device valid on return for the resizing paths. The identity
/// (device-to-device memcpy) path does NOT sync internally — callers must
/// synchronize before reading @p output_device in that case. No per-op syncs
/// are issued (lesson L:265-266: per-op syncs break async pipelining).
///
/// @param input_device  fp16 NCHW [1, channels, in_h, in_w], device-resident.
///                      Treated as read-only; never freed by this function.
/// @param in_h, in_w, channels  Input dimensions.
/// @param out_h, out_w  Target dimensions.
/// @param runtime       IRuntime providing TensorAllocator + ATB context + stream.
/// @param output_device fp16 NCHW [1, channels, out_h, out_w], pre-allocated
///                      by the caller. Filled with the resampled output.
/// @return ACL_SUCCESS on success; any non-zero aclError on failure (with
///         every intermediate device tensor freed before return).
aclError NpuBicubicResizeAASmallOp(const void* input_device,
                                    int in_h, int in_w, int channels,
                                    int out_h, int out_w,
                                    IRuntime* runtime,
                                    void* output_device);

}  // namespace atb_llm
