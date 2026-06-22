#pragma once
#include "atb_llm/types.h"

namespace atb_llm {

class IRuntime;

/// Cross-platform (910B + 310P) NPU bicubic resize via aclnnUpsampleBicubic2d.
/// CANN 9.0.0 documentation confirms both platforms support this operator.
/// Input/output are fp16 on NPU, layout (C, H, W) contiguous.
Status NpuBicubicResize(IRuntime* runtime,
                        const void* input_npu,
                        int32_t channels, int32_t in_h, int32_t in_w,
                        int32_t out_h, int32_t out_w,
                        void* output_npu);

}  // namespace atb_llm
