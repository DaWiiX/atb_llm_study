#pragma once
// Debug-dump helpers gated on ATB_DEBUG_VISION env var.
//
// All entry points are no-ops when ATB_DEBUG_VISION is unset, so they
// can sit in hot paths without runtime cost beyond a getenv() probe.
//
// File format for produced .bin files (matches the Python reference
// loaders under tests/python_reference/):
//   int64_t  count   (number of elements)
//   <count>  values  (element width depends on dtype)

#include "atb_llm/runtime.h"
#include "atb_llm/types.h"
#include "atb/atb_infer.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace atb_llm {
namespace debug {

/// Returns true iff ATB_DEBUG_VISION env var is set (any non-empty value).
inline bool VisionDumpEnabled() {
    const char* v = std::getenv("ATB_DEBUG_VISION");
    return v != nullptr && v[0] != '\0';
}

/// Returns the integer value of ATB_DEBUG_VISION (0 if unset / not a number).
/// Used to distinguish ATB_DEBUG_VISION=1 (dump only) from
/// ATB_DEBUG_VISION=2 (dump + controlled feed).
inline int VisionDumpLevel() {
    const char* v = std::getenv("ATB_DEBUG_VISION");
    return v ? std::atoi(v) : 0;
}

/// Copy @p count fp16 elements from @p npu_tensor to host then write to
/// @p path as the standard [int64 count][uint16_t * count] format.
/// Synchronises the runtime stream before fopen.  No-op when
/// VisionDumpEnabled() is false.
Status DumpNpuFp16(IRuntime* rt, const atb::Tensor& npu_tensor,
                   int64_t count, const char* path);

/// Write @p count host-resident fp16 elements to @p path in standard
/// format.  No-op when VisionDumpEnabled() is false.
Status DumpHostFp16(const uint16_t* host_data, int64_t count, const char* path);

/// Write @p count host-resident int64 elements to @p path.
/// No-op when VisionDumpEnabled() is false.
Status DumpHostInt64(const int64_t* host_data, int64_t count, const char* path);

/// CONTROLLED FEED helper.
///
/// When ATB_DEBUG_VISION=2, read @p count fp16 elements from @p path
/// and overwrite @p dst_npu with them (H2D).  Use to inject a Python
/// reference tensor into the C++ pipeline at a specific stage to
/// isolate where divergence starts.
///
/// Returns STATUS_OK on success (including when no-op because env var
/// isn't 2).  Logs WARN and returns STATUS_OK on dim mismatch -- this
/// is a debug tool, not production fail-fast.
Status MaybeFeedNpuFp16(IRuntime* rt, atb::Tensor& dst_npu,
                        int64_t count, const char* path);

}  // namespace debug
}  // namespace atb_llm
