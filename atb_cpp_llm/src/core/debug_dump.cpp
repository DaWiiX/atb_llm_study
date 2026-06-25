#include "core/debug_dump.h"
#include "core/tensor_allocator.h"
#include "log/logger.h"
#include <vector>

namespace atb_llm {
namespace debug {

namespace {
// Write [int64 count][raw bytes] to @p path.  Returns true on success.
bool WriteCountedBlob(const char* path, int64_t count,
                      const void* data, size_t bytes) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        LOG_WARN("debug::Dump: fopen %s failed", path);
        return false;
    }
    std::fwrite(&count, sizeof(int64_t), 1, f);
    std::fwrite(data, 1, bytes, f);
    std::fclose(f);
    LOG_DEBUG("Saved %s (%ld values)", path, static_cast<long>(count));
    return true;
}
}  // namespace

Status DumpNpuFp16(IRuntime* rt, const atb::Tensor& npu_tensor,
                   int64_t count, const char* path) {
    if (!VisionDumpEnabled()) return STATUS_OK;
    if (!rt || count <= 0 || !path) return ERROR_INVALID_PARAM;

    // Synchronize BEFORE the D2H copy: ATB runs on its own stream, so a
    // synchronous aclrtMemcpy D2H does not wait for in-flight ATB work. With
    // per-op sync off by default (H1), dumping without this sync reads stale
    // data and misleads debugging. (Debug-only path; gated by ATB_DEBUG_VISION.)
    rt->Synchronize();
    std::vector<uint16_t> host(count);
    Status s = rt->GetAllocator()->CopyToHost(
        host.data(), npu_tensor, count * sizeof(uint16_t));
    if (s != STATUS_OK) {
        LOG_WARN("debug::DumpNpuFp16: CopyToHost failed for %s", path);
        return s;
    }
    WriteCountedBlob(path, count, host.data(), count * sizeof(uint16_t));
    return STATUS_OK;
}

Status DumpHostFp16(const uint16_t* host_data, int64_t count, const char* path) {
    if (!VisionDumpEnabled()) return STATUS_OK;
    if (!host_data || count <= 0 || !path) return ERROR_INVALID_PARAM;
    WriteCountedBlob(path, count, host_data, count * sizeof(uint16_t));
    return STATUS_OK;
}

Status DumpHostInt64(const int64_t* host_data, int64_t count, const char* path) {
    if (!VisionDumpEnabled()) return STATUS_OK;
    if (!host_data || count <= 0 || !path) return ERROR_INVALID_PARAM;
    WriteCountedBlob(path, count, host_data, count * sizeof(int64_t));
    return STATUS_OK;
}

Status MaybeFeedNpuFp16(IRuntime* rt, atb::Tensor& dst_npu,
                        int64_t count, const char* path) {
    if (VisionDumpLevel() != 2) return STATUS_OK;  // only fires at level 2
    if (!rt || count <= 0 || !path) return ERROR_INVALID_PARAM;

    FILE* f = std::fopen(path, "rb");
    if (!f) {
        LOG_WARN("debug::MaybeFeedNpuFp16: %s not found, skipping override", path);
        return STATUS_OK;
    }
    int64_t file_count = 0;
    if (std::fread(&file_count, sizeof(int64_t), 1, f) != 1) {
        std::fclose(f);
        LOG_WARN("debug::MaybeFeedNpuFp16: %s header read failed", path);
        return STATUS_OK;
    }
    if (file_count != count) {
        std::fclose(f);
        LOG_WARN("debug::MaybeFeedNpuFp16: dim mismatch (%ld in file, %ld expected) — skipping",
                 static_cast<long>(file_count), static_cast<long>(count));
        return STATUS_OK;
    }
    std::vector<uint16_t> buf(count);
    size_t got = std::fread(buf.data(), sizeof(uint16_t), count, f);
    std::fclose(f);
    if (got != static_cast<size_t>(count)) {
        LOG_WARN("debug::MaybeFeedNpuFp16: short read %zu/%ld from %s",
                 got, static_cast<long>(count), path);
        return STATUS_OK;
    }
    Status s = rt->GetAllocator()->CopyToDevice(
        dst_npu, buf.data(), count * sizeof(uint16_t));
    rt->Synchronize();
    LOG_DEBUG("CONTROLLED FEED: overwrote tensor with %s (%ld values)",
             path, static_cast<long>(count));
    return s;
}

}  // namespace debug
}  // namespace atb_llm
