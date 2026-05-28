#pragma once
#include "atb_llm/types.h"
#include <cstdint>
#include <mutex>
#include <utility>
#include <acl/acl.h>

namespace atb_llm {

/// Per-runtime workspace memory pool.
/// Manages a single large device buffer that grows on demand.
/// NOT a global singleton -- each IRuntime instance owns its own BufferPool.
class BufferPool {
public:
    BufferPool();
    ~BufferPool();

    // Non-copyable
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    /// Pre-allocate buffer to the given size (bytes).
    /// If already larger, no-op.
    Status SetBufferSize(int64_t size);

    /// Get workspace pointer, growing if needed.
    /// Thread-safe: holds mutex during potential realloc.
    /// Returns {nullptr, ERROR_NPU_MEMORY} on allocation failure.
    std::pair<uint8_t*, Status> GetWorkspace(uint64_t required_size);

    /// Release workspace memory.
    void Free();

    /// Get current buffer size
    uint64_t GetSize() const { return buffer_size_; }

private:
    uint8_t* buffer_ = nullptr;
    uint64_t buffer_size_ = 0;
    std::mutex mutex_;
};

} // namespace atb_llm
