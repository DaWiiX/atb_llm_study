#include "core/buffer_pool.h"
#include "log/logger.h"

namespace atb_llm {

BufferPool::BufferPool() = default;

BufferPool::~BufferPool() {
    Free();
}

Status BufferPool::SetBufferSize(int64_t size) {
    if (size <= 0) return STATUS_OK;

    std::lock_guard<std::mutex> lock(mutex_);

    if (buffer_ && buffer_size_ >= static_cast<uint64_t>(size)) {
        return STATUS_OK;  // Already large enough
    }

    // Free old buffer
    if (buffer_) {
        aclrtFree(buffer_);
        buffer_ = nullptr;
        buffer_size_ = 0;
    }

    // Allocate new buffer
    aclError ret = aclrtMalloc(reinterpret_cast<void**>(&buffer_),
                                static_cast<size_t>(size),
                                ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("BufferPool: aclrtMalloc(%ld) failed: %d",
                  static_cast<long>(size), static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }

    buffer_size_ = static_cast<uint64_t>(size);
    LOG_INFO("BufferPool: allocated %ld bytes", static_cast<long>(size));
    return STATUS_OK;
}

std::pair<uint8_t*, Status> BufferPool::GetWorkspace(uint64_t required_size) {
    if (required_size == 0) return {nullptr, STATUS_OK};

    std::lock_guard<std::mutex> lock(mutex_);

    if (buffer_ && buffer_size_ >= required_size) {
        return {buffer_, STATUS_OK};
    }

    // Need to grow: free old, allocate new
    if (buffer_) {
        aclrtFree(buffer_);
        buffer_ = nullptr;
        buffer_size_ = 0;
    }

    // Allocate with some headroom to avoid frequent realloc
    uint64_t alloc_size = required_size + (required_size >> 2);  // +25%

    aclError ret = aclrtMalloc(reinterpret_cast<void**>(&buffer_),
                                static_cast<size_t>(alloc_size),
                                ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("BufferPool: aclrtMalloc(%lu) failed: %d",
                  alloc_size, static_cast<int>(ret));
        return {nullptr, ERROR_NPU_MEMORY};
    }

    buffer_size_ = alloc_size;
    LOG_INFO("BufferPool: grew to %lu bytes (requested %lu)", alloc_size, required_size);
    return {buffer_, STATUS_OK};
}

void BufferPool::Free() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_) {
        aclrtFree(buffer_);
        buffer_ = nullptr;
        buffer_size_ = 0;
    }
}

} // namespace atb_llm
