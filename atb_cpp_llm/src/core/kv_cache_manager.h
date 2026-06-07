#pragma once
#include "atb_llm/types.h"
#include <cstdint>
#include <vector>

namespace atb_llm {

/// KV Cache Manager - future extension point for efficient autoregressive decoding.
///
/// This is a placeholder interface for KV (Key-Value) cache management in
/// autoregressive inference. When implemented, this class will:
///
/// - Pre-allocate GPU memory for K/V tensors across all attention layers
/// - Manage cache slots with page-based allocation (PagedAttention)
/// - Support dynamic batch insertion/removal without full cache copy
/// - Enable prefix caching for common prompt prefixes
///
/// Future implementation will work with:
/// - Block allocation for K/V tensors on NPU
/// - Block table management for virtual-to-physical mapping
/// - Score-based eviction policies for memory pressure
///
/// This interface is intentionally minimal to allow flexible future design.
/// Actual KV Cache logic is NOT implemented yet.
class KVCacheManager {
public:
    virtual ~KVCacheManager() = default;

    // Non-copyable
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    /// Initialize the KV cache with specified dimensions.
    ///
    /// @param num_layers Number of transformer layers
    /// @param num_heads Number of attention heads
    /// @param head_dim Dimension per head
    /// @param max_seq_len Maximum sequence length to support
    /// @return STATUS_OK on success, error code otherwise
    virtual Status Initialize(int32_t num_layers,
                              int32_t num_heads,
                              int32_t head_dim,
                              int32_t max_seq_len) = 0;

    /// Allocate a cache slot for a new sequence.
    ///
    /// @param sequence_id Unique identifier for the sequence
    /// @param num_tokens Number of tokens to reserve space for
    /// @return STATUS_OK on success, ERROR_NPU_MEMORY if cache is full
    virtual Status AllocateSlot(int32_t sequence_id,
                                int32_t num_tokens) = 0;

    /// Free cache slot for a completed sequence.
    ///
    /// @param sequence_id Sequence to deallocate
    virtual void FreeSlot(int32_t sequence_id) = 0;

    /// Get pointer to K/V cache for a specific layer and head.
    ///
    /// @param layer_idx Layer index (0-based)
    /// @param head_idx Head index (0-based)
    /// @param sequence_id Sequence to get cache for
    /// @return Pointer to device memory, or nullptr if not allocated
    virtual void* GetCachePtr(int32_t layer_idx,
                              int32_t head_idx,
                              int32_t sequence_id) const = 0;

    /// Get total memory allocated for KV cache (bytes).
    virtual int64_t GetMemoryUsage() const = 0;

    /// Get capacity - maximum number of sequences that can be cached.
    virtual int32_t GetCapacity() const = 0;

protected:
    KVCacheManager() = default;
};

}  // namespace atb_llm
