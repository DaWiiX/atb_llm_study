#pragma once
#include <cstdint>

namespace atb_llm {

/// KV Cache configuration for generative models.
struct KVCacheConfig {
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t head_dim = 0;
    int64_t max_seq_len = 0;
    int64_t batch_size = 1;
};

/// KV Cache manager stub for generative models.
///
/// This is a placeholder for future implementation.
/// Generative models (Qwen3, DeepSeek-V2/V3, etc.) will use this
/// to manage key-value caches across decoding steps.
///
/// Future: manages pre-allocated KV cache buffers per layer,
/// provides Get/Put operations for incremental decoding.
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& cfg) : cfg_(cfg) {}
    ~KVCacheManager() = default;

    /// Returns true if KV cache is available (always false for now).
    bool IsAvailable() const { return false; }

    const KVCacheConfig& GetConfig() const { return cfg_; }

private:
    KVCacheConfig cfg_;
};

}  // namespace atb_llm
