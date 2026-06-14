#pragma once
#include "atb_llm/runtime.h"
#include "core/context_manager.h"
#include "core/tensor_allocator.h"
#include "core/buffer_pool.h"
#include "io/weight_loader.h"
#include <memory>

namespace atb_llm {

/// Concrete implementation of IRuntime.
/// Owns all NPU resources: Context, Stream, Allocator, BufferPool, WeightLoader.
/// Use the static Create() factory method for full initialization.
class RuntimeImpl : public IRuntime {
public:
    ~RuntimeImpl() override;

    // Non-copyable, non-movable (owns NPU resources)
    RuntimeImpl(const RuntimeImpl&) = delete;
    RuntimeImpl& operator=(const RuntimeImpl&) = delete;

    /// Factory: creates a fully-initialized RuntimeImpl.
    /// Returns STATUS_OK on success, or an error code on failure.
    /// On failure, `out` is set to nullptr.
    static Status Create(int device_id, int64_t buffer_size, std::unique_ptr<IRuntime>& out);

    // IRuntime interface
    atb::Context* GetContext() override;
    aclrtStream GetStream() override;
    Status Synchronize() override;
    TensorAllocator* GetAllocator() override;
    std::pair<uint8_t*, Status> GetWorkspace(uint64_t required_size) override;
    Status SetBufferSize(uint64_t size_bytes) override;
    WeightLoader* GetWeightLoader() override;

private:
    RuntimeImpl(int device_id, int64_t buffer_size);
    int device_id_;
    int64_t buffer_size_;
    std::unique_ptr<ContextManager> ctx_mgr_;
    std::unique_ptr<TensorAllocator> allocator_;
    std::unique_ptr<BufferPool> buffer_pool_;
    std::unique_ptr<WeightLoader> weight_loader_;
};

/// Factory function to create a Runtime.
/// Returns nullptr on failure (e.g. invalid device_id).
std::unique_ptr<IRuntime> CreateRuntime(int device_id, int64_t buffer_size);

} // namespace atb_llm
