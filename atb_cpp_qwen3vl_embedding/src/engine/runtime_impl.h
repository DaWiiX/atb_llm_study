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
class RuntimeImpl : public IRuntime {
public:
    RuntimeImpl(int device_id, int64_t buffer_size);
    ~RuntimeImpl() override;

    // IRuntime interface
    atb::Context* GetContext() override;
    aclrtStream GetStream() override;
    Status Synchronize() override;
    TensorAllocator* GetAllocator() override;
    std::pair<uint8_t*, Status> GetWorkspace(uint64_t required_size) override;
    WeightLoader* GetWeightLoader() override;

private:
    std::unique_ptr<ContextManager> ctx_mgr_;
    std::unique_ptr<TensorAllocator> allocator_;
    std::unique_ptr<BufferPool> buffer_pool_;
    std::unique_ptr<WeightLoader> weight_loader_;
};

/// Factory function to create a Runtime
std::unique_ptr<IRuntime> CreateRuntime(int device_id, int64_t buffer_size);

} // namespace atb_llm
