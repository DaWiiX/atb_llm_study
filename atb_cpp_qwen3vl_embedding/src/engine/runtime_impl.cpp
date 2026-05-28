#include "engine/runtime_impl.h"
#include "log/logger.h"

namespace atb_llm {

RuntimeImpl::RuntimeImpl(int device_id, int64_t buffer_size) {
    Status s = ContextManager::Create(device_id, ctx_mgr_);
    if (s != STATUS_OK) {
        LOG_ERROR("ContextManager::Create failed for device %d", device_id);
        return;  // ctx_mgr_ is nullptr; GetContext()/GetStream() will return nullptr
    }
    allocator_ = std::make_unique<TensorAllocator>(ctx_mgr_->GetContext(), ctx_mgr_->GetStream());
    buffer_pool_ = std::make_unique<BufferPool>();
    weight_loader_ = std::make_unique<WeightLoader>();

    if (buffer_size > 0) {
        s = buffer_pool_->SetBufferSize(buffer_size);
        if (s != STATUS_OK) {
            LOG_WARN("Failed to pre-allocate buffer pool: %ld bytes", static_cast<long>(buffer_size));
        }
    }
}

RuntimeImpl::~RuntimeImpl() = default;

atb::Context* RuntimeImpl::GetContext() {
    return ctx_mgr_ ? ctx_mgr_->GetContext() : nullptr;
}

aclrtStream RuntimeImpl::GetStream() {
    return ctx_mgr_ ? ctx_mgr_->GetStream() : nullptr;
}

Status RuntimeImpl::Synchronize() {
    if (!ctx_mgr_) return ERROR_NPU_MEMORY;
    return ctx_mgr_->Synchronize();
}

TensorAllocator* RuntimeImpl::GetAllocator() {
    return allocator_.get();
}

std::pair<uint8_t*, Status> RuntimeImpl::GetWorkspace(uint64_t required_size) {
    return buffer_pool_->GetWorkspace(required_size);
}

WeightLoader* RuntimeImpl::GetWeightLoader() {
    return weight_loader_.get();
}

std::unique_ptr<IRuntime> CreateRuntime(int device_id, int64_t buffer_size) {
    return std::make_unique<RuntimeImpl>(device_id, buffer_size);
}

} // namespace atb_llm
