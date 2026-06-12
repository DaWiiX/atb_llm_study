#include "engine/runtime_impl.h"
#include "log/logger.h"
#include "util/cpp11_compat.h"

namespace atb_llm {

RuntimeImpl::RuntimeImpl(int device_id, int64_t buffer_size)
    : device_id_(device_id), buffer_size_(buffer_size) {}

Status RuntimeImpl::Create(int device_id, int64_t buffer_size, std::unique_ptr<IRuntime>& out) {
    auto impl = std::unique_ptr<RuntimeImpl>(new RuntimeImpl(device_id, buffer_size));

    // Initialize ContextManager -- the critical resource that may fail on invalid device_id
    Status s = ContextManager::Create(device_id, impl->ctx_mgr_);
    if (s != STATUS_OK) {
        LOG_ERROR("ContextManager::Create failed for device %d", device_id);
        out.reset();
        return s;
    }

    // Initialize remaining NPU resources
    impl->allocator_ = atb_llm::make_unique<TensorAllocator>(
        impl->ctx_mgr_->GetContext(), impl->ctx_mgr_->GetStream());
    impl->buffer_pool_ = atb_llm::make_unique<BufferPool>();
    impl->weight_loader_ = atb_llm::make_unique<WeightLoader>();

    if (buffer_size > 0) {
        s = impl->buffer_pool_->SetBufferSize(buffer_size);
        if (s != STATUS_OK) {
            LOG_WARN("Failed to pre-allocate buffer pool: %ld bytes",
                     static_cast<long>(buffer_size));
        }
    }

    LOG_INFO("RuntimeImpl created for device %d", device_id);
    out = std::move(impl);
    return STATUS_OK;
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

Status RuntimeImpl::SetBufferSize(uint64_t size_bytes) {
    if (!buffer_pool_) return ERROR_NPU_MEMORY;
    return buffer_pool_->SetBufferSize(static_cast<int64_t>(size_bytes));
}

WeightLoader* RuntimeImpl::GetWeightLoader() {
    return weight_loader_.get();
}

std::unique_ptr<IRuntime> CreateRuntime(int device_id, int64_t buffer_size) {
    // Allow ASCEND_DEVICE_ID env var to override device_id when
    // the caller passes the default (0).  Explicit non-zero values
    // always take precedence over the environment variable.
    if (device_id == 0) {
        const char* env = std::getenv("ASCEND_DEVICE_ID");
        if (env != nullptr && env[0] != '\0') {
            device_id = std::atoi(env);
        }
    }
    std::unique_ptr<IRuntime> rt;
    Status s = RuntimeImpl::Create(device_id, buffer_size, rt);
    if (s != STATUS_OK) {
        LOG_ERROR("CreateRuntime failed: device_id=%d status=%d", device_id, static_cast<int>(s));
    }
    return rt;
}

} // namespace atb_llm
