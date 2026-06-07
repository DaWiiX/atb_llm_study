#include "core/context_manager.h"
#include "log/logger.h"

namespace atb_llm {

ContextManager::ContextManager(int device_id) : device_id_(device_id) {}

Status ContextManager::Create(int device_id, std::unique_ptr<ContextManager>& out) {
    out = std::unique_ptr<ContextManager>(new ContextManager(device_id));

    Status s = out->InitACL();
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to initialize ACL for device %d", device_id);
        out.reset();
        return s;
    }
    s = out->InitStream();
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to create stream for device %d", device_id);
        out.reset();
        return s;
    }
    s = out->InitContext();
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to create ATB context for device %d", device_id);
        out.reset();
        return s;
    }
    LOG_INFO("ContextManager initialized for device %d", device_id);
    return STATUS_OK;
}

ContextManager::~ContextManager() {
    // Destroy stream before context
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    // ContextHandle destructor calls atb::DestroyContext
    // Finalize ACL
    if (acl_initialized_) {
        aclFinalize();
        acl_initialized_ = false;
    }
    LOG_INFO("ContextManager destroyed for device %d", device_id_);
}

atb::Context* ContextManager::GetContext() {
    return ctx_.get();
}

aclrtStream ContextManager::GetStream() {
    return stream_;
}

Status ContextManager::Synchronize() {
    if (!stream_) return ERROR_NPU_MEMORY;
    aclError ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("Stream synchronization failed: %d", static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }
    return STATUS_OK;
}

Status ContextManager::InitACL() {
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
        LOG_ERROR("aclInit failed: %d", static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }
    acl_initialized_ = true;

    ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice(%d) failed: %d", device_id_, static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }

    return STATUS_OK;
}

Status ContextManager::InitStream() {
    aclError ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtCreateStream failed: %d", static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }
    return STATUS_OK;
}

Status ContextManager::InitContext() {
    atb::Context* raw_ctx = nullptr;
    atb::Status ret = atb::CreateContext(&raw_ctx);
    if (ret != atb::NO_ERROR) {
        LOG_ERROR("atb::CreateContext failed: %d", static_cast<int>(ret));
        return static_cast<Status>(ERROR_ATB_BASE + ret);
    }
    ctx_ = ContextHandle(raw_ctx);

    // Set the stream on the ATB context
    ret = ctx_.get()->SetExecuteStream(stream_);
    if (ret != atb::NO_ERROR) {
        LOG_ERROR("SetExecuteStream failed: %d", static_cast<int>(ret));
        return static_cast<Status>(ERROR_ATB_BASE + ret);
    }

    // Enable GRAPH_LAUNCH_MODE for better performance
    // NOTE: Disabled — causes stream sync failures (507057) on vision block graphs
    // when Setup() is called before each Execute(). Re-enable after investigating.
    // ret = ctx_.get()->SetLaunchMode(atb::GRAPH_LAUNCH_MODE);
    // if (ret != atb::NO_ERROR) {
    //     LOG_WARN("SetLaunchMode(GRAPH_LAUNCH_MODE) failed: %d, using default", static_cast<int>(ret));
    // }

    return STATUS_OK;
}

} // namespace atb_llm
