#pragma once
#include "atb_llm/types.h"
#include "atb/context.h"
#include "core/raii.h"
#include <acl/acl.h>
#include <memory>

namespace atb_llm {

/// Manages ACL initialization, ATB Context, and Stream lifecycle.
/// RAII: destructor automatically cleans up Context, Stream, and ACL.
/// Use the static Create() factory method for full initialization.
class ContextManager {
public:
    ~ContextManager();

    // Non-copyable, non-movable (owns global resources)
    ContextManager(const ContextManager&) = delete;
    ContextManager& operator=(const ContextManager&) = delete;

    /// Factory: creates a fully-initialized ContextManager.
    /// Returns STATUS_OK on success, or an error code on failure.
    /// On failure, `out` is set to nullptr.
    static Status Create(int device_id, std::unique_ptr<ContextManager>& out);

    /// Get the ATB context (owned)
    atb::Context* GetContext();

    /// Get the ACL stream (owned)
    aclrtStream GetStream();

    /// Synchronize the stream
    Status Synchronize();

    /// Get the device ID
    int GetDeviceId() const { return device_id_; }

private:
    explicit ContextManager(int device_id);
    int device_id_;
    aclrtStream stream_ = nullptr;
    ContextHandle ctx_;
    bool acl_initialized_ = false;

    Status InitACL();
    Status InitContext();
    Status InitStream();
};

} // namespace atb_llm
