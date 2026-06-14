#pragma once
#include "atb/atb_infer.h"
#include "atb_llm/operation_handle.h"
#include "utils/cpp11_compat.h"
#include <utility>

namespace atb_llm {

/// RAII wrapper for atb::Context*
/// Calls atb::DestroyContext on destruction.
class ContextHandle {
public:
    explicit ContextHandle(atb::Context* ctx = nullptr) : ctx_(ctx) {}
    ~ContextHandle() { if (ctx_) atb::DestroyContext(ctx_); }

    // Move-only
    ContextHandle(ContextHandle&& o) noexcept : ctx_(atb_llm::exchange(o.ctx_, static_cast<atb::Context*>(nullptr))) {}
    ContextHandle& operator=(ContextHandle&& o) noexcept {
        if (this != &o) {
            if (ctx_) atb::DestroyContext(ctx_);
            ctx_ = atb_llm::exchange(o.ctx_, static_cast<atb::Context*>(nullptr));
        }
        return *this;
    }
    ContextHandle(const ContextHandle&) = delete;
    ContextHandle& operator=(const ContextHandle&) = delete;

    atb::Context* get() const { return ctx_; }
    atb::Context* release() { return atb_llm::exchange(ctx_, static_cast<atb::Context*>(nullptr)); }
    explicit operator bool() const { return ctx_ != nullptr; }

private:
    atb::Context* ctx_ = nullptr;
};

} // namespace atb_llm
