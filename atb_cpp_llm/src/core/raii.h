#pragma once
#include "atb/atb_infer.h"
#include "util/cpp11_compat.h"
#include <utility>

namespace atb_llm {

/// RAII wrapper for atb::Operation*
/// Calls atb::DestroyOperation on destruction.
class OperationHandle {
public:
    explicit OperationHandle(atb::Operation* op = nullptr) : op_(op) {}
    ~OperationHandle() { if (op_) atb::DestroyOperation(op_); }

    // Move-only
    OperationHandle(OperationHandle&& o) noexcept : op_(atb_llm::exchange(o.op_, static_cast<atb::Operation*>(nullptr))) {}
    OperationHandle& operator=(OperationHandle&& o) noexcept {
        if (this != &o) {
            if (op_) atb::DestroyOperation(op_);
            op_ = atb_llm::exchange(o.op_, static_cast<atb::Operation*>(nullptr));
        }
        return *this;
    }
    OperationHandle(const OperationHandle&) = delete;
    OperationHandle& operator=(const OperationHandle&) = delete;

    atb::Operation* get() const { return op_; }
    atb::Operation* release() { return atb_llm::exchange(op_, static_cast<atb::Operation*>(nullptr)); }
    explicit operator bool() const { return op_ != nullptr; }

private:
    atb::Operation* op_ = nullptr;
};

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
