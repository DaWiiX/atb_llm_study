#pragma once
#include "atb/atb_infer.h"
#include <utility>

namespace atb_llm {

/// RAII wrapper for atb::Operation*
/// Calls atb::DestroyOperation on destruction.
class OperationHandle {
public:
    explicit OperationHandle(atb::Operation* op = nullptr) : op_(op) {}
    ~OperationHandle() { if (op_) atb::DestroyOperation(op_); }

    // Move-only
    OperationHandle(OperationHandle&& o) noexcept : op_(std::exchange(o.op_, nullptr)) {}
    OperationHandle& operator=(OperationHandle&& o) noexcept {
        if (this != &o) {
            if (op_) atb::DestroyOperation(op_);
            op_ = std::exchange(o.op_, nullptr);
        }
        return *this;
    }
    OperationHandle(const OperationHandle&) = delete;
    OperationHandle& operator=(const OperationHandle&) = delete;

    atb::Operation* get() const { return op_; }
    atb::Operation* release() { return std::exchange(op_, nullptr); }
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
    ContextHandle(ContextHandle&& o) noexcept : ctx_(std::exchange(o.ctx_, nullptr)) {}
    ContextHandle& operator=(ContextHandle&& o) noexcept {
        if (this != &o) {
            if (ctx_) atb::DestroyContext(ctx_);
            ctx_ = std::exchange(o.ctx_, nullptr);
        }
        return *this;
    }
    ContextHandle(const ContextHandle&) = delete;
    ContextHandle& operator=(const ContextHandle&) = delete;

    atb::Context* get() const { return ctx_; }
    atb::Context* release() { return std::exchange(ctx_, nullptr); }
    explicit operator bool() const { return ctx_ != nullptr; }

private:
    atb::Context* ctx_ = nullptr;
};

} // namespace atb_llm
