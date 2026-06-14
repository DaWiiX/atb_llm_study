#pragma once
#include "atb/atb_infer.h"
#include <utility>

namespace atb_llm {

// std::exchange (C++14) equivalent.
// Used by RAII wrappers for move-only semantics.
template <typename T, typename U = T>
T exchange(T& obj, U&& new_value) {
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}

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

} // namespace atb_llm
