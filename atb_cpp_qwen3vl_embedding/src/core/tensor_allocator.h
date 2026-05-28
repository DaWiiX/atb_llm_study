#pragma once
#include "atb_llm/types.h"
#include "atb/types.h"
#include <vector>
#include <cstdint>
#include <acl/acl.h>

namespace atb {
class Context;  // forward declaration
}

namespace atb_llm {

/// NPU memory allocator for atb::Tensor.
/// Allocates device memory with 512-byte alignment and manages tensor lifecycle.
class TensorAllocator {
public:
    TensorAllocator(atb::Context* ctx, aclrtStream stream);
    ~TensorAllocator();

    // Non-copyable
    TensorAllocator(const TensorAllocator&) = delete;
    TensorAllocator& operator=(const TensorAllocator&) = delete;

    /// Allocate NPU memory and set TensorDesc for float16
    Status AllocFloat16(atb::Tensor& tensor, std::vector<int64_t> shape);

    /// Allocate NPU memory and set TensorDesc for float32
    Status AllocFloat32(atb::Tensor& tensor, std::vector<int64_t> shape);

    /// Allocate NPU memory and set TensorDesc for int64
    Status AllocInt64(atb::Tensor& tensor, std::vector<int64_t> shape);

    /// Host -> Device copy (async via stream)
    Status CopyToDevice(atb::Tensor& tensor, const void* host_data, size_t size);

    /// Device -> Host copy (sync)
    Status CopyToHost(void* host_data, const atb::Tensor& tensor, size_t size);

    /// Release NPU memory for a single tensor
    void Free(atb::Tensor& tensor);

    /// Release all tracked tensors (called on model destruction)
    void FreeAll();

    /// Get number of tracked tensors
    size_t NumTracked() const { return allocations_.size(); }

private:
    atb::Context* ctx_;
    aclrtStream stream_;

    struct Allocation {
        void* device_ptr = nullptr;
        size_t size = 0;
    };
    std::vector<Allocation> allocations_;

    Status AllocTensor(atb::Tensor& tensor, std::vector<int64_t> shape,
                       aclDataType dtype, size_t element_size);
    static size_t Align512(size_t size);
};

} // namespace atb_llm
