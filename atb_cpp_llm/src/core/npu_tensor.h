#pragma once
#include "atb/types.h"
#include <acl/acl.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace atb_llm {

/// RAII wrapper for NPU-resident atb::Tensor.
/// Allocates device memory on construction, frees on destruction.
/// Move-only: copying is prohibited to prevent double-free.
class NpuTensor {
public:
    /// Default-construct an empty (null) tensor.
    NpuTensor() = default;

    /// Construct and allocate an NPU tensor with the given shape and dtype.
    /// On allocation failure, the tensor remains null (check with operator bool).
    NpuTensor(std::vector<int64_t> shape, aclDataType dtype, size_t element_size);

    ~NpuTensor();

    // Move-only
    NpuTensor(NpuTensor&& other) noexcept;
    NpuTensor& operator=(NpuTensor&& other) noexcept;
    NpuTensor(const NpuTensor&) = delete;
    NpuTensor& operator=(const NpuTensor&) = delete;

    /// Access the underlying atb::Tensor (null if released or moved-from).
    atb::Tensor* Get();
    const atb::Tensor* Get() const;

    [[deprecated("Use Get() for read access. Release() transfers raw NPU pointer ownership — caller must aclrtFree().")]]
    atb::Tensor* Release();

    /// True if this NpuTensor owns a non-null device allocation.
    explicit operator bool() const;

    /// Adopt an already-allocated device tensor. Takes ownership: the
    /// destructor will aclrtFree @p tensor.deviceData. @p tensor is left
    /// empty (deviceData nulled, dataSize zeroed) after the call.
    ///
    /// @warning @p tensor.deviceData MUST be freeable by a single aclrtFree.
    /// The safe sources are: aclrtMalloc, NpuTensor/AllocNpu*, or a
    /// TensorAllocator allocation that has been Detach()'d. Adopting a tensor
    /// still tracked by TensorAllocator causes a double-free (NpuTensor's
    /// aclrtFree here + TensorAllocator::~TensorAllocator's FreeAll).
    static NpuTensor Adopt(atb::Tensor& tensor);

private:
    atb::Tensor tensor_{};
    bool owns_ = false;

    void FreeIfOwned();
    static size_t Align512(size_t size);
};

/// Convenience factories matching TensorAllocator's API.
NpuTensor AllocNpuFloat16(std::vector<int64_t> shape);
NpuTensor AllocNpuFloat32(std::vector<int64_t> shape);
NpuTensor AllocNpuInt64(std::vector<int64_t> shape);
NpuTensor AllocNpuInt32(std::vector<int64_t> shape);

} // namespace atb_llm
