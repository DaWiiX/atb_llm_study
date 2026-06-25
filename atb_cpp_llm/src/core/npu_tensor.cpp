#include "core/npu_tensor.h"
#include "log/logger.h"
#include <cstring>

namespace atb_llm {

size_t NpuTensor::Align512(size_t size) {
    return (size + 511) & ~static_cast<size_t>(511);
}

NpuTensor::NpuTensor(std::vector<int64_t> shape, aclDataType dtype, size_t element_size) {
    if (shape.empty() || shape.size() > atb::MAX_DIM) {
        LOG_ERROR("NpuTensor: invalid shape dimensions: %zu", shape.size());
        return;
    }

    size_t num_elements = 1;
    for (auto dim : shape) {
        if (dim <= 0) {
            LOG_ERROR("NpuTensor: invalid dimension value: %ld", static_cast<long>(dim));
            return;
        }
        num_elements *= static_cast<size_t>(dim);
    }

    size_t raw_size = num_elements * element_size;
    size_t alloc_size = Align512(raw_size);

    void* dev_ptr = nullptr;
    aclError ret = aclrtMalloc(&dev_ptr, alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("NpuTensor: aclrtMalloc failed: size=%zu, error=%d",
                  alloc_size, static_cast<int>(ret));
        return;
    }

    tensor_.desc.dtype = dtype;
    tensor_.desc.format = ACL_FORMAT_ND;
    tensor_.desc.shape.dimNum = static_cast<uint64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        tensor_.desc.shape.dims[i] = shape[i];
    }
    tensor_.deviceData = dev_ptr;
    tensor_.hostData = nullptr;
    tensor_.dataSize = raw_size;
    owns_ = true;
}

NpuTensor::~NpuTensor() {
    FreeIfOwned();
}

NpuTensor::NpuTensor(NpuTensor&& other) noexcept
    : tensor_(other.tensor_), owns_(other.owns_) {
    other.tensor_ = {};
    other.owns_ = false;
}

NpuTensor& NpuTensor::operator=(NpuTensor&& other) noexcept {
    if (this != &other) {
        FreeIfOwned();
        tensor_ = other.tensor_;
        owns_ = other.owns_;
        other.tensor_ = {};
        other.owns_ = false;
    }
    return *this;
}

atb::Tensor* NpuTensor::Get() {
    return owns_ ? &tensor_ : nullptr;
}

const atb::Tensor* NpuTensor::Get() const {
    return owns_ ? &tensor_ : nullptr;
}

atb::Tensor* NpuTensor::Release() {
    if (!owns_) return nullptr;
    owns_ = false;
    return &tensor_;
}

NpuTensor::operator bool() const {
    return owns_ && tensor_.deviceData != nullptr;
}

NpuTensor NpuTensor::Adopt(atb::Tensor& tensor) {
    NpuTensor t;
    t.tensor_ = tensor;                       // shallow copy: desc + deviceData + dataSize
    t.owns_ = (tensor.deviceData != nullptr); // empty input -> stays non-owning
    // Transfer ownership: source no longer holds the device pointer.
    tensor.deviceData = nullptr;
    tensor.dataSize = 0;
    return t;  // move-constructed on return
}

void NpuTensor::FreeIfOwned() {
    if (owns_ && tensor_.deviceData) {
        aclrtFree(tensor_.deviceData);
        tensor_ = {};
        owns_ = false;
    }
}

// ── Convenience factories ─────────────────────────────────

NpuTensor AllocNpuFloat16(std::vector<int64_t> shape) {
    return NpuTensor(std::move(shape), ACL_FLOAT16, 2);
}

NpuTensor AllocNpuFloat32(std::vector<int64_t> shape) {
    return NpuTensor(std::move(shape), ACL_FLOAT, 4);
}

NpuTensor AllocNpuInt64(std::vector<int64_t> shape) {
    return NpuTensor(std::move(shape), ACL_INT64, 8);
}

NpuTensor AllocNpuInt32(std::vector<int64_t> shape) {
    return NpuTensor(std::move(shape), ACL_INT32, 4);
}

} // namespace atb_llm
