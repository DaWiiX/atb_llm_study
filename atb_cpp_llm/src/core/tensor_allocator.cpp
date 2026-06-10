#include "core/tensor_allocator.h"
#include "log/logger.h"
#include <cstring>

namespace atb_llm {

TensorAllocator::TensorAllocator(atb::Context* ctx, aclrtStream stream)
    : ctx_(ctx), stream_(stream) {}

TensorAllocator::~TensorAllocator() {
    FreeAll();
}

size_t TensorAllocator::Align512(size_t size) {
    return (size + 511) & ~static_cast<size_t>(511);
}

Status TensorAllocator::AllocFloat16(atb::Tensor& tensor, std::vector<int64_t> shape) {
    return AllocTensor(tensor, std::move(shape), ACL_FLOAT16, 2);
}

Status TensorAllocator::AllocFloat32(atb::Tensor& tensor, std::vector<int64_t> shape) {
    return AllocTensor(tensor, std::move(shape), ACL_FLOAT, 4);
}

Status TensorAllocator::AllocInt64(atb::Tensor& tensor, std::vector<int64_t> shape) {
    return AllocTensor(tensor, std::move(shape), ACL_INT64, 8);
}

Status TensorAllocator::AllocInt32(atb::Tensor& tensor, std::vector<int64_t> shape) {
    return AllocTensor(tensor, std::move(shape), ACL_INT32, 4);
}

Status TensorAllocator::AllocTensor(atb::Tensor& tensor, std::vector<int64_t> shape,
                                     aclDataType dtype, size_t element_size) {
    if (shape.empty() || shape.size() > atb::MAX_DIM) {
        LOG_ERROR("Invalid shape dimensions: %zu", shape.size());
        return ERROR_INVALID_PARAM;
    }

    // Calculate total elements
    size_t num_elements = 1;
    for (auto dim : shape) {
        if (dim <= 0) {
            LOG_ERROR("Invalid dimension value: %ld", static_cast<long>(dim));
            return ERROR_INVALID_PARAM;
        }
        num_elements *= static_cast<size_t>(dim);
    }

    size_t raw_size = num_elements * element_size;
    size_t alloc_size = Align512(raw_size);

    // Allocate device memory
    void* dev_ptr = nullptr;
    aclError ret = aclrtMalloc(&dev_ptr, alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc failed: size=%zu, error=%d", alloc_size, static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }

    // Set tensor descriptor
    tensor.desc.dtype = dtype;
    tensor.desc.format = ACL_FORMAT_ND;
    tensor.desc.shape.dimNum = static_cast<uint64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
        tensor.desc.shape.dims[i] = shape[i];
    }

    tensor.deviceData = dev_ptr;
    tensor.hostData = nullptr;
    tensor.dataSize = raw_size;  // must match ATB's inferred size (shape * element_size), not aligned alloc

    // Track allocation (keyed by device pointer for O(1) lookup)
    allocations_.emplace(dev_ptr, Allocation{dev_ptr, alloc_size});

    return STATUS_OK;
}

Status TensorAllocator::CopyToDevice(atb::Tensor& tensor, const void* host_data, size_t size) {
    if (!tensor.deviceData || !host_data) {
        LOG_ERROR("CopyToDevice: null pointer");
        return ERROR_INVALID_PARAM;
    }
    if (size > tensor.dataSize) {
        LOG_ERROR("CopyToDevice: size %zu exceeds tensor dataSize %zu", size, tensor.dataSize);
        return ERROR_INVALID_PARAM;
    }

    aclError ret = aclrtMemcpy(tensor.deviceData, tensor.dataSize,
                                host_data, size,
                                ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy H2D failed: %d", static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }

    return STATUS_OK;
}

Status TensorAllocator::CopyToHost(void* host_data, const atb::Tensor& tensor, size_t size) {
    if (!host_data || !tensor.deviceData) {
        LOG_ERROR("CopyToHost: null pointer");
        return ERROR_INVALID_PARAM;
    }
    if (size > tensor.dataSize) {
        LOG_ERROR("CopyToHost: size %zu exceeds tensor dataSize %zu", size, tensor.dataSize);
        return ERROR_INVALID_PARAM;
    }

    aclError ret = aclrtMemcpy(host_data, size,
                                tensor.deviceData, size,
                                ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy D2H failed: %d", static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }

    return STATUS_OK;
}

Status TensorAllocator::CopyToDevice(atb::Tensor& tensor, const void* host_data,
                                      size_t size, size_t dst_offset_bytes) {
    if (!tensor.deviceData || !host_data) {
        LOG_ERROR("CopyToDevice(offset): null pointer");
        return ERROR_INVALID_PARAM;
    }
    if (dst_offset_bytes + size > tensor.dataSize) {
        LOG_ERROR("CopyToDevice(offset): offset %zu + size %zu exceeds tensor dataSize %zu",
                  dst_offset_bytes, size, tensor.dataSize);
        return ERROR_INVALID_PARAM;
    }

    void* dst_ptr = static_cast<uint8_t*>(tensor.deviceData) + dst_offset_bytes;
    aclError ret = aclrtMemcpy(dst_ptr, size,
                                host_data, size,
                                ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy H2D (offset) failed: %d", static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }

    return STATUS_OK;
}

Status TensorAllocator::CopyToHost(void* host_data, const atb::Tensor& tensor,
                                    size_t size, size_t src_offset_bytes) {
    if (!host_data || !tensor.deviceData) {
        LOG_ERROR("CopyToHost(offset): null pointer");
        return ERROR_INVALID_PARAM;
    }
    if (src_offset_bytes + size > tensor.dataSize) {
        LOG_ERROR("CopyToHost(offset): offset %zu + size %zu exceeds tensor dataSize %zu",
                  src_offset_bytes, size, tensor.dataSize);
        return ERROR_INVALID_PARAM;
    }

    void* src_ptr = static_cast<uint8_t*>(tensor.deviceData) + src_offset_bytes;
    aclError ret = aclrtMemcpy(host_data, size,
                                src_ptr, size,
                                ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy D2H (offset) failed: %d", static_cast<int>(ret));
        return ERROR_NPU_MEMORY;
    }

    return STATUS_OK;
}

void TensorAllocator::Free(atb::Tensor& tensor) {
    if (tensor.deviceData) {
        aclrtFree(tensor.deviceData);
        // O(1) removal from tracked allocations to prevent double-free in FreeAll()
        allocations_.erase(tensor.deviceData);
        tensor.deviceData = nullptr;
        tensor.dataSize = 0;
    }
}

void TensorAllocator::FreeAll() {
    for (auto it = allocations_.begin(); it != allocations_.end(); ++it) {
        Allocation& alloc = it->second;
        if (alloc.device_ptr) {
            aclrtFree(alloc.device_ptr);
            alloc.device_ptr = nullptr;
        }
    }
    allocations_.clear();
}

} // namespace atb_llm
