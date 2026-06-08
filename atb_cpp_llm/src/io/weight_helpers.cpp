#include "io/weight_helpers.h"
#include "utils/float_utils.h"
#include "io/safetensors_reader.h"
#include "log/logger.h"
#include "safetensors.hh"
#include <cstring>
#include <vector>

namespace atb_llm {
namespace io {

// ═════════════════════════════════════════════════════════════════════
// Helper: copy weight tensor to NPU as float16
// ═════════════════════════════════════════════════════════════════════

Status CopyWeightToFp16NPU(WeightLoader& loader,
                           const std::string& key,
                           TensorAllocator& alloc,
                           atb::Tensor& dst) {
    WeightInfo info;
    Status s = loader.GetTensor(key, info);
    if (s != STATUS_OK) {
        LOG_ERROR("Weight not found: %s", key.c_str());
        return s;
    }

    const uint8_t* data = loader.GetTensorData(key);
    if (!data) {
        LOG_ERROR("No data for weight: %s", key.c_str());
        return ERROR_WEIGHT_LOAD;
    }

    std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
    size_t num_elements = 1;
    for (auto d : shape)
        num_elements *= static_cast<size_t>(d);

    auto st_dtype = static_cast<safetensors::dtype>(info.dtype);

    if (st_dtype == safetensors::kFLOAT16) {
        // Already fp16, copy directly
        s = alloc.AllocFloat16(dst, shape);
        if (s != STATUS_OK) return s;
        return alloc.CopyToDevice(dst, data, info.nbytes);

    } else if (st_dtype == safetensors::kBFLOAT16) {
        // Convert bf16 -> fp16
        std::vector<uint16_t> fp16_buf(num_elements);
        atb_llm::Bf16ToFp16Buffer(reinterpret_cast<const uint16_t*>(data),
                                  fp16_buf.data(), num_elements);
        s = alloc.AllocFloat16(dst, shape);
        if (s != STATUS_OK) return s;
        return alloc.CopyToDevice(dst, fp16_buf.data(), num_elements * sizeof(uint16_t));

    } else if (st_dtype == safetensors::kFLOAT32) {
        // Convert fp32 -> fp16 directly (round-to-nearest-even, matches Python)
        const float* f32_ptr = reinterpret_cast<const float*>(data);
        std::vector<uint16_t> fp16_buf(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            fp16_buf[i] = atb_llm::Fp32ToFp16(f32_ptr[i]);
        }
        s = alloc.AllocFloat16(dst, shape);
        if (s != STATUS_OK) return s;
        return alloc.CopyToDevice(dst, fp16_buf.data(), num_elements * sizeof(uint16_t));

    } else if (st_dtype == safetensors::kINT64) {
        // int64: copy as-is (for position embeddings etc.)
        s = alloc.AllocInt64(dst, shape);
        if (s != STATUS_OK) return s;
        return alloc.CopyToDevice(dst, data, info.nbytes);

    } else {
        LOG_ERROR("Unsupported dtype for %s: %d", key.c_str(), info.dtype);
        return ERROR_UNSUPPORTED;
    }
}

// ═════════════════════════════════════════════════════════════════════
// Helper: copy weight tensor to host as float16
// ═════════════════════════════════════════════════════════════════════

Status CopyWeightToFp16Host(WeightLoader& loader,
                            const std::string& key,
                            void* host_dst,
                            size_t dst_capacity_bytes) {
    WeightInfo info;
    Status s = loader.GetTensor(key, info);
    if (s != STATUS_OK) return s;

    const uint8_t* data = loader.GetTensorData(key);
    if (!data) return ERROR_WEIGHT_LOAD;

    size_t num_elements = 1;
    for (auto d : info.shape)
        num_elements *= d;

    auto st_dtype = static_cast<safetensors::dtype>(info.dtype);
    size_t required_bytes = num_elements * sizeof(uint16_t);

    if (dst_capacity_bytes < required_bytes) {
        LOG_ERROR("Host buffer too small for %s: need %zu, have %zu",
                  key.c_str(), required_bytes, dst_capacity_bytes);
        return ERROR_INVALID_PARAM;
    }

    if (st_dtype == safetensors::kFLOAT16) {
        std::memcpy(host_dst, data, info.nbytes);
    } else if (st_dtype == safetensors::kBFLOAT16) {
        atb_llm::Bf16ToFp16Buffer(reinterpret_cast<const uint16_t*>(data),
                                  reinterpret_cast<uint16_t*>(host_dst), num_elements);
    } else if (st_dtype == safetensors::kFLOAT32) {
        const float* f32_ptr = reinterpret_cast<const float*>(data);
        uint16_t* dst16 = reinterpret_cast<uint16_t*>(host_dst);
        for (size_t i = 0; i < num_elements; i++) {
            dst16[i] = atb_llm::Fp32ToFp16(f32_ptr[i]);
        }
    } else {
        LOG_ERROR("Unsupported dtype for host copy %s: %d", key.c_str(), info.dtype);
        return ERROR_UNSUPPORTED;
    }

    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// Batch loading
// ═════════════════════════════════════════════════════════════════════

Status LoadLinearWeights(WeightLoader& loader,
                          TensorAllocator& alloc,
                          const std::string& prefix,
                          const WeightLoadEntry* entries,
                          size_t num_entries) {
    for (size_t i = 0; i < num_entries; i++) {
        std::string key = prefix + entries[i].suffix;
        Status s = CopyWeightToFp16NPU(loader, key, alloc, *entries[i].dst);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to load weight: %s", key.c_str());
            return s;
        }
    }
    return STATUS_OK;
}

}  // namespace io
}  // namespace atb_llm
