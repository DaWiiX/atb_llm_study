#include "io/weight_loader.h"
#include "core/tensor_allocator.h"
#include "log/logger.h"
#include "safetensors.hh"
#include <cstring>

namespace atb_llm {

WeightLoader::WeightLoader() = default;
WeightLoader::~WeightLoader() = default;

Status WeightLoader::LoadFromFile(const std::string& path) {
    return reader_.LoadFromFile(path);
}

Status WeightLoader::GetTensor(const std::string& key, WeightInfo& info) const {
    return reader_.GetTensor(key, info);
}

std::vector<std::string> WeightLoader::GetKeysByPrefix(const std::string& prefix) const {
    return reader_.GetKeysByPrefix(prefix);
}

const uint8_t* WeightLoader::GetTensorData(const std::string& key) const {
    return reader_.GetTensorData(key);
}

Status WeightLoader::CopyToNPU(const std::string& key, atb::Tensor& dst, TensorAllocator& alloc) {
    WeightInfo info;
    Status s = GetTensor(key, info);
    if (s != STATUS_OK) return s;

    const uint8_t* data = GetTensorData(key);
    if (!data) {
        LOG_ERROR("No data for tensor: %s", key.c_str());
        return ERROR_WEIGHT_LOAD;
    }

    // Convert shape from size_t to int64_t
    std::vector<int64_t> shape(info.shape.begin(), info.shape.end());

    // Determine ATB dtype and handle bf16->f16 conversion
    auto st_dtype = static_cast<safetensors::dtype>(info.dtype);

    if (st_dtype == safetensors::kFLOAT16) {
        ATB_LLM_CHECK(alloc.AllocFloat16(dst, shape));
        ATB_LLM_CHECK(alloc.CopyToDevice(dst, data, info.nbytes));
    } else if (st_dtype == safetensors::kBFLOAT16) {
        // Convert bf16 to float32 on host, then copy to NPU as float32
        // bf16: 1 sign + 8 exp + 7 mantissa, stored as 2 bytes
        size_t num_elements = info.nbytes / 2;
        std::vector<float> f32_buf(num_elements);
        const uint16_t* bf16_ptr = reinterpret_cast<const uint16_t*>(data);
        for (size_t i = 0; i < num_elements; i++) {
            // Reinterpret bf16 as f32 by shifting bits
            uint32_t bits = static_cast<uint32_t>(bf16_ptr[i]) << 16;
            std::memcpy(&f32_buf[i], &bits, sizeof(float));
        }
        size_t f32_bytes = num_elements * sizeof(float);
        ATB_LLM_CHECK(alloc.AllocFloat32(dst, shape));
        ATB_LLM_CHECK(alloc.CopyToDevice(dst, f32_buf.data(), f32_bytes));
    } else if (st_dtype == safetensors::kFLOAT32) {
        ATB_LLM_CHECK(alloc.AllocFloat32(dst, shape));
        ATB_LLM_CHECK(alloc.CopyToDevice(dst, data, info.nbytes));
    } else if (st_dtype == safetensors::kINT64) {
        ATB_LLM_CHECK(alloc.AllocInt64(dst, shape));
        ATB_LLM_CHECK(alloc.CopyToDevice(dst, data, info.nbytes));
    } else {
        LOG_ERROR("Unsupported dtype for tensor %s: %d", key.c_str(), info.dtype);
        return ERROR_UNSUPPORTED;
    }

    return STATUS_OK;
}

size_t WeightLoader::NumTensors() const {
    return reader_.NumTensors();
}

} // namespace atb_llm
