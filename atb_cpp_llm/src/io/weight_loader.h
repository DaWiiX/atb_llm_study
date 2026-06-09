#pragma once
#include "atb_llm/types.h"
#include "atb/types.h"
#include "io/safetensors_reader.h"
#include <string>
#include <vector>
#include <cstdint>

namespace atb_llm {

class TensorAllocator;

/// Loads weights from safetensors and copies them to NPU memory
class WeightLoader {
public:
    WeightLoader();
    ~WeightLoader();

    /// Load safetensors file
    Status LoadFromFile(const std::string& path);

    /// Get tensor metadata
    Status GetTensor(const std::string& key, WeightInfo& info) const;

    /// Get all keys with a given prefix
    std::vector<std::string> GetKeysByPrefix(const std::string& prefix) const;

    /// Get raw data pointer for a tensor
    const uint8_t* GetTensorData(const std::string& key) const;

    /// Copy a weight tensor to NPU, converting dtype if needed
    /// Supported conversions: bf16->f32, f16->f16, f32->f32
    Status CopyToNPU(const std::string& key, atb::Tensor& dst, TensorAllocator& alloc);

    /// Get number of tensors
    size_t NumTensors() const;

    /// Access underlying reader
    const SafetensorsReader& GetReader() const { return reader_; }

private:
    SafetensorsReader reader_;
};

} // namespace atb_llm
