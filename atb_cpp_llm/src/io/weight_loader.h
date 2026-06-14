#pragma once
#include "atb_llm/types.h"
#include "io/safetensors_reader.h"
#include <string>
#include <vector>
#include <cstdint>

namespace atb_llm {

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

    /// Get number of tensors
    size_t NumTensors() const;

    /// Access underlying reader
    const SafetensorsReader& GetReader() const { return reader_; }

private:
    SafetensorsReader reader_;
};

} // namespace atb_llm
