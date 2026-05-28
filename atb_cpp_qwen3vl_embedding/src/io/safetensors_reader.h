#pragma once
#include "atb_llm/types.h"
#include <string>
#include <vector>
#include <cstdint>

// Forward declare to avoid exposing safetensors.hh
namespace safetensors {
struct safetensors_t;
}

namespace atb_llm {

/// Weight tensor metadata (without the actual data pointer)
struct WeightInfo {
    std::vector<size_t> shape;
    size_t offset_begin = 0;  // byte offset in data region
    size_t offset_end = 0;
    size_t nbytes = 0;
    int dtype = -1;  // safetensors::dtype enum value
};

/// RAII wrapper around safetensors::safetensors_t
class SafetensorsReader {
public:
    SafetensorsReader();
    ~SafetensorsReader();

    // Move-only
    SafetensorsReader(SafetensorsReader&& other) noexcept;
    SafetensorsReader& operator=(SafetensorsReader&& other) noexcept;
    SafetensorsReader(const SafetensorsReader&) = delete;
    SafetensorsReader& operator=(const SafetensorsReader&) = delete;

    /// Load from file
    Status LoadFromFile(const std::string& path);

    /// Get tensor metadata by key
    Status GetTensor(const std::string& key, WeightInfo& info) const;

    /// Get tensor data pointer (into mmap'd or loaded storage)
    const uint8_t* GetTensorData(const std::string& key) const;

    /// Get all keys with a given prefix
    std::vector<std::string> GetKeysByPrefix(const std::string& prefix) const;

    /// Get all keys
    std::vector<std::string> GetAllKeys() const;

    /// Check if key exists
    bool HasKey(const std::string& key) const;

    /// Get number of tensors
    size_t NumTensors() const;

private:
    safetensors::safetensors_t* st_ = nullptr;
};

} // namespace atb_llm
