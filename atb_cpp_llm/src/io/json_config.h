#pragma once
#include "atb_llm/types.h"
#include <string>
#include <vector>

// Forward declare cJSON to avoid exposing C header
struct cJSON;

namespace atb_llm {

/// cJSON RAII wrapper -- type-safe JSON config reader
class JsonConfig {
public:
    JsonConfig();
    ~JsonConfig();

    // Move-only
    JsonConfig(JsonConfig&& other) noexcept;
    JsonConfig& operator=(JsonConfig&& other) noexcept;
    JsonConfig(const JsonConfig&) = delete;
    JsonConfig& operator=(const JsonConfig&) = delete;

    /// Load JSON from file path
    static JsonConfig Load(const std::string& path);

    /// Load JSON from string
    static JsonConfig Parse(const std::string& json_str);

    /// Type-safe accessors
    int GetInt(const std::string& key, int default_val = 0) const;
    float GetFloat(const std::string& key, float default_val = 0.0f) const;
    std::string GetString(const std::string& key, const std::string& default_val = "") const;
    bool GetBool(const std::string& key, bool default_val = false) const;
    std::vector<int> GetIntArray(const std::string& key) const;
    std::vector<float> GetFloatArray(const std::string& key) const;

    /// Get sub-object as a new JsonConfig
    JsonConfig GetSubConfig(const std::string& key) const;

    /// Check if key exists
    bool HasKey(const std::string& key) const;

    /// Check if the config is valid (non-null root)
    bool IsValid() const;

private:
    explicit JsonConfig(cJSON* root);
    cJSON* root_ = nullptr;
};

} // namespace atb_llm
