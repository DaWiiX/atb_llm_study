#pragma once
#include "atb_llm/model.h"
#include <vector>
#include <string>

namespace atb_llm {

/// Global model factory registry (singleton).
class ModelRegistry {
public:
    static ModelRegistry& Instance();

    /// Register a RegistryEntry
    void Register(RegistryEntry entry);

    /// Legacy register: creates a RegistryEntry with priority=0, no compat_check
    void Register(const std::string& model_type, ModelFactory factory);

    /// Exact match by model_type
    std::unique_ptr<IModel> Create(const std::string& model_type) const;

    /// Fallback: exact match first, then compat_check by priority (descending)
    std::unique_ptr<IModel> CreateWithFallback(const std::string& model_type,
                                               const JsonConfig& cfg) const;

    bool Has(const std::string& model_type) const;

private:
    ModelRegistry() = default;
    std::vector<RegistryEntry> entries_;
};

} // namespace atb_llm
