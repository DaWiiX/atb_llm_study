#pragma once
#include "atb_llm/model.h"
#include <map>
#include <string>

namespace atb_llm {

/// Global model factory registry (singleton).
class ModelRegistry {
public:
    static ModelRegistry& Instance();

    void Register(const std::string& model_type, ModelFactory factory);
    std::unique_ptr<IModel> Create(const std::string& model_type) const;
    bool Has(const std::string& model_type) const;

private:
    ModelRegistry() = default;
    std::map<std::string, ModelFactory> registry_;
};

} // namespace atb_llm
