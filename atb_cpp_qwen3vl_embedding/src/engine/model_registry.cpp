#include "engine/model_registry.h"
#include "io/json_config.h"
#include "log/logger.h"

namespace atb_llm {

// ── ModelRegistry ────────────────────────────────────────

ModelRegistry& ModelRegistry::Instance() {
    static ModelRegistry instance;
    return instance;
}

void ModelRegistry::Register(const std::string& model_type, ModelFactory factory) {
    registry_[model_type] = std::move(factory);
    LOG_INFO("Registered model factory: %s", model_type.c_str());
}

std::unique_ptr<IModel> ModelRegistry::Create(const std::string& model_type) const {
    auto it = registry_.find(model_type);
    if (it != registry_.end()) {
        return it->second();
    }
    return nullptr;
}

bool ModelRegistry::Has(const std::string& model_type) const {
    return registry_.count(model_type) > 0;
}

// ── Free functions (declared in model.h) ─────────────────

void RegisterModelFactory(const std::string& model_type, ModelFactory factory) {
    ModelRegistry::Instance().Register(model_type, std::move(factory));
}

std::unique_ptr<IModel> CreateModel(const std::string& model_dir) {
    auto cfg = JsonConfig::Load(model_dir + "/config.json");
    if (!cfg.IsValid()) {
        LOG_ERROR("Failed to load config.json from %s", model_dir.c_str());
        return nullptr;
    }

    std::string model_type = cfg.GetString("model_type", "");
    if (model_type.empty()) {
        LOG_ERROR("No model_type in config.json");
        return nullptr;
    }

    // Try exact match first
    auto& registry = ModelRegistry::Instance();
    auto model = registry.Create(model_type);
    if (model) {
        LOG_INFO("Created model: %s", model_type.c_str());
        return model;
    }

    // Prefix matching fallback
    if (model_type.find("qwen3") != std::string::npos) {
        if (cfg.HasKey("vision_config")) {
            model = registry.Create("qwen3vl_embedding");
        } else {
            model = registry.Create("qwen3");
        }
    } else if (model_type.find("deepseek") != std::string::npos) {
        model = registry.Create("deepseek_v2");
    }

    if (model) {
        LOG_INFO("Created model via prefix match: %s", model_type.c_str());
    } else {
        LOG_ERROR("No registered factory for model type: %s", model_type.c_str());
    }

    return model;
}

} // namespace atb_llm
