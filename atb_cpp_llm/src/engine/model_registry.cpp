#include "engine/model_registry.h"
#include "io/json_config.h"
#include "log/logger.h"
#include <algorithm>

namespace atb_llm {

// ── ModelRegistry ────────────────────────────────────────

ModelRegistry& ModelRegistry::Instance() {
    static ModelRegistry instance;
    return instance;
}

void ModelRegistry::Register(RegistryEntry entry) {
    LOG_INFO("Registered model entry: %s (priority=%d, has_compat=%s)",
             entry.model_type.c_str(),
             static_cast<int>(entry.priority),
             entry.compat_check ? "yes" : "no");
    entries_.push_back(std::move(entry));
}

void ModelRegistry::Register(const std::string& model_type, ModelFactory factory) {
    RegistryEntry entry;
    entry.model_type = model_type;
    entry.factory = std::move(factory);
    entry.priority = 0;
    entry.compat_check = nullptr;
    Register(std::move(entry));
}

std::unique_ptr<IModel> ModelRegistry::Create(const std::string& model_type) const {
    for (const auto& e : entries_) {
        if (e.model_type == model_type) {
            return e.factory();
        }
    }
    return nullptr;
}

std::unique_ptr<IModel> ModelRegistry::CreateWithFallback(const std::string& model_type,
                                                           const JsonConfig& cfg) const {
    // Step 1: Exact match
    for (const auto& e : entries_) {
        if (e.model_type == model_type) {
            return e.factory();
        }
    }

    // Step 2: Compatibility check (sorted by priority descending)
    // Collect matching entries with compat_check
    std::vector<const RegistryEntry*> matches;
    for (const auto& e : entries_) {
        if (e.compat_check && e.compat_check(model_type, cfg)) {
            matches.push_back(&e);
        }
    }

    if (!matches.empty()) {
        // Sort by priority descending (highest priority first)
        std::sort(matches.begin(), matches.end(),
                  [](const RegistryEntry* a, const RegistryEntry* b) {
                      return a->priority > b->priority;
                  });
        return matches.front()->factory();
    }

    return nullptr;
}

bool ModelRegistry::Has(const std::string& model_type) const {
    for (const auto& e : entries_) {
        if (e.model_type == model_type) {
            return true;
        }
    }
    return false;
}

// ── Free functions (declared in model.h) ─────────────────

void RegisterModelFactory(const std::string& model_type, ModelFactory factory) {
    ModelRegistry::Instance().Register(model_type, std::move(factory));
}

void RegisterModelEntry(RegistryEntry entry) {
    ModelRegistry::Instance().Register(std::move(entry));
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

    auto& registry = ModelRegistry::Instance();

    // Step 1: Exact match
    auto model = registry.Create(model_type);
    if (model) {
        LOG_INFO("Created model (exact match): %s", model_type.c_str());
        return model;
    }

    // Step 2: Compatibility check (by priority descending)
    model = registry.CreateWithFallback(model_type, cfg);
    if (model) {
        LOG_INFO("Created model (compat match): %s", model_type.c_str());
        return model;
    }

    LOG_ERROR("No registered factory for model type: %s", model_type.c_str());
    return nullptr;
}

} // namespace atb_llm
