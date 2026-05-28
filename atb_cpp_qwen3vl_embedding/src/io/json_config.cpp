#include "io/json_config.h"
#include "log/logger.h"
#include "cJSON.h"
#include <fstream>
#include <sstream>

namespace atb_llm {

JsonConfig::JsonConfig() : root_(nullptr) {}

JsonConfig::JsonConfig(cJSON* root) : root_(root) {}

JsonConfig::~JsonConfig() {
    if (root_) {
        cJSON_Delete(root_);
        root_ = nullptr;
    }
}

JsonConfig::JsonConfig(JsonConfig&& other) noexcept : root_(other.root_) {
    other.root_ = nullptr;
}

JsonConfig& JsonConfig::operator=(JsonConfig&& other) noexcept {
    if (this != &other) {
        if (root_) cJSON_Delete(root_);
        root_ = other.root_;
        other.root_ = nullptr;
    }
    return *this;
}

JsonConfig JsonConfig::Load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        LOG_ERROR("Failed to open JSON file: %s", path.c_str());
        return JsonConfig(nullptr);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return Parse(ss.str());
}

JsonConfig JsonConfig::Parse(const std::string& json_str) {
    cJSON* root = cJSON_Parse(json_str.c_str());
    if (!root) {
        LOG_ERROR("Failed to parse JSON: %s", cJSON_GetErrorPtr());
    }
    return JsonConfig(root);
}

int JsonConfig::GetInt(const std::string& key, int default_val) const {
    if (!root_) return default_val;
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root_, key.c_str());
    if (item && cJSON_IsNumber(item)) {
        return item->valueint;
    }
    return default_val;
}

float JsonConfig::GetFloat(const std::string& key, float default_val) const {
    if (!root_) return default_val;
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root_, key.c_str());
    if (item && cJSON_IsNumber(item)) {
        return static_cast<float>(item->valuedouble);
    }
    return default_val;
}

std::string JsonConfig::GetString(const std::string& key, const std::string& default_val) const {
    if (!root_) return default_val;
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root_, key.c_str());
    if (item && cJSON_IsString(item) && item->valuestring) {
        return item->valuestring;
    }
    return default_val;
}

bool JsonConfig::GetBool(const std::string& key, bool default_val) const {
    if (!root_) return default_val;
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root_, key.c_str());
    if (item && cJSON_IsBool(item)) {
        return cJSON_IsTrue(item);
    }
    return default_val;
}

std::vector<int> JsonConfig::GetIntArray(const std::string& key) const {
    std::vector<int> result;
    if (!root_) return result;
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root_, key.c_str());
    if (item && cJSON_IsArray(item)) {
        int size = cJSON_GetArraySize(item);
        result.reserve(size);
        for (int i = 0; i < size; i++) {
            cJSON* elem = cJSON_GetArrayItem(item, i);
            if (elem && cJSON_IsNumber(elem)) {
                result.push_back(elem->valueint);
            }
        }
    }
    return result;
}

std::vector<float> JsonConfig::GetFloatArray(const std::string& key) const {
    std::vector<float> result;
    if (!root_) return result;
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root_, key.c_str());
    if (item && cJSON_IsArray(item)) {
        int size = cJSON_GetArraySize(item);
        result.reserve(size);
        for (int i = 0; i < size; i++) {
            cJSON* elem = cJSON_GetArrayItem(item, i);
            if (elem && cJSON_IsNumber(elem)) {
                result.push_back(static_cast<float>(elem->valuedouble));
            }
        }
    }
    return result;
}

JsonConfig JsonConfig::GetSubConfig(const std::string& key) const {
    if (!root_) return JsonConfig(nullptr);
    cJSON* item = cJSON_GetObjectItemCaseSensitive(root_, key.c_str());
    if (item && cJSON_IsObject(item)) {
        // Detach from parent so it can be independently managed
        cJSON* copy = cJSON_Duplicate(item, 1);
        return JsonConfig(copy);
    }
    return JsonConfig(nullptr);
}

bool JsonConfig::HasKey(const std::string& key) const {
    if (!root_) return false;
    return cJSON_HasObjectItem(root_, key.c_str()) != 0;
}

bool JsonConfig::IsValid() const {
    return root_ != nullptr;
}

} // namespace atb_llm
