#include "io/weight_loader.h"

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

size_t WeightLoader::NumTensors() const {
    return reader_.NumTensors();
}

} // namespace atb_llm
