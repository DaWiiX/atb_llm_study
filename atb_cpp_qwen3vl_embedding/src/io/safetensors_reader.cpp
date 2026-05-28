#include "io/safetensors_reader.h"
#include "log/logger.h"
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

namespace atb_llm {

SafetensorsReader::SafetensorsReader() : st_(new safetensors::safetensors_t()) {}

SafetensorsReader::~SafetensorsReader() {
    delete st_;
    st_ = nullptr;
}

SafetensorsReader::SafetensorsReader(SafetensorsReader&& other) noexcept : st_(other.st_) {
    other.st_ = nullptr;
}

SafetensorsReader& SafetensorsReader::operator=(SafetensorsReader&& other) noexcept {
    if (this != &other) {
        delete st_;
        st_ = other.st_;
        other.st_ = nullptr;
    }
    return *this;
}

Status SafetensorsReader::LoadFromFile(const std::string& path) {
    if (!st_) {
        st_ = new safetensors::safetensors_t();
    }
    std::string warn, err;
    bool ok = safetensors::load_from_file(path, st_, &warn, &err);
    if (!warn.empty()) {
        LOG_WARN("safetensors warning: %s", warn.c_str());
    }
    if (!ok) {
        LOG_ERROR("Failed to load safetensors: %s", err.c_str());
        return ERROR_FILE_NOT_FOUND;
    }
    LOG_INFO("Loaded safetensors: %zu tensors", st_->tensors.size());
    return STATUS_OK;
}

Status SafetensorsReader::GetTensor(const std::string& key, WeightInfo& info) const {
    if (!st_) return ERROR_WEIGHT_LOAD;

    safetensors::tensor_t tensor;
    if (!st_->tensors.at(key, &tensor)) {
        LOG_ERROR("Tensor not found: %s", key.c_str());
        return ERROR_WEIGHT_LOAD;
    }

    info.shape = tensor.shape;
    info.offset_begin = tensor.data_offsets[0];
    info.offset_end = tensor.data_offsets[1];
    info.nbytes = tensor.data_offsets[1] - tensor.data_offsets[0];
    info.dtype = static_cast<int>(tensor.dtype);
    return STATUS_OK;
}

const uint8_t* SafetensorsReader::GetTensorData(const std::string& key) const {
    if (!st_) return nullptr;

    safetensors::tensor_t tensor;
    if (!st_->tensors.at(key, &tensor)) {
        return nullptr;
    }

    size_t offset = tensor.data_offsets[0];

    if (st_->mmaped && st_->databuffer_addr) {
        return st_->databuffer_addr + offset;
    }

    if (!st_->storage.empty()) {
        return st_->storage.data() + offset;
    }

    return nullptr;
}

std::vector<std::string> SafetensorsReader::GetKeysByPrefix(const std::string& prefix) const {
    std::vector<std::string> result;
    if (!st_) return result;

    const auto& keys = st_->tensors.keys();
    for (const auto& key : keys) {
        if (key.size() >= prefix.size() &&
            key.compare(0, prefix.size(), prefix) == 0) {
            result.push_back(key);
        }
    }
    return result;
}

std::vector<std::string> SafetensorsReader::GetAllKeys() const {
    std::vector<std::string> result;
    if (!st_) return result;
    const auto& keys = st_->tensors.keys();
    result.assign(keys.begin(), keys.end());
    return result;
}

bool SafetensorsReader::HasKey(const std::string& key) const {
    if (!st_) return false;
    return st_->tensors.count(key);
}

size_t SafetensorsReader::NumTensors() const {
    if (!st_) return 0;
    return st_->tensors.size();
}

} // namespace atb_llm
