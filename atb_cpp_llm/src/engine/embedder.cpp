#include "atb_llm/embedder.h"
#include "log/logger.h"

namespace atb_llm {

Status Qwen3VLEmbedder::Load(const std::string& model_dir) {
    EngineConfig cfg;
    cfg.model_dir = model_dir;
    cfg.device_id = 0;
    cfg.buffer_size = 5ULL * 1024 * 1024 * 1024;
    cfg.normalize = true;  // L2-normalise output (matches Qwen3VLEmbedder)

    Status s = LLMEngine::Create(cfg, engine_);
    if (s != STATUS_OK) {
        LOG_ERROR("Qwen3VLEmbedder::Load: LLMEngine::Create failed: %d",
                  static_cast<int>(s));
        return s;
    }

    LOG_INFO("Qwen3VLEmbedder loaded from %s", model_dir.c_str());
    return STATUS_OK;
}

Status Qwen3VLEmbedder::Encode(const InferRequest& request,
                                 InferResult& result) {
    if (!engine_) {
        LOG_ERROR("Qwen3VLEmbedder::Encode: not loaded");
        return ERROR_INVALID_PARAM;
    }
    return engine_->Encode(request, result);
}

Status Qwen3VLEmbedder::EncodeWithTiming(const InferRequest& request,
                                          InferResult& result,
                                          StageTimings& timings) {
    if (!engine_) {
        LOG_ERROR("Qwen3VLEmbedder::EncodeWithTiming: not loaded");
        return ERROR_INVALID_PARAM;
    }
    return engine_->EncodeWithTiming(request, result, timings);
}

} // namespace atb_llm
