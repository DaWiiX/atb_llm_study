#include "atb_llm/embedder.h"
#include "log/logger.h"

namespace atb_llm {

namespace {
// Embedder-specific input/output invariants.  Failing any of these means
// the caller violated the Qwen3VLEmbedder contract documented in
// include/atb_llm/embedder.h.
Status ValidateEncodeInput(const InferRequest& request) {
    if (request.text.batch_size != 1) {
        LOG_ERROR("Qwen3VLEmbedder requires batch_size=1, got %ld",
                  static_cast<long>(request.text.batch_size));
        return ERROR_INVALID_PARAM;
    }
    if (request.text.seq_length <= 0 || request.text.input_ids == nullptr) {
        LOG_ERROR("Qwen3VLEmbedder requires non-empty input_ids");
        return ERROR_INVALID_PARAM;
    }
    return STATUS_OK;
}

Status ValidateEncodeOutput(const InferResult& result) {
    if (result.shape.size() != 1) {
        LOG_ERROR("Qwen3VLEmbedder expected rank-1 output, got rank=%zu",
                  result.shape.size());
        return ERROR_INFERENCE;
    }
    if (result.dtype != ACL_FLOAT16) {
        LOG_ERROR("Qwen3VLEmbedder expected fp16 output, got dtype=%d",
                  static_cast<int>(result.dtype));
        return ERROR_INFERENCE;
    }
    return STATUS_OK;
}
}  // namespace

Status Qwen3VLEmbedder::Load(const std::string& model_dir) {
    EngineConfig cfg;
    cfg.model_dir = model_dir;
    cfg.device_id = 0;
    cfg.buffer_size = 5ULL * 1024 * 1024 * 1024;
    // NOTE: L2 normalization is fixed-true for Qwen3VL embedding by the
    // model config (Qwen3VLConfig::normalize defaults to true and is not
    // exposed through EngineConfig).  An embedder always L2-normalises.

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
    // Validate caller contract first — these checks are independent of
    // engine_ state and make the failure mode clear at the API boundary.
    Status s = ValidateEncodeInput(request);
    if (s != STATUS_OK) return s;

    if (!engine_) {
        LOG_ERROR("Qwen3VLEmbedder::Encode: not loaded");
        return ERROR_INVALID_PARAM;
    }

    s = engine_->Encode(request, result);
    if (s != STATUS_OK) return s;

    return ValidateEncodeOutput(result);
}

Status Qwen3VLEmbedder::EncodeWithTiming(const InferRequest& request,
                                          InferResult& result,
                                          StageTimings& timings) {
    Status s = ValidateEncodeInput(request);
    if (s != STATUS_OK) return s;

    if (!engine_) {
        LOG_ERROR("Qwen3VLEmbedder::EncodeWithTiming: not loaded");
        return ERROR_INVALID_PARAM;
    }

    s = engine_->EncodeWithTiming(request, result, timings);
    if (s != STATUS_OK) return s;

    return ValidateEncodeOutput(result);
}

} // namespace atb_llm
