#include "engine/llm_engine.h"
#include "log/logger.h"
#include "util/cpp11_compat.h"
#include <chrono>

namespace atb_llm {

// ── IModel default: ForwardWithTiming wraps Forward with e2e timer ──

Status IModel::ForwardWithTiming(const InferRequest& request,
                                  InferResult& result,
                                  StageTimings& timings) {
    auto t0 = std::chrono::high_resolution_clock::now();
    Status s = Forward(request, result);
    auto t1 = std::chrono::high_resolution_clock::now();
    timings.e2e_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return s;
}

// ── LLMEngine::Impl ─────────────────────────────────────

Status LLMEngine::Impl::Init(const EngineConfig& config) {
    // 1. Initialize ACL + ATB Context + Stream
    runtime_ = CreateRuntime(config.device_id, config.buffer_size);
    if (!runtime_) {
        LOG_ERROR("Failed to create runtime");
        return ERROR_NPU_MEMORY;
    }

    // 2. Open weight file
    weight_path_ = config.model_dir + "/model.safetensors";
    Status s = runtime_->GetWeightLoader()->LoadFromFile(weight_path_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to load weights from %s", weight_path_.c_str());
        return s;
    }

    // 3. Auto-detect model type and create
    model_ = CreateModel(config.model_dir);
    if (!model_) {
        LOG_ERROR("Failed to create model from %s", config.model_dir.c_str());
        return ERROR_UNSUPPORTED;
    }

    // 4. Load model (model handles graph building + weight loading)
    s = model_->Load(config.model_dir, runtime_.get());
    if (s != STATUS_OK) {
        LOG_ERROR("Model Load() failed: %d", static_cast<int>(s));
        return s;
    }

    config_ = config;
    LOG_INFO("LLMEngine initialized: model=%s, device=%d",
             model_->GetName(), config.device_id);
    return STATUS_OK;
}

Status LLMEngine::Impl::Forward(const InferRequest& req, InferResult& res) {
    return model_->Forward(req, res);
}

Status LLMEngine::Impl::Encode(const InferRequest& req, InferResult& res) {
    return model_->Forward(req, res);  // For embedding models, Encode == Forward
}

Status LLMEngine::Impl::EncodeWithTiming(const InferRequest& req, InferResult& res,
                                          StageTimings& timings) {
    return model_->ForwardWithTiming(req, res, timings);
}

// ── LLMEngine ───────────────────────────────────────────

LLMEngine::LLMEngine() : impl_(atb_llm::make_unique<Impl>()) {}

LLMEngine::~LLMEngine() = default;

Status LLMEngine::Create(const EngineConfig& config,
                          std::unique_ptr<LLMEngine>& engine) {
    if (config.model_dir.empty()) {
        LOG_ERROR("EngineConfig.model_dir is empty");
        return ERROR_INVALID_PARAM;
    }

    engine.reset(new LLMEngine());
    Status s = engine->impl_->Init(config);
    if (s != STATUS_OK) {
        engine.reset();
        return s;
    }
    return STATUS_OK;
}

Status LLMEngine::Forward(const InferRequest& request, InferResult& result) {
    if (!impl_) return ERROR_INVALID_PARAM;
    return impl_->Forward(request, result);
}

Status LLMEngine::Encode(const InferRequest& request, InferResult& result) {
    if (!impl_) return ERROR_INVALID_PARAM;
    return impl_->Encode(request, result);
}

Status LLMEngine::EncodeWithTiming(const InferRequest& request,
                                    InferResult& result,
                                    StageTimings& timings) {
    if (!impl_) return ERROR_INVALID_PARAM;
    return impl_->EncodeWithTiming(request, result, timings);
}

} // namespace atb_llm
