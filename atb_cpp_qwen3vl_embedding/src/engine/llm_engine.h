#pragma once
#include "atb_llm/engine.h"
#include "engine/runtime_impl.h"
#include "engine/model_registry.h"
#include "log/logger.h"

namespace atb_llm {

/// LLMEngine::Impl -- the actual implementation hidden behind PImpl
class LLMEngine::Impl {
public:
    Status Init(const EngineConfig& config);
    Status Forward(const InferRequest& req, InferResult& res);
    Status Encode(const InferRequest& req, InferResult& res);

private:
    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<IModel> model_;
    std::string weight_path_;
    EngineConfig config_;
};

} // namespace atb_llm
