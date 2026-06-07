#pragma once
#include "atb_llm/types.h"
#include "atb_llm/model.h"
#include <memory>

namespace atb_llm {

class LLMEngine {
public:
    static Status Create(const EngineConfig& config,
                         std::unique_ptr<LLMEngine>& engine);
    Status Forward(const InferRequest& request, InferResult& result);
    Status Encode(const InferRequest& request, InferResult& result);
    Status EncodeWithTiming(const InferRequest& request,
                             InferResult& result,
                             StageTimings& timings);
    ~LLMEngine();

private:
    LLMEngine();
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace atb_llm
