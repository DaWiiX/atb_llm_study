#pragma once
#include "atb_llm/engine.h"
#include "atb_llm/types.h"
#include <memory>
#include <string>

namespace atb_llm {

/// Production-grade Qwen3VL Embedding deployment interface.
///
/// Thin wrapper around LLMEngine that guarantees:
///   1. L2 normalization is ON (matches Qwen3VLEmbedder.process)
///   2. attention_mask-based last-token pooling (via InferRequest.text.attention_mask)
///
/// Tokenization is done OFFLINE (Python tokenizer → token_ids).
/// This class receives pre-tokenized input_ids.  Images must be
/// preprocessed into (pixel_values, grid_thw) fp16 by the upstream layer.
///
/// Usage:
///   Qwen3VLEmbedder emb;
///   emb.Load("/path/to/model");
///   InferRequest req;
///   // ... fill req.text.input_ids, req.text.seq_length,
///   //     req.preprocessed.pixel_values, etc.
///   // Optionally set req.text.attention_mask for padding-aware pooling.
///   InferResult out;
///   emb.Encode(req, out);
///   // out is (hidden_size,) fp16, L2-normalised
class Qwen3VLEmbedder {
public:
    Status Load(const std::string& model_dir);

    Status Encode(const InferRequest& request, InferResult& result);
    Status EncodeWithTiming(const InferRequest& request,
                             InferResult& result,
                             StageTimings& timings);

private:
    std::unique_ptr<LLMEngine> engine_;
};

} // namespace atb_llm
