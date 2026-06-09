#pragma once
#include "atb_llm/engine.h"
#include "atb_llm/types.h"
#include <memory>
#include <string>

namespace atb_llm {

/// Production-grade Qwen3VL Embedding deployment interface.
///
/// Thin wrapper around LLMEngine that adds embedder-specific invariants
/// on top of the generic Forward/Encode contract:
///
///   * Output is L2-normalised by construction (Qwen3VLConfig::normalize
///     is fixed-true; this class does not expose a toggle).
///   * Single-sample only: request.text.batch_size MUST be 1.  Batched
///     embedding should use multiple Encode calls or a future Batch API.
///   * Pooling is attention_mask-aware: if request.text.attention_mask
///     is set, pooling uses the last non-padded token; otherwise it
///     falls back to seq_len-1.  Embedder callers SHOULD provide the
///     mask whenever the input contains padding.
///   * Tokenisation happens OFFLINE (see chat_tokenizer.py for the
///     Qwen3VL chat template).  This class consumes pre-tokenised
///     input_ids only.
///
/// Output:
///   InferResult.shape = {hidden_size}, dtype = ACL_FLOAT16, L2-normed.
///
/// Usage:
///   Qwen3VLEmbedder emb;
///   emb.Load("/path/to/model");
///   InferRequest req;
///   // ... fill req.text.input_ids, req.text.seq_length,
///   //     req.preprocessed.{pixel_values, num_patches, grid_thw} for image.
///   // Optionally set req.text.attention_mask for padded inputs.
///   InferResult out;
///   emb.Encode(req, out);
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
