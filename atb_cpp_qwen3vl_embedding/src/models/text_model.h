#pragma once
#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace models {

/// Qwen3VL Text Model runner.
///
/// Split-graph strategy: a single DecoderLayer graph is built once and
/// looped 28 times at runtime with per-layer weights.
/// A separate FinalNorm graph handles the output normalization.
///
/// Equivalent to Python text_model.py.
class TextModel {
public:
    /// Text model configuration.
    struct Config {
        int32_t num_heads = 16;
        int32_t num_kv_heads = 16;
        int32_t head_dim = 128;
        int32_t intermediate_size = 4096;
        int32_t num_layers = 28;
        float epsilon = 1e-6f;
    };

    /// Constructor.
    explicit TextModel(const Config& cfg);

    /// Build all required ATB graphs.
    /// @param seq_len  Sequence length for the decoder layer graph
    /// @return STATUS_OK on success
    Status Build(int32_t seq_len);

    /// Get the decoder layer graph (shared across all 28 layers).
    OperationHandle& GetLayerGraph() { return layer_graph_; }

    /// Get the final norm graph.
    OperationHandle& GetNormGraph() { return norm_graph_; }

    /// Get the config.
    const Config& GetConfig() const { return cfg_; }

private:
    Config cfg_;
    OperationHandle layer_graph_;
    OperationHandle norm_graph_;
};

/// Build a causal mask for text attention.
/// @param seq_len  Sequence length
/// @param mask_out Output: (seq_len, seq_len) float32 additive mask, pre-allocated
void MakeCausalMask(int32_t seq_len, float* mask_out);

} // namespace models
} // namespace atb_llm
