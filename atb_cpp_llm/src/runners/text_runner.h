#pragma once
#include "atb_llm/types.h"
#include "atb_llm/layer_desc.h"
#include "core/raii.h"
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace runners {

/// Text Runner — manages ATB graph lifecycle for text decoder.
///
/// Split-graph strategy: a single DecoderLayer graph is built once and
/// looped N times at runtime with per-layer weights.
/// A separate FinalNorm graph handles the output normalization.
///
/// This runner only manages graph building and caching.
/// Execution orchestration (layer loop, deepstack injection, etc.)
/// is the responsibility of the adapter/family layer.
class TextRunner {
public:
    /// Text runner configuration.
    struct Config {
        int32_t num_heads = 16;
        int32_t num_kv_heads = 16;
        int32_t head_dim = 128;
        int32_t intermediate_size = 4096;
        int32_t num_layers = 28;
        int32_t batch_size = 1;         // batch dimension (future use)
        float epsilon = 1e-6f;
        bool use_qk_norm = true;
        int32_t rotary_dim = 2;
        bool use_mask = true;

        /// Layer descriptor for config-driven graph building.
        /// When set, EnsureBuilt() uses layer_desc fields instead of
        /// the flat Config fields above. The default values match Qwen3VL-2B.
        LayerDescriptor layer_desc;
    };

    /// Constructor.
    explicit TextRunner(const Config& cfg);

    /// Build or rebuild ATB graphs for the given sequence length.
    /// If seq_len hasn't changed since last Build(), this is a no-op.
    /// @param seq_len  Sequence length for the decoder layer graph
    /// @return STATUS_OK on success
    Status EnsureBuilt(int32_t seq_len);

    /// Get the decoder layer graph (shared across all layers).
    OperationHandle& GetLayerGraph() { return layer_graph_; }

    /// Get the final norm graph.
    OperationHandle& GetNormGraph() { return norm_graph_; }

    /// Get the config.
    const Config& GetConfig() const { return cfg_; }

    /// Get the cached sequence length (0 = not yet built).
    int32_t cached_seq_len() const { return cached_seq_len_; }

private:
    Config cfg_;
    OperationHandle layer_graph_;
    OperationHandle norm_graph_;
    int32_t cached_seq_len_ = 0;
};

/// Build a causal mask for text attention.
/// @param seq_len  Sequence length
/// @param mask_out Output: (seq_len, seq_len) float32 additive mask, pre-allocated
void MakeCausalMask(int32_t seq_len, float* mask_out);

/// Build a causal mask for text attention, directly in fp16.
/// Eliminates the fp32 intermediate + fp32→fp16 conversion loop.
/// @param seq_len  Sequence length
/// @param mask_out Output: (seq_len, seq_len) fp16 additive mask, pre-allocated
void MakeCausalMaskFp16(int32_t seq_len, uint16_t* mask_out);

/// Build a causal mask for text attention, directly in NZ (FRACTAL_NZ) layout.
/// Output shape: [1, n1, s_pad, 16]  where n1 = ceil(seq_len/16), s_pad = n1*16.
/// Only writes the upper-triangular mask positions; padding is zero-filled.
/// @param seq_len  Sequence length
/// @param mask_out Output: NZ-layout fp16 mask, pre-allocated to n1*s_pad*16 elements
/// @param s_pad    Padded sequence length = ceil(seq_len/16)*16
/// @param n1       Number of 16-column blocks = ceil(seq_len/16)
void MakeCausalMaskNzFp16(int32_t seq_len, uint16_t* mask_out,
                           int64_t s_pad, int64_t n1);

} // namespace runners
} // namespace atb_llm
