#include "runners/text_runner.h"
#include "components/common/rms_norm_graph.h"
#include "components/text/decoder_layer_graph.h"
#include "log/logger.h"

namespace atb_llm {
namespace runners {

TextRunner::TextRunner(const Config& cfg) : cfg_(cfg) {}

Status TextRunner::EnsureBuilt(int32_t seq_len) {
    // No-op if already built for this sequence length
    if (cached_seq_len_ == seq_len && layer_graph_) {
        return STATUS_OK;
    }

    // Build the shared decoder layer graph (built once, looped N times)
    Status s = components::text::TextDecoderLayerGraph::Build(
        "TextDecoderLayer",
        cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim,
        seq_len, cfg_.epsilon, cfg_.use_mask,
        layer_graph_, cfg_.use_qk_norm, cfg_.rotary_dim);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build TextDecoderLayer graph");
        return s;
    }

    // Build the final RMSNorm graph
    s = components::RmsNormGraph::Build("TextFinalNorm", cfg_.epsilon, norm_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build TextFinalNorm graph");
        return s;
    }

    cached_seq_len_ = seq_len;
    LOG_INFO("TextRunner built: %d layers, nh=%d, kv_nh=%d, hd=%d, S=%d",
             cfg_.num_layers, cfg_.num_heads, cfg_.num_kv_heads,
             cfg_.head_dim, seq_len);
    return STATUS_OK;
}

void MakeCausalMask(int32_t seq_len, float* mask_out) {
    // Create additive causal mask for ATB MASK_TYPE_NORM.
    // 0 = attend, -65504 (fp16 min) = mask.
    const float mask_value = -65504.0f;
    for (int32_t i = 0; i < seq_len; i++) {
        for (int32_t j = 0; j < seq_len; j++) {
            mask_out[i * seq_len + j] = (j > i) ? mask_value : 0.0f;
        }
    }
}

} // namespace runners
} // namespace atb_llm
