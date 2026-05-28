#include "models/text_model.h"
#include "components/norm/rms_norm_graph.h"
#include "layers/text_decoder_layer.h"
#include "log/logger.h"
#include <cstring>
#include <limits>

namespace atb_llm {
namespace models {

TextModel::TextModel(const Config& cfg) : cfg_(cfg) {}

Status TextModel::Build(int32_t seq_len) {
    // Build the shared decoder layer graph (built once, looped 28 times)
    Status s = layers::TextDecoderLayerGraph::Build(
        "TextDecoderLayer",
        cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim,
        seq_len, cfg_.epsilon, /*use_mask=*/true,
        layer_graph_);
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

    LOG_INFO("TextModel built: %d layers, nh=%d, kv_nh=%d, hd=%d, S=%d",
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

} // namespace models
} // namespace atb_llm
