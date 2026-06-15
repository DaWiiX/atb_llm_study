#include "runners/text_runner.h"
#include "components/common/rms_norm_graph.h"
#include "components/text/decoder_layer_graph.h"
#include "utils/float_utils.h"
#include "log/logger.h"

namespace atb_llm {
namespace runners {

TextRunner::TextRunner(const Config& cfg) : cfg_(cfg) {}

Status TextRunner::EnsureBuilt(int32_t seq_len) {
    // No-op if already built for this sequence length
    if (cached_seq_len_ == seq_len && layer_graph_) {
        return STATUS_OK;
    }

    // Sync flat Config fields into layer_desc so that legacy usage
    // (setting cfg.num_heads, etc.) still works.  The layer_desc
    // fields take precedence, but users who only set flat fields
    // expect them to be reflected here.
    auto& ld = cfg_.layer_desc;
    ld.attn.num_heads    = cfg_.num_heads;
    ld.attn.num_kv_heads = cfg_.num_kv_heads;
    ld.attn.head_dim     = cfg_.head_dim;
    ld.attn.epsilon      = cfg_.epsilon;
    ld.attn.use_qk_norm  = cfg_.use_qk_norm;
    ld.attn.rotary_dim   = cfg_.rotary_dim;
    ld.attn.use_mask     = cfg_.use_mask;
    ld.attn.seq_len      = seq_len;
    ld.mlp.intermediate_size = cfg_.intermediate_size;
    ld.input_norm.epsilon   = cfg_.epsilon;
    ld.post_norm.epsilon    = cfg_.epsilon;

    const auto& attn = ld.attn;

    // Build the shared decoder layer graph using LayerDescriptor
    Status s = components::text::TextDecoderLayerGraph::Build(
        "TextDecoderLayer",
        attn.num_heads, attn.num_kv_heads, attn.head_dim,
        seq_len, attn.epsilon, attn.use_mask,
        layer_graph_, attn.use_qk_norm, attn.rotary_dim);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build TextDecoderLayer graph");
        return s;
    }

    // Build the final normalization graph using LayerDescriptor
    s = components::RmsNormGraph::Build("TextFinalNorm",
        ld.post_norm, norm_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build TextFinalNorm graph");
        return s;
    }

    cached_seq_len_ = seq_len;
    LOG_INFO("TextRunner built: %d layers, nh=%d, kv_nh=%d, hd=%d, S=%d",
             cfg_.num_layers, attn.num_heads, attn.num_kv_heads,
             attn.head_dim, seq_len);
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

void MakeCausalMaskFp16(int32_t seq_len, uint16_t* mask_out) {
    // Direct fp16 generation — skips the fp32 intermediate allocation
    // and the element-by-element fp32→fp16 conversion loop.
    // -65504 = fp16 minimum representable value (used as mask sentinel).
    const uint16_t mask_fp16 = atb_llm::Fp32ToFp16(-65504.0f);
    const uint16_t zero_fp16  = atb_llm::Fp32ToFp16(0.0f);
    for (int32_t i = 0; i < seq_len; i++) {
        for (int32_t j = 0; j < seq_len; j++) {
            mask_out[i * seq_len + j] = (j > i) ? mask_fp16 : zero_fp16;
        }
    }
}

void MakeCausalMaskNzFp16(int32_t seq_len, uint16_t* mask_out,
                           int64_t s_pad, int64_t n1) {
    // Generate causal mask directly in NZ (FRACTAL_NZ) layout.
    // Shape: [1, n1, s_pad, 16]  where n1 = ceil(seq_len/16), s_pad = n1*16.
    // NZ[0, col/16, row, col%16] = ND[row][col]
    //
    // Only the upper-triangular positions are non-zero (-65504 = masked).
    // All padding and lower-triangular positions default to zero (attend).
    const int64_t total = n1 * s_pad * 16;
    const uint16_t mask_fp16 = atb_llm::Fp32ToFp16(-65504.0f);
    // Zero-fill everything (padding + lower triangle = attend)
    std::memset(mask_out, 0, static_cast<size_t>(total) * sizeof(uint16_t));
    // Write upper-triangular mask positions
    for (int32_t row = 0; row < seq_len; row++) {
        for (int32_t col = row + 1; col < seq_len; col++) {  // col > row = masked
            int64_t block_col = col / 16;
            int64_t col_in_block = col % 16;
            int64_t nz_idx = block_col * s_pad * 16 + row * 16 + col_in_block;
            mask_out[nz_idx] = mask_fp16;
        }
    }
}

} // namespace runners
} // namespace atb_llm
