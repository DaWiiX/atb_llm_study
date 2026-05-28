#pragma once
#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "io/weight_loader.h"
#include "core/tensor_allocator.h"
#include "atb/types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace adapters {

/// Pre-loaded NPU-resident weights for a single text decoder layer.
/// All tensors are float16 on NPU.
/// Order matches Python get_text_layer_weights:
///   q_w, k_w, v_w, o_w, qn_w, kn_w, gate_w, up_w, down_w, input_ln_w, post_ln_w
struct TextLayerWeights {
    atb::Tensor q_weight;
    atb::Tensor k_weight;
    atb::Tensor v_weight;
    atb::Tensor o_weight;
    atb::Tensor q_norm_weight;
    atb::Tensor k_norm_weight;
    atb::Tensor gate_weight;
    atb::Tensor up_weight;
    atb::Tensor down_weight;
    atb::Tensor input_ln_weight;
    atb::Tensor post_ln_weight;
};

/// Pre-loaded NPU-resident weights for a single vision block.
/// All tensors are float16 on NPU.
/// Order matches Python get_vision_block_weights:
///   qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b, n1_w, n1_b, n2_w, n2_b
struct VisionBlockWeights {
    atb::Tensor qkv_weight;
    atb::Tensor qkv_bias;
    atb::Tensor proj_weight;
    atb::Tensor proj_bias;
    atb::Tensor fc1_weight;
    atb::Tensor fc1_bias;
    atb::Tensor fc2_weight;
    atb::Tensor fc2_bias;
    atb::Tensor n1_weight;
    atb::Tensor n1_bias;
    atb::Tensor n2_weight;
    atb::Tensor n2_bias;
};

/// Pre-loaded NPU-resident merger weights.
/// Order: n_w, n_b, f1_w, f1_b, f2_w, f2_b
struct MergerWeights {
    atb::Tensor norm_weight;
    atb::Tensor norm_bias;
    atb::Tensor fc1_weight;
    atb::Tensor fc1_bias;
    atb::Tensor fc2_weight;
    atb::Tensor fc2_bias;
};

/// Pre-loaded patch embedding weights.
struct PatchEmbedWeights {
    atb::Tensor weight;  // (hidden_size, C*tp*p*p)
    atb::Tensor bias;    // (hidden_size,)
};

/// All pre-loaded weights for Qwen3VL model.
/// All tensors are float16 on NPU (except embed_weight which stays on CPU).
struct Qwen3VLWeights {
    // Text
    std::vector<TextLayerWeights> text_layers;    // 28 layers
    atb::Tensor text_norm_weight;                  // final norm

    // Vision
    std::vector<VisionBlockWeights> vis_blocks;    // 24 blocks
    PatchEmbedWeights vis_patch_embed;
    atb::Tensor vis_pos_embed;                     // (num_pos_embed, hidden_size) on NPU

    // Merger
    MergerWeights merger;
    std::vector<MergerWeights> deepstack_mergers;  // one per ds index

    // Embedding (stays on CPU for F.embedding lookup)
    // We store it as a host-side buffer
    void* embed_weight_host = nullptr;
    int64_t embed_vocab_size = 0;
    int64_t embed_hidden_size = 0;
};

/// Load all model weights from safetensors and copy to NPU.
/// Converts bf16 -> float16 for all NPU-resident weights.
Status LoadQwen3VLWeights(const std::string& model_dir,
                          const Qwen3VLConfig& config,
                          WeightLoader& loader,
                          TensorAllocator& alloc,
                          Qwen3VLWeights& weights);

/// Convert bf16 (uint16 raw bits) to float16 (uint16 raw bits).
/// Returns the float16 bits.
uint16_t Bf16ToFp16(uint16_t bf16_bits);

/// Convert bf16 buffer to fp16 buffer (element-wise).
void Bf16ToFp16Buffer(const uint16_t* src, uint16_t* dst, size_t num_elements);

} // namespace adapters
} // namespace atb_llm
