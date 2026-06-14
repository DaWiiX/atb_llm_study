#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "io/weight_helpers.h"
#include "utils/float_utils.h"
#include "io/safetensors_reader.h"
#include "log/logger.h"
#include "safetensors.hh"
#include <cstring>
#include <vector>

namespace atb_llm {
namespace adapters {

// ═════════════════════════════════════════════════════════════════════
// Weight loading
// ═════════════════════════════════════════════════════════════════════

static Status LoadTextLayerWeights(WeightLoader& loader,
                                   const Qwen3VLConfig& config,
                                   TensorAllocator& alloc,
                                   int32_t layer_idx,
                                   TextLayerWeights& w) {
    (void)config;  // reserved for future per-layer config lookups (e.g. layer-specific quant scales)
    std::string pfx = "model.language_model.layers." + std::to_string(layer_idx) + ".";
    io::WeightLoadEntry entries[] = {
        {"self_attn.q_proj.weight", &w.q_weight},
        {"self_attn.k_proj.weight", &w.k_weight},
        {"self_attn.v_proj.weight", &w.v_weight},
        {"self_attn.o_proj.weight", &w.o_weight},
        {"self_attn.q_norm.weight", &w.q_norm_weight},
        {"self_attn.k_norm.weight", &w.k_norm_weight},
        {"mlp.gate_proj.weight", &w.gate_weight},
        {"mlp.up_proj.weight", &w.up_weight},
        {"mlp.down_proj.weight", &w.down_weight},
        {"input_layernorm.weight", &w.input_ln_weight},
        {"post_attention_layernorm.weight", &w.post_ln_weight},
    };
    return io::LoadLinearWeights(loader, alloc, pfx, entries, sizeof(entries) / sizeof(entries[0]));
}

static Status LoadVisionBlockWeights(WeightLoader& loader,
                                     const Qwen3VLConfig& config,
                                     TensorAllocator& alloc,
                                     int32_t block_idx,
                                     VisionBlockWeights& w) {
    (void)config;  // reserved for future per-block config lookups
    std::string pfx = "model.visual.blocks." + std::to_string(block_idx) + ".";
    io::WeightLoadEntry entries[] = {
        {"attn.qkv.weight", &w.qkv_weight},
        {"attn.qkv.bias", &w.qkv_bias},
        {"attn.proj.weight", &w.proj_weight},
        {"attn.proj.bias", &w.proj_bias},
        {"mlp.linear_fc1.weight", &w.fc1_weight},
        {"mlp.linear_fc1.bias", &w.fc1_bias},
        {"mlp.linear_fc2.weight", &w.fc2_weight},
        {"mlp.linear_fc2.bias", &w.fc2_bias},
        {"norm1.weight", &w.n1_weight},
        {"norm1.bias", &w.n1_bias},
        {"norm2.weight", &w.n2_weight},
        {"norm2.bias", &w.n2_bias},
    };
    return io::LoadLinearWeights(loader, alloc, pfx, entries, sizeof(entries) / sizeof(entries[0]));
}

static Status LoadMergerWeights(WeightLoader& loader,
                                TensorAllocator& alloc,
                                const std::string& prefix,
                                MergerWeights& w) {
    io::WeightLoadEntry entries[] = {
        {"norm.weight", &w.norm_weight},
        {"norm.bias", &w.norm_bias},
        {"linear_fc1.weight", &w.fc1_weight},
        {"linear_fc1.bias", &w.fc1_bias},
        {"linear_fc2.weight", &w.fc2_weight},
        {"linear_fc2.bias", &w.fc2_bias},
    };
    return io::LoadLinearWeights(loader, alloc, prefix, entries, sizeof(entries) / sizeof(entries[0]));
}

Status LoadQwen3VLWeights(const std::string& model_dir,
                          const Qwen3VLConfig& config,
                          WeightLoader& loader,
                          TensorAllocator& alloc,
                          Qwen3VLWeights& weights) {
    Status s;

    // ── Load safetensors ──────────────────────────────────
    std::string st_path = model_dir + "/model.safetensors";
    s = loader.LoadFromFile(st_path);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to load safetensors: %s", st_path.c_str());
        return s;
    }

    // ── Text layers (28) ──────────────────────────────────
    weights.text_layers.resize(config.text_num_layers);
    for (int32_t i = 0; i < config.text_num_layers; i++) {
        s = LoadTextLayerWeights(loader, config, alloc, i, weights.text_layers[i]);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to load text layer %d weights", i);
            return s;
        }
    }
    LOG_INFO("Loaded %d text layer weights", config.text_num_layers);

    // ── Text final norm ───────────────────────────────────
    s = io::CopyWeightToFp16NPU(loader, "model.language_model.norm.weight",
                            alloc, weights.text_norm_weight);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to load text norm weight");
        return s;
    }

    // ── Embedding (stays on CPU) ──────────────────────────
    {
        WeightInfo emb_info;
        s = loader.GetTensor("model.language_model.embed_tokens.weight", emb_info);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to get embed weight info");
            return s;
        }
        const uint8_t* emb_data = loader.GetTensorData(
            "model.language_model.embed_tokens.weight");
        if (!emb_data) {
            LOG_ERROR("No data for embed weight");
            return ERROR_WEIGHT_LOAD;
        }

        // Validate embedding shape and compute vocab size
        if (config.text_hidden_size <= 0) {
            LOG_ERROR("Embed vocab size: text_hidden_size=%ld is not positive",
                      static_cast<long>(config.text_hidden_size));
            return ERROR_WEIGHT_LOAD;
        }
        if (emb_info.shape.size() != 2) {
            LOG_ERROR("Embed vocab size: expected rank-2 embedding tensor, got rank=%zu",
                      emb_info.shape.size());
            return ERROR_WEIGHT_LOAD;
        }
        int64_t total_elems = static_cast<int64_t>(emb_info.shape[0]) * static_cast<int64_t>(emb_info.shape[1]);
        if (total_elems % config.text_hidden_size != 0) {
            LOG_ERROR("Embed vocab size: total_elems=%ld not divisible by text_hidden_size=%ld",
                      static_cast<long>(total_elems), static_cast<long>(config.text_hidden_size));
            return ERROR_WEIGHT_LOAD;
        }
        weights.embed_vocab_size = total_elems / config.text_hidden_size;
        weights.embed_hidden_size = config.text_hidden_size;

        size_t num_elements = static_cast<size_t>(weights.embed_vocab_size) * config.text_hidden_size;
        size_t host_bytes = num_elements * sizeof(uint16_t);
        weights.embed_weight_host.resize(num_elements);

        // Copy embedding to host as fp16
        auto st_dtype = static_cast<safetensors::dtype>(emb_info.dtype);
        if (st_dtype == safetensors::kBFLOAT16) {
            atb_llm::Bf16ToFp16Buffer(reinterpret_cast<const uint16_t*>(emb_data),
                                      weights.embed_weight_host.data(),
                                      num_elements);
        } else if (st_dtype == safetensors::kFLOAT16) {
            std::memcpy(weights.embed_weight_host.data(), emb_data, host_bytes);
        } else if (st_dtype == safetensors::kFLOAT32) {
            // Convert fp32 -> fp16 (round-to-nearest-even, matches Python)
            const float* f32 = reinterpret_cast<const float*>(emb_data);
            uint16_t* dst16 = weights.embed_weight_host.data();
            for (size_t i = 0; i < num_elements; i++) {
                dst16[i] = atb_llm::Fp32ToFp16(f32[i]);
            }
        }
        LOG_INFO("Embedding weight loaded to host: vocab=%ld, hs=%ld",
                 static_cast<long>(weights.embed_vocab_size),
                 static_cast<long>(weights.embed_hidden_size));
    }

    // ── Vision blocks (24) ────────────────────────────────
    weights.vis_blocks.resize(config.vis_depth);
    for (int32_t i = 0; i < config.vis_depth; i++) {
        s = LoadVisionBlockWeights(loader, config, alloc, i, weights.vis_blocks[i]);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to load vision block %d weights", i);
            return s;
        }
    }
    LOG_INFO("Loaded %d vision block weights", config.vis_depth);

    // ── Vision patch embed ────────────────────────────────
    {
        // proj.weight shape: (hidden_size, C, tp, p, p) -> reshape to (hidden_size, C*tp*p*p)
        WeightInfo pe_info;
        s = loader.GetTensor("model.visual.patch_embed.proj.weight", pe_info);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to get patch_embed weight info");
            return s;
        }
        const uint8_t* pe_data = loader.GetTensorData("model.visual.patch_embed.proj.weight");
        if (!pe_data) {
            LOG_ERROR("No data for patch_embed weight");
            return ERROR_WEIGHT_LOAD;
        }

        int64_t hs = config.vis_hidden_size;
        int64_t ksize = config.vis_in_channels * config.vis_temporal_patch_size *
                        config.vis_patch_size * config.vis_patch_size;

        // Copy to host, convert to fp16, reshape
        size_t num_pe_elements = 1;
        for (auto d : pe_info.shape)
            num_pe_elements *= d;
        std::vector<uint16_t> pe_fp16(num_pe_elements);

        auto st_dtype = static_cast<safetensors::dtype>(pe_info.dtype);
        if (st_dtype == safetensors::kBFLOAT16) {
            atb_llm::Bf16ToFp16Buffer(reinterpret_cast<const uint16_t*>(pe_data),
                                      pe_fp16.data(), num_pe_elements);
        } else if (st_dtype == safetensors::kFLOAT16) {
            std::memcpy(pe_fp16.data(), pe_data, num_pe_elements * sizeof(uint16_t));
        } else {
            LOG_ERROR("Unsupported patch_embed dtype: %d", pe_info.dtype);
            return ERROR_UNSUPPORTED;
        }

        // Weight: (hs, ksize) - already contiguous after reshape
        s = alloc.AllocFloat16(weights.vis_patch_embed.weight, {hs, ksize});
        if (s != STATUS_OK) return s;
        s = alloc.CopyToDevice(weights.vis_patch_embed.weight,
                               pe_fp16.data(), hs * ksize * sizeof(uint16_t));
        if (s != STATUS_OK) return s;

        // Bias: (hs,)
        s = io::CopyWeightToFp16NPU(loader, "model.visual.patch_embed.proj.bias",
                                alloc, weights.vis_patch_embed.bias);
        if (s != STATUS_OK) return s;
    }

    // ── Vision position embedding ─────────────────────────
    s = io::CopyWeightToFp16NPU(loader, "model.visual.pos_embed.weight",
                            alloc, weights.vis_pos_embed);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to load vision pos_embed weight");
        return s;
    }

    // ── Main merger ───────────────────────────────────────
    s = LoadMergerWeights(loader, alloc, "model.visual.merger.", weights.merger);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to load main merger weights");
        return s;
    }

    // ── Deepstack mergers ─────────────────────────────────
    weights.deepstack_mergers.resize(config.vis_deepstack_visual_indexes.size());
    for (size_t i = 0; i < config.vis_deepstack_visual_indexes.size(); i++) {
        std::string ds_pfx = "model.visual.deepstack_merger_list." + std::to_string(i) + ".";
        s = LoadMergerWeights(loader, alloc, ds_pfx, weights.deepstack_mergers[i]);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to load deepstack merger %zu weights", i);
            return s;
        }
    }

    LOG_INFO("All Qwen3VL weights loaded successfully");
    return STATUS_OK;
}

}  // namespace adapters
}  // namespace atb_llm
