#pragma once
#include "atb_llm/types.h"
#include "io/json_config.h"
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace adapters {

/// Qwen3VL Embedding model configuration.
/// Parsed from config.json and preprocessor_config.json.
/// Equivalent to Python engine_utils.py:load_config.
struct Qwen3VLConfig {
    // ── Top-level ──────────────────────────────────────────
    int64_t image_token_id = 151655;

    // ── Text config ────────────────────────────────────────
    int32_t text_hidden_size = 2048;
    int32_t text_num_heads = 16;
    int32_t text_num_kv_heads = 8;
    int32_t text_head_dim = 128;
    int32_t text_intermediate_size = 6144;
    int32_t text_num_layers = 28;
    float text_rms_norm_eps = 1e-6f;
    float text_rope_theta = 5000000.0f;
    int64_t text_vocab_size = 151936;
    std::vector<int32_t> text_mrope_section = {24, 20, 20};

    // ── Vision config ──────────────────────────────────────
    int32_t vis_hidden_size = 1024;
    int32_t vis_num_heads = 16;
    int32_t vis_intermediate_size = 4096;
    int32_t vis_depth = 24;
    int32_t vis_in_channels = 3;
    int32_t vis_temporal_patch_size = 2;
    int32_t vis_patch_size = 16;
    int32_t vis_spatial_merge_size = 2;
    int32_t vis_num_position_embeddings = 2304;
    int32_t vis_out_hidden_size = 2048;
    std::vector<int32_t> vis_deepstack_visual_indexes = {5, 11, 17};
    float vis_epsilon = 1e-6f;

    // ── Embedding output ─────────────────────────────────
    bool normalize = true;  // L2-normalize output embedding (Python default)

    // ── Preprocessor config ────────────────────────────────
    int32_t pp_patch_size = 16;
    int32_t pp_temporal_patch_size = 2;
    int32_t pp_merge_size = 2;
    int32_t pp_min_pixels = 4096;
    int32_t pp_max_pixels = 1310720;
    std::vector<float> pp_image_mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> pp_image_std = {0.5f, 0.5f, 0.5f};

    // ── Derived ────────────────────────────────────────────
    int32_t vis_head_dim() const { return vis_hidden_size / vis_num_heads; }
    int32_t num_grid() const {
        return static_cast<int32_t>(std::sqrt(vis_num_position_embeddings));
    }
};

/// Load Qwen3VL config from model directory.
/// Reads config.json and preprocessor_config.json.
Status LoadQwen3VLConfig(const std::string& model_dir, Qwen3VLConfig& config);

} // namespace adapters
} // namespace atb_llm
