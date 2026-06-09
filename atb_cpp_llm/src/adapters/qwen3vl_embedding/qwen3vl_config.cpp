#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "log/logger.h"
#include <cmath>

namespace atb_llm {
namespace adapters {

Status LoadQwen3VLConfig(const std::string& model_dir, Qwen3VLConfig& config) {
    // ── Load config.json ──────────────────────────────────
    std::string config_path = model_dir + "/config.json";
    auto cfg = JsonConfig::Load(config_path);
    if (!cfg.IsValid()) {
        LOG_ERROR("Failed to load config.json from %s", config_path.c_str());
        return ERROR_FILE_NOT_FOUND;
    }

    // Top-level
    config.image_token_id = cfg.GetInt("image_token_id", 151655);
    config.vision_start_token_id = cfg.GetInt("vision_start_token_id", 151652);

    // Text config
    auto text_cfg = cfg.GetSubConfig("text_config");
    if (text_cfg.IsValid()) {
        config.text_hidden_size = text_cfg.GetInt("hidden_size", 2048);
        config.text_num_heads = text_cfg.GetInt("num_attention_heads", 16);
        config.text_num_kv_heads = text_cfg.GetInt("num_key_value_heads", 8);
        config.text_head_dim = text_cfg.GetInt("head_dim", 128);
        config.text_intermediate_size = text_cfg.GetInt("intermediate_size", 6144);
        config.text_num_layers = text_cfg.GetInt("num_hidden_layers", 28);
        config.text_rms_norm_eps = text_cfg.GetFloat("rms_norm_eps", 1e-6f);
        config.text_rope_theta = text_cfg.GetFloat("rope_theta", 5000000.0f);
        config.text_vocab_size = text_cfg.GetInt("vocab_size", 151936);

        auto rope_scaling = text_cfg.GetSubConfig("rope_scaling");
        if (rope_scaling.IsValid()) {
            config.text_mrope_section = rope_scaling.GetIntArray("mrope_section");
        }
    }

    // Vision config
    auto vis_cfg = cfg.GetSubConfig("vision_config");
    if (vis_cfg.IsValid()) {
        config.vis_hidden_size = vis_cfg.GetInt("hidden_size", 1024);
        config.vis_num_heads = vis_cfg.GetInt("num_heads", 16);
        config.vis_intermediate_size = vis_cfg.GetInt("intermediate_size", 4096);
        config.vis_depth = vis_cfg.GetInt("depth", 24);
        config.vis_in_channels = vis_cfg.GetInt("in_channels", 3);
        config.vis_temporal_patch_size = vis_cfg.GetInt("temporal_patch_size", 2);
        config.vis_patch_size = vis_cfg.GetInt("patch_size", 16);
        config.vis_spatial_merge_size = vis_cfg.GetInt("spatial_merge_size", 2);
        config.vis_num_position_embeddings = vis_cfg.GetInt("num_position_embeddings", 2304);
        config.vis_out_hidden_size = vis_cfg.GetInt("out_hidden_size", 2048);
        config.vis_deepstack_visual_indexes = vis_cfg.GetIntArray("deepstack_visual_indexes");
        config.vis_epsilon = vis_cfg.GetFloat("layer_norm_eps", 1e-6f);
    }

    // ── Load preprocessor_config.json ─────────────────────
    std::string pp_path = model_dir + "/preprocessor_config.json";
    auto pp = JsonConfig::Load(pp_path);
    if (pp.IsValid()) {
        config.pp_patch_size = pp.GetInt("patch_size", 16);
        config.pp_temporal_patch_size = pp.GetInt("temporal_patch_size", 2);
        config.pp_merge_size = pp.GetInt("merge_size", 2);
        config.pp_min_pixels = pp.GetInt("min_pixels", 4096);
        config.pp_max_pixels = pp.GetInt("max_pixels", 1310720);
        config.pp_image_mean = pp.GetFloatArray("image_mean");
        config.pp_image_std = pp.GetFloatArray("image_std");
        if (config.pp_image_mean.empty()) config.pp_image_mean = {0.5f, 0.5f, 0.5f};
        if (config.pp_image_std.empty()) config.pp_image_std = {0.5f, 0.5f, 0.5f};
    } else {
        LOG_WARN("preprocessor_config.json not found, using defaults");
    }

    LOG_INFO("Qwen3VL config loaded: text_hs=%d nh=%d kv=%d hd=%d layers=%d, "
             "vis_hs=%d nh=%d depth=%d, image_token=%ld",
             config.text_hidden_size, config.text_num_heads, config.text_num_kv_heads,
             config.text_head_dim, config.text_num_layers,
             config.vis_hidden_size, config.vis_num_heads, config.vis_depth,
             static_cast<long>(config.image_token_id));

    return STATUS_OK;
}

} // namespace adapters
} // namespace atb_llm
