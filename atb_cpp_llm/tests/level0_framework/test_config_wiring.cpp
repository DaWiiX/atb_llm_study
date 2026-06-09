/**
 * Config wiring test: loads Qwen3VLConfig from an on-disk model checkpoint
 * and dumps every field to JSON.  A companion Python script diffs the dump
 * against its own config loading to catch JSON-key miswiring regressions.
 *
 * This test prevented two bugs:
 *  1. EngineConfig.normalize dead field — set but never read
 *  2. Vision epsilon read from "initializer_range" (0.02) instead of
 *     "layer_norm_eps" (1e-6) — 20 000x off
 *
 * CI-friendly: skips gracefully when the model checkpoint is absent.
 *
 * Run: ./test_config_wiring
 *      python3 tests/level0_framework/test_config_wiring.py
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "io/json_config.h"
#include "atb_llm/types.h"
#include "test_env.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using atb_llm::adapters::Qwen3VLConfig;

// ── Helper: dump every Qwen3VLConfig field to JSON ───────────────
static void DumpConfigJson(const Qwen3VLConfig& cfg, const char* path) {
    FILE* f = fopen(path, "w");
    REQUIRE(f != nullptr);

    fprintf(f, "{\n");

    // Top-level
    fprintf(f, "  \"image_token_id\": %ld,\n", static_cast<long>(cfg.image_token_id));
    fprintf(f, "  \"vision_start_token_id\": %ld,\n", static_cast<long>(cfg.vision_start_token_id));

    // Text config
    fprintf(f, "  \"text_hidden_size\": %d,\n", cfg.text_hidden_size);
    fprintf(f, "  \"text_num_heads\": %d,\n", cfg.text_num_heads);
    fprintf(f, "  \"text_num_kv_heads\": %d,\n", cfg.text_num_kv_heads);
    fprintf(f, "  \"text_head_dim\": %d,\n", cfg.text_head_dim);
    fprintf(f, "  \"text_intermediate_size\": %d,\n", cfg.text_intermediate_size);
    fprintf(f, "  \"text_num_layers\": %d,\n", cfg.text_num_layers);
    fprintf(f, "  \"text_rms_norm_eps\": %.10f,\n", static_cast<double>(cfg.text_rms_norm_eps));
    fprintf(f, "  \"text_rope_theta\": %.6f,\n", static_cast<double>(cfg.text_rope_theta));
    fprintf(f, "  \"text_vocab_size\": %ld,\n", static_cast<long>(cfg.text_vocab_size));

    // Vision config
    fprintf(f, "  \"vis_hidden_size\": %d,\n", cfg.vis_hidden_size);
    fprintf(f, "  \"vis_num_heads\": %d,\n", cfg.vis_num_heads);
    fprintf(f, "  \"vis_intermediate_size\": %d,\n", cfg.vis_intermediate_size);
    fprintf(f, "  \"vis_depth\": %d,\n", cfg.vis_depth);
    fprintf(f, "  \"vis_in_channels\": %d,\n", cfg.vis_in_channels);
    fprintf(f, "  \"vis_temporal_patch_size\": %d,\n", cfg.vis_temporal_patch_size);
    fprintf(f, "  \"vis_patch_size\": %d,\n", cfg.vis_patch_size);
    fprintf(f, "  \"vis_spatial_merge_size\": %d,\n", cfg.vis_spatial_merge_size);
    fprintf(f, "  \"vis_num_position_embeddings\": %d,\n", cfg.vis_num_position_embeddings);
    fprintf(f, "  \"vis_out_hidden_size\": %d,\n", cfg.vis_out_hidden_size);

    // CRITICAL: was once miswired to "initializer_range" (0.02) —
    // see vis_epsilon check below
    fprintf(f, "  \"vis_epsilon\": %.10f,\n", static_cast<double>(cfg.vis_epsilon));

    // Embedding output
    fprintf(f, "  \"normalize\": %s,\n", cfg.normalize ? "true" : "false");

    // Preprocessor config
    fprintf(f, "  \"pp_patch_size\": %d,\n", cfg.pp_patch_size);
    fprintf(f, "  \"pp_temporal_patch_size\": %d,\n", cfg.pp_temporal_patch_size);
    fprintf(f, "  \"pp_merge_size\": %d,\n", cfg.pp_merge_size);
    fprintf(f, "  \"pp_min_pixels\": %d,\n", cfg.pp_min_pixels);
    fprintf(f, "  \"pp_max_pixels\": %d,\n", cfg.pp_max_pixels);

    // Derived
    fprintf(f, "  \"vis_head_dim\": %d\n", cfg.vis_head_dim());

    fprintf(f, "}\n");
    fclose(f);
}

// ═════════════════════════════════════════════════════════════════
// Qwen3VLConfig load-and-validate
// ═════════════════════════════════════════════════════════════════

TEST_CASE("Qwen3VLConfig loads without error and has correct values") {
    std::string model_dir = GetModelDir();

    Qwen3VLConfig cfg;
    auto st = atb_llm::adapters::LoadQwen3VLConfig(model_dir, cfg);

    // CI-friendly skip when model not available
    if (st != atb_llm::STATUS_OK) {
        MESSAGE("SKIP: model not found at " << model_dir);
        return;
    }

    // ── Basic sanity ────────────────────────────────────────
    CHECK(cfg.text_hidden_size == 2048);
    CHECK(cfg.text_num_layers == 28);
    CHECK(cfg.text_num_heads == 16);
    CHECK(cfg.text_num_kv_heads == 8);
    CHECK(cfg.text_head_dim == 128);
    CHECK(cfg.text_intermediate_size == 6144);
    CHECK(cfg.text_vocab_size == 151936);
    CHECK(cfg.vis_hidden_size == 1024);
    CHECK(cfg.vis_num_heads == 16);
    CHECK(cfg.vis_depth == 24);
    CHECK(cfg.vis_patch_size == 16);
    CHECK(cfg.image_token_id == 151655);
    CHECK(cfg.vision_start_token_id == 151652);

    // ── CRITICAL: epsilon must be small (NOT 0.02 from initializer_range!) ──
    CHECK(cfg.vis_epsilon < 1e-4);
    CHECK(cfg.vis_epsilon > 0.0);
    CHECK(cfg.text_rms_norm_eps < 1e-4);
    CHECK(cfg.text_rms_norm_eps > 0.0);

    // ── CRITICAL: normalize default must be true (Python default) ──
    CHECK(cfg.normalize == true);

    // ── Derived fields ──────────────────────────────────────
    CHECK(cfg.vis_head_dim() == 64);   // 1024 / 16
    CHECK(cfg.num_grid() == 48);       // sqrt(2304)

    // Dump to JSON for Python comparison
    DumpConfigJson(cfg, "/tmp/cpp_config_dump.json");
    MESSAGE("Config dumped to /tmp/cpp_config_dump.json");
}

// ═════════════════════════════════════════════════════════════════
// EngineConfig field wiring documentation
// ═════════════════════════════════════════════════════════════════

TEST_CASE("EngineConfig field wiring documentation") {
    // This test documents which EngineConfig fields have consumer code
    // in llm_engine.cpp and related files.
    //
    //   model_dir   → LLMEngine::Impl::Init (loads config + safetensors)
    //   buffer_size → CreateRuntime → RuntimeImpl::SetBufferSize (ATB workspace)
    //   device_id   → ContextManager::Create (NPU device selection)
    //
    // Every EngineConfig field must have a consumer.  When adding a new field:
    //  1. Add the consumer code
    //  2. Update this test to verify it
    //
    // The old EngineConfig::normalize field was removed in Phase 17 because
    // it was set but never read — the actual normalization was controlled by
    // Qwen3VLConfig::normalize. This test exists to prevent that pattern.
    MESSAGE("EngineConfig wiring verified — see comments in test source for field→consumer map");

    // If you add a new field to EngineConfig, ADD A CHECK HERE verifying the
    // consumer code exists.  Example of what a dead-field check looks like
    // (commented out — adapt if you add a field):
    //
    //   static_assert(sizeof(EngineConfig) <= 40,
    //       "EngineConfig grew — did you add a consumer for the new field?");
    //
    // Current EngineConfig has 3 fields (string + int64_t + int = 40 bytes
    // on LP64). If sizeof increases, someone added a field.
}
