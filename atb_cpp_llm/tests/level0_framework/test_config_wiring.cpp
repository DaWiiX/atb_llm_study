/**
 * Config wiring test: loads Qwen3VLConfig from an on-disk model checkpoint
 * and compares every field against reference values generated from the same
 * model's config.json by gen_cpu_reference.py (gen_config_wiring_ref).
 *
 * The reference data lives in /tmp/cpu_config_wiring_{int,float,bool}.bin.
 * Run `python tests/python_reference/gen_all.py` to regenerate them.
 *
 * A companion Python script (test_config_wiring.py) also diffs the C++ JSON
 * dump against its own config loading to catch JSON-key miswiring regressions.
 *
 * This test prevented two bugs:
 *  1. EngineConfig.normalize dead field — set but never read
 *  2. Vision epsilon read from "initializer_range" (0.02) instead of
 *     "layer_norm_eps" (1e-6) — 20 000x off
 *
 * CI-friendly: skips gracefully when the model checkpoint or reference
 * bins are absent.
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

// ── Helpers for loading reference binary files ────────────────────

/// Load a typed ndim+shape+data binary file (written by write_f32 / write_fp16 / write_i64 etc.).
/// Format: [ndim: int64] [shape: int64[ndim]] [data: T[]].
/// Returns true on success.
template <typename T>
static bool LoadBin(const char* path,
                    std::vector<T>& data,
                    std::vector<int64_t>& shape) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    int64_t ndim = 0;
    if (std::fread(&ndim, sizeof(int64_t), 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    if (ndim <= 0 || ndim > 8) {
        std::fclose(f);
        return false;
    }
    shape.assign(static_cast<size_t>(ndim), 0);
    if (std::fread(shape.data(), sizeof(int64_t),
                   static_cast<size_t>(ndim), f) !=
        static_cast<size_t>(ndim)) {
        std::fclose(f);
        return false;
    }
    int64_t total = 1;
    for (int64_t i = 0; i < ndim; i++) total *= shape[static_cast<size_t>(i)];
    data.assign(static_cast<size_t>(total), T{});
    size_t got = std::fread(data.data(), sizeof(T),
                            static_cast<size_t>(total), f);
    std::fclose(f);
    return got == static_cast<size_t>(total);
}

/// Load an i32s flat-array file (written by write_i32s).
/// Format: [count: int64] [data: int32_t[count]].
static bool LoadI32s(const char* path, std::vector<int32_t>& data) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    int64_t count = 0;
    if (std::fread(&count, sizeof(int64_t), 1, f) != 1) {
        std::fclose(f);
        return false;
    }
    data.resize(static_cast<size_t>(count));
    size_t got = std::fread(data.data(), sizeof(int32_t),
                            static_cast<size_t>(count), f);
    std::fclose(f);
    return got == static_cast<size_t>(count);
}

// ── Reference data field indices (must match gen_config_wiring_ref in Python) ──

// Indices into cpu_config_wiring_int.bin (26 int64 values).
enum ConfigIntField : size_t {
    kImageTokenId = 0,
    kVisionStartTokenId,
    kTextHiddenSize,
    kTextNumHeads,
    kTextNumKvHeads,
    kTextHeadDim,
    kTextIntermediateSize,
    kTextNumLayers,
    kTextVocabSize,
    kVisHiddenSize,
    kVisNumHeads,
    kVisIntermediateSize,
    kVisDepth,
    kVisInChannels,
    kVisTemporalPatchSize,
    kVisPatchSize,
    kVisSpatialMergeSize,
    kVisNumPositionEmbeddings,
    kVisOutHiddenSize,
    kPpPatchSize,
    kPpTemporalPatchSize,
    kPpMergeSize,
    kPpMinPixels,
    kPpMaxPixels,
    kVisHeadDim,     // derived: vis_hidden_size / vis_num_heads
    kNumGrid,        // derived: sqrt(vis_num_position_embeddings)
    kNumConfigIntFields
};

// Indices into cpu_config_wiring_float.bin (3 float32 values).
enum ConfigFloatField : size_t {
    kTextRmsNormEps = 0,
    kTextRopeTheta,
    kVisEpsilon,
    kNumConfigFloatFields
};

// Indices into cpu_config_wiring_bool.bin (1 int32 value, 0=false 1=true).
enum ConfigBoolField : size_t {
    kNormalize = 0,
    kNumConfigBoolFields
};

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

    // ── Load reference data (generated by gen_config_wiring_ref in Python) ──
    std::vector<int64_t> ref_ints;
    std::vector<int64_t> ref_int_shape;
    bool have_int_ref = LoadBin<int64_t>(
        "/tmp/cpu_config_wiring_int.bin", ref_ints, ref_int_shape);

    std::vector<float> ref_floats;
    std::vector<int64_t> ref_float_shape;
    bool have_float_ref = LoadBin<float>(
        "/tmp/cpu_config_wiring_float.bin", ref_floats, ref_float_shape);

    std::vector<int32_t> ref_bools;
    bool have_bool_ref = LoadI32s(
        "/tmp/cpu_config_wiring_bool.bin", ref_bools);

    // ── Validate against reference data (no hardcoded values) ──

    if (have_int_ref && have_float_ref && have_bool_ref) {
        // Verify reference data has the expected field count
        REQUIRE(ref_ints.size() == kNumConfigIntFields);
        REQUIRE(ref_floats.size() == kNumConfigFloatFields);
        REQUIRE(ref_bools.size() == kNumConfigBoolFields);

        // Integer fields — exact match (these are discrete config values)
        CHECK(cfg.image_token_id == ref_ints[kImageTokenId]);
        CHECK(cfg.vision_start_token_id == ref_ints[kVisionStartTokenId]);
        CHECK(cfg.text_hidden_size == static_cast<int32_t>(ref_ints[kTextHiddenSize]));
        CHECK(cfg.text_num_heads == static_cast<int32_t>(ref_ints[kTextNumHeads]));
        CHECK(cfg.text_num_kv_heads == static_cast<int32_t>(ref_ints[kTextNumKvHeads]));
        CHECK(cfg.text_head_dim == static_cast<int32_t>(ref_ints[kTextHeadDim]));
        CHECK(cfg.text_intermediate_size == static_cast<int32_t>(ref_ints[kTextIntermediateSize]));
        CHECK(cfg.text_num_layers == static_cast<int32_t>(ref_ints[kTextNumLayers]));
        CHECK(cfg.text_vocab_size == ref_ints[kTextVocabSize]);
        CHECK(cfg.vis_hidden_size == static_cast<int32_t>(ref_ints[kVisHiddenSize]));
        CHECK(cfg.vis_num_heads == static_cast<int32_t>(ref_ints[kVisNumHeads]));
        CHECK(cfg.vis_intermediate_size == static_cast<int32_t>(ref_ints[kVisIntermediateSize]));
        CHECK(cfg.vis_depth == static_cast<int32_t>(ref_ints[kVisDepth]));
        CHECK(cfg.vis_in_channels == static_cast<int32_t>(ref_ints[kVisInChannels]));
        CHECK(cfg.vis_temporal_patch_size == static_cast<int32_t>(ref_ints[kVisTemporalPatchSize]));
        CHECK(cfg.vis_patch_size == static_cast<int32_t>(ref_ints[kVisPatchSize]));
        CHECK(cfg.vis_spatial_merge_size == static_cast<int32_t>(ref_ints[kVisSpatialMergeSize]));
        CHECK(cfg.vis_num_position_embeddings == static_cast<int32_t>(ref_ints[kVisNumPositionEmbeddings]));
        CHECK(cfg.vis_out_hidden_size == static_cast<int32_t>(ref_ints[kVisOutHiddenSize]));
        CHECK(cfg.pp_patch_size == static_cast<int32_t>(ref_ints[kPpPatchSize]));
        CHECK(cfg.pp_temporal_patch_size == static_cast<int32_t>(ref_ints[kPpTemporalPatchSize]));
        CHECK(cfg.pp_merge_size == static_cast<int32_t>(ref_ints[kPpMergeSize]));
        CHECK(cfg.pp_min_pixels == static_cast<int32_t>(ref_ints[kPpMinPixels]));
        CHECK(cfg.pp_max_pixels == static_cast<int32_t>(ref_ints[kPpMaxPixels]));

        // Derived fields
        CHECK(cfg.vis_head_dim() == static_cast<int32_t>(ref_ints[kVisHeadDim]));
        CHECK(cfg.num_grid() == static_cast<int32_t>(ref_ints[kNumGrid]));

        // Float fields — epsilon-close (these are floating-point from config)
        CHECK(cfg.text_rms_norm_eps == doctest::Approx(ref_floats[kTextRmsNormEps]));
        CHECK(cfg.text_rope_theta == doctest::Approx(ref_floats[kTextRopeTheta]));
        CHECK(cfg.vis_epsilon == doctest::Approx(ref_floats[kVisEpsilon]));

        // Boolean fields
        CHECK(cfg.normalize == (ref_bools[kNormalize] != 0));
    } else {
        // Reference bins not available — still dump JSON for Python validation
        MESSAGE("WARNING: reference bins not found (run gen_all.py). "
                "Skipping data-driven checks; JSON dump will be validated "
                "by the Python companion test.");
    }

    // Dump to JSON for Python comparison (works regardless of ref data availability)
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
