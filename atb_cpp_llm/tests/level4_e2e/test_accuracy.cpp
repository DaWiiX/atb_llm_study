/**
 * C++ Engine Accuracy Test — covers all input modes.
 *
 * Uses identical inputs as Python test_accuracy.py:
 *   - Image: 720x1280 gradient pattern (triggers resize path)
 *   - Text: "Describe the image." → tokens [74785, 279, 2168, 13]
 *
 * Tests:
 *   1. TEXT_ONLY: "Describe the image."
 *   2. IMAGE_ONLY: 720x1280 image only
 *   3. IMAGE_AND_TEXT: "Describe" + [image] + " the image."
 *
 * All modes must achieve cosine similarity ≥ 0.99 vs Python.
 *
 * Run: ./test_accuracy
 * Then: python tests/test_accuracy.py
 */

#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#include "test_env.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

static const std::string MODEL_DIR = GetModelDir();

// ── Shared test constants (must match test_accuracy.py exactly) ────
static constexpr int32_t IMG_H = 720;       // triggers resize: 720→704 (round 22.5→22, banker's)
static constexpr int32_t IMG_W = 1280;      // 1280 is already divisible by 32
static constexpr int32_t IMG_C = 3;

// "Describe the image." token IDs (from Qwen3-VL tokenizer)
static constexpr int64_t TOK_DESCRIBE  = 74785;
static constexpr int64_t TOK_THE       = 279;
static constexpr int64_t TOK_IMAGE     = 2168;
static constexpr int64_t TOK_DOT       = 13;

static constexpr int64_t IMAGE_TOKEN_ID = 151655;  // from config image_token_id

// ── Helper: create test image (must match Python create_test_image) ──
static std::vector<uint8_t> CreateTestImage(int32_t channels, int32_t height, int32_t width) {
    std::vector<uint8_t> image(channels * height * width);
    for (int32_t c = 0; c < channels; c++) {
        for (int32_t h = 0; h < height; h++) {
            for (int32_t w = 0; w < width; w++) {
                uint8_t value = static_cast<uint8_t>((h * 255 / height + w * 255 / width + c * 85) % 256);
                image[c * height * width + h * width + w] = value;
            }
        }
    }
    return image;
}

// ── Helper: preprocess image (shared between IMAGE_ONLY and IMAGE_AND_TEXT) ──
struct PreprocessedImage {
    std::vector<uint16_t> pixel_values;
    int64_t num_patches;
    int64_t grid_thw[3];
    int64_t merged_tokens;   // = num_patches / merge_size^2
};

static bool PreprocessTestImage(const atb_llm::adapters::Qwen3VLConfig& config,
                                 PreprocessedImage& out) {
    std::vector<uint8_t> image = CreateTestImage(IMG_C, IMG_H, IMG_W);

    int32_t factor = config.pp_patch_size * config.pp_merge_size;
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(IMG_H, IMG_W, factor,
                                   config.pp_min_pixels, config.pp_max_pixels,
                                   new_h, new_w);

    int32_t grid_h = new_h / config.pp_patch_size;
    int32_t grid_w = new_w / config.pp_patch_size;
    int32_t grid_t = 1;
    int64_t num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;
    int64_t patch_dim = static_cast<int64_t>(IMG_C) * config.pp_temporal_patch_size *
                        config.pp_patch_size * config.pp_patch_size;

    out.pixel_values.resize(num_patches * patch_dim, 0);
    out.num_patches = 0;
    out.grid_thw[0] = out.grid_thw[1] = out.grid_thw[2] = 0;

    atb_llm::Status s = atb_llm::adapters::PreprocessImage(
        image.data(), IMG_C, IMG_H, IMG_W, config,
        out.pixel_values.data(), out.num_patches, out.grid_thw);

    if (!IS_OK(s)) {
        LOG_ERROR("PreprocessImage failed: %d", static_cast<int>(s));
        return false;
    }

    int64_t merge_size = config.pp_merge_size;
    out.merged_tokens = out.num_patches / (merge_size * merge_size);

    LOG_INFO("Preprocessed %dx%d → %dx%d, grid=[%ld,%ld,%ld], patches=%ld, merged=%ld",
             IMG_H, IMG_W, new_h, new_w,
             static_cast<long>(out.grid_thw[0]), static_cast<long>(out.grid_thw[1]),
             static_cast<long>(out.grid_thw[2]),
             static_cast<long>(out.num_patches), static_cast<long>(out.merged_tokens));

    // Save pixel_values for comparison with Python
    {
        int64_t total = out.num_patches * patch_dim;
        FILE* f = fopen("/tmp/cpp_pixel_values.bin", "wb");
        if (f) {
            fwrite(&total, sizeof(int64_t), 1, f);
            fwrite(out.pixel_values.data(), sizeof(uint16_t), total, f);
            fclose(f);
            LOG_INFO("Saved /tmp/cpp_pixel_values.bin (%ld fp16 values)", static_cast<long>(total));
        }
    }

    return true;
}

// ── Helper: save result to binary file ─────────────────────────────
static bool SaveResult(const char* path, const atb_llm::InferResult& result) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        LOG_ERROR("Failed to open output file: %s", path);
        return false;
    }

    int64_t dim = result.shape[0];
    fwrite(&dim, sizeof(int64_t), 1, f);

    if (result.dtype == ACL_FLOAT16) {
        const uint16_t* fp16_data = result.As<uint16_t>();
        std::vector<float> fp32_data(dim);
        for (int64_t i = 0; i < dim; i++) {
            fp32_data[i] = atb_llm::Fp16ToF32(fp16_data[i]);
        }
        fwrite(fp32_data.data(), sizeof(float), dim, f);
    } else if (result.dtype == ACL_FLOAT) {
        fwrite(result.As<float>(), sizeof(float), dim, f);
    } else {
        LOG_ERROR("Unexpected dtype: %d", static_cast<int>(result.dtype));
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// ── Helper: print first N values ───────────────────────────────────
static void PrintFirstN(const char* label, const atb_llm::InferResult& result, int n) {
    LOG_INFO("%s:", label);
    if (result.dtype == ACL_FLOAT16) {
        const uint16_t* data = result.As<uint16_t>();
        for (int i = 0; i < n && i < static_cast<int>(result.shape[0]); i++) {
            fprintf(stderr, "  [%d] = %.6f (0x%04x)\n", i, atb_llm::Fp16ToF32(data[i]), data[i]);
        }
    } else if (result.dtype == ACL_FLOAT) {
        const float* data = result.As<float>();
        for (int i = 0; i < n && i < static_cast<int>(result.shape[0]); i++) {
            fprintf(stderr, "  [%d] = %.6f\n", i, data[i]);
        }
    }
}

int main(int argc, char** argv) {
    LOG_INFO("=== C++ Engine Accuracy Test ===");
    LOG_INFO("Image: %dx%d, Text: 'Describe the image.'", IMG_H, IMG_W);

    // ── 1. Create engine ─────────────────────────────────────
    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 15LL * 1024 * 1024 * 1024;  // 15GB
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    if (!IS_OK(s) || !engine) {
        LOG_ERROR("Failed to create engine: %d", static_cast<int>(s));
        return 1;
    }
    LOG_INFO("Engine created successfully");

    // Load config for preprocessing
    atb_llm::adapters::Qwen3VLConfig pp_config;

    int tests_passed = 0;
    int tests_total = 0;

    // ═══════════════════════════════════════════════════════════════
    // Test 1: TEXT_ONLY — "Describe the image."
    // ═══════════════════════════════════════════════════════════════
    {
        LOG_INFO("\n=== Test 1: TEXT_ONLY ===");
        tests_total++;

        int64_t input_ids[] = {TOK_DESCRIBE, TOK_THE, TOK_IMAGE, TOK_DOT};
        int64_t seq_len = 4;

        atb_llm::InferRequest request;
        request.mode = atb_llm::InputMode::TEXT_ONLY;
        request.text.input_ids = input_ids;
        request.text.batch_size = 1;
        request.text.seq_length = seq_len;

        atb_llm::InferResult result;
        s = engine->Encode(request, result);
        if (!IS_OK(s)) {
            LOG_ERROR("TEXT_ONLY Encode failed: %d", static_cast<int>(s));
        } else {
            PrintFirstN("First 8 values", result, 8);
            if (SaveResult("/tmp/cpp_text_only.bin", result)) {
                LOG_INFO("Saved to /tmp/cpp_text_only.bin (dim=%ld)", static_cast<long>(result.shape[0]));
                tests_passed++;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 2: IMAGE_ONLY — 720x1280 image
    // ═══════════════════════════════════════════════════════════════
    {
        LOG_INFO("\n=== Test 2: IMAGE_ONLY ===");
        tests_total++;

        PreprocessedImage img;
        if (!PreprocessTestImage(pp_config, img)) {
            LOG_ERROR("Image preprocessing failed");
        } else {
            std::vector<int64_t> input_ids(img.merged_tokens, IMAGE_TOKEN_ID);

            atb_llm::InferRequest request;
            request.mode = atb_llm::InputMode::PREPROCESSED;
            request.text.input_ids = input_ids.data();
            request.text.batch_size = 1;
            request.text.seq_length = img.merged_tokens;
            request.preprocessed.pixel_values = img.pixel_values.data();
            request.preprocessed.num_patches = img.num_patches;
            request.preprocessed.grid_thw = img.grid_thw;
            request.preprocessed.dtype = ACL_FLOAT16;

            atb_llm::InferResult result;
            s = engine->Encode(request, result);
            if (!IS_OK(s)) {
                LOG_ERROR("IMAGE_ONLY Encode failed: %d", static_cast<int>(s));
            } else {
                PrintFirstN("First 8 values", result, 8);
                if (SaveResult("/tmp/cpp_image_only.bin", result)) {
                    LOG_INFO("Saved to /tmp/cpp_image_only.bin (dim=%ld)", static_cast<long>(result.shape[0]));
                    tests_passed++;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 3: IMAGE_AND_TEXT — "Describe" + [image] + " the image."
    // ═══════════════════════════════════════════════════════════════
    {
        LOG_INFO("\n=== Test 3: IMAGE_AND_TEXT ===");
        tests_total++;

        PreprocessedImage img;
        if (!PreprocessTestImage(pp_config, img)) {
            LOG_ERROR("Image preprocessing failed");
        } else {
            // input_ids: [Describe] + [img_tok]*merged + [the, image, .]
            std::vector<int64_t> input_ids;
            input_ids.push_back(TOK_DESCRIBE);
            for (int64_t i = 0; i < img.merged_tokens; i++) {
                input_ids.push_back(IMAGE_TOKEN_ID);
            }
            input_ids.push_back(TOK_THE);
            input_ids.push_back(TOK_IMAGE);
            input_ids.push_back(TOK_DOT);

            int64_t seq_len = static_cast<int64_t>(input_ids.size());
            LOG_INFO("input_ids length: %ld (text=1 + img=%ld + text=3)",
                     static_cast<long>(seq_len), static_cast<long>(img.merged_tokens));

            atb_llm::InferRequest request;
            request.mode = atb_llm::InputMode::PREPROCESSED;
            request.text.input_ids = input_ids.data();
            request.text.batch_size = 1;
            request.text.seq_length = seq_len;
            request.preprocessed.pixel_values = img.pixel_values.data();
            request.preprocessed.num_patches = img.num_patches;
            request.preprocessed.grid_thw = img.grid_thw;
            request.preprocessed.dtype = ACL_FLOAT16;

            atb_llm::InferResult result;
            s = engine->Encode(request, result);
            if (!IS_OK(s)) {
                LOG_ERROR("IMAGE_AND_TEXT Encode failed: %d", static_cast<int>(s));
            } else {
                PrintFirstN("First 8 values", result, 8);
                if (SaveResult("/tmp/cpp_image_text.bin", result)) {
                    LOG_INFO("Saved to /tmp/cpp_image_text.bin (dim=%ld)", static_cast<long>(result.shape[0]));
                    tests_passed++;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    LOG_INFO("\n=== Summary ===");
    LOG_INFO("Tests passed: %d/%d", tests_passed, tests_total);
    LOG_INFO("Run: python tests/test_accuracy.py");

    return (tests_passed == tests_total) ? 0 : 1;
}
