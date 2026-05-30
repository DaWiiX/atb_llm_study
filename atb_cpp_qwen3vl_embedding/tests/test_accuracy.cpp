/**
 * C++ Engine Accuracy Test — covers all input modes.
 *
 * Tests:
 *   1. TEXT_ONLY: pure text inference
 *   2. IMAGE_ONLY: pure image inference (preprocessed)
 *   3. IMAGE_AND_TEXT: image + text inference (preprocessed)
 *
 * Each test saves output to /tmp/cpp_*.bin for comparison with Python.
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

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

static const char* MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B";

// ── Helper: create test image ──────────────────────────────────────
// Creates a simple gradient image for testing
static std::vector<uint8_t> CreateTestImage(int32_t channels, int32_t height, int32_t width) {
    std::vector<uint8_t> image(channels * height * width);
    for (int32_t c = 0; c < channels; c++) {
        for (int32_t h = 0; h < height; h++) {
            for (int32_t w = 0; w < width; w++) {
                // Simple gradient pattern
                uint8_t value = static_cast<uint8_t>((h * 255 / height + w * 255 / width + c * 85) % 256);
                image[c * height * width + h * width + w] = value;
            }
        }
    }
    return image;
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

    // ── 1. Create engine ─────────────────────────────────────
    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 15LL * 1024 * 1024 * 1024;  // 15GB (increased for vision tests)
    config.device_id = 0;
    config.normalize = true;  // L2-normalize output

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
    // Test 1: TEXT_ONLY
    // ═══════════════════════════════════════════════════════════════
    {
        LOG_INFO("\n=== Test 1: TEXT_ONLY ===");
        tests_total++;

        // Fixed input: token IDs [151643, 15339, 1879] (seq_len=3)
        int64_t input_ids[] = {151643, 15339, 1879};
        int64_t seq_len = 3;

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
                LOG_INFO("Saved to /tmp/cpp_text_only.bin");
                tests_passed++;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 2: PREPROCESSED (Image-Only)
    // ═══════════════════════════════════════════════════════════════
    {
        LOG_INFO("\n=== Test 2: PREPROCESSED (Image-Only) ===");
        tests_total++;

        // Create a 64x64 test image
        int32_t img_h = 64, img_w = 64, img_c = 3;
        std::vector<uint8_t> image = CreateTestImage(img_c, img_h, img_w);

        // Preprocess image
        int32_t factor = pp_config.pp_patch_size * pp_config.pp_merge_size;
        int32_t new_h, new_w;
        atb_llm::adapters::SmartResize(img_h, img_w, factor,
                                       pp_config.pp_min_pixels, pp_config.pp_max_pixels,
                                       new_h, new_w);

        int32_t grid_h = new_h / pp_config.pp_patch_size;
        int32_t grid_w = new_w / pp_config.pp_patch_size;
        int32_t grid_t = 1;
        int64_t num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;
        int64_t patch_dim = static_cast<int64_t>(img_c) * pp_config.pp_temporal_patch_size *
                            pp_config.pp_patch_size * pp_config.pp_patch_size;

        std::vector<uint16_t> pixel_values(num_patches * patch_dim, 0);
        int64_t out_num_patches = 0;
        int64_t grid_thw[3] = {};

        s = atb_llm::adapters::PreprocessImage(
            image.data(), img_c, img_h, img_w, pp_config,
            pixel_values.data(), out_num_patches, grid_thw);

        if (!IS_OK(s)) {
            LOG_ERROR("PreprocessImage failed: %d", static_cast<int>(s));
        } else {
            LOG_INFO("Preprocessed: num_patches=%ld, grid_thw=[%ld, %ld, %ld]",
                     static_cast<long>(out_num_patches),
                     static_cast<long>(grid_thw[0]),
                     static_cast<long>(grid_thw[1]),
                     static_cast<long>(grid_thw[2]));

            // For image-only, we need input_ids with image tokens
            // Vision merger outputs merged_tokens = num_patches / (merge_size^2)
            int64_t merge_size = pp_config.pp_merge_size;
            int64_t merged_tokens = out_num_patches / (merge_size * merge_size);
            int64_t image_token_id = 151655;  // from config
            std::vector<int64_t> input_ids(merged_tokens, image_token_id);

            atb_llm::InferRequest request;
            request.mode = atb_llm::InputMode::PREPROCESSED;
            request.text.input_ids = input_ids.data();
            request.text.batch_size = 1;
            request.text.seq_length = merged_tokens;
            request.preprocessed.pixel_values = pixel_values.data();
            request.preprocessed.num_patches = out_num_patches;
            request.preprocessed.grid_thw = grid_thw;
            request.preprocessed.dtype = ACL_FLOAT16;

            atb_llm::InferResult result;
            s = engine->Encode(request, result);
            if (!IS_OK(s)) {
                LOG_ERROR("IMAGE_ONLY Encode failed: %d", static_cast<int>(s));
            } else {
                PrintFirstN("First 8 values", result, 8);
                if (SaveResult("/tmp/cpp_image_only.bin", result)) {
                    LOG_INFO("Saved to /tmp/cpp_image_only.bin");
                    tests_passed++;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Test 3: PREPROCESSED (Image + Text)
    // ═══════════════════════════════════════════════════════════════
    {
        LOG_INFO("\n=== Test 3: PREPROCESSED (Image + Text) ===");
        tests_total++;

        // Create a 32x32 test image
        int32_t img_h = 32, img_w = 32, img_c = 3;
        std::vector<uint8_t> image = CreateTestImage(img_c, img_h, img_w);

        // Preprocess image
        int32_t factor = pp_config.pp_patch_size * pp_config.pp_merge_size;
        int32_t new_h, new_w;
        atb_llm::adapters::SmartResize(img_h, img_w, factor,
                                       pp_config.pp_min_pixels, pp_config.pp_max_pixels,
                                       new_h, new_w);

        int32_t grid_h = new_h / pp_config.pp_patch_size;
        int32_t grid_w = new_w / pp_config.pp_patch_size;
        int32_t grid_t = 1;
        int64_t num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;
        int64_t patch_dim = static_cast<int64_t>(img_c) * pp_config.pp_temporal_patch_size *
                            pp_config.pp_patch_size * pp_config.pp_patch_size;

        std::vector<uint16_t> pixel_values(num_patches * patch_dim, 0);
        int64_t out_num_patches = 0;
        int64_t grid_thw[3] = {};

        s = atb_llm::adapters::PreprocessImage(
            image.data(), img_c, img_h, img_w, pp_config,
            pixel_values.data(), out_num_patches, grid_thw);

        if (!IS_OK(s)) {
            LOG_ERROR("PreprocessImage failed: %d", static_cast<int>(s));
        } else {
            LOG_INFO("Preprocessed: num_patches=%ld, grid_thw=[%ld, %ld, %ld]",
                     static_cast<long>(out_num_patches),
                     static_cast<long>(grid_thw[0]),
                     static_cast<long>(grid_thw[1]),
                     static_cast<long>(grid_thw[2]));

            // Create input_ids: text + image + text
            // Vision merger outputs merged_tokens = num_patches / (merge_size^2)
            int64_t merge_size = pp_config.pp_merge_size;
            int64_t merged_tokens = out_num_patches / (merge_size * merge_size);
            int64_t image_token_id = 151655;  // from config
            // Text tokens: [151643, 15339, 1879] (same as text-only test)
            std::vector<int64_t> input_ids;
            // Add some text tokens before image
            input_ids.push_back(151643);
            // Add image tokens (merged tokens, not patches)
            for (int64_t i = 0; i < merged_tokens; i++) {
                input_ids.push_back(image_token_id);
            }
            // Add more text tokens after image
            input_ids.push_back(15339);
            input_ids.push_back(1879);

            int64_t seq_len = input_ids.size();

            atb_llm::InferRequest request;
            request.mode = atb_llm::InputMode::PREPROCESSED;
            request.text.input_ids = input_ids.data();
            request.text.batch_size = 1;
            request.text.seq_length = seq_len;
            request.preprocessed.pixel_values = pixel_values.data();
            request.preprocessed.num_patches = out_num_patches;
            request.preprocessed.grid_thw = grid_thw;
            request.preprocessed.dtype = ACL_FLOAT16;

            atb_llm::InferResult result;
            s = engine->Encode(request, result);
            if (!IS_OK(s)) {
                LOG_ERROR("IMAGE_AND_TEXT Encode failed: %d", static_cast<int>(s));
            } else {
                PrintFirstN("First 8 values", result, 8);
                if (SaveResult("/tmp/cpp_image_text.bin", result)) {
                    LOG_INFO("Saved to /tmp/cpp_image_text.bin");
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
