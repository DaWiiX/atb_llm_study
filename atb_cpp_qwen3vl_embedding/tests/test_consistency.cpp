/**
 * C++ vs Python consistency test.
 *
 * Runs text-only inference with fixed input_ids and saves the output
 * embedding as a binary file for comparison with the Python engine.
 *
 * Binary format:
 *   [int64_t] hidden_dim
 *   [float32 * hidden_dim] embedding data (converted from fp16)
 *
 * Run: ./test_consistency
 * Then: python tests/test_consistency.py
 */

#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

static const char* MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B";
static const char* OUTPUT_PATH = "/tmp/cpp_embedding.bin";

int main(int argc, char** argv) {
    LOG_INFO("=== C++ vs Python Consistency Test ===");

    // ── 1. Create engine ─────────────────────────────────────
    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;  // 10GB
    config.device_id = 0;
    config.normalize = true;  // L2-normalize output

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    if (!IS_OK(s) || !engine) {
        LOG_ERROR("Failed to create engine: %d", static_cast<int>(s));
        return 1;
    }
    LOG_INFO("Engine created successfully");

    // ── 2. Prepare input ─────────────────────────────────────
    // Fixed input: token IDs [151643, 15339, 1879] (seq_len=3)
    int64_t input_ids[] = {151643, 15339, 1879};
    int64_t seq_len = 3;

    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::TEXT_ONLY;
    request.text.input_ids = input_ids;
    request.text.batch_size = 1;
    request.text.seq_length = seq_len;

    // ── 3. Run inference ─────────────────────────────────────
    atb_llm::InferResult result;
    s = engine->Encode(request, result);
    if (!IS_OK(s)) {
        LOG_ERROR("Encode failed: %d", static_cast<int>(s));
        return 1;
    }

    LOG_INFO("Inference succeeded: shape=(%ld), dtype=%d",
             static_cast<long>(result.shape[0]), static_cast<int>(result.dtype));

    // ── 4. Convert to float32 ────────────────────────────────
    int64_t dim = result.shape[0];
    std::vector<float> fp32_data(dim);

    if (result.dtype == ACL_FLOAT16) {
        const uint16_t* fp16_data = result.As<uint16_t>();
        for (int64_t i = 0; i < dim; i++) {
            fp32_data[i] = atb_llm::Fp16ToF32(fp16_data[i]);
        }
        LOG_INFO("Converted fp16 -> fp32 (%ld elements)", static_cast<long>(dim));
    } else if (result.dtype == ACL_FLOAT) {
        const float* f32_data = result.As<float>();
        std::memcpy(fp32_data.data(), f32_data, dim * sizeof(float));
        LOG_INFO("Output already fp32 (%ld elements)", static_cast<long>(dim));
    } else {
        LOG_ERROR("Unexpected dtype: %d", static_cast<int>(result.dtype));
        return 1;
    }

    // ── 5. Print first 8 values ──────────────────────────────
    LOG_INFO("First 8 embedding values:");
    for (int i = 0; i < 8 && i < static_cast<int>(dim); i++) {
        fprintf(stderr, "  [%d] = %.6f\n", i, fp32_data[i]);
    }

    // ── 6. Save to binary file ───────────────────────────────
    FILE* f = fopen(OUTPUT_PATH, "wb");
    if (!f) {
        LOG_ERROR("Failed to open output file: %s", OUTPUT_PATH);
        return 1;
    }

    // Write header: int64_t dimension
    fwrite(&dim, sizeof(int64_t), 1, f);
    // Write data: float32 embedding
    fwrite(fp32_data.data(), sizeof(float), dim, f);
    fclose(f);

    LOG_INFO("Saved embedding to %s (dim=%ld, %ld bytes)",
             OUTPUT_PATH, static_cast<long>(dim),
             static_cast<long>(sizeof(int64_t) + dim * sizeof(float)));

    LOG_INFO("=== Consistency test C++ side complete ===");
    LOG_INFO("Run: python tests/test_consistency.py");

    return 0;
}
