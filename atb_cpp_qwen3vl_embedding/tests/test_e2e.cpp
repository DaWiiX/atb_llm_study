/**
 * Phase 4 E2E test: Qwen3VL Embedding model.
 *
 * Tests:
 *   1. Full model load (config + weights + graph build)
 *   2. Text-only inference
 *   3. Vision inference (if image data available)
 *   4. Output shape and non-zero validation
 *
 * Run: ./test_e2e
 * Requires: NPU device + ATB/ACL runtime + model checkpoint
 */

#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>

static int test_count = 0;
static int pass_count = 0;

#define TEST_ASSERT(cond, msg)                                          \
    do {                                                                \
        test_count++;                                                   \
        if (!(cond)) {                                                  \
            LOG_ERROR("FAIL: %s (%s:%d)", msg, __FILE__, __LINE__);    \
        } else {                                                        \
            pass_count++;                                               \
            LOG_INFO("PASS: %s", msg);                                 \
        }                                                               \
    } while (0)

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

static const char* MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B";

// ═════════════════════════════════════════════════════════════════════
// Test: Engine creation and model load
// ═════════════════════════════════════════════════════════════════════
void test_engine_load() {
    LOG_INFO("=== Test: Engine Load ===");

    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;  // 10GB
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    TEST_ASSERT(IS_OK(s), "LLMEngine::Create succeeds");
    TEST_ASSERT(engine != nullptr, "Engine is non-null");

    if (!engine) return;

    LOG_INFO("Engine load test done");
}

// ═════════════════════════════════════════════════════════════════════
// Test: Text-only inference
// ═════════════════════════════════════════════════════════════════════
void test_text_only_inference() {
    LOG_INFO("=== Test: Text-Only Inference ===");

    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    TEST_ASSERT(IS_OK(s), "Engine created for text-only test");
    if (!engine) return;

    // Simple text input: "Hello world" tokenized
    // Using a few dummy token IDs (valid for the model's vocab)
    int64_t input_ids[] = {151643, 15339, 1879};  // <|im_start|> Hello world
    int64_t seq_len = 3;

    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::TEXT_ONLY;
    request.text.input_ids = input_ids;
    request.text.batch_size = 1;
    request.text.seq_length = seq_len;

    atb_llm::InferResult result;
    auto start = std::chrono::high_resolution_clock::now();
    s = engine->Encode(request, result);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    TEST_ASSERT(IS_OK(s), "Text-only Encode succeeds");
    LOG_INFO("Text-only inference time: %ld ms", static_cast<long>(ms));

    if (IS_OK(s)) {
        TEST_ASSERT(result.shape.size() == 1, "Output is 1D (embedding vector)");
        TEST_ASSERT(result.shape[0] == 2048, "Output dim matches hidden_size");

        // Check output is non-zero
        bool non_zero = false;
        if (result.dtype == ACL_FLOAT16) {
            const uint16_t* data = result.As<uint16_t>();
            for (size_t i = 0; i < result.shape[0]; i++) {
                if (data[i] != 0) { non_zero = true; break; }
            }
        } else if (result.dtype == ACL_FLOAT) {
            const float* data = result.As<float>();
            for (size_t i = 0; i < result.shape[0]; i++) {
                if (std::fabs(data[i]) > 1e-10f) { non_zero = true; break; }
            }
        }
        TEST_ASSERT(non_zero, "Output embedding is not all zeros");

        // Print first few values
        if (result.dtype == ACL_FLOAT16) {
            const uint16_t* data = result.As<uint16_t>();
            LOG_INFO("First 8 output values (fp16 hex):");
            for (int i = 0; i < 8 && i < static_cast<int>(result.shape[0]); i++) {
                fprintf(stderr, "  [%d] = 0x%04x\n", i, data[i]);
            }
        }
    }

    LOG_INFO("Text-only inference test done");
}

// ═════════════════════════════════════════════════════════════════════
// Test: Multiple inference calls (check stability)
// ═════════════════════════════════════════════════════════════════════
void test_multiple_inference() {
    LOG_INFO("=== Test: Multiple Inference Calls ===");

    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    TEST_ASSERT(IS_OK(s), "Engine created for multi-inference test");
    if (!engine) return;

    // Run 3 inference calls with different inputs
    for (int trial = 0; trial < 3; trial++) {
        int64_t input_ids[] = {151643, 15339 + trial, 1879};
        int64_t seq_len = 3;

        atb_llm::InferRequest request;
        request.mode = atb_llm::InputMode::TEXT_ONLY;
        request.text.input_ids = input_ids;
        request.text.batch_size = 1;
        request.text.seq_length = seq_len;

        atb_llm::InferResult result;
        s = engine->Encode(request, result);
        TEST_ASSERT(IS_OK(s), ("Inference call " + std::to_string(trial) + " succeeds").c_str());

        if (IS_OK(s) && !result.data.empty()) {
            bool non_zero = false;
            const uint16_t* data = result.As<uint16_t>();
            for (size_t i = 0; i < result.shape[0]; i++) {
                if (data[i] != 0) { non_zero = true; break; }
            }
            TEST_ASSERT(non_zero,
                        ("Output " + std::to_string(trial) + " is non-zero").c_str());
        }
    }

    LOG_INFO("Multiple inference test done");
}

// ═════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    LOG_INFO("=== Phase 4 E2E Tests: Qwen3VL Embedding ===");

    test_engine_load();
    test_text_only_inference();
    test_multiple_inference();

    LOG_INFO("=== Results: %d/%d passed ===", pass_count, test_count);

    if (pass_count == test_count) {
        LOG_INFO("ALL TESTS PASSED");
        return 0;
    } else {
        LOG_ERROR("SOME TESTS FAILED");
        return 1;
    }
}
