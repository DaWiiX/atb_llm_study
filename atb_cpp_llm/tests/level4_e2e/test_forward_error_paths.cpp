/**
 * Forward() error path tests for Qwen3VLModel.
 *
 * Tests:
 *   1. Forward with null input_ids → ERROR_INVALID_PARAM
 *   2. Forward with batch_size != 1 → ERROR_INVALID_PARAM
 *   3. Forward with seq_len = 0 → ERROR_INVALID_PARAM (from AllocTensor dim<=0 check)
 *   4. Forward with invalid batch_size → ERROR_INVALID_PARAM
 *
 * Run: ./test_forward_error_paths
 * Requires: NPU device + ATB/ACL runtime + model checkpoint
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "log/logger.h"
#include "test_env.h"

#include <cstdio>
#include <cstring>
#include <vector>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

static const std::string MODEL_DIR = GetModelDir();

// ═════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (MODEL_DIR.empty()) {
        std::fprintf(stderr,
            "QWEN3VL_EMB_MODEL_DIR is not set. "
            "Source .env via build_and_test.sh or export the variable.\n");
        return 1;
    }

    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}

// ═════════════════════════════════════════════════════════════════════
// Test: Forward with null input_ids
// ═════════════════════════════════════════════════════════════════════
TEST_CASE("Forward Null Input IDs") {
    LOG_INFO("=== Test: Forward Null Input IDs ===");

    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    CHECK(IS_OK(s));
    if (!engine) return;

    // Test with null input_ids
    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::TEXT_ONLY;
    request.text.input_ids = nullptr;  // NULL input_ids
    request.text.batch_size = 1;
    request.text.seq_length = 3;

    atb_llm::InferResult result;
    s = engine->Forward(request, result);

    // Should return ERROR_INVALID_PARAM (not crash)
    CHECK(s == atb_llm::ERROR_INVALID_PARAM);

    LOG_INFO("Null input_ids test done");
}

// ═════════════════════════════════════════════════════════════════════
// Test: Forward with batch_size != 1
// ═════════════════════════════════════════════════════════════════════
TEST_CASE("Forward Invalid Batch Size") {
    LOG_INFO("=== Test: Forward Invalid Batch Size ===");

    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    CHECK(IS_OK(s));
    if (!engine) return;

    // Test with batch_size = 2 (only batch_size=1 is supported)
    int64_t input_ids[] = {151643, 15339, 1879};
    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::TEXT_ONLY;
    request.text.input_ids = input_ids;
    request.text.batch_size = 2;  // Invalid: only 1 supported
    request.text.seq_length = 3;

    atb_llm::InferResult result;
    s = engine->Forward(request, result);

    // Should return ERROR_INVALID_PARAM (not crash)
    CHECK(s == atb_llm::ERROR_INVALID_PARAM);

    // Test with batch_size = 0
    request.text.batch_size = 0;
    s = engine->Forward(request, result);

    // Should return ERROR_INVALID_PARAM (not crash)
    CHECK(s == atb_llm::ERROR_INVALID_PARAM);

    LOG_INFO("Invalid batch_size test done");
}

// ═════════════════════════════════════════════════════════════════════
// Test: Forward with seq_len = 0
// ═════════════════════════════════════════════════════════════════════
TEST_CASE("Forward Zero Seq Len") {
    LOG_INFO("=== Test: Forward Zero Seq Len ===");

    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    CHECK(IS_OK(s));
    if (!engine) return;

    // Test with seq_len = 0
    // This should fail at AllocFloat16(hidden_npu, {0, hidden_size})
    // because AllocTensor checks dim <= 0
    int64_t input_ids[] = {151643};
    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::TEXT_ONLY;
    request.text.input_ids = input_ids;
    request.text.batch_size = 1;
    request.text.seq_length = 0;  // Zero seq_len

    atb_llm::InferResult result;
    s = engine->Forward(request, result);

    // Should return an error (not crash)
    // AllocTensor returns ERROR_INVALID_PARAM for dim <= 0
    CHECK(s != atb_llm::STATUS_OK);
    CHECK(s == atb_llm::ERROR_INVALID_PARAM);

    LOG_INFO("Zero seq_len test done");
}
