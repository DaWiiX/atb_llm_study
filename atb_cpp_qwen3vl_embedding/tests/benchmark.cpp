/**
 * Phase 4 Benchmark: Qwen3VL Embedding inference performance.
 *
 * Measures end-to-end inference latency (text-only).
 * Uses same input format as Python benchmark for fair comparison.
 *
 * Run: ./benchmark [--iter N] [--warmup M] [--seq S] [--cmp]
 *   --cmp: output in compact format for comparison script
 * Requires: NPU device + ATB/ACL runtime + model checkpoint
 */

#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "log/logger.h"
#include "test_env.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

static const std::string MODEL_DIR = GetModelDir();

int main(int argc, char** argv) {
    int num_iter = 10;
    int num_warmup = 3;
    int seq_len = 4;
    bool cmp_mode = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            num_iter = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            num_warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cmp") == 0) {
            cmp_mode = true;
        }
    }

    if (!cmp_mode) {
        LOG_INFO("=== Qwen3VL Embedding Benchmark ===");
        LOG_INFO("Model: %s", MODEL_DIR.c_str());
        LOG_INFO("Sequence length: %d", seq_len);
        LOG_INFO("Iterations: %d (warmup: %d)", num_iter, num_warmup);
    }

    // Create engine
    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    auto t0 = std::chrono::high_resolution_clock::now();
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (s != atb_llm::STATUS_OK || !engine) {
        LOG_ERROR("Failed to create engine: %d", static_cast<int>(s));
        return 1;
    }
    if (!cmp_mode) LOG_INFO("Engine load time: %ld ms", static_cast<long>(load_ms));

    // Prepare input — same format as Python:
    //   [151643, 15339, ..., 15339, 151645]   where middle tokens repeat
    std::vector<int64_t> input_ids_vec(seq_len);
    input_ids_vec[0] = 151643;            // <|im_start|>
    for (int i = 1; i < seq_len - 1; i++) {
        input_ids_vec[i] = 15339;         // filler token
    }
    input_ids_vec[seq_len - 1] = 151645;  // <|im_end|>

    auto run_inference = [&]() -> double {
        atb_llm::InferRequest request;
        request.mode = atb_llm::InputMode::TEXT_ONLY;
        request.text.input_ids = input_ids_vec.data();
        request.text.batch_size = 1;
        request.text.seq_length = seq_len;

        atb_llm::InferResult result;
        auto start = std::chrono::high_resolution_clock::now();
        s = engine->Encode(request, result);
        auto end = std::chrono::high_resolution_clock::now();

        if (s != atb_llm::STATUS_OK) return -1.0;
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    // Warmup
    if (!cmp_mode) LOG_INFO("Running %d warmup iterations...", num_warmup);
    for (int i = 0; i < num_warmup; i++) {
        double ms = run_inference();
        if (ms < 0) {
            LOG_ERROR("Warmup iteration %d failed", i);
            return 1;
        }
        if (!cmp_mode) LOG_INFO("  Warmup %d: %.2f ms", i, ms);
    }

    // Benchmark
    if (!cmp_mode) LOG_INFO("Running %d benchmark iterations...", num_iter);
    std::vector<double> times(num_iter);
    for (int i = 0; i < num_iter; i++) {
        double ms = run_inference();
        if (ms < 0) {
            LOG_ERROR("Benchmark iteration %d failed", i);
            return 1;
        }
        times[i] = ms;
        if (!cmp_mode) LOG_INFO("  Iter %d: %.2f ms", i, ms);
    }

    // Statistics
    std::sort(times.begin(), times.end());
    double total = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = total / num_iter;
    double median = (num_iter % 2 == 0)
        ? (times[num_iter/2 - 1] + times[num_iter/2]) / 2.0
        : times[num_iter/2];
    double min_val = times.front();
    double max_val = times.back();

    // Standard deviation
    double sq_sum = 0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / num_iter);

    if (cmp_mode) {
        // Compact output for comparison script
        printf("CPP_RESULT: mean=%.2f median=%.2f min=%.2f max=%.2f std=%.2f p95=%.2f p99=%.2f\n",
               mean, median, min_val, max_val, stddev,
               times[static_cast<int>(num_iter * 0.95)],
               times[static_cast<int>(num_iter * 0.99)]);
    } else {
        LOG_INFO("=== Benchmark Results ===");
        LOG_INFO("  Sequence length: %d", seq_len);
        LOG_INFO("  Iterations: %d", num_iter);
        LOG_INFO("  Mean:    %.2f ms", mean);
        LOG_INFO("  Median:  %.2f ms", median);
        LOG_INFO("  Min:     %.2f ms", min_val);
        LOG_INFO("  Max:     %.2f ms", max_val);
        LOG_INFO("  Stddev:  %.2f ms", stddev);
        LOG_INFO("  P95:     %.2f ms", times[static_cast<int>(num_iter * 0.95)]);
        LOG_INFO("  P99:     %.2f ms", times[static_cast<int>(num_iter * 0.99)]);
    }

    return 0;
}
