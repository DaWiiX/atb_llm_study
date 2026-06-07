/**
 * C++ Multimodal Benchmark: reads preprocessed inputs from files,
 * runs full image+text inference through ATB engine.
 *
 * Input files (in /tmp/mm_inputs/):
 *   {tag}_pixel_values.bin  — float16 pixel_values
 *   {tag}_input_ids.bin     — int64 input_ids
 *   {tag}_grid_thw.bin      — int64 grid_thw
 *   {tag}_meta.bin          — (num_patches, patch_dim, seq_len, vis_tokens)
 *
 * Run: ./bench_mm --tag 416x672 [--iter N] [--warmup M] [--cmp]
 */
#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "log/logger.h"
#include "test_env.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

static const std::string MODEL_DIR = GetModelDir();
static const char* INPUT_DIR = "/tmp/mm_inputs";

// Read binary file into vector
template <typename T>
std::vector<T> read_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { LOG_ERROR("Cannot open: %s", path.c_str()); return {}; }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg() / sizeof(T);
    f.seekg(0, std::ios::beg);
    std::vector<T> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz * sizeof(T));
    return data;
}

int main(int argc, char** argv) {
    int num_iter = 10, num_warmup = 3;
    std::string tag = "416x672";
    bool cmp_mode = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            num_iter = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            num_warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tag") == 0 && i + 1 < argc) {
            tag = argv[++i];
        } else if (std::strcmp(argv[i], "--cmp") == 0) {
            cmp_mode = true;
        }
    }

    // Read inputs
    std::string base = std::string(INPUT_DIR) + "/" + tag;
    auto pv = read_binary<uint16_t>(base + "_pixel_values.bin");
    auto iids = read_binary<int64_t>(base + "_input_ids.bin");
    auto gth = read_binary<int64_t>(base + "_grid_thw.bin");

    // Read meta: num_patches, patch_dim, seq_len, vis_tokens
    std::ifstream mf(base + "_meta.bin", std::ios::binary);
    int64_t meta[4] = {};
    mf.read(reinterpret_cast<char*>(meta), sizeof(meta));
    int64_t num_patches = meta[0], patch_dim = meta[1], seq_len = meta[2];

    if (pv.empty() || iids.empty()) {
        LOG_ERROR("Failed to read inputs for tag=%s", tag.c_str());
        return 1;
    }

    if (!cmp_mode) {
        LOG_INFO("=== C++ Multimodal Benchmark ===");
        LOG_INFO("Tag: %s, S=%lld, patches=%lld", tag.c_str(),
                 static_cast<long long>(seq_len),
                 static_cast<long long>(num_patches));
    }

    // Create engine
    atb_llm::EngineConfig config;
    config.model_dir = MODEL_DIR;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(config, engine);
    if (s != atb_llm::STATUS_OK || !engine) {
        LOG_ERROR("Failed to create engine");
        return 1;
    }

    // Setup request
    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::PREPROCESSED;
    request.text.input_ids = iids.data();
    request.text.batch_size = 1;
    request.text.seq_length = seq_len;
    request.preprocessed.pixel_values = pv.data();
    request.preprocessed.num_patches = num_patches;
    request.preprocessed.patch_dim = patch_dim;
    request.preprocessed.grid_thw = gth.data();
    request.preprocessed.dtype = ACL_FLOAT16;

    auto run_inference = [&]() -> double {
        atb_llm::InferResult result;
        auto start = std::chrono::high_resolution_clock::now();
        s = engine->Encode(request, result);
        auto end = std::chrono::high_resolution_clock::now();
        if (s != atb_llm::STATUS_OK) return -1.0;
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        double ms = run_inference();
        if (ms < 0) { LOG_ERROR("Warmup %d failed", i); return 1; }
        if (!cmp_mode) LOG_INFO("  Warmup %d: %.2f ms", i, ms);
    }

    // Benchmark
    std::vector<double> times(num_iter);
    for (int i = 0; i < num_iter; i++) {
        double ms = run_inference();
        if (ms < 0) { LOG_ERROR("Iter %d failed", i); return 1; }
        times[i] = ms;
        if (!cmp_mode) LOG_INFO("  Iter %d: %.2f ms", i, ms);
    }

    // Statistics
    std::sort(times.begin(), times.end());
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / num_iter;
    double median = (num_iter % 2 == 0)
        ? (times[num_iter/2 - 1] + times[num_iter/2]) / 2.0 : times[num_iter/2];
    double sq_sum = 0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / num_iter);

    if (cmp_mode) {
        printf("CPP_RESULT: mean=%.2f median=%.2f min=%.2f max=%.2f std=%.2f p95=%.2f p99=%.2f\n",
               mean, median, times.front(), times.back(), stddev,
               times[static_cast<int>(num_iter * 0.95)],
               times[static_cast<int>(num_iter * 0.99)]);
    } else {
        LOG_INFO("=== Results (S=%lld, patches=%lld) ===",
                 static_cast<long long>(seq_len), static_cast<long long>(num_patches));
        LOG_INFO("  Mean:    %.2f ms", mean);
        LOG_INFO("  Median:  %.2f ms", median);
        LOG_INFO("  P95:     %.2f ms", times[static_cast<int>(num_iter * 0.95)]);
    }
    return 0;
}
