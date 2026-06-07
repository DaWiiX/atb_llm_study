/**
 * Unified Qwen3VL Embedding Benchmark with per-stage timing.
 *
 * Matches the Python benchmark.py 6-stage breakdown for fair comparison.
 *
 * Run: ./benchmark [--mode text|mm|both] [--iter N] [--warmup M]
 *                  [--seq S] [--width W --height H] [--cmp]
 *
 *   --mode text   : text-only benchmark
 *   --mode mm     : multimodal benchmark (image + text)
 *   --mode both   : run both modes (default)
 *   --iter N      : benchmark iterations (default: 5)
 *   --warmup M    : warmup iterations (default: 3)
 *   --seq S       : text-only sequence length (default: 64)
 *   --width W     : image width for mm mode (default: 416)
 *   --height H    : image height for mm mode (default: 672)
 *   --cmp         : compact output for scripts
 *
 * Requires: NPU device + ATB/ACL runtime + model checkpoint
 */

#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "log/logger.h"
#include "test_env.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <functional>

static const std::string MODEL_DIR = GetModelDir();

// ── Statistics helper ────────────────────────────────────────
struct Stats {
    double mean = 0, median = 0, min_val = 0, max_val = 0, stddev = 0;
};

Stats ComputeStats(std::vector<double>& times) {
    Stats st;
    int n = static_cast<int>(times.size());
    std::sort(times.begin(), times.end());
    double total = std::accumulate(times.begin(), times.end(), 0.0);
    st.mean = total / n;
    st.median = (n % 2 == 0)
        ? (times[n/2 - 1] + times[n/2]) / 2.0
        : times[n/2];
    st.min_val = times.front();
    st.max_val = times.back();
    double sq_sum = 0;
    for (double t : times) sq_sum += (t - st.mean) * (t - st.mean);
    st.stddev = std::sqrt(sq_sum / n);
    return st;
}

// ── Run benchmark iterations ─────────────────────────────────
// Returns: vector of (StageTimings, e2e_ms) pairs
struct TimedResult {
    atb_llm::StageTimings timings;
    double e2e_ms = 0;
};

std::vector<TimedResult> RunBenchmark(
        atb_llm::LLMEngine* engine,
        const atb_llm::InferRequest& request,
        int num_warmup, int num_iter, bool verbose) {
    atb_llm::Status s;

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        atb_llm::InferResult result;
        atb_llm::StageTimings timings;
        s = engine->EncodeWithTiming(request, result, timings);
        if (s != atb_llm::STATUS_OK) {
            LOG_ERROR("Warmup iteration %d failed: %d", i, static_cast<int>(s));
            return {};
        }
        if (verbose) {
            LOG_INFO("  Warmup %d: %.2f ms", i, timings.e2e_ms);
        }
    }

    // Benchmark
    std::vector<TimedResult> results(num_iter);
    for (int i = 0; i < num_iter; i++) {
        atb_llm::InferResult result;
        atb_llm::StageTimings timings;
        s = engine->EncodeWithTiming(request, result, timings);
        if (s != atb_llm::STATUS_OK) {
            LOG_ERROR("Benchmark iteration %d failed: %d", i, static_cast<int>(s));
            return {};
        }
        results[i].timings = timings;
        results[i].e2e_ms = timings.e2e_ms;
        if (verbose) {
            LOG_INFO("  Iter %d: %.2f ms", i, timings.e2e_ms);
        }
    }
    return results;
}

// ── Compute mean of a stage field ────────────────────────────
double MeanStage(const std::vector<TimedResult>& results,
                 std::function<double(const atb_llm::StageTimings&)> getter) {
    double sum = 0;
    for (const auto& r : results) sum += getter(r.timings);
    return sum / results.size();
}

// ── Report stage timings (human-readable) ────────────────────
void ReportStages(const std::vector<TimedResult>& results,
                  const char* label, int seq_len, int vis_tokens,
                  const Stats& e2e_stats) {
    double preprocess = MeanStage(results, [](auto& t) { return t.preprocess_ms; });
    double vision_pos = MeanStage(results, [](auto& t) { return t.vision_pos_ms; });
    double vision_model = MeanStage(results, [](auto& t) { return t.vision_model_ms; });
    double text_embed = MeanStage(results, [](auto& t) { return t.text_embed_ms; });
    double position_ids = MeanStage(results, [](auto& t) { return t.position_ids_ms; });
    double text_model = MeanStage(results, [](auto& t) { return t.text_model_ms; });
    double pooling = MeanStage(results, [](auto& t) { return t.pooling_ms; });

    double staged_sum = preprocess + vision_pos + vision_model +
                        text_embed + position_ids + text_model + pooling;

    auto pct = [&](double v) { return staged_sum > 0 ? v / staged_sum * 100.0 : 0.0; };

    LOG_INFO("------------------------------------------------------------");
    LOG_INFO("  %s   S=%d   vision_tokens=%d", label, seq_len, vis_tokens);
    LOG_INFO("------------------------------------------------------------");
    LOG_INFO("  Stage              Mean (ms)   %% of staged");
    LOG_INFO("  %-18s %8.2f      %5.1f%%", "Preprocess", preprocess, pct(preprocess));
    LOG_INFO("  %-18s %8.2f      %5.1f%%", "Vision PosEmb", vision_pos, pct(vision_pos));
    LOG_INFO("  %-18s %8.2f      %5.1f%%", "Vision Model", vision_model, pct(vision_model));
    LOG_INFO("  %-18s %8.2f      %5.1f%%", "Text Embed+Inj", text_embed, pct(text_embed));
    LOG_INFO("  %-18s %8.2f      %5.1f%%", "Position IDs", position_ids, pct(position_ids));
    LOG_INFO("  %-18s %8.2f      %5.1f%%", "Text Model", text_model, pct(text_model));
    LOG_INFO("  %-18s %8.2f      %5.1f%%", "Pooling", pooling, pct(pooling));
    LOG_INFO("  ---------------- --------------- ------------");
    LOG_INFO("  %-18s %8.2f", "Staged sum", staged_sum);
    LOG_INFO("  %-18s %8.2f +/- %.2f", "E2E (no sync)",
             e2e_stats.mean, e2e_stats.stddev);
}

// ── Report stage timings (compact) ───────────────────────────
void ReportStagesCompact(const std::vector<TimedResult>& results,
                         const char* mode, const char* resolution,
                         int seq_len, int vis_tokens,
                         const Stats& e2e_stats) {
    double preprocess = MeanStage(results, [](auto& t) { return t.preprocess_ms; });
    double vision_pos = MeanStage(results, [](auto& t) { return t.vision_pos_ms; });
    double vision_model = MeanStage(results, [](auto& t) { return t.vision_model_ms; });
    double text_embed = MeanStage(results, [](auto& t) { return t.text_embed_ms; });
    double position_ids = MeanStage(results, [](auto& t) { return t.position_ids_ms; });
    double text_model = MeanStage(results, [](auto& t) { return t.text_model_ms; });
    double pooling = MeanStage(results, [](auto& t) { return t.pooling_ms; });
    double staged_sum = preprocess + vision_pos + vision_model +
                        text_embed + position_ids + text_model + pooling;

    printf("BENCH_RESULT: mode=%s resolution=%s S=%d vis=%d "
           "preprocess=%.2f vision_pos=%.2f vision_model=%.2f "
           "text_embed=%.2f position_ids=%.2f text_model=%.2f "
           "pooling=%.2f staged=%.2f e2e_mean=%.2f e2e_std=%.2f\n",
           mode, resolution, seq_len, vis_tokens,
           preprocess, vision_pos, vision_model,
           text_embed, position_ids, text_model,
           pooling, staged_sum, e2e_stats.mean, e2e_stats.stddev);
}

// ═══════════════════════════════════════════════════════════════
// Text-only benchmark
// ═══════════════════════════════════════════════════════════════

int RunTextBenchmark(atb_llm::LLMEngine* engine,
                     int seq_len, int num_warmup, int num_iter,
                     bool cmp_mode) {
    // Input: [151643, 15339, ..., 15339, 151645]
    std::vector<int64_t> input_ids(seq_len);
    input_ids[0] = 151643;
    for (int i = 1; i < seq_len - 1; i++) {
        input_ids[i] = 15339;
    }
    input_ids[seq_len - 1] = 151645;

    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::TEXT_ONLY;
    request.text.input_ids = input_ids.data();
    request.text.batch_size = 1;
    request.text.seq_length = seq_len;

    if (!cmp_mode) {
        LOG_INFO("=== Text-Only Benchmark ===");
        LOG_INFO("Sequence length: %d", seq_len);
        LOG_INFO("Iterations: %d (warmup: %d)", num_iter, num_warmup);
    }

    auto results = RunBenchmark(engine, request, num_warmup, num_iter, !cmp_mode);
    if (results.empty()) return 1;

    std::vector<double> e2e_times;
    for (auto& r : results) e2e_times.push_back(r.e2e_ms);
    Stats e2e_stats = ComputeStats(e2e_times);

    if (cmp_mode) {
        ReportStagesCompact(results, "text", "N/A", seq_len, 0, e2e_stats);
    } else {
        ReportStages(results, "text-only", seq_len, 0, e2e_stats);
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════
// Multimodal benchmark
// ═══════════════════════════════════════════════════════════════

int RunMultimodalBenchmark(atb_llm::LLMEngine* engine,
                           int img_w, int img_h,
                           int num_warmup, int num_iter,
                           bool cmp_mode) {
    // Load Qwen3VL config for preprocessing
    atb_llm::adapters::Qwen3VLConfig config;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(MODEL_DIR, config);
    if (s != atb_llm::STATUS_OK) {
        LOG_ERROR("Failed to load Qwen3VL config from %s", MODEL_DIR.c_str());
        return 1;
    }

    // Generate synthetic image: solid blue, (C, H, W) uint8
    std::vector<uint8_t> image(3 * img_h * img_w);
    // R=0, G=0, B=200 (blue-ish)
    for (int i = 0; i < img_h * img_w; i++) {
        image[i] = 0;                    // R
        image[img_h * img_w + i] = 0;    // G
        image[2 * img_h * img_w + i] = 200; // B
    }

    // Preprocess image
    // First, compute how many patches we'll get to size the output buffer
    int32_t factor = config.pp_patch_size * config.pp_temporal_patch_size;
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(img_h, img_w, factor,
                                    config.pp_min_pixels, config.pp_max_pixels,
                                    new_h, new_w);
    int64_t grid_h = new_h / config.vis_patch_size;
    int64_t grid_w = new_w / config.vis_patch_size;
    int64_t grid_t = 2;  // temporal dimension is always 2 for Qwen3VL
    int64_t num_patches = grid_t * grid_h * grid_w;
    int64_t patch_dim = static_cast<int64_t>(config.vis_in_channels) *
                        config.vis_temporal_patch_size *
                        config.vis_patch_size * config.vis_patch_size;

    std::vector<uint16_t> pixel_values(num_patches * patch_dim);
    int64_t actual_patches = 0;
    int64_t grid_thw[3] = {};
    s = atb_llm::adapters::PreprocessImage(
        image.data(), 3, img_h, img_w, config,
        pixel_values.data(), actual_patches, grid_thw);
    if (s != atb_llm::STATUS_OK) {
        LOG_ERROR("PreprocessImage failed");
        return 1;
    }

    int64_t vis_tokens = actual_patches / (config.vis_spatial_merge_size * config.vis_spatial_merge_size);
    int64_t image_token_id = config.image_token_id;

    // Build input_ids: [151643] + [image_token_id] * vis_tokens + [15339] * 10 + [151645]
    int text_fill = 10;
    int64_t seq_len = 1 + vis_tokens + text_fill + 1;
    std::vector<int64_t> input_ids(seq_len);
    input_ids[0] = 151643;
    for (int64_t i = 0; i < vis_tokens; i++) {
        input_ids[1 + i] = image_token_id;
    }
    for (int i = 0; i < text_fill; i++) {
        input_ids[1 + vis_tokens + i] = 15339;
    }
    input_ids[seq_len - 1] = 151645;

    // Setup request
    atb_llm::InferRequest request;
    request.mode = atb_llm::InputMode::PREPROCESSED;
    request.text.input_ids = input_ids.data();
    request.text.batch_size = 1;
    request.text.seq_length = seq_len;
    request.preprocessed.pixel_values = pixel_values.data();
    request.preprocessed.num_patches = actual_patches;
    request.preprocessed.patch_dim = patch_dim;
    request.preprocessed.grid_thw = grid_thw;
    request.preprocessed.dtype = ACL_FLOAT16;

    char resolution[32];
    std::snprintf(resolution, sizeof(resolution), "%dx%d", img_w, img_h);

    if (!cmp_mode) {
        LOG_INFO("=== Multimodal Benchmark ===");
        LOG_INFO("Image: %dx%d -> %ldx%ld, patches=%ld, vis_tokens=%ld",
                 img_w, img_h,
                 static_cast<long>(new_h), static_cast<long>(new_w),
                 static_cast<long>(actual_patches),
                 static_cast<long>(vis_tokens));
        LOG_INFO("Sequence length: %ld", static_cast<long>(seq_len));
        LOG_INFO("Iterations: %d (warmup: %d)", num_iter, num_warmup);
    }

    auto results = RunBenchmark(engine, request, num_warmup, num_iter, !cmp_mode);
    if (results.empty()) return 1;

    std::vector<double> e2e_times;
    for (auto& r : results) e2e_times.push_back(r.e2e_ms);
    Stats e2e_stats = ComputeStats(e2e_times);

    if (cmp_mode) {
        ReportStagesCompact(results, "mm", resolution,
                            static_cast<int>(seq_len),
                            static_cast<int>(vis_tokens), e2e_stats);
    } else {
        char label[64];
        std::snprintf(label, sizeof(label), "%dx%d", img_w, img_h);
        ReportStages(results, label,
                     static_cast<int>(seq_len),
                     static_cast<int>(vis_tokens), e2e_stats);
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    int num_iter = 5;
    int num_warmup = 3;
    int seq_len = 64;
    int img_w = 416;
    int img_h = 672;
    bool cmp_mode = false;
    std::string mode = "both";

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iter") == 0 && i + 1 < argc) {
            num_iter = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            num_warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            img_w = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            img_h = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (std::strcmp(argv[i], "--cmp") == 0) {
            cmp_mode = true;
        }
    }

    if (!cmp_mode) {
        LOG_INFO("=== Qwen3VL Embedding Benchmark ===");
        LOG_INFO("Model: %s", MODEL_DIR.c_str());
        LOG_INFO("Mode: %s", mode.c_str());
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

    int ret = 0;
    if (mode == "text") {
        ret = RunTextBenchmark(engine.get(), seq_len, num_warmup, num_iter, cmp_mode);
    } else if (mode == "mm") {
        ret = RunMultimodalBenchmark(engine.get(), img_w, img_h, num_warmup, num_iter, cmp_mode);
    } else if (mode == "both") {
        ret = RunTextBenchmark(engine.get(), seq_len, num_warmup, num_iter, cmp_mode);
        if (ret == 0) {
            ret = RunMultimodalBenchmark(engine.get(), img_w, img_h, num_warmup, num_iter, cmp_mode);
        }
    } else {
        LOG_ERROR("Unknown mode: %s (use text|mm|both)", mode.c_str());
        return 1;
    }

    return ret;
}
