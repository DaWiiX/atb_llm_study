/**
 * Unified Qwen3VL Embedding Benchmark with per-stage timing.
 *
 * Matches the Python benchmark.py 6-stage breakdown for fair comparison.
 *
 * Run: ./benchmark [--mode text|mm|io|all|bench|compare] [--iter N] [--warmup M]
 *                  [--seq S] [--width W --height H] [--cmp]
 *
 *   --mode text    : text-only benchmark
 *   --mode mm      : multimodal benchmark (image + text)
 *   --mode io      : image-only benchmark (no text tokens)
 *   --mode all     : run all three modes (default)
 *   --mode bench   : run image-only at 4 fixed resolutions (224x224, 416x672, 672x416, 896x896)
 *   --mode compare : run full test matrix: TEXT_ONLY + IMAGE_ONLY x4 + IMAGE_AND_TEXT x4
 *                    saves pooler output .bin for each combination
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
#include "utils/float_utils.h"
#include "test_env.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <tuple>
#include <functional>

// ── Statistics helper ────────────────────────────────────────
struct Stats {
    double mean = 0, median = 0, min_val = 0, max_val = 0, stddev = 0;
};

// ── Cold-start metrics ──────────────────────────────────────
struct ColdMetrics {
    double cold_e2e_ms = 0;
    double warm_e2e_ms = 0;
    double penalty_pct = 0;
    bool valid = false;
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

// ── Helper: load token IDs saved by Python benchmark --save-tokens ─
// Format: [int32_t count] [int64_t * count]
// Returns empty vector on failure (caller should fallback to hardcoded).
static std::vector<int64_t> LoadTokenIds(const std::string& path) {
    std::vector<int64_t> ids;
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        return ids;  // file doesn't exist — caller should fallback
    }
    int32_t count = 0;
    if (fread(&count, sizeof(int32_t), 1, fp) != 1) {
        LOG_ERROR("Failed to read token count from %s", path.c_str());
        fclose(fp);
        return {};
    }
    if (count <= 0) {
        LOG_ERROR("Token file %s has invalid count %d", path.c_str(), count);
        fclose(fp);
        return {};
    }
    ids.resize(static_cast<size_t>(count));
    size_t read_n = fread(ids.data(), sizeof(int64_t), static_cast<size_t>(count), fp);
    fclose(fp);
    if (read_n != static_cast<size_t>(count)) {
        LOG_ERROR("Failed to read token data from %s (expected %d, got %zu)",
                  path.c_str(), count, read_n);
        return {};
    }
    LOG_INFO("Loaded %d token IDs from %s", count, path.c_str());
    return ids;
}

// ── Helper: create gradient test image (NCHW layout, matches Python benchmark.py) ─
static std::vector<uint8_t> CreateGradientImage(int32_t channels, int32_t height, int32_t width) {
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

// ── Helper: save pixel_values (fp16) to binary file ──────────────
// Format: [int32_t num_values] [uint16_t * num_values]
// Used by RunCompareMode so Python benchmark --load-pixel-values can
// consume the identical preprocessed input.
static void SavePixelValues(const std::string& path,
                            const uint16_t* data, int32_t count) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        LOG_ERROR("Failed to open pixel_values file: %s", path.c_str());
        return;
    }
    fwrite(&count, sizeof(int32_t), 1, fp);
    fwrite(data, sizeof(uint16_t), count, fp);
    fclose(fp);
    LOG_INFO("Saved pixel_values to %s (count=%d)", path.c_str(), count);
}

// ── Helper: save InferResult (fp16→fp32) to binary file ────────
// Format: [int64_t dim] [float32 * dim]
static bool SavePoolerOutput(const char* path, const atb_llm::InferResult& result) {
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
    LOG_INFO("Saved %s (dim=%ld)", path, static_cast<long>(dim));
    return true;
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
        int num_warmup, int num_iter, bool verbose,
        const char* save_bin_path = nullptr,
        ColdMetrics* cold = nullptr) {
    atb_llm::Status s;

    if (cold && num_warmup < 1) {
        cold->valid = false;
    }

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        atb_llm::InferResult result;
        atb_llm::StageTimings timings;
        s = engine->EncodeWithTiming(request, result, timings);
        if (s != atb_llm::STATUS_OK) {
            LOG_ERROR("Warmup iteration %d failed: %d", i, static_cast<int>(s));
            return {};
        }
        // All ATB graphs were compiled during LLMEngine::Create(); no lazy graph init in cold path
        // Capture cold start from the very first forward
        if (cold && i == 0) {
            cold->cold_e2e_ms = timings.e2e_ms;
        }
        if (verbose) {
            LOG_INFO("  Warmup %d: %.2f ms", i, timings.e2e_ms);
        }
    }

    // Benchmark
    std::vector<TimedResult> results(num_iter);
    atb_llm::InferResult last_result;
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
        if (save_bin_path && i == num_iter - 1) {
            last_result = std::move(result);
        }
        if (verbose) {
            LOG_INFO("  Iter %d: %.2f ms", i, timings.e2e_ms);
        }
    }

    // Compute cold-start penalty
    if (cold && cold->cold_e2e_ms > 0 && results.size() > 0) {
        double sum = 0;
        for (auto& r : results) sum += r.e2e_ms;
        cold->warm_e2e_ms = sum / results.size();
        if (cold->warm_e2e_ms > 0) {
            cold->penalty_pct = (cold->cold_e2e_ms - cold->warm_e2e_ms)
                                / cold->warm_e2e_ms * 100.0;
            cold->valid = true;
        }
    }

    if (save_bin_path) {
        if (!SavePoolerOutput(save_bin_path, last_result)) {
            LOG_ERROR("Failed to save pooler output to %s", save_bin_path);
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
    double preprocess = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.preprocess_ms; });
    double vision_pos = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.vision_pos_ms; });
    double vision_model = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.vision_model_ms; });
    double text_embed = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.text_embed_ms; });
    double position_ids = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.position_ids_ms; });
    double text_model = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.text_model_ms; });
    double pooling = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.pooling_ms; });

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
    double preprocess = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.preprocess_ms; });
    double vision_pos = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.vision_pos_ms; });
    double vision_model = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.vision_model_ms; });
    double text_embed = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.text_embed_ms; });
    double position_ids = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.position_ids_ms; });
    double text_model = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.text_model_ms; });
    double pooling = MeanStage(results, [](const atb_llm::StageTimings& t) { return t.pooling_ms; });
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

// ── Report cold-start results (dual-format) ────────────────────
void ReportColdStart(const std::vector<std::tuple<std::string, std::string, ColdMetrics>>& cold_results) {
    if (cold_results.empty()) return;

    // Human-readable table
    LOG_INFO("============================================================");
    LOG_INFO("  === Cold-Start Benchmark ===");
    LOG_INFO("============================================================");
    LOG_INFO("  mode   resolution     cold_e2e_ms  warm_e2e_ms  penalty_pct");
    LOG_INFO("  ------ -------------- ------------ ------------ -----------");
    for (const auto& entry : cold_results) {
        const std::string& mode = std::get<0>(entry);
        const std::string& resolution = std::get<1>(entry);
        const ColdMetrics& cm = std::get<2>(entry);
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%+.1f%%", cm.penalty_pct);
        LOG_INFO("  %-6s %-14s %12.2f %12.2f %11s",
                 mode.c_str(), resolution.c_str(),
                 cm.cold_e2e_ms, cm.warm_e2e_ms, buf);
    }
    LOG_INFO("============================================================");

    // Machine-parseable lines
    for (const auto& entry : cold_results) {
        const std::string& mode = std::get<0>(entry);
        const std::string& resolution = std::get<1>(entry);
        const ColdMetrics& cm = std::get<2>(entry);
        printf("BENCH_COLD: mode=%s resolution=%s cold_ms=%.2f warm_ms=%.2f penalty_pct=%.1f\n",
               mode.c_str(), resolution.c_str(), cm.cold_e2e_ms, cm.warm_e2e_ms, cm.penalty_pct);
    }
}

// ═══════════════════════════════════════════════════════════════
// Text-only benchmark
// ═══════════════════════════════════════════════════════════════

int RunTextBenchmark(atb_llm::LLMEngine* engine,
                     int seq_len, int num_warmup, int num_iter,
                     bool cmp_mode,
                     ColdMetrics* cold = nullptr) {
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

    auto results = RunBenchmark(engine, request, num_warmup, num_iter, !cmp_mode,
                                nullptr, cold);
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
                           const std::string& model_dir,
                           int img_w, int img_h,
                           int num_warmup, int num_iter,
                           bool cmp_mode,
                           ColdMetrics* cold = nullptr) {
    // Load Qwen3VL config for preprocessing
    atb_llm::adapters::Qwen3VLConfig config;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(model_dir, config);
    if (s != atb_llm::STATUS_OK) {
        LOG_ERROR("Failed to load Qwen3VL config from %s", model_dir.c_str());
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
    auto pre_start = std::chrono::high_resolution_clock::now();
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
    auto pre_end = std::chrono::high_resolution_clock::now();

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

    auto results = RunBenchmark(engine, request, num_warmup, num_iter, !cmp_mode,
                                nullptr, cold);
    if (results.empty()) return 1;
    double pre_ms = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
    for (auto& r : results) {
        r.timings.preprocess_ms = pre_ms;
    }

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
// Image-only benchmark
// ═══════════════════════════════════════════════════════════════

int RunImageOnlyBenchmark(atb_llm::LLMEngine* engine,
                          const std::string& model_dir,
                          int img_w, int img_h,
                          int num_warmup, int num_iter,
                          bool cmp_mode,
                          ColdMetrics* cold = nullptr) {
    // Load Qwen3VL config for preprocessing
    atb_llm::adapters::Qwen3VLConfig config;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(model_dir, config);
    if (s != atb_llm::STATUS_OK) {
        LOG_ERROR("Failed to load Qwen3VL config from %s", model_dir.c_str());
        return 1;
    }

    // Generate synthetic image: gradient pattern
    std::vector<uint8_t> image(3 * img_h * img_w);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < img_h; h++) {
            for (int w = 0; w < img_w; w++) {
                image[c * img_h * img_w + h * img_w + w] =
                    static_cast<uint8_t>((h * 255 / img_h + w * 255 / img_w + c * 85) % 256);
            }
        }
    }

    // Preprocess image
    int32_t factor = config.pp_patch_size * config.pp_temporal_patch_size;
    int32_t new_h, new_w;
    auto pre_start = std::chrono::high_resolution_clock::now();
    atb_llm::adapters::SmartResize(img_h, img_w, factor,
                                    config.pp_min_pixels, config.pp_max_pixels,
                                    new_h, new_w);
    int64_t grid_h = new_h / config.vis_patch_size;
    int64_t grid_w = new_w / config.vis_patch_size;
    int64_t grid_t = 2;
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
    auto pre_end = std::chrono::high_resolution_clock::now();

    int64_t vis_tokens = actual_patches / (config.vis_spatial_merge_size * config.vis_spatial_merge_size);
    int64_t image_token_id = config.image_token_id;

    // Load chat-templated token IDs for image-only
    char io_tok_path[256];
    std::snprintf(io_tok_path, sizeof(io_tok_path),
                  "/tmp/tokens_chat_io_%dx%d.bin", img_w, img_h);
    std::vector<int64_t> input_ids = LoadTokenIds(io_tok_path);
    if (input_ids.empty()) {
        LOG_WARN("Chat-templated token file not found: %s — "
                 "using bare image tokens. Run 'python atb_cpp_llm/scripts/gen_compare_tokens.py' "
                 "to generate accurate token files.",
                 io_tok_path);
        // Fallback to bare image tokens
        input_ids.resize(vis_tokens);
        for (int64_t i = 0; i < vis_tokens; i++)
            input_ids[i] = image_token_id;
    }
    int64_t seq_len = static_cast<int64_t>(input_ids.size());

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
        LOG_INFO("=== Image-Only Benchmark ===");
        LOG_INFO("Image: %dx%d -> %ldx%ld, patches=%ld, vis_tokens=%ld",
                 img_w, img_h,
                 static_cast<long>(new_h), static_cast<long>(new_w),
                 static_cast<long>(actual_patches),
                 static_cast<long>(vis_tokens));
        LOG_INFO("Sequence length: %ld (image tokens only)", static_cast<long>(seq_len));
        LOG_INFO("Iterations: %d (warmup: %d)", num_iter, num_warmup);
    }

    auto results = RunBenchmark(engine, request, num_warmup, num_iter, !cmp_mode,
                                nullptr, cold);
    if (results.empty()) return 1;
    double pre_ms = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
    for (auto& r : results) {
        r.timings.preprocess_ms = pre_ms;
    }

    std::vector<double> e2e_times;
    for (auto& r : results) e2e_times.push_back(r.e2e_ms);
    Stats e2e_stats = ComputeStats(e2e_times);

    if (cmp_mode) {
        ReportStagesCompact(results, "io", resolution,
                            static_cast<int>(seq_len),
                            static_cast<int>(vis_tokens), e2e_stats);
    } else {
        char label[64];
        std::snprintf(label, sizeof(label), "image-only %dx%d", img_w, img_h);
        ReportStages(results, label,
                     static_cast<int>(seq_len),
                     static_cast<int>(vis_tokens), e2e_stats);
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════
// Compare mode — runs full test matrix and saves .bin outputs
// ═══════════════════════════════════════════════════════════════

int RunCompareMode(atb_llm::LLMEngine* engine,
                   const std::string& model_dir,
                   int num_warmup, int num_iter) {
    // Load Qwen3VL config for preprocessing
    atb_llm::adapters::Qwen3VLConfig config;
    atb_llm::Status s = atb_llm::adapters::LoadQwen3VLConfig(model_dir, config);
    if (s != atb_llm::STATUS_OK) {
        LOG_ERROR("Failed to load Qwen3VL config from %s", model_dir.c_str());
        return 1;
    }

    // Token constants: "Describe the image." (from Qwen3-VL tokenizer)
    static constexpr int64_t TOK_DESCRIBE = 74785;
    static constexpr int64_t TOK_THE      = 279;
    static constexpr int64_t TOK_IMAGE    = 2168;
    static constexpr int64_t TOK_DOT      = 13;
    int64_t image_token_id = config.image_token_id;

    // 4 resolutions (matching Python benchmark.py)
    struct { int w; int h; } resolutions[] = {
        {416, 672},
        {720, 1280},
        {1080, 1920},
        {1440, 2560},
    };
    int num_res = static_cast<int>(sizeof(resolutions) / sizeof(resolutions[0]));

    LOG_INFO("============================================================");
    LOG_INFO("  Compare Mode — 13 combinations (5x TEXT + 4x IO + 4x MM)");
    LOG_INFO("  Warmup: %d, Iterations: %d", num_warmup, num_iter);
    LOG_INFO("============================================================");

    int ret = 0;

    // ── 1. TEXT_ONLY: multiple sequence lengths (chat-templated) ──
    {
        constexpr int64_t text_seq_lengths[] = {100, 512, 1024, 2048, 4096};
        constexpr int num_text = static_cast<int>(sizeof(text_seq_lengths) / sizeof(text_seq_lengths[0]));

        for (int ti = 0; ti < num_text; ti++) {
            int64_t seq = text_seq_lengths[ti];

            // Load chat-templated token IDs from Python (chat_tokenizer.py)
            char tok_path[256];
            std::snprintf(tok_path, sizeof(tok_path),
                          "/tmp/tokens_chat_text_only_%ld.bin", static_cast<long>(seq));
            std::vector<int64_t> input_ids = LoadTokenIds(tok_path);
            if (input_ids.empty()) {
                LOG_WARN("Chat-templated token file not found: %s — "
                         "using fallback tokens. Run 'python atb_cpp_llm/scripts/gen_compare_tokens.py' "
                         "to generate accurate token files.",
                         tok_path);
                // Fallback: construct simple token sequence of nominal seq_len
                input_ids.resize(static_cast<size_t>(seq));
                input_ids[0] = 151643;
                for (int64_t j = 1; j < seq - 1; j++)
                    input_ids[j] = 15339;
                input_ids[static_cast<size_t>(seq) - 1] = 151645;
            }
            int64_t actual_seq = static_cast<int64_t>(input_ids.size());

            LOG_INFO("------------------------------------------------------------");
            LOG_INFO("  [%d/13] TEXT_ONLY  S=%ld (chat template: %ld tokens)",
                     ti + 1, static_cast<long>(seq), static_cast<long>(actual_seq));
            LOG_INFO("------------------------------------------------------------");

            atb_llm::InferRequest request;
            request.mode = atb_llm::InputMode::TEXT_ONLY;
            request.text.input_ids = input_ids.data();
            request.text.batch_size = 1;
            request.text.seq_length = actual_seq;

            char save_path[256];
            std::snprintf(save_path, sizeof(save_path),
                          "/tmp/cpp_text_only_%ld.bin", static_cast<long>(seq));

            auto results = RunBenchmark(engine, request, num_warmup, num_iter, true,
                                         save_path);
            if (results.empty()) return 1;

            std::vector<double> e2e_times;
            for (auto& r : results) e2e_times.push_back(r.e2e_ms);
            Stats e2e_stats = ComputeStats(e2e_times);
            ReportStagesCompact(results, "text", "N/A",
                                static_cast<int>(actual_seq), 0, e2e_stats);
        }
    }

    // ── 2. IMAGE_ONLY and IMAGE_AND_TEXT for each resolution ─────
    for (int i = 0; i < num_res && ret == 0; i++) {
        int img_w = resolutions[i].w;
        int img_h = resolutions[i].h;

        LOG_INFO("============================================================");
        LOG_INFO("  Resolution %dx%d", img_w, img_h);
        LOG_INFO("============================================================");

        // Generate gradient test image (matches Python benchmark.py)
        auto pre_start = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> image = CreateGradientImage(3, img_h, img_w);

        // Compute output buffer size
        int32_t factor = config.pp_patch_size * config.pp_temporal_patch_size;
        int32_t new_h, new_w;
        atb_llm::adapters::SmartResize(img_h, img_w, factor,
                                        config.pp_min_pixels, config.pp_max_pixels,
                                        new_h, new_w);
        int64_t grid_h = new_h / config.vis_patch_size;
        int64_t grid_w = new_w / config.vis_patch_size;
        int64_t grid_t = 2;  // Qwen3VL: always 2 temporal patches
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
        auto pre_end = std::chrono::high_resolution_clock::now();
        if (s != atb_llm::STATUS_OK) {
            LOG_ERROR("PreprocessImage failed for %dx%d", img_w, img_h);
            ret = 1;
            break;
        }
        double pre_ms = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();

        // Save preprocessed pixel_values so Python --load-pixel-values
        // can consume the identical input (fixes input mismatch).
        {
            char pv_path[256];
            std::snprintf(pv_path, sizeof(pv_path),
                          "/tmp/cpp_pv_%dx%d.bin", img_w, img_h);
            SavePixelValues(pv_path, pixel_values.data(),
                            static_cast<int32_t>(actual_patches * patch_dim));
        }

        int64_t vis_tokens = actual_patches /
            (config.vis_spatial_merge_size * config.vis_spatial_merge_size);

        char resolution_str[32];
        std::snprintf(resolution_str, sizeof(resolution_str), "%dx%d", img_w, img_h);

        LOG_INFO("  SmartResize: %dx%d → %dx%d, patches=%ld, vis_tokens=%ld",
                 img_w, img_h, new_h, new_w,
                 static_cast<long>(actual_patches),
                 static_cast<long>(vis_tokens));

        // ── IMAGE_ONLY ────────────────────────────────────────────
        {
            int combo_idx = 6 + i * 2;  // 6,8,10,12
            LOG_INFO("  [%d/13] IMAGE_ONLY  %dx%d", combo_idx, img_w, img_h);

            // Load chat-templated token IDs (matches Python reference)
            char io_tok_path[256];
            std::snprintf(io_tok_path, sizeof(io_tok_path),
                          "/tmp/tokens_chat_io_%dx%d.bin", img_w, img_h);
            std::vector<int64_t> input_ids = LoadTokenIds(io_tok_path);
            if (input_ids.empty()) {
                LOG_WARN("Chat-templated token file not found: %s — "
                         "using bare image tokens. Run 'python atb_cpp_llm/scripts/gen_compare_tokens.py' "
                         "to generate accurate token files.",
                         io_tok_path);
                // Fallback to bare image tokens
                input_ids.resize(vis_tokens, image_token_id);
            }
            int64_t io_seq_len = static_cast<int64_t>(input_ids.size());

            atb_llm::InferRequest request;
            request.mode = atb_llm::InputMode::PREPROCESSED;
            request.text.input_ids = input_ids.data();
            request.text.batch_size = 1;
            request.text.seq_length = io_seq_len;
            request.preprocessed.pixel_values = pixel_values.data();
            request.preprocessed.num_patches = actual_patches;
            request.preprocessed.patch_dim = patch_dim;
            request.preprocessed.grid_thw = grid_thw;
            request.preprocessed.dtype = ACL_FLOAT16;

            char save_path[256];
            std::snprintf(save_path, sizeof(save_path),
                          "/tmp/cpp_io_%dx%d.bin", img_w, img_h);

            auto results = RunBenchmark(engine, request, num_warmup, num_iter, true,
                                         save_path);
            if (results.empty()) { ret = 1; break; }
            for (auto& r : results) {
                r.timings.preprocess_ms = pre_ms;
            }

            std::vector<double> e2e_times;
            for (auto& r : results) e2e_times.push_back(r.e2e_ms);
            Stats e2e_stats = ComputeStats(e2e_times);
            ReportStagesCompact(results, "io", resolution_str,
                                static_cast<int>(vis_tokens),
                                static_cast<int>(vis_tokens), e2e_stats);
        }

        // ── IMAGE_AND_TEXT ────────────────────────────────────────
        if (ret == 0) {
            int combo_idx = 7 + i * 2;  // 7,9,11,13
            LOG_INFO("  [%d/13] IMAGE_AND_TEXT  %dx%d", combo_idx, img_w, img_h);

            // Load chat-templated token IDs from Python (chat_tokenizer.py)
            char token_path[256];
            std::snprintf(token_path, sizeof(token_path),
                          "/tmp/tokens_chat_mm_%dx%d.bin", img_w, img_h);
            std::vector<int64_t> input_ids = LoadTokenIds(token_path);

            if (input_ids.empty()) {
                // Fallback: hardcoded construction
                LOG_WARN("Chat-templated token file not found: %s — "
                         "using hardcoded tokens. Run 'python atb_cpp_llm/scripts/gen_compare_tokens.py' "
                         "to generate accurate token files.", token_path);
                input_ids.push_back(TOK_DESCRIBE);
                for (int64_t j = 0; j < vis_tokens; j++) {
                    input_ids.push_back(image_token_id);
                }
                input_ids.push_back(TOK_THE);
                input_ids.push_back(TOK_IMAGE);
                input_ids.push_back(TOK_DOT);
            } else {
                // Verify image token count consistency
                int64_t loaded_img_tokens = 0;
                for (auto tid : input_ids) {
                    if (tid == image_token_id) loaded_img_tokens++;
                }
                if (loaded_img_tokens != vis_tokens) {
                    LOG_WARN("Loaded MM tokens have %ld image tokens but "
                                "preprocess produced %ld — possible mismatch.",
                                static_cast<long>(loaded_img_tokens),
                                static_cast<long>(vis_tokens));
                }
            }

            int64_t seq_len = static_cast<int64_t>(input_ids.size());

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

            char save_path[256];
            std::snprintf(save_path, sizeof(save_path),
                          "/tmp/cpp_mm_%dx%d.bin", img_w, img_h);

            auto results = RunBenchmark(engine, request, num_warmup, num_iter, true,
                                         save_path);
            if (results.empty()) { ret = 1; break; }
            for (auto& r : results) {
                r.timings.preprocess_ms = pre_ms;
            }

            std::vector<double> e2e_times;
            for (auto& r : results) e2e_times.push_back(r.e2e_ms);
            Stats e2e_stats = ComputeStats(e2e_times);
            ReportStagesCompact(results, "mm", resolution_str,
                                static_cast<int>(seq_len),
                                static_cast<int>(vis_tokens), e2e_stats);
        }
    }

    if (ret == 0) {
        LOG_INFO("============================================================");
        LOG_INFO("  Compare mode complete. Output files:");
        LOG_INFO("    /tmp/cpp_text_only_100.bin");
        LOG_INFO("    /tmp/cpp_text_only_512.bin");
        LOG_INFO("    /tmp/cpp_text_only_1024.bin");
        LOG_INFO("    /tmp/cpp_text_only_2048.bin");
        LOG_INFO("    /tmp/cpp_text_only_4096.bin");
        for (int i = 0; i < num_res; i++) {
            LOG_INFO("    /tmp/cpp_pv_%dx%d.bin", resolutions[i].w, resolutions[i].h);
            LOG_INFO("    /tmp/cpp_io_%dx%d.bin", resolutions[i].w, resolutions[i].h);
            LOG_INFO("    /tmp/cpp_mm_%dx%d.bin", resolutions[i].w, resolutions[i].h);
        }
        LOG_INFO("============================================================");
    }

    return ret;
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
    std::string mode = "all";

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

    // Validate model directory (was static global — now checked after LOG init)
    std::string model_dir = GetModelDir();
    if (model_dir.empty()) {
        LOG_ERROR("QWEN3VL_EMB_MODEL_DIR is not set (see stderr for details).");
        return 1;
    }

    if (!cmp_mode) {
        LOG_INFO("=== Qwen3VL Embedding Benchmark ===");
        LOG_INFO("Model: %s", model_dir.c_str());
        LOG_INFO("Mode: %s", mode.c_str());
    }

    // Engine config (shared by all factory calls)
    atb_llm::EngineConfig config;
    config.model_dir = model_dir;
    config.buffer_size = 10LL * 1024 * 1024 * 1024;
    config.device_id = 0;

    // Helper: create engine and return via unique_ptr
    auto CreateEngine = [&]() -> std::unique_ptr<atb_llm::LLMEngine> {
        std::unique_ptr<atb_llm::LLMEngine> eng;
        auto t0 = std::chrono::high_resolution_clock::now();
        atb_llm::Status st = atb_llm::LLMEngine::Create(config, eng);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (st != atb_llm::STATUS_OK || !eng) {
            LOG_ERROR("Failed to create engine: %d", static_cast<int>(st));
            return nullptr;
        }
        if (!cmp_mode) LOG_INFO("Engine load time: %ld ms", static_cast<long>(load_ms));
        return eng;
    };

    // Create engine for single modes; all/cold create engines per mode below
    std::unique_ptr<atb_llm::LLMEngine> engine;
    if (mode != "all" && mode != "cold") {
        engine = CreateEngine();
        if (!engine) return 1;
    }

    int ret = 0;
    if (mode == "text") {
        ret = RunTextBenchmark(engine.get(), seq_len, num_warmup, num_iter, cmp_mode);
    } else if (mode == "mm") {
        ret = RunMultimodalBenchmark(engine.get(), model_dir, img_w, img_h, num_warmup, num_iter, cmp_mode);
    } else if (mode == "io") {
        ret = RunImageOnlyBenchmark(engine.get(), model_dir, img_w, img_h, num_warmup, num_iter, cmp_mode);
    } else if (mode == "all") {
        std::vector<std::tuple<std::string, std::string, ColdMetrics>> cold_results;

        // ── Text mode (per-mode engine) ─────────────────────────
        {
            auto eng = CreateEngine();
            if (!eng) return 1;
            ColdMetrics cold;
            ret = RunTextBenchmark(eng.get(), seq_len, num_warmup, num_iter, cmp_mode, &cold);
            if (ret == 0 && cold.valid) {
                cold_results.push_back({"text", "seq=" + std::to_string(seq_len), cold});
            }
        }

        // ── Image-only mode (per-mode engine) ───────────────────
        if (ret == 0) {
            auto eng = CreateEngine();
            if (!eng) return 1;
            ColdMetrics cold;
            char res[32];
            std::snprintf(res, sizeof(res), "%dx%d", img_w, img_h);
            ret = RunImageOnlyBenchmark(eng.get(), model_dir, img_w, img_h,
                                        num_warmup, num_iter, cmp_mode, &cold);
            if (ret == 0 && cold.valid) {
                cold_results.push_back({"io", res, cold});
            }
        }

        // ── Multimodal mode (per-mode engine) ───────────────────
        if (ret == 0) {
            auto eng = CreateEngine();
            if (!eng) return 1;
            ColdMetrics cold;
            char res[32];
            std::snprintf(res, sizeof(res), "%dx%d", img_w, img_h);
            ret = RunMultimodalBenchmark(eng.get(), model_dir, img_w, img_h,
                                         num_warmup, num_iter, cmp_mode, &cold);
            if (ret == 0 && cold.valid) {
                cold_results.push_back({"mm", res, cold});
            }
        }

        ReportColdStart(cold_results);
    } else if (mode == "cold") {
        std::vector<std::tuple<std::string, std::string, ColdMetrics>> cold_results;

        // ── Text cold: seq=2048 ─────────────────────────────────
        {
            auto eng = CreateEngine();
            if (!eng) return 1;
            ColdMetrics cold;
            ret = RunTextBenchmark(eng.get(), 2048, num_warmup, num_iter, cmp_mode, &cold);
            if (ret == 0 && cold.valid) {
                cold_results.push_back({"text", "seq=2048", cold});
            }
        }

        // ── IO cold: 1080x1920 ──────────────────────────────────
        if (ret == 0) {
            auto eng = CreateEngine();
            if (!eng) return 1;
            ColdMetrics cold;
            ret = RunImageOnlyBenchmark(eng.get(), model_dir, 1080, 1920,
                                        num_warmup, num_iter, cmp_mode, &cold);
            if (ret == 0 && cold.valid) {
                cold_results.push_back({"io", "1080x1920", cold});
            }
        }

        // ── MM cold: 1080x1920 ──────────────────────────────────
        if (ret == 0) {
            auto eng = CreateEngine();
            if (!eng) return 1;
            ColdMetrics cold;
            ret = RunMultimodalBenchmark(eng.get(), model_dir, 1080, 1920,
                                         num_warmup, num_iter, cmp_mode, &cold);
            if (ret == 0 && cold.valid) {
                cold_results.push_back({"mm", "1080x1920", cold});
            }
        }

        ReportColdStart(cold_results);
    } else if (mode == "bench") {
        struct { int w, h; } resolutions[] = {
            {224, 224},
            {416, 672},
            {672, 416},
            {896, 896},
        };
        int num_res = static_cast<int>(sizeof(resolutions) / sizeof(resolutions[0]));
        for (int i = 0; i < num_res; i++) {
            if (!cmp_mode && i > 0) {
                LOG_INFO("============================================================");
            }
            ret = RunImageOnlyBenchmark(engine.get(), model_dir,
                                         resolutions[i].w, resolutions[i].h,
                                         num_warmup, num_iter, cmp_mode);
            if (ret != 0) break;
        }
    } else if (mode == "compare") {
        ret = RunCompareMode(engine.get(), model_dir, num_warmup, num_iter);
    } else {
        LOG_ERROR("Unknown mode: %s (use text|mm|io|all|bench|compare|cold)", mode.c_str());
        return 1;
    }

    return ret;
}
