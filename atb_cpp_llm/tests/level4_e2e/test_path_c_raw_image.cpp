/**
 * Path C end-to-end test — raw_image (in-engine NPU preprocess) vs
 * PREPROCESSED (external CPU preprocess fed via H2D).
 *
 * Same image + same text, two requests:
 *   A. PREPROCESSED: CPU PreprocessImage -> pixel_values -> Encode (H2D inside).
 *   B. raw_image:    uint8 image -> Encode (NPU preprocess inside, no H2D).
 *
 * Acceptance (path C):
 *   - final embedding cosine(raw_image, PREPROCESSED) >= 0.99
 *   - raw_image path fills StageTimings.preprocess_ms (>0)
 *   - PREPROCESSED path leaves StageTimings.preprocess_ms == 0
 *
 * Image: 96x96. SmartResize factor=32, 96/32=3.0 -> 96 (identity, non-downsample),
 * so the NPU bicubic uses the non-AA variant on 910B and matches CPU to cos>=0.99.
 * This isolates path C plumbing (device tensor feeds the vision pipeline with no
 * D2H/H2D round trip) from the AA-vs-non-AA divergence that the bicubic spike
 * test gates separately.
 */

#include "atb_llm/types.h"
#include "atb_llm/engine.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#include "test_env.h"

#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

static const std::string MODEL_DIR = GetModelDir();

// 96x96: identity SmartResize (non-downsample -> non-AA bicubic on 910B).
static constexpr int32_t IMG_H = 96;
static constexpr int32_t IMG_W = 96;
static constexpr int32_t IMG_C = 3;

// "Describe the image." token IDs (match test_accuracy.cpp).
static constexpr int64_t TOK_DESCRIBE    = 74785;
static constexpr int64_t TOK_THE         = 279;
static constexpr int64_t TOK_IMAGE       = 2168;
static constexpr int64_t TOK_DOT         = 13;
static constexpr int64_t IMAGE_TOKEN_ID  = 151655;

static std::vector<uint8_t> CreateTestImage(int32_t channels, int32_t height, int32_t width) {
    std::vector<uint8_t> image(static_cast<size_t>(channels) * height * width);
    for (int32_t c = 0; c < channels; c++) {
        for (int32_t h = 0; h < height; h++) {
            for (int32_t w = 0; w < width; w++) {
                uint8_t v = static_cast<uint8_t>(
                    (h * 255 / height + w * 255 / width + c * 85) % 256);
                image[static_cast<size_t>(c) * height * width + h * width + w] = v;
            }
        }
    }
    return image;
}

static double CosineSim(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

static std::vector<float> ResultToFp32(const atb_llm::InferResult& r) {
    int64_t dim = r.shape.empty() ? 0 : r.shape[0];
    std::vector<float> out(dim);
    if (r.dtype == ACL_FLOAT16) {
        const uint16_t* d = r.As<uint16_t>();
        for (int64_t i = 0; i < dim; i++) out[i] = atb_llm::Fp16ToF32(d[i]);
    } else if (r.dtype == ACL_FLOAT) {
        const float* d = r.As<float>();
        for (int64_t i = 0; i < dim; i++) out[i] = d[i];
    } else {
        LOG_ERROR("ResultToFp32: unexpected dtype %d", static_cast<int>(r.dtype));
    }
    return out;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    if (MODEL_DIR.empty()) {
        std::fprintf(stderr,
            "QWEN3VL_EMB_MODEL_DIR is not set. Source .env via build_and_test.sh "
            "or export the variable.\n");
        return 1;
    }

    LOG_INFO("=== Path C e2e: raw_image (NPU preprocess) vs PREPROCESSED (CPU preprocess) ===");
    LOG_INFO("Image: %dx%d (identity SmartResize -> non-AA bicubic on 910B)", IMG_H, IMG_W);

    atb_llm::EngineConfig cfg;
    cfg.model_dir = MODEL_DIR;
    cfg.buffer_size = 15LL * 1024 * 1024 * 1024;  // 15GB
    cfg.device_id = 0;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::Status s = atb_llm::LLMEngine::Create(cfg, engine);
    if (!IS_OK(s) || !engine) {
        LOG_ERROR("Failed to create engine: %d", static_cast<int>(s));
        return 1;
    }

    // Load the same config the engine uses, so the CPU preprocess (path A) and
    // the in-engine NPU preprocess (path B) share identical preprocessor params.
    atb_llm::adapters::Qwen3VLConfig pp_cfg;
    s = atb_llm::adapters::LoadQwen3VLConfig(MODEL_DIR, pp_cfg);
    if (!IS_OK(s)) {
        LOG_ERROR("LoadQwen3VLConfig failed: %d", static_cast<int>(s));
        return 1;
    }

    std::vector<uint8_t> image = CreateTestImage(IMG_C, IMG_H, IMG_W);

    // SmartResize to learn merged_tokens (shared; needed to build input_ids).
    int32_t factor = pp_cfg.pp_patch_size * pp_cfg.pp_merge_size;
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(IMG_H, IMG_W, factor,
                                   pp_cfg.pp_min_pixels, pp_cfg.pp_max_pixels,
                                   new_h, new_w);
    int64_t grid_h = new_h / pp_cfg.pp_patch_size;
    int64_t grid_w = new_w / pp_cfg.pp_patch_size;
    int64_t num_patches = grid_h * grid_w;  // grid_t=1
    int64_t merge_size = pp_cfg.pp_merge_size;
    int64_t merged_tokens = num_patches / (merge_size * merge_size);
    int64_t patch_dim = static_cast<int64_t>(IMG_C) * pp_cfg.pp_temporal_patch_size *
                        pp_cfg.pp_patch_size * pp_cfg.pp_patch_size;

    LOG_INFO("SmartResize: %dx%d -> %dx%d, grid=(1,%ld,%ld), patches=%ld, merged=%ld",
             IMG_H, IMG_W, new_h, new_w,
             static_cast<long>(grid_h), static_cast<long>(grid_w),
             static_cast<long>(num_patches), static_cast<long>(merged_tokens));

    // input_ids: [Describe] + [img_tok]*merged + [the, image, .]
    std::vector<int64_t> input_ids;
    input_ids.push_back(TOK_DESCRIBE);
    for (int64_t i = 0; i < merged_tokens; i++) input_ids.push_back(IMAGE_TOKEN_ID);
    input_ids.push_back(TOK_THE);
    input_ids.push_back(TOK_IMAGE);
    input_ids.push_back(TOK_DOT);
    int64_t seq_len = static_cast<int64_t>(input_ids.size());

    // ── Path A: PREPROCESSED (CPU preprocess -> H2D inside engine) ──
    std::vector<uint16_t> pixel_values(static_cast<size_t>(num_patches) * patch_dim, 0);
    int64_t np_a = 0;
    int64_t grid_thw_a[3] = {0, 0, 0};
    s = atb_llm::adapters::PreprocessImage(
        image.data(), IMG_C, IMG_H, IMG_W, pp_cfg,
        pixel_values.data(), np_a, grid_thw_a);
    if (!IS_OK(s)) {
        LOG_ERROR("PreprocessImage (CPU) failed: %d", static_cast<int>(s));
        return 1;
    }

    atb_llm::InferRequest reqA;
    reqA.mode = atb_llm::InputMode::PREPROCESSED;
    reqA.text.input_ids = input_ids.data();
    reqA.text.batch_size = 1;
    reqA.text.seq_length = seq_len;
    reqA.preprocessed.pixel_values = pixel_values.data();
    reqA.preprocessed.num_patches = np_a;
    reqA.preprocessed.grid_thw = grid_thw_a;
    reqA.preprocessed.dtype = ACL_FLOAT16;

    atb_llm::InferResult resA;
    atb_llm::StageTimings tA{};
    s = engine->EncodeWithTiming(reqA, resA, tA);
    if (!IS_OK(s)) {
        LOG_ERROR("Path A (PREPROCESSED) Encode failed: %d", static_cast<int>(s));
        return 1;
    }
    LOG_INFO("Path A (PREPROCESSED): preprocess_ms=%.3f e2e_ms=%.3f",
             tA.preprocess_ms, tA.e2e_ms);

    // ── Path B: raw_image (path C — NPU preprocess inside engine, no H2D) ──
    // IMAGE_ONLY vs IMAGE_AND_TEXT: ForwardWithTiming requires input_ids !=
    // null && seq_len > 0 for ALL modes (existing gate near the top of
    // ForwardWithTiming, not path-C introduced), so IMAGE_ONLY still needs a
    // non-empty text input and is behaviorally identical to IMAGE_AND_TEXT on
    // the raw_image path — both set has_raw_image and share the same vision+
    // text code. IMAGE_AND_TEXT therefore covers the raw_image mode space; no
    // separate IMAGE_ONLY case is needed (and would just duplicate this encode).
    atb_llm::InferRequest reqB;
    reqB.mode = atb_llm::InputMode::IMAGE_AND_TEXT;
    reqB.text.input_ids = input_ids.data();
    reqB.text.batch_size = 1;
    reqB.text.seq_length = seq_len;
    reqB.raw_image.data = image.data();
    reqB.raw_image.channels = IMG_C;
    reqB.raw_image.height = IMG_H;
    reqB.raw_image.width = IMG_W;

    atb_llm::InferResult resB;
    atb_llm::StageTimings tB{};
    s = engine->EncodeWithTiming(reqB, resB, tB);
    if (!IS_OK(s)) {
        LOG_ERROR("Path B (raw_image) Encode failed: %d", static_cast<int>(s));
        return 1;
    }
    LOG_INFO("Path B (raw_image):    preprocess_ms=%.3f e2e_ms=%.3f",
             tB.preprocess_ms, tB.e2e_ms);

    // ── Compare embeddings ──
    std::vector<float> embA = ResultToFp32(resA);
    std::vector<float> embB = ResultToFp32(resB);
    if (embA.size() != embB.size() || embA.empty()) {
        LOG_ERROR("Embedding size mismatch/empty: A=%zu B=%zu", embA.size(), embB.size());
        return 1;
    }
    int64_t dim = static_cast<int64_t>(embA.size());
    double cos = CosineSim(embA.data(), embB.data(), dim);

    LOG_INFO("================================================================");
    LOG_INFO("[PATH C GATE] raw_image vs PREPROCESSED embedding cos=%.6f (dim=%ld)",
             cos, static_cast<long>(dim));
    LOG_INFO("  preprocess_ms: PREPROCESSED=%.3f (expect 0) | raw_image=%.3f (expect >0)",
             tA.preprocess_ms, tB.preprocess_ms);
    LOG_INFO("================================================================");

    int failures = 0;
    if (cos < 0.99) {
        LOG_ERROR("FAIL: cosine %.6f < 0.99 (path C produced a different embedding)", cos);
        failures++;
    }
    if (tB.preprocess_ms <= 0.0) {
        LOG_ERROR("FAIL: raw_image path did not fill preprocess_ms (got %.3f)", tB.preprocess_ms);
        failures++;
    }
    if (tA.preprocess_ms != 0.0) {
        LOG_ERROR("FAIL: PREPROCESSED path should leave preprocess_ms=0 (got %.3f)", tA.preprocess_ms);
        failures++;
    }

    if (failures == 0) {
        LOG_INFO("PASS: path C (raw_image) matches PREPROCESSED at cos=%.6f >= 0.99", cos);
        return 0;
    }
    return 1;
}
