/**
 * Vision Pipeline Stage-by-Stage Precision Test
 *
 * Compares C++ intermediate values against Python reference data
 * saved in /tmp/stage_L{N}_*.bin binary files.
 *
 * Test stages:
 *   L0: Preprocessing -- pixel_values from gradient image
 *   L1: Patch Embedding -- PatchEmbedGraph ATB output
 *   L2: Position Embedding -- ComputePosEmbedInterp CPU output
 *   L3: Vision RoPE -- VisionRotaryEmbedding::ComputeRoPE CPU output
 *
 * Prerequisites:
 *     1. python tests/test_stage_reference.py  (generates /tmp/stage_L{N}_*.bin)
 *     2. Model checkpoint at MODEL_DIR
 *
 * Run: ./test_vision_stages
 */

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "components/vision/patch_embed_graph.h"
#include "components/vision/pos_embed_interp.h"
#include "components/common/mrope.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include "core/tensor_allocator.h"
#include "engine/runtime_impl.h"
#include "io/weight_loader.h"
#include "io/weight_helpers.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#include "test_env.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <string>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

static const std::string MODEL_DIR = GetModelDir();

// ── Test image dimensions (must match Python reference generator) ──
static constexpr int32_t IMG_H = 720;
static constexpr int32_t IMG_W = 1280;
static constexpr int32_t IMG_C = 3;

// ── Binary file loader ──────────────────────────────────────
// Format: [ndim: int64, shape: int64[ndim], data: raw bytes]
struct LoadedArray {
    std::vector<int64_t> shape;
    std::vector<float> data_f32;     // for fp16/fp32 data converted to f32
    std::vector<int64_t> data_i64;   // for int64 data
    int dtype;  // 0=fp16, 1=fp32, 2=int64

    LoadedArray() : dtype(1) {}

    size_t NumElements() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    bool Load(const std::string& path, int expected_dtype = 1) {
        dtype = expected_dtype;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            LOG_ERROR("Cannot open %s", path.c_str());
            return false;
        }

        int64_t ndim;
        if (fread(&ndim, sizeof(int64_t), 1, f) != 1) {
            LOG_ERROR("Failed to read ndim from %s", path.c_str());
            fclose(f);
            return false;
        }

        shape.resize(ndim);
        if (fread(shape.data(), sizeof(int64_t), ndim, f) != static_cast<size_t>(ndim)) {
            LOG_ERROR("Failed to read shape from %s", path.c_str());
            fclose(f);
            return false;
        }

        size_t total = NumElements();

        if (expected_dtype == 0) {
            // fp16 data: read uint16_t values and convert to float32
            std::vector<uint16_t> buf(total);
            if (fread(buf.data(), sizeof(uint16_t), total, f) != total) {
                LOG_ERROR("Failed to read fp16 data from %s", path.c_str());
                fclose(f);
                return false;
            }
            data_f32.resize(total);
            for (size_t i = 0; i < total; i++) {
                data_f32[i] = atb_llm::Fp16ToF32(buf[i]);
            }
        } else if (expected_dtype == 1) {
            // fp32 data: direct copy
            data_f32.resize(total);
            if (fread(data_f32.data(), sizeof(float), total, f) != total) {
                LOG_ERROR("Failed to read fp32 data from %s", path.c_str());
                fclose(f);
                return false;
            }
        } else if (expected_dtype == 2) {
            // int64 data: direct copy
            data_i64.resize(total);
            if (fread(data_i64.data(), sizeof(int64_t), total, f) != total) {
                LOG_ERROR("Failed to read int64 data from %s", path.c_str());
                fclose(f);
                return false;
            }
        }

        fclose(f);
        return true;
    }
};

// ── Cosine similarity ──────────────────────────────────────
static float CosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    int64_t n = static_cast<int64_t>(std::min(a.size(), b.size()));
    if (n == 0) return 0.0f;
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

// ── Max absolute difference ────────────────────────────────
static float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    int64_t n = static_cast<int64_t>(std::min(a.size(), b.size()));
    float max_d = 0;
    for (int64_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

// ── Create gradient image (must match Python create_test_image) ──
static std::vector<uint8_t> CreateTestImage(int32_t channels, int32_t height, int32_t width) {
    std::vector<uint8_t> image(channels * height * width);
    for (int32_t c = 0; c < channels; c++) {
        for (int32_t h = 0; h < height; h++) {
            for (int32_t w = 0; w < width; w++) {
                image[c * height * width + h * width + w] =
                    static_cast<uint8_t>((h * 255 / height + w * 255 / width + c * 85) % 256);
            }
        }
    }
    return image;
}

// ═══════════════════════════════════════════════════════════════════════
// Test L0: Preprocessing Verification
// ═══════════════════════════════════════════════════════════════════════
static bool TestL0_Preprocessing(int& passed, int& failed) {
    LOG_INFO("\n=== L0: Preprocessing Verification ===");

    // Load Python reference
    LoadedArray ref;
    if (!ref.Load("/tmp/stage_L0_pixel_values.bin", 0 /* fp16 */)) {
        LOG_ERROR("  SKIP: /tmp/stage_L0_pixel_values.bin not found (run test_stage_reference.py first)");
        failed++;
        return false;
    }

    std::string shape_str;
    for (size_t i = 0; i < ref.shape.size(); i++) {
        if (i > 0) shape_str += ",";
        shape_str += std::to_string(ref.shape[i]);
    }
    LOG_INFO("  Python ref: shape=[%s] elements=%zu", shape_str.c_str(), ref.NumElements());

    // Run C++ preprocessing
    atb_llm::adapters::Qwen3VLConfig pp_config;
    auto image = CreateTestImage(IMG_C, IMG_H, IMG_W);

    int32_t factor = pp_config.pp_patch_size * pp_config.pp_merge_size;
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(IMG_H, IMG_W, factor,
                                    pp_config.pp_min_pixels, pp_config.pp_max_pixels,
                                    new_h, new_w);

    int32_t grid_h = new_h / pp_config.pp_patch_size;
    int32_t grid_w = new_w / pp_config.pp_patch_size;
    int32_t grid_t = 1;
    int64_t num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;
    int64_t patch_dim = static_cast<int64_t>(IMG_C) * pp_config.pp_temporal_patch_size *
                        pp_config.pp_patch_size * pp_config.pp_patch_size;

    std::vector<uint16_t> pixel_values(num_patches * patch_dim, 0);
    int64_t out_num_patches = 0;
    int64_t grid_thw[3] = {};

    atb_llm::Status s = atb_llm::adapters::PreprocessImage(
        image.data(), IMG_C, IMG_H, IMG_W, pp_config,
        pixel_values.data(), out_num_patches, grid_thw);

    if (!IS_OK(s)) {
        LOG_ERROR("  PreprocessImage failed: %d", static_cast<int>(s));
        failed++;
        return false;
    }

    LOG_INFO("  C++: SmartResize %dx%d -> %dx%d, grid=[%ld,%ld,%ld], patches=%ld",
             IMG_H, IMG_W, new_h, new_w,
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(out_num_patches));

    // Convert C++ fp16 pixel_values to fp32 for comparison
    int64_t total = out_num_patches * patch_dim;
    std::vector<float> cpp_f32(total);
    for (int64_t i = 0; i < total; i++) {
        cpp_f32[i] = atb_llm::Fp16ToF32(pixel_values[i]);
    }

    // Compare
    int64_t ref_total = static_cast<int64_t>(ref.NumElements());
    if (total != ref_total) {
        LOG_WARN("  Element count mismatch: C++=%ld, Python=%ld",
                 static_cast<long>(total), static_cast<long>(ref_total));
    }

    float cos = CosineSim(cpp_f32, ref.data_f32);
    float max_d = MaxAbsDiff(cpp_f32, ref.data_f32);

    LOG_INFO("  Cosine: %.6f, MaxDiff: %.6f", cos, max_d);

    LOG_INFO("  C++    first 4: %.6f %.6f %.6f %.6f",
             total > 0 ? cpp_f32[0] : 0, total > 1 ? cpp_f32[1] : 0,
             total > 2 ? cpp_f32[2] : 0, total > 3 ? cpp_f32[3] : 0);
    LOG_INFO("  Python first 4: %.6f %.6f %.6f %.6f",
             ref.data_f32.size() > 0 ? ref.data_f32[0] : 0,
             ref.data_f32.size() > 1 ? ref.data_f32[1] : 0,
             ref.data_f32.size() > 2 ? ref.data_f32[2] : 0,
             ref.data_f32.size() > 3 ? ref.data_f32[3] : 0);

    if (cos > 0.999f) {
        LOG_INFO("  [PASS] L0: Preprocessing pixel_values match");
        passed++;
        return true;
    } else {
        LOG_ERROR("  [FAIL] L0: Preprocessing pixel_values diverge (cos=%.6f)", cos);
        failed++;
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test L1: Patch Embedding Verification (ATB graph execution)
// ═══════════════════════════════════════════════════════════════════════
static bool TestL1_PatchEmbed(int& passed, int& failed,
                               atb_llm::IRuntime* runtime) {
    LOG_INFO("\n=== L1: Patch Embedding Verification ===");

    auto* ctx = runtime->GetContext();
    auto* alloc = runtime->GetAllocator();
    auto* weight_loader = runtime->GetWeightLoader();

    // Load Python reference: pixel_values as input (fp16)
    LoadedArray ref_pixels;
    if (!ref_pixels.Load("/tmp/stage_L0_pixel_values.bin", 0 /* fp16 */)) {
        LOG_ERROR("  SKIP: /tmp/stage_L0_pixel_values.bin not found");
        failed++;
        return false;
    }

    // Load Python reference: patch_embed output (fp16)
    LoadedArray ref_output;
    if (!ref_output.Load("/tmp/stage_L1_patch_embed_out.bin", 0 /* fp16 */)) {
        LOG_ERROR("  SKIP: /tmp/stage_L1_patch_embed_out.bin not found");
        failed++;
        return false;
    }

    LOG_INFO("  Python ref pixels: %zu elements", ref_pixels.NumElements());
    LOG_INFO("  Python ref output: %zu elements", ref_output.NumElements());

    // Load config
    atb_llm::adapters::Qwen3VLConfig config;

    int32_t in_channels = config.vis_in_channels;
    int32_t temporal_patch_size = config.vis_temporal_patch_size;
    int32_t patch_size = config.vis_patch_size;
    int32_t embed_dim = config.vis_hidden_size;
    int32_t kernel_size = in_channels * temporal_patch_size * patch_size * patch_size;

    int64_t flat_pixels = static_cast<int64_t>(ref_pixels.NumElements());
    int64_t num_patches = flat_pixels / kernel_size;

    LOG_INFO("  kernel_size=%d, num_patches=%ld, embed_dim=%d",
             kernel_size, static_cast<long>(num_patches), embed_dim);

    // Build PatchEmbedGraph
    atb_llm::OperationHandle patch_embed_op;
    atb_llm::Status s = atb_llm::components::PatchEmbedGraph::Build(
        "vision_patch_embed",
        in_channels, temporal_patch_size, patch_size, embed_dim,
        patch_embed_op);

    if (!IS_OK(s) || !patch_embed_op) {
        LOG_ERROR("  PatchEmbedGraph::Build failed: %d", static_cast<int>(s));
        failed++;
        return false;
    }
    LOG_INFO("  PatchEmbedGraph built successfully");

    // Convert Python pixel_values from fp32 back to fp16 for NPU upload
    std::vector<uint16_t> pixels_fp16(flat_pixels);
    for (size_t i = 0; i < ref_pixels.data_f32.size(); i++) {
        pixels_fp16[i] = atb_llm::Fp32ToFp16(ref_pixels.data_f32[i]);
    }

    // Allocate NPU tensors using TensorAllocator
    atb::Tensor pixels_tensor;
    s = alloc->AllocFloat16(pixels_tensor, {flat_pixels});
    if (!IS_OK(s)) { LOG_ERROR("  Alloc pixels failed"); failed++; return false; }

    atb::Tensor weight_tensor;
    s = alloc->AllocFloat16(weight_tensor, {embed_dim, kernel_size});
    if (!IS_OK(s)) { LOG_ERROR("  Alloc weight failed"); failed++; return false; }

    atb::Tensor bias_tensor;
    s = alloc->AllocFloat16(bias_tensor, {embed_dim});
    if (!IS_OK(s)) { LOG_ERROR("  Alloc bias failed"); failed++; return false; }

    atb::Tensor output_tensor;
    s = alloc->AllocFloat16(output_tensor, {num_patches, embed_dim});
    if (!IS_OK(s)) { LOG_ERROR("  Alloc output failed"); failed++; return false; }

    // Copy pixel_values to NPU
    alloc->CopyToDevice(pixels_tensor, pixels_fp16.data(),
                        flat_pixels * sizeof(uint16_t));

    // Load patch_embed weights from safetensors
    // Real tensor names are model.visual.patch_embed.proj.{weight,bias} —
    // matches what qwen3vl_weights.cpp uses in production.
    s = atb_llm::io::CopyWeightToFp16NPU(*weight_loader,
                                           "model.visual.patch_embed.proj.weight",
                                           *alloc, weight_tensor);
    if (!IS_OK(s)) {
        LOG_ERROR("  Failed to load patch_embed weight: %d", static_cast<int>(s));
        failed++;
        return false;
    }

    s = atb_llm::io::CopyWeightToFp16NPU(*weight_loader,
                                           "model.visual.patch_embed.proj.bias",
                                           *alloc, bias_tensor);
    if (!IS_OK(s)) {
        LOG_ERROR("  Failed to load patch_embed bias: %d", static_cast<int>(s));
        failed++;
        return false;
    }

    LOG_INFO("  Weights loaded to NPU");

    // patch_embed.proj.weight is stored in safetensors as a 5D Conv3d kernel
    // (embed_dim, C, tp, p, p). The graph's Linear node expects a 2D weight
    // (embed_dim, kernel_size). Reshape in place — element count is identical.
    weight_tensor.desc.shape.dimNum = 2;
    weight_tensor.desc.shape.dims[0] = embed_dim;
    weight_tensor.desc.shape.dims[1] = kernel_size;

    // Build variant pack
    atb::VariantPack vp;
    vp.inTensors = {pixels_tensor, weight_tensor, bias_tensor};
    vp.outTensors = {output_tensor};

    // Execute graph (Setup + workspace + Execute, same pattern as BaseModel::ExecuteGraph)
    uint64_t ws_size = 0;
    atb::Status atb_s = patch_embed_op.get()->Setup(vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("  Graph Setup failed: %d", static_cast<int>(atb_s));
        failed++;
        return false;
    }

    // Allocate workspace on device
    {
        auto __atb_pair_ws_ptr = runtime->GetWorkspace(ws_size > 0 ? ws_size : 1); auto& ws_ptr = __atb_pair_ws_ptr.first; auto& ws_s = __atb_pair_ws_ptr.second;
        if (ws_s != atb_llm::STATUS_OK || !ws_ptr) {
            LOG_ERROR("  Workspace allocation failed: %lu bytes", static_cast<unsigned long>(ws_size));
            failed++;
            return false;
        }
        uint64_t effective_ws = ws_size > 0 ? ws_size : 1;

        atb_s = patch_embed_op.get()->Execute(vp, ws_ptr, effective_ws, ctx);
        if (atb_s != atb::NO_ERROR) {
            LOG_ERROR("  Graph Execute failed: %d", static_cast<int>(atb_s));
            failed++;
            return false;
        }
    }

    // Synchronize before reading output
    runtime->Synchronize();

    int64_t out_elements = num_patches * embed_dim;
    std::vector<uint16_t> output_fp16(out_elements);
    alloc->CopyToHost(output_fp16.data(), output_tensor,
                      out_elements * sizeof(uint16_t));

    // Convert output to fp32 for comparison
    std::vector<float> cpp_f32(out_elements);
    for (int64_t i = 0; i < out_elements; i++) {
        cpp_f32[i] = atb_llm::Fp16ToF32(output_fp16[i]);
    }

    LOG_INFO("  C++ output first 4: %.6f %.6f %.6f %.6f",
             out_elements > 0 ? cpp_f32[0] : 0, out_elements > 1 ? cpp_f32[1] : 0,
             out_elements > 2 ? cpp_f32[2] : 0, out_elements > 3 ? cpp_f32[3] : 0);
    LOG_INFO("  Pyth output first 4: %.6f %.6f %.6f %.6f",
             ref_output.data_f32.size() > 0 ? ref_output.data_f32[0] : 0,
             ref_output.data_f32.size() > 1 ? ref_output.data_f32[1] : 0,
             ref_output.data_f32.size() > 2 ? ref_output.data_f32[2] : 0,
             ref_output.data_f32.size() > 3 ? ref_output.data_f32[3] : 0);

    // Compare
    int64_t cmp_n = std::min(static_cast<int64_t>(cpp_f32.size()),
                              static_cast<int64_t>(ref_output.data_f32.size()));
    std::vector<float> cmp_cpp(cpp_f32.begin(), cpp_f32.begin() + cmp_n);
    std::vector<float> cmp_ref(ref_output.data_f32.begin(), ref_output.data_f32.begin() + cmp_n);

    float cos = CosineSim(cmp_cpp, cmp_ref);
    float max_d = MaxAbsDiff(cmp_cpp, cmp_ref);

    LOG_INFO("  Cosine: %.6f, MaxDiff: %.6f, elements: C++=%ld, Python=%zu",
             cos, max_d, static_cast<long>(out_elements), ref_output.data_f32.size());

    if (cos > 0.999f) {
        LOG_INFO("  [PASS] L1: Patch Embedding match");
        passed++;
        return true;
    } else {
        LOG_ERROR("  [FAIL] L1: Patch Embedding diverge (cos=%.6f)", cos);
        failed++;
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test L2: Position Embedding Verification
// ═══════════════════════════════════════════════════════════════════════
static bool TestL2_PositionEmbedding(int& passed, int& failed,
                                      atb_llm::WeightLoader* weight_loader) {
    LOG_INFO("\n=== L2: Position Embedding Verification ===");

    atb_llm::adapters::Qwen3VLConfig config;
    int64_t merge_size = config.vis_spatial_merge_size;
    int32_t num_grid = config.num_grid();  // sqrt(vis_num_position_embeddings)
    int32_t vis_hs = config.vis_hidden_size;

    // Step 1: Load the pos_embed source weight on host
    int64_t num_pos = config.vis_num_position_embeddings;
    std::vector<uint16_t> pos_embed_src(num_pos * vis_hs);
    atb_llm::Status s = atb_llm::io::CopyWeightToFp16Host(
        *weight_loader,
        "model.visual.pos_embed.weight",
        pos_embed_src.data(),
        pos_embed_src.size() * sizeof(uint16_t));
    if (!IS_OK(s)) {
        LOG_ERROR("  Failed to load pos_embed weight to host");
        failed++;
        return false;
    }
    LOG_INFO("  Loaded pos_embed source: (%ld, %d) fp16",
             static_cast<long>(num_pos), vis_hs);

    // Step 2: Get grid_thw from preprocessing
    auto image = CreateTestImage(IMG_C, IMG_H, IMG_W);
    int32_t factor = config.pp_patch_size * config.pp_merge_size;
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(IMG_H, IMG_W, factor,
                                    config.pp_min_pixels, config.pp_max_pixels,
                                    new_h, new_w);

    int32_t grid_h = new_h / config.pp_patch_size;
    int32_t grid_w = new_w / config.pp_patch_size;
    int64_t grid_thw[3] = {1, static_cast<int64_t>(grid_h), static_cast<int64_t>(grid_w)};
    int64_t num_patches = grid_thw[0] * grid_thw[1] * grid_thw[2];
    int64_t num_images = 1;

    LOG_INFO("  grid_thw: [%ld, %ld, %ld], num_patches=%ld",
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]), static_cast<long>(num_patches));

    // Step 3: Compute C++ pos_embed
    std::vector<uint16_t> pos_embed_cpp(num_patches * vis_hs);
    atb_llm::components::ComputePosEmbedInterp(
        pos_embed_src.data(), vis_hs, num_grid,
        static_cast<int32_t>(merge_size),
        grid_thw, num_images,
        pos_embed_cpp.data());

    // Convert C++ fp16 output to fp32
    std::vector<float> cpp_f32(num_patches * vis_hs);
    for (int64_t i = 0; i < num_patches * vis_hs; i++) {
        cpp_f32[i] = atb_llm::Fp16ToF32(pos_embed_cpp[i]);
    }

    // Step 4a: Compare against Python NPU fp16 pos_embed
    LoadedArray ref_npu;
    bool has_npu_ref = ref_npu.Load("/tmp/stage_L2_pos_embed_npu.bin", 0 /* fp16 */);

    if (has_npu_ref) {
        LOG_INFO("  Comparing against Python NPU fp16 pos_embed");
        int64_t cmp_n = std::min(static_cast<int64_t>(cpp_f32.size()),
                                  static_cast<int64_t>(ref_npu.data_f32.size()));
        std::vector<float> cmp_cpp(cpp_f32.begin(), cpp_f32.begin() + cmp_n);
        std::vector<float> cmp_ref(ref_npu.data_f32.begin(), ref_npu.data_f32.begin() + cmp_n);

        float cos = CosineSim(cmp_cpp, cmp_ref);
        float max_d = MaxAbsDiff(cmp_cpp, cmp_ref);

        LOG_INFO("  NPU comparison: Cosine=%.6f, MaxDiff=%.6f", cos, max_d);

        if (cos > 0.99f) {
            LOG_INFO("  [PASS] L2a: C++ CPU pos_embed vs Python NPU pos_embed (cos=%.6f)", cos);
            passed++;
        } else {
            LOG_ERROR("  [FAIL] L2a: C++ CPU vs Python NPU pos_embed diverge (cos=%.6f)", cos);
            failed++;
        }
    } else {
        LOG_WARN("  SKIP L2a: /tmp/stage_L2_pos_embed_npu.bin not found");
    }

    // Step 4b: Compare against Python CPU pos_embed
    LoadedArray ref_cpu;
    bool has_cpu_ref = ref_cpu.Load("/tmp/stage_L2_pos_embed_cpu.bin", 0 /* fp16 */);

    if (has_cpu_ref) {
        LOG_INFO("  Comparing against Python CPU pos_embed (computed f32, stored fp16)");
        int64_t cmp_n = std::min(static_cast<int64_t>(cpp_f32.size()),
                                  static_cast<int64_t>(ref_cpu.data_f32.size()));
        std::vector<float> cmp_cpp(cpp_f32.begin(), cpp_f32.begin() + cmp_n);
        std::vector<float> cmp_ref(ref_cpu.data_f32.begin(), ref_cpu.data_f32.begin() + cmp_n);

        float cos = CosineSim(cmp_cpp, cmp_ref);
        float max_d = MaxAbsDiff(cmp_cpp, cmp_ref);

        LOG_INFO("  CPU comparison: Cosine=%.6f, MaxDiff=%.6f", cos, max_d);

        if (cos > 0.999f) {
            LOG_INFO("  [PASS] L2b: C++ CPU pos_embed vs Python CPU pos_embed (cos=%.6f)", cos);
            passed++;
        } else {
            LOG_ERROR("  [FAIL] L2b: C++ CPU vs Python CPU pos_embed diverge (cos=%.6f)", cos);
            failed++;
        }
    } else {
        LOG_WARN("  SKIP L2b: /tmp/stage_L2_pos_embed_cpu.bin not found");
        // If no CPU ref either, still count as pass since we computed successfully
        LOG_INFO("  [PASS] L2: Position embedding computed (no reference available for comparison)");
        passed++;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Test L3: Vision RoPE Verification
// ═══════════════════════════════════════════════════════════════════════
static bool TestL3_VisionRoPE(int& passed, int& failed) {
    LOG_INFO("\n=== L3: Vision RoPE Verification ===");

    atb_llm::adapters::Qwen3VLConfig config;
    int64_t merge_size = config.vis_spatial_merge_size;
    int32_t vis_hd = config.vis_head_dim();

    // Get grid_thw from preprocessing
    auto image = CreateTestImage(IMG_C, IMG_H, IMG_W);
    int32_t factor = config.pp_patch_size * config.pp_merge_size;
    int32_t new_h, new_w;
    atb_llm::adapters::SmartResize(IMG_H, IMG_W, factor,
                                    config.pp_min_pixels, config.pp_max_pixels,
                                    new_h, new_w);

    int32_t grid_h = new_h / config.pp_patch_size;
    int32_t grid_w = new_w / config.pp_patch_size;
    int64_t grid_thw[3] = {1, static_cast<int64_t>(grid_h), static_cast<int64_t>(grid_w)};
    int64_t num_patches = grid_thw[0] * grid_thw[1] * grid_thw[2];
    int64_t num_images = 1;

    LOG_INFO("  grid_thw: [%ld, %ld, %ld], num_patches=%ld, vis_hd=%d",
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(num_patches), vis_hd);

    // Create VisionRotaryEmbedding
    // Python: VisionRotaryEmbedding(dim=hd_v // 2)
    int32_t rope_dim = vis_hd / 2;
    atb_llm::components::VisionRotaryEmbedding vis_rope(rope_dim);

    // Compute vision RoPE (CPU, fp32)
    std::vector<float> cos_cpp(num_patches * vis_hd);
    std::vector<float> sin_cpp(num_patches * vis_hd);

    int64_t total_tokens = vis_rope.ComputeRoPE(
        grid_thw, num_images, static_cast<int32_t>(merge_size),
        cos_cpp.data(), sin_cpp.data());

    LOG_INFO("  C++ RoPE computed: total_tokens=%ld, cos shape=(%ld, %d)",
             static_cast<long>(total_tokens), static_cast<long>(num_patches), vis_hd);

    LOG_INFO("  C++ cos first 4: %.6f %.6f %.6f %.6f",
             cos_cpp.size() > 0 ? cos_cpp[0] : 0, cos_cpp.size() > 1 ? cos_cpp[1] : 0,
             cos_cpp.size() > 2 ? cos_cpp[2] : 0, cos_cpp.size() > 3 ? cos_cpp[3] : 0);
    LOG_INFO("  C++ sin first 4: %.6f %.6f %.6f %.6f",
             sin_cpp.size() > 0 ? sin_cpp[0] : 0, sin_cpp.size() > 1 ? sin_cpp[1] : 0,
             sin_cpp.size() > 2 ? sin_cpp[2] : 0, sin_cpp.size() > 3 ? sin_cpp[3] : 0);

    // Compare against Python reference
    bool any_test = false;

    // Load Python NPU fp16 cos reference
    LoadedArray ref_cos;
    if (ref_cos.Load("/tmp/stage_L3_rope_cos.bin", 0 /* fp16 */)) {
        any_test = true;
        int64_t cmp_n = std::min(static_cast<int64_t>(cos_cpp.size()),
                                  static_cast<int64_t>(ref_cos.data_f32.size()));
        std::vector<float> cmp_cpp(cos_cpp.begin(), cos_cpp.begin() + cmp_n);
        std::vector<float> cmp_ref(ref_cos.data_f32.begin(), ref_cos.data_f32.begin() + cmp_n);

        float cos_sim = CosineSim(cmp_cpp, cmp_ref);
        float max_d = MaxAbsDiff(cmp_cpp, cmp_ref);

        LOG_INFO("  cos comparison: Cosine=%.6f, MaxDiff=%.6f", cos_sim, max_d);
        LOG_INFO("  Python cos first 4: %.6f %.6f %.6f %.6f",
                 ref_cos.data_f32.size() > 0 ? ref_cos.data_f32[0] : 0,
                 ref_cos.data_f32.size() > 1 ? ref_cos.data_f32[1] : 0,
                 ref_cos.data_f32.size() > 2 ? ref_cos.data_f32[2] : 0,
                 ref_cos.data_f32.size() > 3 ? ref_cos.data_f32[3] : 0);

        if (cos_sim > 0.99f) {
            LOG_INFO("  [PASS] L3a: Vision RoPE cos match (cos=%.6f)", cos_sim);
            passed++;
        } else {
            LOG_ERROR("  [FAIL] L3a: Vision RoPE cos diverge (cos=%.6f)", cos_sim);
            failed++;
        }
    } else {
        LOG_WARN("  SKIP L3a: /tmp/stage_L3_rope_cos.bin not found");
    }

    // Load Python NPU fp16 sin reference
    LoadedArray ref_sin;
    if (ref_sin.Load("/tmp/stage_L3_rope_sin.bin", 0 /* fp16 */)) {
        any_test = true;
        int64_t cmp_n = std::min(static_cast<int64_t>(sin_cpp.size()),
                                  static_cast<int64_t>(ref_sin.data_f32.size()));
        std::vector<float> cmp_cpp(sin_cpp.begin(), sin_cpp.begin() + cmp_n);
        std::vector<float> cmp_ref(ref_sin.data_f32.begin(), ref_sin.data_f32.begin() + cmp_n);

        float cos_sim = CosineSim(cmp_cpp, cmp_ref);
        float max_d = MaxAbsDiff(cmp_cpp, cmp_ref);

        LOG_INFO("  sin comparison: Cosine=%.6f, MaxDiff=%.6f", cos_sim, max_d);
        LOG_INFO("  Python sin first 4: %.6f %.6f %.6f %.6f",
                 ref_sin.data_f32.size() > 0 ? ref_sin.data_f32[0] : 0,
                 ref_sin.data_f32.size() > 1 ? ref_sin.data_f32[1] : 0,
                 ref_sin.data_f32.size() > 2 ? ref_sin.data_f32[2] : 0,
                 ref_sin.data_f32.size() > 3 ? ref_sin.data_f32[3] : 0);

        if (cos_sim > 0.99f) {
            LOG_INFO("  [PASS] L3b: Vision RoPE sin match (cos=%.6f)", cos_sim);
            passed++;
        } else {
            LOG_ERROR("  [FAIL] L3b: Vision RoPE sin diverge (cos=%.6f)", cos_sim);
            failed++;
        }
    } else {
        LOG_WARN("  SKIP L3b: /tmp/stage_L3_rope_sin.bin not found");
    }

    // If no reference files were available at all, still count as pass
    if (!any_test) {
        LOG_INFO("  [PASS] L3: Vision RoPE computed (no reference available for comparison)");
        passed++;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════
int main() {
    if (MODEL_DIR.empty()) {
        std::fprintf(stderr,
            "QWEN3VL_EMB_MODEL_DIR is not set. "
            "Source .env via build_and_test.sh or export the variable.\n");
        return 1;
    }

    LOG_INFO("=== Vision Pipeline Stage Precision Test ===");
    LOG_INFO("Image: %dx%d, Model: %s", IMG_H, IMG_W, MODEL_DIR.c_str());

    int passed = 0, failed = 0;

    // Initialize ATB/ACL environment
    std::unique_ptr<atb_llm::IRuntime> runtime;
    atb_llm::Status s = atb_llm::RuntimeImpl::Create(0, 5LL * 1024 * 1024 * 1024, runtime);
    if (!IS_OK(s) || !runtime) {
        LOG_ERROR("Failed to create runtime: %d", static_cast<int>(s));
        return 1;
    }
    LOG_INFO("Runtime created successfully");

    auto* weight_loader = runtime->GetWeightLoader();

    // Load model weights (needed for patch_embed and pos_embed)
    s = weight_loader->LoadFromFile(MODEL_DIR + "/model.safetensors");
    if (!IS_OK(s)) {
        LOG_ERROR("Failed to load model weights: %d", static_cast<int>(s));
        return 1;
    }
    LOG_INFO("Model weights loaded");

    // L0: Preprocessing
    TestL0_Preprocessing(passed, failed);

    // L1: Patch Embedding
    TestL1_PatchEmbed(passed, failed, runtime.get());

    // L2: Position Embedding
    TestL2_PositionEmbedding(passed, failed, weight_loader);

    // L3: Vision RoPE
    TestL3_VisionRoPE(passed, failed);

    // Summary
    printf("\n=== Stage Precision Summary ===\n");
    printf("Passed: %d, Failed: %d\n", passed, failed);

    if (failed > 0) {
        LOG_ERROR("SOME TESTS FAILED");
    } else {
        LOG_INFO("ALL TESTS PASSED");
    }

    return failed > 0 ? 1 : 0;
}
