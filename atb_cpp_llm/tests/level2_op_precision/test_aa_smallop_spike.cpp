/**
 * 310P AA bicubic small-op spike (Batch A, single resolution gate).
 *
 * PURPOSE:
 *   aclnnUpsampleBicubic2dAA is unsupported on Atlas Inference Series
 *   (aclnnStatus=561103 — see docs/STATUS.md:67), so 310P cannot reach
 *   cos >= 0.99 vs official PIL via the fused aclnn op. This spike validates
 *   that ATB Linear + Transpose stacked into a PIL-equivalent separable
 *   bicubic AA path can hit the precision bar on the worst-case
 *   production resolution (1080x1920 -> 992x1792 double-axis downsample,
 *   the regime where AA matters most).
 *
 *   Batch A is single-resolution algorithm validation — Batch B expands
 *   to all 4 production resolutions plus the full preprocess pipeline,
 *   Batch C engineers the dispatch change in qwen3vl_preprocess.cpp. This
 *   file only contains Batch A's gate.
 *
 * Inputs:
 *   /tmp/bicubic_prod_1080x1920_input.bin       — float32 [3,1080,1920] (uint8-quantized)
 *   /tmp/bicubic_prod_1080x1920_pil_output.bin  — float32 [3,992,1792]  (PIL bit-exact target)
 *
 * Both bins are produced by tests/python_reference/gen_cpu_reference.py
 * (--stage bicubic_preprocess); see CMakeLists REFDATA_DEPENDENT_TESTS.
 *
 * Gate: cos >= 0.99 against the PIL output. Logs cos + max_abs_diff.
 *
 * Run: ./build/test_aa_smallop_spike
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/runtime.h"
#include "atb_llm/types.h"
#include "atb_llm/operation_handle.h"
#include "components/vision/smallop_bicubic_aa.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "core/tensor_allocator.h"
#include "families/base_model.h"   // ExecuteOperation
#include "ops/elewise_op.h"
#include "ops/transpose_op.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include "acl/acl.h"
#include "atb/atb_infer.h"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

namespace {

struct LoadedArrayF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        int64_t ndim;
        if (fread(&ndim, sizeof(int64_t), 1, f) != 1) { fclose(f); return false; }
        shape.resize(ndim);
        int64_t total = 1;
        for (int64_t i = 0; i < ndim; i++) {
            if (fread(&shape[i], sizeof(int64_t), 1, f) != 1) { fclose(f); return false; }
            total *= shape[i];
        }
        data.resize(total);
        if (fread(data.data(), sizeof(float), total, f) != static_cast<size_t>(total)) {
            fclose(f); return false;
        }
        fclose(f);
        return true;
    }
};

double CosineSim(const float* a, const float* b, int64_t n) {
    if (n <= 0) return 0.0;
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

float MaxAbsDiff(const float* a, const float* b, int64_t n) {
    float m = 0;
    for (int64_t i = 0; i < n; i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Manually-strung NPU preprocess pipeline using NpuBicubicResizeAASmallOp in
// place of the aclnn bicubic op. Mirrors PreprocessImageNpuInternal
// (qwen3vl_preprocess.cpp:369-729) step-for-step — SmartResize(CPU) → H2D fp16
// → small-op AA resize → normalize (3 Elewise) → AsStrided + 8-D Transpose
// patchify → D2H — but injects the small-op resize so TC3 validates the
// small-op path end-to-end WITHOUT touching the production dispatch (Batch C).
// All device intermediates are freed before return; pixel_values_out receives
// the host-side patchified fp16 output.
atb_llm::Status RunNpuPipelineSmallOpAA(
        atb_llm::IRuntime* runtime,
        const uint8_t* image, int32_t channels, int32_t height, int32_t width,
        const atb_llm::adapters::Qwen3VLConfig& cfg,
        std::vector<uint16_t>& pixel_values_out,
        int64_t& num_patches, int64_t* grid_thw) {
    using namespace atb_llm;
    auto* alloc = runtime->GetAllocator();

    const int32_t patch_size = cfg.pp_patch_size;
    const int32_t tp         = cfg.pp_temporal_patch_size;
    const int32_t merge_size = cfg.pp_merge_size;
    const int32_t factor     = patch_size * merge_size;

    int32_t new_h = 0, new_w = 0;
    adapters::SmartResize(height, width, factor,
                          cfg.pp_min_pixels, cfg.pp_max_pixels, new_h, new_w);

    // Host uint8 -> fp16 (matches preprocess.cpp step 2).
    const int64_t in_elems = static_cast<int64_t>(channels) * height * width;
    std::vector<uint16_t> in_fp16(in_elems);
    for (int64_t i = 0; i < in_elems; i++) {
        in_fp16[i] = Fp32ToFp16(static_cast<float>(image[i]));
    }

    // All device handles declared up front for goto-cleanup discipline.
    atb::Tensor input_t{}, resized_t{}, normalize_tmp{}, mean_bc{}, inv_std_bc{},
                asstrided_out{}, transpose_out{};
    Status ret = STATUS_OK;

    ret = alloc->AllocFloat16(input_t, {1, channels, height, width});
    if (ret != STATUS_OK) goto cleanup;
    ret = alloc->CopyToDevice(input_t, in_fp16.data(), in_elems * sizeof(uint16_t));
    if (ret != STATUS_OK) goto cleanup;

    ret = alloc->AllocFloat16(resized_t, {1, channels, new_h, new_w});
    if (ret != STATUS_OK) goto cleanup;

    // Step 3: small-op AA bicubic (the path under test).
    {
        aclError ae = NpuBicubicResizeAASmallOp(
            input_t.deviceData, height, width, channels, new_h, new_w,
            runtime, resized_t.deviceData);
        if (ae != ACL_SUCCESS) { ret = ERROR_INFERENCE; goto cleanup; }
    }
    alloc->Free(input_t);

    {
        // Step 4: normalize (x/255 - mean) / std via 3 broadcast Elewise ops.
        float mean[3] = {0.5f, 0.5f, 0.5f};
        float std_val[3] = {0.5f, 0.5f, 0.5f};
        if (cfg.pp_image_mean.size() >= 3)
            for (int c = 0; c < 3; c++) mean[c] = cfg.pp_image_mean[c];
        if (cfg.pp_image_std.size() >= 3)
            for (int c = 0; c < 3; c++) std_val[c] = cfg.pp_image_std[c];

        ret = alloc->AllocFloat16(normalize_tmp, {1, channels, new_h, new_w});
        if (ret != STATUS_OK) goto cleanup;

        std::vector<uint16_t> mean_neg(static_cast<size_t>(channels));
        std::vector<uint16_t> inv_std(static_cast<size_t>(channels));
        for (int32_t c = 0; c < channels; c++) {
            mean_neg[c] = Fp32ToFp16(-mean[c]);
            inv_std[c]  = Fp32ToFp16(1.0f / std_val[c]);
        }
        ret = alloc->AllocFloat16(mean_bc, {1, channels, 1, 1});
        if (ret != STATUS_OK) goto cleanup;
        ret = alloc->AllocFloat16(inv_std_bc, {1, channels, 1, 1});
        if (ret != STATUS_OK) goto cleanup;
        ret = alloc->CopyToDevice(mean_bc, mean_neg.data(), mean_neg.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) goto cleanup;
        ret = alloc->CopyToDevice(inv_std_bc, inv_std.data(), inv_std.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) goto cleanup;

        // 4a. MULS(1/255): resized -> normalize_tmp
        {
            atb::VariantPack vp;
            vp.inTensors = {resized_t};
            vp.outTensors = {normalize_tmp};
            OperationHandle op = ops::ElewiseOp::MakeMuls(1.0f / 255.0f);
            uint64_t ws = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws);
            if (ret != STATUS_OK) goto cleanup;
        }
        // 4b. ADD(-mean broadcast): normalize_tmp + mean_bc -> resized_t
        {
            atb::VariantPack vp;
            vp.inTensors = {normalize_tmp, mean_bc};
            vp.outTensors = {resized_t};
            OperationHandle op = ops::ElewiseOp::MakeAdd();
            uint64_t ws = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws);
            if (ret != STATUS_OK) goto cleanup;
        }
        // 4c. MUL(1/std broadcast): resized_t * inv_std_bc -> normalize_tmp
        {
            atb::VariantPack vp;
            vp.inTensors = {resized_t, inv_std_bc};
            vp.outTensors = {normalize_tmp};
            OperationHandle op = ops::ElewiseOp::MakeMul();
            uint64_t ws = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws);
            if (ret != STATUS_OK) goto cleanup;
        }

        ret = runtime->Synchronize();
        if (ret != STATUS_OK) goto cleanup;
        alloc->Free(mean_bc);
        alloc->Free(inv_std_bc);
        alloc->Free(resized_t);

        // Step 5: AsStrided + 8-D Transpose patchify (perm {2,5,3,6,1,0,4,7}).
        const int32_t grid_t = 1;
        const int32_t grid_h = new_h / patch_size;
        const int32_t grid_w = new_w / patch_size;
        const int64_t merged_h = grid_h / merge_size;
        const int64_t merged_w = grid_w / merge_size;
        grid_thw[0] = grid_t;
        grid_thw[1] = grid_h;
        grid_thw[2] = grid_w;
        num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;
        const int64_t patch_dim =
            static_cast<int64_t>(channels) * tp * patch_size * patch_size;

        const int64_t ms = merge_size;
        const int64_t ps = patch_size;
        std::vector<int64_t> shape7 = {channels, merged_h, ms, ps, merged_w, ms, ps};
        std::vector<int64_t> shape8 = {tp, channels, merged_h, ms, ps, merged_w, ms, ps};
        std::vector<int64_t> stride7(7);
        {
            int64_t s = 1;
            for (int i = 6; i >= 0; --i) { stride7[i] = s; s *= shape7[i]; }
        }
        std::vector<int64_t> stride8;
        stride8.push_back(0);
        for (int64_t s7 : stride7) stride8.push_back(s7);

        // 5a. AsStrided
        {
            atb::infer::AsStridedParam param;
            for (int64_t s : shape8)  param.size.push_back(s);
            for (int64_t s : stride8) param.stride.push_back(s);
            param.offset.push_back(0);
            atb::Operation* raw = nullptr;
            atb::Status as = atb::CreateOperation(param, &raw);
            if (as != atb::NO_ERROR || raw == nullptr) { ret = ERROR_INFERENCE; goto cleanup; }
            OperationHandle op(raw);
            ret = alloc->AllocFloat16(asstrided_out, shape8);
            if (ret != STATUS_OK) goto cleanup;
            atb::VariantPack vp;
            vp.inTensors  = {normalize_tmp};
            vp.outTensors = {asstrided_out};
            uint64_t ws = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws);
            if (ret != STATUS_OK) goto cleanup;
        }
        // 5b. Transpose
        {
            const std::vector<int32_t> perm = {2, 5, 3, 6, 1, 0, 4, 7};
            std::vector<int64_t> out_shape(8);
            for (int i = 0; i < 8; i++) out_shape[i] = shape8[perm[i]];
            OperationHandle op = ops::TransposeOp::Create(perm);
            if (!op) { ret = ERROR_INFERENCE; goto cleanup; }
            ret = alloc->AllocFloat16(transpose_out, out_shape);
            if (ret != STATUS_OK) goto cleanup;
            atb::VariantPack vp;
            vp.inTensors  = {asstrided_out};
            vp.outTensors = {transpose_out};
            uint64_t ws = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws);
            if (ret != STATUS_OK) goto cleanup;
        }

        ret = runtime->Synchronize();
        if (ret != STATUS_OK) goto cleanup;

        // Step 6: D2H.
        const int64_t total = num_patches * patch_dim;
        pixel_values_out.resize(static_cast<size_t>(total));
        if (aclrtMemcpy(pixel_values_out.data(), total * sizeof(uint16_t),
                        transpose_out.deviceData, total * sizeof(uint16_t),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ERROR_INFERENCE;
            goto cleanup;
        }
    }

cleanup:
    if (transpose_out.deviceData != nullptr) alloc->Free(transpose_out);
    if (asstrided_out.deviceData != nullptr) alloc->Free(asstrided_out);
    if (inv_std_bc.deviceData    != nullptr) alloc->Free(inv_std_bc);
    if (mean_bc.deviceData       != nullptr) alloc->Free(mean_bc);
    if (normalize_tmp.deviceData != nullptr) alloc->Free(normalize_tmp);
    if (resized_t.deviceData     != nullptr) alloc->Free(resized_t);
    if (input_t.deviceData       != nullptr) alloc->Free(input_t);
    return ret;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// TC1: NpuBicubicResizeAASmallOp 1080x1920 -> 992x1792 vs PIL.
// Worst-case double-axis downsample at the largest production resolution
// (1.5MB fp16 input). All inner-loop precision dominates; boundary effects
// are negligible at this scale, so a CHECK on cos >= 0.99 is honest.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("smallop-aa-spike: 1080x1920 -> 992x1792 vs PIL ground truth") {
    LOG_INFO("=== Batch A spike: SmallOp AA bicubic single-resolution algorithm gate ===");

    // Reference data: PIL ground truth (gen_cpu_reference.py bicubic_preprocess).
    LoadedArrayF32 in_ref, pil_ref;
    REQUIRE(in_ref.Load("/tmp/bicubic_prod_1080x1920_input.bin"));
    REQUIRE(pil_ref.Load("/tmp/bicubic_prod_1080x1920_pil_output.bin"));

    REQUIRE(in_ref.shape.size() == 3);
    REQUIRE(pil_ref.shape.size() == 3);

    const int32_t channels = static_cast<int32_t>(in_ref.shape[0]);
    const int32_t in_h     = static_cast<int32_t>(in_ref.shape[1]);
    const int32_t in_w     = static_cast<int32_t>(in_ref.shape[2]);
    const int32_t out_h    = static_cast<int32_t>(pil_ref.shape[1]);
    const int32_t out_w    = static_cast<int32_t>(pil_ref.shape[2]);
    // Confirm we're actually testing the targeted production resolution.
    REQUIRE(channels == 3);
    REQUIRE(in_h == 1080);
    REQUIRE(in_w == 1920);
    REQUIRE(out_h == 992);
    REQUIRE(out_w == 1792);

    const int64_t in_elems  = static_cast<int64_t>(channels) * in_h  * in_w;
    const int64_t out_elems = static_cast<int64_t>(channels) * out_h * out_w;
    REQUIRE(static_cast<int64_t>(in_ref.data.size())  == in_elems);
    REQUIRE(static_cast<int64_t>(pil_ref.data.size()) == out_elems);

    // Spin up runtime (1 GB workspace pool — plenty for one image worth of
    // small-op weights + intermediate buffers; matches other spike tests).
    auto runtime = atb_llm::CreateRuntime(0, 1LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    // ── Host fp32 -> fp16 ──
    std::vector<uint16_t> in_fp16(in_elems);
    for (int64_t i = 0; i < in_elems; i++) {
        in_fp16[i] = atb_llm::Fp32ToFp16(in_ref.data[i]);
    }

    // ── H2D upload (raw aclrtMalloc — caller-owned, matches the wrapper's
    //    contract: input_device + output_device are owned by the caller) ──
    const size_t in_bytes  = static_cast<size_t>(in_elems)  * sizeof(uint16_t);
    const size_t out_bytes = static_cast<size_t>(out_elems) * sizeof(uint16_t);
    void* d_in  = nullptr;
    void* d_out = nullptr;
    REQUIRE(aclrtMalloc(&d_in,  in_bytes,  ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
    REQUIRE(aclrtMalloc(&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
    REQUIRE(aclrtMemcpy(d_in, in_bytes, in_fp16.data(), in_bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);

    // ── Run the small-op AA pipeline ──
    aclError ae = atb_llm::NpuBicubicResizeAASmallOp(
        d_in, in_h, in_w, channels, out_h, out_w,
        runtime.get(), d_out);
    REQUIRE(ae == ACL_SUCCESS);

    // Wrapper syncs intermediates before cleanup; caller still syncs before
    // reading output (identity memcpy path does not sync internally). L:265-266.
    REQUIRE(aclrtSynchronizeStream(runtime->GetStream()) == ACL_SUCCESS);

    // ── D2H + fp16 -> fp32 ──
    std::vector<uint16_t> out_fp16(out_elems);
    REQUIRE(aclrtMemcpy(out_fp16.data(), out_bytes, d_out, out_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
    std::vector<float> got(out_elems);
    for (int64_t i = 0; i < out_elems; i++) {
        got[i] = atb_llm::Fp16ToF32(out_fp16[i]);
    }

    const double cos = CosineSim(got.data(), pil_ref.data.data(), out_elems);
    const float  diff = MaxAbsDiff(got.data(), pil_ref.data.data(), out_elems);

    LOG_INFO("[SmallOp AA TC1] %dx%d -> %dx%d  cos=%.6f  max_diff=%.4f",
             in_h, in_w, out_h, out_w, cos, diff);

    aclrtFree(d_in);
    aclrtFree(d_out);

    LOG_INFO("================================================================");
    LOG_INFO("[Batch A gate] cos >= 0.99 required to proceed to Batch B");
    LOG_INFO("================================================================");
    CHECK(cos >= 0.99);
}

// ═════════════════════════════════════════════════════════════════
// TC2: NpuBicubicResizeAASmallOp vs PIL at all 4 production resolutions.
//   416x672  -> 416x672   identity (per-axis skip -> memcpy, cos == 1.0)
//   720x1280 -> 704x1280   V-only (height shrinks, width skipped)
//   1080x1920-> 992x1792   double-axis downsample
//   1440x2560-> 992x1792   double-axis downsample (worst case)
// out_h/out_w are read from each pil_output bin header (not hard-coded).
// ═════════════════════════════════════════════════════════════════
TEST_CASE("smallop-aa-spike TC2: 4 production resolutions vs PIL") {
    auto runtime = atb_llm::CreateRuntime(0, 1LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    const char* names[] = {
        "bicubic_prod_416x672",
        "bicubic_prod_720x1280",
        "bicubic_prod_1080x1920",
        "bicubic_prod_1440x2560",
    };

    for (const char* base : names) {
        LoadedArrayF32 in_ref, pil_ref;
        REQUIRE(in_ref.Load(std::string("/tmp/") + base + "_input.bin"));
        REQUIRE(pil_ref.Load(std::string("/tmp/") + base + "_pil_output.bin"));
        REQUIRE(in_ref.shape.size() == 3);
        REQUIRE(pil_ref.shape.size() == 3);

        const int32_t channels = static_cast<int32_t>(in_ref.shape[0]);
        const int32_t in_h     = static_cast<int32_t>(in_ref.shape[1]);
        const int32_t in_w     = static_cast<int32_t>(in_ref.shape[2]);
        const int32_t out_h    = static_cast<int32_t>(pil_ref.shape[1]);
        const int32_t out_w    = static_cast<int32_t>(pil_ref.shape[2]);
        const int64_t in_elems  = static_cast<int64_t>(channels) * in_h * in_w;
        const int64_t out_elems = static_cast<int64_t>(channels) * out_h * out_w;
        REQUIRE(static_cast<int64_t>(in_ref.data.size())  == in_elems);
        REQUIRE(static_cast<int64_t>(pil_ref.data.size()) == out_elems);

        std::vector<uint16_t> in_fp16(in_elems);
        for (int64_t i = 0; i < in_elems; i++)
            in_fp16[i] = atb_llm::Fp32ToFp16(in_ref.data[i]);

        const size_t in_bytes  = static_cast<size_t>(in_elems)  * sizeof(uint16_t);
        const size_t out_bytes = static_cast<size_t>(out_elems) * sizeof(uint16_t);
        void* d_in = nullptr;
        void* d_out = nullptr;
        REQUIRE(aclrtMalloc(&d_in,  in_bytes,  ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        REQUIRE(aclrtMalloc(&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        REQUIRE(aclrtMemcpy(d_in, in_bytes, in_fp16.data(), in_bytes,
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);

        aclError ae = atb_llm::NpuBicubicResizeAASmallOp(
            d_in, in_h, in_w, channels, out_h, out_w, runtime.get(), d_out);
        REQUIRE(ae == ACL_SUCCESS);
        REQUIRE(aclrtSynchronizeStream(runtime->GetStream()) == ACL_SUCCESS);

        std::vector<uint16_t> out_fp16(out_elems);
        REQUIRE(aclrtMemcpy(out_fp16.data(), out_bytes, d_out, out_bytes,
                            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
        std::vector<float> got(out_elems);
        for (int64_t i = 0; i < out_elems; i++)
            got[i] = atb_llm::Fp16ToF32(out_fp16[i]);

        const double cos = CosineSim(got.data(), pil_ref.data.data(), out_elems);
        const float diff = MaxAbsDiff(got.data(), pil_ref.data.data(), out_elems);
        const bool identity = (out_h == in_h && out_w == in_w);

        LOG_INFO("[SmallOp AA TC2] %dx%d -> %dx%d  cos=%.6f  max_diff=%.4f%s",
                 in_h, in_w, out_h, out_w, cos, diff,
                 identity ? "  [identity memcpy]" : "");

        aclrtFree(d_in);
        aclrtFree(d_out);

        if (identity) CHECK(cos >= 0.9999);  // memcpy is lossless -> cos == 1.0
        CHECK(cos >= 0.99);
    }
}

// ═════════════════════════════════════════════════════════════════
// TC3: full NPU preprocess pipeline (small-op AA resize + normalize +
// patchify) vs CPU PreprocessImage (PIL 8bpc bit-exact path) at all 4
// production resolutions. Validates that the small-op AA output feeds
// normalize + patchify and stays cos >= 0.99 against the CPU PIL pipeline
// end-to-end. No dispatch change in qwen3vl_preprocess.cpp — the NPU
// pipeline is strung manually in RunNpuPipelineSmallOpAA (Batch C engineers
// the production dispatch).
// ═════════════════════════════════════════════════════════════════
TEST_CASE("smallop-aa-spike TC3: full NPU pipeline (small-op AA) vs CPU PreprocessImage") {
    auto runtime = atb_llm::CreateRuntime(0, 1LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    atb_llm::adapters::Qwen3VLConfig cfg{};
    cfg.pp_patch_size = 16;
    cfg.pp_temporal_patch_size = 2;
    cfg.pp_merge_size = 2;
    cfg.pp_min_pixels = 4096;
    cfg.pp_max_pixels = 1843200;
    // image_mean/std default to {0.5,0.5,0.5} (preprocessor_config.json).

    const int32_t channels = 3;
    const char* names[] = {
        "bicubic_prod_416x672",
        "bicubic_prod_720x1280",
        "bicubic_prod_1080x1920",
        "bicubic_prod_1440x2560",
    };

    for (const char* base : names) {
        LoadedArrayF32 in_ref;
        REQUIRE(in_ref.Load(std::string("/tmp/") + base + "_input.bin"));
        REQUIRE(in_ref.shape.size() == 3);
        REQUIRE(static_cast<int32_t>(in_ref.shape[0]) == channels);
        const int32_t H = static_cast<int32_t>(in_ref.shape[1]);
        const int32_t W = static_cast<int32_t>(in_ref.shape[2]);

        // The input bin is the ORIGINAL image (pre-SmartResize); quantize to
        // uint8 (clip+round) exactly like the Python generator.
        std::vector<uint8_t> image(in_ref.data.size());
        for (size_t i = 0; i < image.size(); i++) {
            float v = std::round(in_ref.data[i]);
            v = std::min(255.0f, std::max(0.0f, v));
            image[i] = static_cast<uint8_t>(v);
        }

        int32_t new_h = 0, new_w = 0;
        atb_llm::adapters::SmartResize(H, W, cfg.pp_patch_size * cfg.pp_merge_size,
                                       cfg.pp_min_pixels, cfg.pp_max_pixels,
                                       new_h, new_w);
        const int64_t grid_h = new_h / cfg.pp_patch_size;
        const int64_t grid_w = new_w / cfg.pp_patch_size;
        const int64_t patch_dim = static_cast<int64_t>(channels)
            * cfg.pp_temporal_patch_size * cfg.pp_patch_size * cfg.pp_patch_size;
        const int64_t total = grid_h * grid_w * patch_dim;

        // CPU reference.
        std::vector<uint16_t> cpu_pv(total);
        int64_t np_cpu = 0;
        int64_t grid_cpu[3] = {0, 0, 0};
        REQUIRE(atb_llm::adapters::PreprocessImage(
                    image.data(), channels, H, W, cfg,
                    cpu_pv.data(), np_cpu, grid_cpu) == atb_llm::STATUS_OK);

        // NPU small-op pipeline.
        std::vector<uint16_t> npu_pv;
        int64_t np_npu = 0;
        int64_t grid_npu[3] = {0, 0, 0};
        REQUIRE(RunNpuPipelineSmallOpAA(
                    runtime.get(), image.data(), channels, H, W, cfg,
                    npu_pv, np_npu, grid_npu) == atb_llm::STATUS_OK);

        REQUIRE(np_npu == np_cpu);
        REQUIRE(static_cast<int64_t>(npu_pv.size()) == total);
        CHECK(grid_npu[0] == grid_cpu[0]);
        CHECK(grid_npu[1] == grid_cpu[1]);
        CHECK(grid_npu[2] == grid_cpu[2]);

        std::vector<float> a(total), b(total);
        for (int64_t i = 0; i < total; i++) {
            a[i] = atb_llm::Fp16ToF32(cpu_pv[i]);
            b[i] = atb_llm::Fp16ToF32(npu_pv[i]);
        }
        const double cos = CosineSim(a.data(), b.data(), total);
        const float diff = MaxAbsDiff(a.data(), b.data(), total);

        LOG_INFO("[SmallOp AA TC3] %dx%d -> %dx%d  grid=(1,%ld,%ld)  patches=%ld  "
                 "cos=%.6f  max_diff=%.4f",
                 H, W, new_h, new_w, static_cast<long>(grid_h),
                 static_cast<long>(grid_w), static_cast<long>(np_npu), cos, diff);

        CHECK(cos >= 0.99);
    }
}

// ═════════════════════════════════════════════════════════════════
// TC4 (H-only branch coverage, Reviewer MINOR-1): width shrinks, height
// unchanged -> exercises the `need_h && !need_v` path (direct-write
// output_view + early goto cleanup, smallop_bicubic_aa.cpp:203-217). The 4
// production resolutions don't isolate this branch (416=identity, 720=V-only,
// 1080/1440=double-axis). [3,64,128] -> [3,64,64] forces H active, V skipped.
// Reference is an independent PIL BICUBIC computed by gen_cpu_reference.py
// (_gen_bicubic_case "honly_64x128"), same mechanism as TC2.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("smallop-aa-spike TC4: H-only (64x128 -> 64x64) vs PIL") {
    LoadedArrayF32 in_ref, pil_ref;
    REQUIRE(in_ref.Load("/tmp/bicubic_honly_64x128_input.bin"));
    REQUIRE(pil_ref.Load("/tmp/bicubic_honly_64x128_pil_output.bin"));
    REQUIRE(in_ref.shape.size() == 3);
    REQUIRE(pil_ref.shape.size() == 3);

    const int32_t channels = static_cast<int32_t>(in_ref.shape[0]);
    const int32_t in_h     = static_cast<int32_t>(in_ref.shape[1]);
    const int32_t in_w     = static_cast<int32_t>(in_ref.shape[2]);
    const int32_t out_h    = static_cast<int32_t>(pil_ref.shape[1]);
    const int32_t out_w    = static_cast<int32_t>(pil_ref.shape[2]);
    // Confirm this really is the H-only regime (width shrinks, height fixed).
    REQUIRE(out_h == in_h);   // V skipped
    REQUIRE(out_w != in_w);   // H active

    const int64_t in_elems  = static_cast<int64_t>(channels) * in_h * in_w;
    const int64_t out_elems = static_cast<int64_t>(channels) * out_h * out_w;
    REQUIRE(static_cast<int64_t>(in_ref.data.size())  == in_elems);
    REQUIRE(static_cast<int64_t>(pil_ref.data.size()) == out_elems);

    auto runtime = atb_llm::CreateRuntime(0, 1LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    std::vector<uint16_t> in_fp16(in_elems);
    for (int64_t i = 0; i < in_elems; i++)
        in_fp16[i] = atb_llm::Fp32ToFp16(in_ref.data[i]);

    const size_t in_bytes  = static_cast<size_t>(in_elems)  * sizeof(uint16_t);
    const size_t out_bytes = static_cast<size_t>(out_elems) * sizeof(uint16_t);
    void* d_in = nullptr;
    void* d_out = nullptr;
    REQUIRE(aclrtMalloc(&d_in,  in_bytes,  ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
    REQUIRE(aclrtMalloc(&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
    REQUIRE(aclrtMemcpy(d_in, in_bytes, in_fp16.data(), in_bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);

    aclError ae = atb_llm::NpuBicubicResizeAASmallOp(
        d_in, in_h, in_w, channels, out_h, out_w, runtime.get(), d_out);
    REQUIRE(ae == ACL_SUCCESS);
    REQUIRE(aclrtSynchronizeStream(runtime->GetStream()) == ACL_SUCCESS);

    std::vector<uint16_t> out_fp16(out_elems);
    REQUIRE(aclrtMemcpy(out_fp16.data(), out_bytes, d_out, out_bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
    std::vector<float> got(out_elems);
    double nonzero_sum = 0.0;
    for (int64_t i = 0; i < out_elems; i++) {
        got[i] = atb_llm::Fp16ToF32(out_fp16[i]);
        nonzero_sum += std::fabs(got[i]);
    }

    const double cos = CosineSim(got.data(), pil_ref.data.data(), out_elems);
    const float diff = MaxAbsDiff(got.data(), pil_ref.data.data(), out_elems);
    LOG_INFO("[SmallOp AA TC4 H-only] %dx%d -> %dx%d  cos=%.6f  max_diff=%.4f",
             in_h, in_w, out_h, out_w, cos, diff);

    aclrtFree(d_in);
    aclrtFree(d_out);

    CHECK(nonzero_sum > 0.0);   // basic sanity: output is not all zeros
    CHECK(cos >= 0.99);
}

// ═════════════════════════════════════════════════════════════════
// TC5 (perf baseline): single-image latency of NpuBicubicResizeAASmallOp at
// the 4 production resolutions. This is the SEPARABLE single-op baseline (5
// ops: Linear x2 + Transpose x2, per-axis-skipped) that the future graph
// (fused) version will be compared against. Timed window = op call + stream
// sync (input already H2D, excludes D2H). Warmup 3, time 10, report mean.
// No precision CHECK (measurement only); a loose upper bound guards against
// pathological regressions.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("smallop-aa-spike TC5-perf: separable single-op latency, 4 resolutions") {
    auto runtime = atb_llm::CreateRuntime(0, 1LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    struct Case { int in_h, in_w, out_h, out_w; };
    // out dims from SmartResize (match the prod bins): 416 identity is skipped
    // (memcpy, not representative of the separable op chain).
    const Case cases[] = {
        {720,  1280, 704,  1280},
        {1080, 1920, 992,  1792},
        {1440, 2560, 992,  1792},
        {416,  672,  416,  672},   // identity (memcpy) — included for completeness
    };
    const int channels = 3;

    for (const Case& c : cases) {
        const int64_t in_elems  = static_cast<int64_t>(channels) * c.in_h  * c.in_w;
        const int64_t out_elems = static_cast<int64_t>(channels) * c.out_h * c.out_w;
        const size_t in_bytes  = static_cast<size_t>(in_elems)  * sizeof(uint16_t);
        const size_t out_bytes = static_cast<size_t>(out_elems) * sizeof(uint16_t);

        std::vector<uint16_t> in_fp16(in_elems);
        for (int64_t i = 0; i < in_elems; i++)
            in_fp16[i] = atb_llm::Fp32ToFp16(static_cast<float>((i * 37) % 256));

        void* d_in = nullptr;
        void* d_out = nullptr;
        REQUIRE(aclrtMalloc(&d_in,  in_bytes,  ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        REQUIRE(aclrtMalloc(&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        REQUIRE(aclrtMemcpy(d_in, in_bytes, in_fp16.data(), in_bytes,
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);

        // Warmup (not timed).
        for (int w = 0; w < 3; w++) {
            REQUIRE(atb_llm::NpuBicubicResizeAASmallOp(
                d_in, c.in_h, c.in_w, channels, c.out_h, c.out_w,
                runtime.get(), d_out) == ACL_SUCCESS);
            REQUIRE(aclrtSynchronizeStream(runtime->GetStream()) == ACL_SUCCESS);
        }

        // Time 10 rounds (op call + sync; input already on device, no D2H).
        const int rounds = 10;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < rounds; r++) {
            REQUIRE(atb_llm::NpuBicubicResizeAASmallOp(
                d_in, c.in_h, c.in_w, channels, c.out_h, c.out_w,
                runtime.get(), d_out) == ACL_SUCCESS);
            REQUIRE(aclrtSynchronizeStream(runtime->GetStream()) == ACL_SUCCESS);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / rounds;

        const bool identity = (c.out_h == c.in_h && c.out_w == c.in_w);
        LOG_INFO("[SmallOp AA PERF] %dx%d -> %dx%d: %.2f ms (separable, 5 ops)%s",
                 c.in_h, c.in_w, c.out_h, c.out_w, ms,
                 identity ? "  [identity memcpy]" : "");

        aclrtFree(d_in);
        aclrtFree(d_out);

        // Loose regression guard (measurement only, no precision implication).
        CHECK(ms < 50.0);
    }
}
