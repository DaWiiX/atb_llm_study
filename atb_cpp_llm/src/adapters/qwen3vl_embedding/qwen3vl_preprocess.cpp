#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#include "utils/cpp11_compat.h"
#include "atb_llm/runtime.h"
#include "ops/elewise_op.h"
#include "ops/transpose_op.h"
#include "components/vision/aclnn_bicubic_resize.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace atb_llm {
namespace adapters {

// ── Bicubic interpolation kernel (Catmull-Rom, a=-0.5) ──────
// Matches PyTorch F.interpolate(mode='bicubic', align_corners=False)
static float CubicWeight(float x) {
    x = std::fabs(x);
    float a = -0.5f;
    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

// ── PIL-style precomputed coefficient table (Resample.c::precompute_coeffs) ──
// For each output pixel o, precompute the 4 tap indices + weights once
// (outSize*4 CubicWeight calls) instead of per output² pixel (outSize²*16).
//
// Mapping (UNCHANGED from original): src = (o+0.5)*in/out - 0.5
//   s = floor(src); d = src - s.
// For m in [-1,2]: idx[o*4+(m+1)] = clamp(s+m, 0, in-1)  [edge-clamp, UNCHANGED]
//                  kk[o*4+(m+1)]  = CubicWeight(d - m)
//
// The 4 Catmull-Rom weights sum to 1.0 for any d in [0,1) (kernel partition
// of unity), so no ww renormalization is needed. edge-clamp keeps all 4 taps
// (unlike PIL's windowed boundary handling, which can drop taps) — matching
// the original implementation's boundary behaviour exactly.
static void PrecomputeCoeffs(int32_t inSize, int32_t outSize,
                             int32_t* idx, double* kk) {
    for (int32_t o = 0; o < outSize; o++) {
        float src = (o + 0.5f) * static_cast<float>(inSize) / outSize - 0.5f;
        int32_t s = static_cast<int32_t>(std::floor(src));
        float d = src - s;
        int32_t* idx_o = idx + o * 4;
        double* kk_o = kk + o * 4;
        for (int m = -1; m <= 2; m++) {
            idx_o[m + 1] = atb_llm::clamp(s + m, 0, inSize - 1);
            kk_o[m + 1] = static_cast<double>(CubicWeight(d - m));
        }
    }
}

// ── Horizontal pass (separable, Resample.c::ImagingResampleHorizontal) ──
// Resize width only: imTemp[c, ih, ow] = sum_n kk_w[ow*4+n] * input[c, ih, idx_w[ow*4+n]]
// imTemp layout: [C, in_h, out_w]. Accumulated in double for precision.
static void HorizontalPass(const uint8_t* input, int32_t in_h, int32_t in_w,
                           int32_t channels, int32_t out_w,
                           const int32_t* idx_w, const double* kk_w,
                           float* imTemp) {
    for (int32_t c = 0; c < channels; c++) {
        const uint8_t* in_c = input + static_cast<size_t>(c) * in_h * in_w;
        float* tmp_c = imTemp + static_cast<size_t>(c) * in_h * out_w;
        for (int32_t ih = 0; ih < in_h; ih++) {
            const uint8_t* in_row = in_c + static_cast<size_t>(ih) * in_w;
            float* tmp_row = tmp_c + static_cast<size_t>(ih) * out_w;
            for (int32_t ow = 0; ow < out_w; ow++) {
                const int32_t* idx = idx_w + ow * 4;
                const double* kk = kk_w + ow * 4;
                double sum = static_cast<double>(in_row[idx[0]]) * kk[0]
                           + static_cast<double>(in_row[idx[1]]) * kk[1]
                           + static_cast<double>(in_row[idx[2]]) * kk[2]
                           + static_cast<double>(in_row[idx[3]]) * kk[3];
                tmp_row[ow] = static_cast<float>(sum);
            }
        }
    }
}

// ── Vertical pass (separable, Resample.c::ImagingResampleVertical) ──
// Resize height + optional normalize fusion:
//   output[c,oh,ow] = (sum_m kk_h[oh*4+m] * imTemp[c, idx_h[oh*4+m], ow] * rescale - mean[c]) / std[c]
// Raw (BicubicResize): rescale=1, mean=0, std=1 → output = sum (uint8 weighted sum).
// Fused (PreprocessImage): rescale=1/255, mean/std from config → (sum/255 - mean)/std.
static void VerticalPass(const float* imTemp, int32_t in_h, int32_t out_w,
                         int32_t channels, int32_t out_h,
                         const int32_t* idx_h, const double* kk_h,
                         const float* mean, const float* std_val, float rescale,
                         float* output) {
    for (int32_t c = 0; c < channels; c++) {
        const float* tmp_c = imTemp + static_cast<size_t>(c) * in_h * out_w;
        float* out_c = output + static_cast<size_t>(c) * out_h * out_w;
        float mc = mean[c];
        float sc = std_val[c];
        for (int32_t oh = 0; oh < out_h; oh++) {
            const int32_t* idx = idx_h + oh * 4;
            const double* kk = kk_h + oh * 4;
            const float* tmp0 = tmp_c + static_cast<size_t>(idx[0]) * out_w;
            const float* tmp1 = tmp_c + static_cast<size_t>(idx[1]) * out_w;
            const float* tmp2 = tmp_c + static_cast<size_t>(idx[2]) * out_w;
            const float* tmp3 = tmp_c + static_cast<size_t>(idx[3]) * out_w;
            float* out_row = out_c + static_cast<size_t>(oh) * out_w;
            for (int32_t ow = 0; ow < out_w; ow++) {
                double sum = static_cast<double>(tmp0[ow]) * kk[0]
                           + static_cast<double>(tmp1[ow]) * kk[1]
                           + static_cast<double>(tmp2[ow]) * kk[2]
                           + static_cast<double>(tmp3[ow]) * kk[3];
                out_row[ow] = static_cast<float>((sum * rescale - mc) / sc);
            }
        }
    }
}

void BicubicResize(const uint8_t* input, int32_t in_h, int32_t in_w,
                   int32_t channels, int32_t out_h, int32_t out_w,
                   float* output) {
    // Precompute horizontal + vertical coefficient tables (outSize*4 CubicWeight
    // calls each, vs original outSize²*16).
    std::vector<int32_t> idx_w(static_cast<size_t>(out_w) * 4);
    std::vector<int32_t> idx_h(static_cast<size_t>(out_h) * 4);
    std::vector<double> kk_w(static_cast<size_t>(out_w) * 4);
    std::vector<double> kk_h(static_cast<size_t>(out_h) * 4);
    PrecomputeCoeffs(in_w, out_w, idx_w.data(), kk_w.data());
    PrecomputeCoeffs(in_h, out_h, idx_h.data(), kk_h.data());

    // Separable two-stage: horizontal pass (uint8 → float32 imTemp), then
    // vertical pass (imTemp → float32 output). 4+4=8 taps/pixel vs 16.
    std::vector<float> imTemp(static_cast<size_t>(channels) * in_h * out_w);
    HorizontalPass(input, in_h, in_w, channels, out_w,
                   idx_w.data(), kk_w.data(), imTemp.data());

    // Raw output: rescale=1, mean=0, std=1 → output = weighted sum (no /255).
    std::vector<float> zmean(channels, 0.0f);
    std::vector<float> ostd(channels, 1.0f);
    VerticalPass(imTemp.data(), in_h, out_w, channels, out_h,
                 idx_h.data(), kk_h.data(), zmean.data(), ostd.data(), 1.0f,
                 output);
}

// Python-style banker's rounding: round half to even (matches Python 3 round())
static int32_t BankersRound(float x) {
    float floor_val = std::floor(x);
    float frac = x - floor_val;
    if (frac < 0.5f) return static_cast<int32_t>(floor_val);
    if (frac > 0.5f) return static_cast<int32_t>(floor_val + 1.0f);
    // Exactly 0.5 — round to nearest even
    int32_t lower = static_cast<int32_t>(floor_val);
    return (lower % 2 == 0) ? lower : lower + 1;
}

void SmartResize(int32_t height, int32_t width,
                 int32_t factor, int32_t min_pixels, int32_t max_pixels,
                 int32_t& new_height, int32_t& new_width) {
    // Round to nearest multiple of factor (banker's rounding, matches Python)
    int32_t h_bar = BankersRound(static_cast<float>(height) / factor) * factor;
    int32_t w_bar = BankersRound(static_cast<float>(width) / factor) * factor;

    int64_t area = static_cast<int64_t>(h_bar) * w_bar;
    if (area > max_pixels) {
        double beta = std::sqrt(static_cast<double>(height * width) / max_pixels);
        h_bar = std::max(factor, static_cast<int32_t>(
                                     std::floor(height / beta / factor)) *
                                     factor);
        w_bar = std::max(factor, static_cast<int32_t>(
                                     std::floor(width / beta / factor)) *
                                     factor);
    } else if (area < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
        h_bar = static_cast<int32_t>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int32_t>(std::ceil(width * beta / factor)) * factor;
    }

    new_height = h_bar;
    new_width = w_bar;
}

Status PreprocessImage(const uint8_t* image,
                       int32_t channels, int32_t height, int32_t width,
                       const Qwen3VLConfig& config,
                       uint16_t* pixel_values,
                       int64_t& num_patches,
                       int64_t* grid_thw) {
    int32_t patch_size = config.pp_patch_size;
    int32_t tp = config.pp_temporal_patch_size;
    int32_t merge_size = config.pp_merge_size;
    int32_t factor = patch_size * merge_size;

    // 1. Smart resize
    int32_t new_h, new_w;
    SmartResize(height, width, factor,
                config.pp_min_pixels, config.pp_max_pixels,
                new_h, new_w);

    // 2+3. Bicubic resize + normalize (FUSED via separable two-stage convolution)
    // Precompute coefficient tables (PIL precompute_coeffs style).
    std::vector<int32_t> idx_w(static_cast<size_t>(new_w) * 4);
    std::vector<int32_t> idx_h(static_cast<size_t>(new_h) * 4);
    std::vector<double> kk_w(static_cast<size_t>(new_w) * 4);
    std::vector<double> kk_h(static_cast<size_t>(new_h) * 4);
    PrecomputeCoeffs(width, new_w, idx_w.data(), kk_w.data());
    PrecomputeCoeffs(height, new_h, idx_h.data(), kk_h.data());

    // Horizontal pass: uint8 [C,H,W] → float32 imTemp [C, H, new_w]
    std::vector<float> imTemp(static_cast<size_t>(channels) * height * new_w);
    HorizontalPass(image, height, width, channels, new_w,
                   idx_w.data(), kk_w.data(), imTemp.data());

    // Vertical pass: imTemp → resized [C, new_h, new_w], fusing rescale/255 + normalize.
    // Saves one full pass over the resized buffer vs separate normalize loop.
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_val[3] = {0.5f, 0.5f, 0.5f};
    if (config.pp_image_mean.size() >= 3) {
        mean[0] = config.pp_image_mean[0];
        mean[1] = config.pp_image_mean[1];
        mean[2] = config.pp_image_mean[2];
    }
    if (config.pp_image_std.size() >= 3) {
        std_val[0] = config.pp_image_std[0];
        std_val[1] = config.pp_image_std[1];
        std_val[2] = config.pp_image_std[2];
    }

    size_t resized_size = static_cast<size_t>(channels) * new_h * new_w;
    std::vector<float> resized(resized_size);
    VerticalPass(imTemp.data(), height, new_w, channels, new_h,
                 idx_h.data(), kk_h.data(), mean, std_val, 1.0f / 255.0f,
                 resized.data());

    // 4. Pad to temporal_patch_size (single image -> 2 frames by duplicating)
    int32_t total_frames = tp;           // For single image, we pad to tp=2
    int32_t grid_t = total_frames / tp;  // = 1
    int32_t grid_h = new_h / patch_size;
    int32_t grid_w = new_w / patch_size;

    grid_thw[0] = grid_t;
    grid_thw[1] = grid_h;
    grid_thw[2] = grid_w;

    // 5. Patch extraction with spatial merge ordering
    // Python: frames.reshape(grid_t, tp, C, grid_h//ms, ms, ps, grid_w//ms, ms, ps)
    //         .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    //         .reshape(grid_t*grid_h*grid_w, C*tp*ps*ps)
    //
    // After permute, the last dim order is: C, tp, ps_h, ps_w
    // So each patch element is indexed by (c, f, ph, pw) with strides:
    //   offset = c * tp*ps*ps + f * ps*ps + ph * ps + pw

    int64_t patch_dim = static_cast<int64_t>(channels) * tp * patch_size * patch_size;
    num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;

    int64_t merged_h = grid_h / merge_size;
    int64_t merged_w = grid_w / merge_size;

    // Single-image input: all tp frames read the same `resized` data, so the
    // f=0 block is computed once and memcpy'd to f=1..tp-1 (halves read+convert
    // work in the temporal dimension).
    size_t ps2 = static_cast<size_t>(patch_size) * patch_size;
    int64_t patch_idx = 0;
    for (int64_t br = 0; br < merged_h; br++) {
        for (int64_t bc = 0; bc < merged_w; bc++) {
            for (int64_t ir = 0; ir < merge_size; ir++) {
                for (int64_t ic = 0; ic < merge_size; ic++) {
                    uint16_t* pixel_base = pixel_values + patch_idx * patch_dim;
                    int64_t row_base = br * merge_size * patch_size + ir * patch_size;
                    int64_t col_base = bc * merge_size * patch_size + ic * patch_size;
                    for (int64_t c = 0; c < channels; c++) {
                        const float* resized_c =
                            resized.data() + c * new_h * new_w;
                        // f=0 block: read each pixel once, convert once.
                        uint16_t* f0 = pixel_base + c * tp * ps2;
                        for (int64_t ph = 0; ph < patch_size; ph++) {
                            const float* src_row =
                                resized_c + (row_base + ph) * new_w + col_base;
                            uint16_t* dst_row = f0 + ph * patch_size;
                            for (int64_t pw = 0; pw < patch_size; pw++) {
                                dst_row[pw] = atb_llm::Fp32ToFp16(src_row[pw]);
                            }
                        }
                        // Duplicate f=0 to f=1..tp-1 (single image → identical frames).
                        for (int32_t f = 1; f < tp; f++) {
                            std::memcpy(f0 + f * ps2, f0, ps2 * sizeof(uint16_t));
                        }
                    }
                    patch_idx++;
                }
            }
        }
    }

    LOG_INFO("Preprocessed image: %dx%d -> %dx%d, grid=(%ld,%ld,%ld), patches=%ld, dim=%ld",
             height, width, new_h, new_w,
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(num_patches), static_cast<long>(patch_dim));

    return STATUS_OK;
}

// ── Internal: NPU preprocess steps 1-5 (SmartResize → H2D → bicubic →
// normalize → AsStrided+Transpose). Shared by both PreprocessImageNpu
// overloads (DRY). Leaves the device-resident patch output in @p transpose_out
// (allocator-tracked; caller owns — must Free or Detach). All intermediates
// are freed here; on error transpose_out is freed too. No D2H: the device
// tensor stays on NPU so path C can feed the vision pipeline directly.
static Status PreprocessImageNpuInternal(IRuntime* runtime,
                                         const uint8_t* image,
                                         int32_t channels, int32_t height, int32_t width,
                                         const Qwen3VLConfig& config,
                                         atb::Tensor& transpose_out,
                                         int64_t& num_patches,
                                         int64_t* grid_thw) {
    if (runtime == nullptr || image == nullptr || grid_thw == nullptr) {
        LOG_ERROR("PreprocessImageNpuInternal: null parameter (runtime=%p, image=%p, grid_thw=%p)",
                  static_cast<const void*>(runtime), static_cast<const void*>(image),
                  static_cast<const void*>(grid_thw));
        return ERROR_INVALID_PARAM;
    }

    int32_t patch_size = config.pp_patch_size;
    int32_t tp = config.pp_temporal_patch_size;
    int32_t merge_size = config.pp_merge_size;
    int32_t factor = patch_size * merge_size;

    // --- Step 1: SmartResize (CPU) ------------------------------------
    int32_t new_h, new_w;
    SmartResize(height, width, factor,
                config.pp_min_pixels, config.pp_max_pixels,
                new_h, new_w);

    // --- Resources (all declared upfront for goto-cleanup) ------------
    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();
    Status ret = STATUS_OK;

    // Value-initialize all tensor handles: deviceData starts as nullptr so
    // the cleanup block can distinguish allocated vs unallocated tensors.
    // transpose_out is the caller-owned out-param (value-initialised by caller).
    atb::Tensor input_tensor{};
    atb::Tensor resized_tensor{};
    atb::Tensor normalize_tmp{};
    atb::Tensor mean_bc{};
    atb::Tensor inv_std_bc{};
    atb::Tensor asstrided_out{};

    // --- Step 2: H2D -- uint8->fp16 conversion + copy to device ---------
    int64_t in_elems = static_cast<int64_t>(channels) * height * width;
    std::vector<uint16_t> in_fp16(in_elems);
    for (int64_t i = 0; i < in_elems; i++) {
        in_fp16[i] = Fp32ToFp16(static_cast<float>(image[i]));
    }

    ret = alloc->AllocFloat16(input_tensor, {1, channels, height, width});
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: AllocFloat16 input failed");
        goto cleanup;
    }
    ret = alloc->CopyToDevice(input_tensor, in_fp16.data(), in_elems * sizeof(uint16_t));
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: CopyToDevice input failed");
        goto cleanup;
    }

    // --- Step 3: Bicubic Resize on NPU ---------------------------------
    ret = alloc->AllocFloat16(resized_tensor, {1, channels, new_h, new_w});
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: AllocFloat16 resized failed");
        goto cleanup;
    }

    // AA pre-filtering is only beneficial when genuinely downsampling.
    // At identity / near-identity scales AA smooths the image and
    // degrades precision vs CPU (cos drops to ~0.950).  On 310P the AA
    // op is not available so we always use non-AA.
    {
        bool downsample = (new_h < height || new_w < width);
        if (Is910B() && downsample) {
            ret = NpuBicubicResizeAA(runtime,
                                     input_tensor.deviceData,
                                     channels, height, width, new_h, new_w,
                                     resized_tensor.deviceData);
        } else {
            ret = NpuBicubicResize(runtime,
                                   input_tensor.deviceData,
                                   channels, height, width, new_h, new_w,
                                   resized_tensor.deviceData);
        }
    }
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: NpuBicubicResize failed");
        goto cleanup;
    }

    // Free input tensor early (no longer needed); Free() sets deviceData to
    // nullptr so the cleanup block will skip it — no double-free.
    alloc->Free(input_tensor);

    // --- Step 4–6: Normalize + D2H + patch extraction -----------------
    // Nested scope isolates all normalization local variables (mean, std_val,
    // mean_neg, inv_std, RunAtbOp lambda) so that gotos from steps 2-3
    // (above) don't cross their initialization — C++ forbids goto over
    // variables with non-trivial destructors.
    {
        // ── Step 4: Normalize on NPU via 3 ATB ElewiseOps ──
        // Formula: (x/255 - mean) / std
        //
        // ATB has no scalar add/subtract (no ELEWISE_ADDS).  We use broadcast
        // tensors instead: ELEWISE_ADD/ELEWISE_MUL broadcast when corresponding
        // dims are equal or at least one is 1.  A (1,C,1,1) tensor broadcasts
        // to (1,C,H,W) -- one scalar per channel, perfect for per-channel mean/std.
        //
        // Ops: 4a. MULS(1/255)
        //      4b. ADD(-mean broadcast)  --  x/255 + (-mean) = x/255 - mean
        //      4c. MUL(1/std broadcast)  --  (x/255 - mean) * (1/std)

        float mean[3] = {0.5f, 0.5f, 0.5f};
        float std_val[3] = {0.5f, 0.5f, 0.5f};
        if (config.pp_image_mean.size() >= 3) {
            mean[0] = config.pp_image_mean[0];
            mean[1] = config.pp_image_mean[1];
            mean[2] = config.pp_image_mean[2];
        }
        if (config.pp_image_std.size() >= 3) {
            std_val[0] = config.pp_image_std[0];
            std_val[1] = config.pp_image_std[1];
            std_val[2] = config.pp_image_std[2];
        }

        // Intermediate tensor (same shape as resized, used as ping-pong buffer)
        ret = alloc->AllocFloat16(normalize_tmp, {1, channels, new_h, new_w});
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: AllocFloat16 normalize_tmp failed");
            goto cleanup;
        }

        // Per-channel broadcast tensors (1, C, 1, 1) -- just 3 elements each
        std::vector<uint16_t> mean_neg(static_cast<size_t>(channels));
        std::vector<uint16_t> inv_std(static_cast<size_t>(channels));
        for (int32_t c = 0; c < channels; c++) {
            mean_neg[c] = Fp32ToFp16(-mean[c]);
            inv_std[c] = Fp32ToFp16(1.0f / std_val[c]);
        }

        ret = alloc->AllocFloat16(mean_bc, {1, channels, 1, 1});
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: AllocFloat16 mean_bc failed");
            goto cleanup;
        }
        ret = alloc->AllocFloat16(inv_std_bc, {1, channels, 1, 1});
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: AllocFloat16 inv_std_bc failed");
            goto cleanup;
        }
        ret = alloc->CopyToDevice(mean_bc, mean_neg.data(), mean_neg.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) goto cleanup;
        ret = alloc->CopyToDevice(inv_std_bc, inv_std.data(), inv_std.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) goto cleanup;

        // Run a single ATB operation (Setup -> GetWorkspace -> Execute).  Works
        // for any atb::Operation wrapped in OperationHandle (Elewise, AsStrided,
        // Transpose, ...).  op is moved in and destroyed when the call returns.
        auto RunAtbOp = [&](OperationHandle op, atb::VariantPack& vp) -> Status {
            uint64_t ws_size = 0;
            atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
            if (atb_s != atb::NO_ERROR) {
                LOG_ERROR("PreprocessImageNpu: ATB op Setup failed, atbStatus=%d",
                          static_cast<int>(atb_s));
                return ERROR_INFERENCE;
            }
            uint8_t* ws = nullptr;
            if (ws_size > 0) {
                auto wp = runtime->GetWorkspace(ws_size);
                ws = wp.first;
                if (wp.second != STATUS_OK) {
                    LOG_ERROR("PreprocessImageNpu: GetWorkspace failed, size=%lu",
                              static_cast<unsigned long>(ws_size));
                    return wp.second;
                }
            }
            atb_s = op.get()->Execute(vp, ws, ws_size, ctx);
            if (atb_s != atb::NO_ERROR) {
                LOG_ERROR("PreprocessImageNpu: ATB op Execute failed, atbStatus=%d",
                          static_cast<int>(atb_s));
                return ERROR_INFERENCE;
            }
            return STATUS_OK;
        };

        // 4a. MULS(1/255): resized -> normalize_tmp
        {
            atb::VariantPack vp;
            vp.inTensors = {resized_tensor};
            vp.outTensors = {normalize_tmp};
            ret = RunAtbOp(ops::ElewiseOp::MakeMuls(1.0f / 255.0f), vp);
            if (ret != STATUS_OK) goto cleanup;
        }

        // 4b. ADD(-mean broadcast): normalize_tmp + mean_bc -> resized_tensor (overwrite)
        {
            atb::VariantPack vp;
            vp.inTensors = {normalize_tmp, mean_bc};
            vp.outTensors = {resized_tensor};
            ret = RunAtbOp(ops::ElewiseOp::MakeAdd(), vp);
            if (ret != STATUS_OK) goto cleanup;
        }

        // 4c. MUL(1/std broadcast): resized_tensor * inv_std_bc -> normalize_tmp (final)
        {
            atb::VariantPack vp;
            vp.inTensors = {resized_tensor, inv_std_bc};
            vp.outTensors = {normalize_tmp};
            ret = RunAtbOp(ops::ElewiseOp::MakeMul(), vp);
            if (ret != STATUS_OK) goto cleanup;
        }

        // Ensure all NPU ops complete before D2H
        ret = runtime->Synchronize();
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: Synchronize after normalize failed");
            goto cleanup;
        }

        // Free broadcast tensors and bicubic output (no longer needed);
        // Free() sets deviceData to nullptr so cleanup skips them.
        alloc->Free(mean_bc);
        alloc->Free(inv_std_bc);
        alloc->Free(resized_tensor);

        // ── Step 5: Patch extraction on NPU (AsStrided + 8-D Transpose) ──
        // Replaces the old D2H + CPU patch loop.  normalize_tmp [1,C,new_h,new_w]
        // is logically [C, new_h, new_w] = [C, merged_h*ms*ps, merged_w*ms*ps].
        // AsStrided reinterprets it as the 8-D patch grid [tp, C, merged_h, ms,
        // ps, merged_w, ms, ps] with a 0 element-stride on tp (broadcasts the
        // single frame to all tp frames — matches the CPU loop's f=0 -> f=1..tp-1
        // memcpy).  Transpose perm [2,5,3,6,1,0,4,7] reorders to [merged_h,
        // merged_w, ms, ms, C, tp, ps, ps], whose row-major layout is exactly
        // [num_patches, patch_dim].  Bit-exact vs CPU (test_patch_transpose_spike).
        int32_t grid_t = 1;
        int32_t grid_h = new_h / patch_size;
        int32_t grid_w = new_w / patch_size;
        int64_t merged_h = grid_h / merge_size;
        int64_t merged_w = grid_w / merge_size;

        grid_thw[0] = grid_t;
        grid_thw[1] = grid_h;
        grid_thw[2] = grid_w;
        num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;

        // 8-D shapes (grid_t=1 squeezed; tp handled by AsStrided broadcast).
        // shape7 = [C, merged_h, ms, ps, merged_w, ms, ps] is the contiguous
        // row-major view of [C, new_h, new_w]; shape8 prepends tp.
        const int64_t ms = merge_size;
        const int64_t ps = patch_size;
        std::vector<int64_t> shape7 = {channels, merged_h, ms, ps, merged_w, ms, ps};
        std::vector<int64_t> shape8 = {tp, channels, merged_h, ms, ps, merged_w, ms, ps};

        // Row-major element strides for shape7 (verified bit-exact in the spike);
        // tp dim gets stride 0 (broadcast).
        std::vector<int64_t> stride7(7);
        {
            int64_t s = 1;
            for (int i = 6; i >= 0; --i) { stride7[i] = s; s *= shape7[i]; }
        }
        std::vector<int64_t> stride8;
        stride8.push_back(0);
        for (int64_t s7 : stride7) stride8.push_back(s7);

        // 5a. AsStrided: normalize_tmp -> asstrided_out [shape8, stride8, offset 0]
        {
            atb::infer::AsStridedParam param;
            for (int64_t s : shape8)  param.size.push_back(s);
            for (int64_t s : stride8) param.stride.push_back(s);
            param.offset.push_back(0);

            atb::Operation* raw = nullptr;
            atb::Status atb_s = atb::CreateOperation(param, &raw);
            if (atb_s != atb::NO_ERROR || raw == nullptr) {
                LOG_ERROR("PreprocessImageNpu: AsStrided CreateOperation failed, atbStatus=%d",
                          static_cast<int>(atb_s));
                ret = ERROR_INFERENCE;
                goto cleanup;
            }
            OperationHandle asstrided_op(raw);

            ret = alloc->AllocFloat16(asstrided_out, shape8);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: AllocFloat16 asstrided_out failed");
                goto cleanup;
            }

            atb::VariantPack vp;
            vp.inTensors  = {normalize_tmp};
            vp.outTensors = {asstrided_out};
            ret = RunAtbOp(std::move(asstrided_op), vp);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: AsStrided execute failed");
                goto cleanup;
            }
        }

        // 5b. Transpose perm [2,5,3,6,1,0,4,7]:
        //   [tp,C,mh,ms,ps,mw,ms,ps] -> [mh,mw,ms,ms,C,tp,ps,ps]
        // whose row-major layout == [num_patches, patch_dim].
        {
            const std::vector<int32_t> perm = {2, 5, 3, 6, 1, 0, 4, 7};
            std::vector<int64_t> perm_out_shape(8);
            for (int i = 0; i < 8; i++) {
                perm_out_shape[i] = shape8[perm[i]];
            }

            OperationHandle transpose_op = ops::TransposeOp::Create(perm);
            if (!transpose_op) {
                LOG_ERROR("PreprocessImageNpu: TransposeOp::Create failed");
                ret = ERROR_INFERENCE;
                goto cleanup;
            }

            ret = alloc->AllocFloat16(transpose_out, perm_out_shape);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: AllocFloat16 transpose_out failed");
                goto cleanup;
            }

            atb::VariantPack vp;
            vp.inTensors  = {asstrided_out};
            vp.outTensors = {transpose_out};
            ret = RunAtbOp(std::move(transpose_op), vp);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: Transpose execute failed");
                goto cleanup;
            }
        }

        // ── Step 6: free intermediates (no D2H — path C keeps the device tensor) ──
        // transpose_out is already laid out as [num_patches, patch_dim] fp16.
        // The sync ensures the async AsStrided+Transpose complete before we free
        // their inputs (normalize_tmp, asstrided_out); it also makes transpose_out
        // valid for the caller (D2H in the CPU overload, direct feed in path C).
        // transpose_out itself is NOT freed here — ownership transfers to the caller.
        ret = runtime->Synchronize();
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpuInternal: Synchronize before freeing intermediates failed");
            goto cleanup;
        }

        // Free patch-pipeline intermediates (normalize_tmp was the AsStrided
        // input; asstrided_out the Transpose input). transpose_out is returned.
        alloc->Free(normalize_tmp);
        alloc->Free(asstrided_out);
    }

    return STATUS_OK;  // transpose_out valid, caller owns (Free or Detach)

cleanup:
    // Free in reverse allocation order; Free() is idempotent (checks deviceData
    // != nullptr and nulls it afterwards), so already-freed tensors are safe.
    if (transpose_out.deviceData != nullptr)   { alloc->Free(transpose_out); }
    if (asstrided_out.deviceData != nullptr)   { alloc->Free(asstrided_out); }
    if (inv_std_bc.deviceData != nullptr)      { alloc->Free(inv_std_bc); }
    if (mean_bc.deviceData != nullptr)         { alloc->Free(mean_bc); }
    if (normalize_tmp.deviceData != nullptr)   { alloc->Free(normalize_tmp); }
    if (resized_tensor.deviceData != nullptr)  { alloc->Free(resized_tensor); }
    if (input_tensor.deviceData != nullptr)    { alloc->Free(input_tensor); }

    return ret;
}

// ── CPU-pointer overload: NPU preprocess + D2H into a host buffer. ──
// Existing public contract (preprocessed pixel_values on host). The device
// tensor from the internal helper is D2H'd here then freed.
Status PreprocessImageNpu(IRuntime* runtime,
                          const uint8_t* image,
                          int32_t channels, int32_t height, int32_t width,
                          const Qwen3VLConfig& config,
                          uint16_t* pixel_values,
                          int64_t& num_patches,
                          int64_t* grid_thw) {
    if (runtime == nullptr || image == nullptr || pixel_values == nullptr || grid_thw == nullptr) {
        LOG_ERROR("PreprocessImageNpu: null parameter (runtime=%p, image=%p, pixel_values=%p, grid_thw=%p)",
                  static_cast<const void*>(runtime), static_cast<const void*>(image),
                  static_cast<const void*>(pixel_values), static_cast<const void*>(grid_thw));
        return ERROR_INVALID_PARAM;
    }

    atb::Tensor transpose_out{};
    Status ret = PreprocessImageNpuInternal(runtime, image, channels, height, width,
                                            config, transpose_out, num_patches, grid_thw);
    if (ret != STATUS_OK) return ret;

    // transpose_out is allocator-tracked and already synchronised by the helper
    // (its free-safety sync also makes the Transpose output valid for D2H), so a
    // single D2H delivers pixel_values directly — no CPU rearrange, no extra sync.
    auto* alloc = runtime->GetAllocator();
    int64_t patch_dim = static_cast<int64_t>(channels) * config.pp_temporal_patch_size *
                        config.pp_patch_size * config.pp_patch_size;
    ret = alloc->CopyToHost(pixel_values, transpose_out,
                            static_cast<size_t>(num_patches) * patch_dim * sizeof(uint16_t));
    alloc->Free(transpose_out);  // tracked; free regardless of D2H result
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: D2H transpose_out failed");
        return ret;
    }

    int32_t patch_size = config.pp_patch_size;
    LOG_INFO("PreprocessImageNpu: %dx%d -> %dx%d, grid=(%ld,%ld,%ld), patches=%ld, dim=%ld",
             height, width,
             static_cast<long>(grid_thw[1]) * patch_size, static_cast<long>(grid_thw[2]) * patch_size,
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(num_patches), static_cast<long>(patch_dim));
    return STATUS_OK;
}

// ── Device-tensor overload (path C): no D2H — the device tensor stays on NPU
// and is handed to the caller (ownership transferred via Detach). The caller
// wraps it with NpuTensor::Adopt (or aclrtFree's it) and feeds it directly to
// the vision pipeline, eliminating the D2H (here) → H2D (ForwardWithTiming)
// round trip of the CPU-pointer overload above. ──
Status PreprocessImageNpu(IRuntime* runtime,
                          const uint8_t* image,
                          int32_t channels, int32_t height, int32_t width,
                          const Qwen3VLConfig& config,
                          atb::Tensor& pixel_values_npu,
                          int64_t& num_patches,
                          int64_t* grid_thw) {
    if (runtime == nullptr || image == nullptr || grid_thw == nullptr) {
        LOG_ERROR("PreprocessImageNpu(device): null parameter (runtime=%p, image=%p, grid_thw=%p)",
                  static_cast<const void*>(runtime), static_cast<const void*>(image),
                  static_cast<const void*>(grid_thw));
        return ERROR_INVALID_PARAM;
    }

    Status ret = PreprocessImageNpuInternal(runtime, image, channels, height, width,
                                            config, pixel_values_npu, num_patches, grid_thw);
    if (ret != STATUS_OK) return ret;

    // Transfer device ownership to the caller: untrack from the allocator so
    // NpuTensor::Adopt (aclrtFree) won't clash with ~TensorAllocator/FreeAll.
    runtime->GetAllocator()->Detach(pixel_values_npu);

    // PatchEmbedGraph reads the input's desc.dims[0] to recover N (it assumes a
    // 1-D flat input of N*patch_dim). The Transpose output is 8-D; reinterpret
    // it as 1-D {num_patches*patch_dim} (same deviceData/dataSize, row-major).
    int64_t patch_dim = static_cast<int64_t>(channels) * config.pp_temporal_patch_size *
                        config.pp_patch_size * config.pp_patch_size;
    pixel_values_npu.desc.shape.dimNum = 1;
    pixel_values_npu.desc.shape.dims[0] = num_patches * patch_dim;
    // dataSize already == num_patches * patch_dim * sizeof(fp16); unchanged.

    int32_t patch_size = config.pp_patch_size;
    LOG_INFO("PreprocessImageNpu(device): %dx%d -> %dx%d, grid=(%ld,%ld,%ld), patches=%ld, dim=%ld [path C, no D2H]",
             height, width,
             static_cast<long>(grid_thw[1]) * patch_size, static_cast<long>(grid_thw[2]) * patch_size,
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(num_patches), static_cast<long>(patch_dim));
    return STATUS_OK;
}

}  // namespace adapters
}  // namespace atb_llm
