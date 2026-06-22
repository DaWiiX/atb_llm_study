#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#include "utils/cpp11_compat.h"
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

}  // namespace adapters
}  // namespace atb_llm
