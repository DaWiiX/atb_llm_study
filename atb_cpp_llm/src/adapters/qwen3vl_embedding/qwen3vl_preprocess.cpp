#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "utils/float_utils.h"
#include "log/logger.h"
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

void BicubicResize(const uint8_t* input, int32_t in_h, int32_t in_w,
                   int32_t channels, int32_t out_h, int32_t out_w,
                   float* output) {
    for (int32_t c = 0; c < channels; c++) {
        const uint8_t* in_c = input + static_cast<size_t>(c) * in_h * in_w;
        float* out_c = output + static_cast<size_t>(c) * out_h * out_w;

        for (int32_t oh = 0; oh < out_h; oh++) {
            // align_corners=False mapping
            float src_h = (oh + 0.5f) * static_cast<float>(in_h) / out_h - 0.5f;
            int32_t sh = static_cast<int32_t>(std::floor(src_h));
            float dh = src_h - sh;

            for (int32_t ow = 0; ow < out_w; ow++) {
                float src_w = (ow + 0.5f) * static_cast<float>(in_w) / out_w - 0.5f;
                int32_t sw = static_cast<int32_t>(std::floor(src_w));
                float dw = src_w - sw;

                float sum = 0.0f;
                for (int32_t m = -1; m <= 2; m++) {
                    float w_h = CubicWeight(dh - m);
                    int32_t ih = std::clamp(sh + m, 0, in_h - 1);
                    for (int32_t n = -1; n <= 2; n++) {
                        float w_w = CubicWeight(dw - n);
                        int32_t iw = std::clamp(sw + n, 0, in_w - 1);
                        sum += in_c[ih * in_w + iw] * w_h * w_w;
                    }
                }
                out_c[oh * out_w + ow] = sum;
            }
        }
    }
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

    // 2. Bicubic resize (uint8 -> float32, matches PyTorch align_corners=False)
    size_t resized_size = static_cast<size_t>(channels) * new_h * new_w;
    std::vector<float> resized(resized_size);
    BicubicResize(image, height, width, channels, new_h, new_w, resized.data());

    // 3. Rescale to [0, 1] and normalize
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

    for (int32_t c = 0; c < channels; c++) {
        for (int32_t i = 0; i < new_h * new_w; i++) {
            size_t idx = static_cast<size_t>(c) * new_h * new_w + i;
            resized[idx] = (resized[idx] / 255.0f - mean[c]) / std_val[c];
        }
    }

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

    int64_t patch_idx = 0;
    for (int64_t br = 0; br < merged_h; br++) {
        for (int64_t bc = 0; bc < merged_w; bc++) {
            for (int64_t ir = 0; ir < merge_size; ir++) {
                for (int64_t ic = 0; ic < merge_size; ic++) {
                    // This is one output patch
                    // Dimension order: C, tp, ps_h, ps_w (matches Python permute)
                    int64_t offset = 0;
                    for (int64_t c = 0; c < channels; c++) {
                        for (int64_t f = 0; f < tp; f++) {
                            for (int64_t ph = 0; ph < patch_size; ph++) {
                                for (int64_t pw = 0; pw < patch_size; pw++) {
                                    int64_t row = br * merge_size * patch_size + ir * patch_size + ph;
                                    int64_t col = bc * merge_size * patch_size + ic * patch_size + pw;
                                    float val = resized[c * new_h * new_w + row * new_w + col];
                                    // Store as fp16
                                    uint16_t fp16_val = atb_llm::Fp32ToFp16(val);
                                    pixel_values[patch_idx * patch_dim + offset] = fp16_val;
                                    offset++;
                                }
                            }
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
