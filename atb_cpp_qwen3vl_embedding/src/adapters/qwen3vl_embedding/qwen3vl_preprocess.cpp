#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "log/logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace atb_llm {
namespace adapters {

void SmartResize(int32_t height, int32_t width,
                 int32_t factor, int32_t min_pixels, int32_t max_pixels,
                 int32_t& new_height, int32_t& new_width) {
    // Round to nearest multiple of factor
    int32_t h_bar = static_cast<int32_t>(std::round(
        static_cast<float>(height) / factor)) * factor;
    int32_t w_bar = static_cast<int32_t>(std::round(
        static_cast<float>(width) / factor)) * factor;

    int64_t area = static_cast<int64_t>(h_bar) * w_bar;
    if (area > max_pixels) {
        double beta = std::sqrt(static_cast<double>(height * width) / max_pixels);
        h_bar = std::max(factor, static_cast<int32_t>(
            std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int32_t>(
            std::floor(width / beta / factor)) * factor);
    } else if (area < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
        h_bar = static_cast<int32_t>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int32_t>(std::ceil(width * beta / factor)) * factor;
    }

    new_height = h_bar;
    new_width = w_bar;
}

void BilinearResize(const uint8_t* input, int32_t in_h, int32_t in_w,
                    int32_t channels, int32_t out_h, int32_t out_w,
                    float* output) {
    // Bilinear interpolation for uint8 -> float32
    for (int32_t oh = 0; oh < out_h; oh++) {
        for (int32_t ow = 0; ow < out_w; ow++) {
            // Map output coord to input coord
            float fy = static_cast<float>(oh) * in_h / out_h;
            float fx = static_cast<float>(ow) * in_w / out_w;

            int32_t y0 = static_cast<int32_t>(fy);
            int32_t x0 = static_cast<int32_t>(fx);
            int32_t y1 = std::min(y0 + 1, in_h - 1);
            int32_t x1 = std::min(x0 + 1, in_w - 1);

            float dy = fy - y0;
            float dx = fx - x0;

            for (int32_t c = 0; c < channels; c++) {
                // Input layout: (C, H, W) -> index: c*H*W + h*W + w
                float v00 = input[c * in_h * in_w + y0 * in_w + x0];
                float v01 = input[c * in_h * in_w + y0 * in_w + x1];
                float v10 = input[c * in_h * in_w + y1 * in_w + x0];
                float v11 = input[c * in_h * in_w + y1 * in_w + x1];

                float val = v00 * (1 - dy) * (1 - dx) +
                            v01 * (1 - dy) * dx +
                            v10 * dy * (1 - dx) +
                            v11 * dy * dx;

                // Output layout: (C, out_h, out_w)
                output[c * out_h * out_w + oh * out_w + ow] = val;
            }
        }
    }
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

    // 2. Bilinear resize (uint8 -> float32)
    size_t resized_size = static_cast<size_t>(channels) * new_h * new_w;
    std::vector<float> resized(resized_size);
    BilinearResize(image, height, width, channels, new_h, new_w, resized.data());

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
    int32_t total_frames = tp;  // For single image, we pad to tp=2
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
    // We implement this directly without 9D reshape.
    // Each output patch (n, :) contains:
    //   For each merge-group (br, bc), for each intra (ir, ic),
    //   for each channel c, for each frame f, for each spatial (ph, pw):
    //     value = normalized[ c, br*ms*ps + ir*ps + ph, bc*ms*ps + ic*ps + pw ]
    //   (duplicated for both frames since single image)

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
                    int64_t offset = 0;
                    // Frame 0 (same as frame 1 for single image)
                    for (int64_t c = 0; c < channels; c++) {
                        for (int64_t ph = 0; ph < patch_size; ph++) {
                            for (int64_t pw = 0; pw < patch_size; pw++) {
                                int64_t row = br * merge_size * patch_size + ir * patch_size + ph;
                                int64_t col = bc * merge_size * patch_size + ic * patch_size + pw;
                                float val = resized[c * new_h * new_w + row * new_w + col];
                                // Store as fp16
                                uint16_t fp16_val = Bf16ToFp16(
                                    static_cast<uint16_t>(
                                        reinterpret_cast<uint32_t&>(val) >> 16));
                                pixel_values[patch_idx * patch_dim + offset] = fp16_val;
                                offset++;
                            }
                        }
                    }
                    // Frame 1 (duplicate for temporal_patch_size=2)
                    for (int64_t c = 0; c < channels; c++) {
                        for (int64_t ph = 0; ph < patch_size; ph++) {
                            for (int64_t pw = 0; pw < patch_size; pw++) {
                                int64_t row = br * merge_size * patch_size + ir * patch_size + ph;
                                int64_t col = bc * merge_size * patch_size + ic * patch_size + pw;
                                float val = resized[c * new_h * new_w + row * new_w + col];
                                uint16_t fp16_val = Bf16ToFp16(
                                    static_cast<uint16_t>(
                                        reinterpret_cast<uint32_t&>(val) >> 16));
                                pixel_values[patch_idx * patch_dim + offset] = fp16_val;
                                offset++;
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

} // namespace adapters
} // namespace atb_llm
