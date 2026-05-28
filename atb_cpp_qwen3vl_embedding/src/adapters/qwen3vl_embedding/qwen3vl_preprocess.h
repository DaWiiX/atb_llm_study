#pragma once
#include "atb_llm/types.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace adapters {

/// Qwen3VL image preprocessor (CPU-side).
///
/// Pipeline:
///   image (C,H,W) uint8 -> smart_resize -> rescale -> normalize
///   -> patch extraction -> pixel_values (N, C*tp*p*p) fp16
///
/// Equivalent to Python preprocess.py.

/// Resize image dimensions to be divisible by factor, within pixel limits.
/// Returns (new_height, new_width).
void SmartResize(int32_t height, int32_t width,
                 int32_t factor, int32_t min_pixels, int32_t max_pixels,
                 int32_t& new_height, int32_t& new_width);

/// Bilinear resize (CPU).
/// @param input    (C, H, W) uint8
/// @param in_h     Input height
/// @param in_w     Input width
/// @param channels Number of channels
/// @param out_h    Output height
/// @param out_w    Output width
/// @param output   (C, out_h, out_w) float32, pre-allocated
void BilinearResize(const uint8_t* input, int32_t in_h, int32_t in_w,
                    int32_t channels, int32_t out_h, int32_t out_w,
                    float* output);

/// Full image preprocessing.
///
/// @param image         (C, H, W) uint8, pixel values in [0, 255]
/// @param channels      Number of channels (typically 3)
/// @param height        Image height
/// @param width         Image width
/// @param config        Qwen3VL config (preprocessor params)
/// @param pixel_values  Output: (N, patch_dim) float16, pre-allocated
///                      patch_dim = C * temporal_patch_size * patch_size^2
/// @param num_patches   Output: N (number of patches)
/// @param grid_thw      Output: (3,) [grid_t, grid_h, grid_w]
Status PreprocessImage(const uint8_t* image,
                       int32_t channels, int32_t height, int32_t width,
                       const Qwen3VLConfig& config,
                       uint16_t* pixel_values,
                       int64_t& num_patches,
                       int64_t* grid_thw);

} // namespace adapters
} // namespace atb_llm
