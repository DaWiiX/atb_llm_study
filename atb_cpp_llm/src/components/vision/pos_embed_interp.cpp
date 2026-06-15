#include "components/vision/pos_embed_interp.h"
#include "utils/float_utils.h"
#include <algorithm>
#include <cstring>
#include <vector>

namespace atb_llm {
namespace components {

void ComputePosEmbedInterp(const uint16_t* pos_embed_src,
                           int32_t hidden_size, int32_t num_grid,
                           int32_t merge_size,
                           const int64_t* grid_thw, int64_t num_images,
                           uint16_t* pos_out) {
    int64_t vis_hs = hidden_size;

    // Interpolate for each image
    int64_t out_offset = 0;
    for (int64_t img = 0; img < num_images; img++) {
        int64_t t = grid_thw[img * 3 + 0];
        int64_t h = grid_thw[img * 3 + 1];
        int64_t w = grid_thw[img * 3 + 2];

        // Bilinear interpolation from (num_grid, num_grid) to (h, w)
        std::vector<float> interpolated(h * w * vis_hs, 0.0f);

        for (int64_t hi = 0; hi < h; hi++) {
            for (int64_t wi = 0; wi < w; wi++) {
                float fy = (h <= 1) ? 0.0f : static_cast<float>(hi) * (num_grid - 1) / (h - 1);
                float fx = (w <= 1) ? 0.0f : static_cast<float>(wi) * (num_grid - 1) / (w - 1);

                int32_t y0 = static_cast<int32_t>(fy);
                int32_t x0 = static_cast<int32_t>(fx);
                int32_t y1 = std::min(y0 + 1, num_grid - 1);
                int32_t x1 = std::min(x0 + 1, num_grid - 1);

                float dy = fy - y0;
                float dx = fx - x0;

                int64_t idx00 = y0 * num_grid + x0;
                int64_t idx01 = y0 * num_grid + x1;
                int64_t idx10 = y1 * num_grid + x0;
                int64_t idx11 = y1 * num_grid + x1;

                float* out_row = interpolated.data() + (hi * w + wi) * vis_hs;
                for (int64_t d = 0; d < vis_hs; d++) {
                    // Read fp16 values and convert to fp32
                    float v00 = atb_llm::Fp16ToF32(pos_embed_src[idx00 * vis_hs + d]);
                    float v01 = atb_llm::Fp16ToF32(pos_embed_src[idx01 * vis_hs + d]);
                    float v10 = atb_llm::Fp16ToF32(pos_embed_src[idx10 * vis_hs + d]);
                    float v11 = atb_llm::Fp16ToF32(pos_embed_src[idx11 * vis_hs + d]);

                    out_row[d] = v00 * (1 - dy) * (1 - dx) + v01 * (1 - dy) * dx +
                                 v10 * dy * (1 - dx) + v11 * dy * dx;
                }
            }
        }

        // Apply spatial merge shuffling and convert to fp16
        // Python: pos_embed.view(t, h//ms, ms, w//ms, ms, -1).permute(0,1,3,2,4,5).flatten(0,4)
        int64_t merged_h = h / merge_size;
        int64_t merged_w = w / merge_size;

        int64_t patch_idx = 0;
        for (int64_t br = 0; br < merged_h; br++) {
            for (int64_t bc = 0; bc < merged_w; bc++) {
                for (int64_t ir = 0; ir < merge_size; ir++) {
                    for (int64_t ic = 0; ic < merge_size; ic++) {
                        int64_t row = br * merge_size + ir;
                        int64_t col = bc * merge_size + ic;
                        const float* src = interpolated.data() + (row * w + col) * vis_hs;
                        uint16_t* dst = pos_out + (out_offset + patch_idx) * vis_hs;
                        for (int64_t d = 0; d < vis_hs; d++) {
                            // float32 -> fp16 (round-to-nearest-even, matches Python)
                            dst[d] = atb_llm::Fp32ToFp16(src[d]);
                        }
                        patch_idx++;
                    }
                }
            }
        }
        // Repeat for temporal dimension
        for (int64_t ti = 1; ti < t; ti++) {
            std::memcpy(pos_out + (out_offset + ti * merged_h * merged_w * merge_size * merge_size) * vis_hs,
                        pos_out + out_offset * vis_hs,
                        merged_h * merged_w * merge_size * merge_size * vis_hs * sizeof(uint16_t));
        }
        out_offset += t * h * w;
    }
}

}  // namespace components
}  // namespace atb_llm
