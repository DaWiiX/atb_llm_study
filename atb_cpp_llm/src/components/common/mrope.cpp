#include "components/common/mrope.h"
#include "log/logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace atb_llm {
namespace components {

// ═════════════════════════════════════════════════════════════════════
// MRoPE
// ═════════════════════════════════════════════════════════════════════

MRoPE::MRoPE(int32_t head_dim, float rope_theta,
             const std::vector<int32_t>& mrope_section)
    : head_dim_(head_dim), rope_theta_(rope_theta), mrope_section_(mrope_section) {
    // inv_freq[i] = 1.0 / (theta^(2i/dim)) for i in [0, dim/2)
    int32_t half_dim = head_dim / 2;
    inv_freq_.resize(half_dim);
    for (int32_t i = 0; i < half_dim; i++) {
        inv_freq_[i] = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
    }
}

void MRoPE::ApplyInterleaved(const float* freqs_3d,
                               int64_t batch_size, int64_t seq_len,
                               float* out) const {
    // freqs_3d: [3, B, S, head_dim/2]
    // out: [B, S, head_dim/2]
    // Start with T values, then interleave H and W at appropriate offsets.
    int64_t half_dim = head_dim_ / 2;
    int64_t plane_stride = batch_size * seq_len * half_dim;

    // Copy T plane as base
    const float* t_plane = freqs_3d;
    std::memcpy(out, t_plane, batch_size * seq_len * half_dim * sizeof(float));

    // Interleave H and W
    for (int dim_idx = 1; dim_idx <= 2; dim_idx++) {
        int32_t section_length = mrope_section_[dim_idx] * 3;
        const float* src = freqs_3d + dim_idx * plane_stride;

        // Replace every 3rd element starting from offset=dim_idx
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t s = 0; s < seq_len; s++) {
                float* dst_row = out + (b * seq_len + s) * half_dim;
                const float* src_row = src + (b * seq_len + s) * half_dim;
                for (int32_t idx = dim_idx; idx < section_length && idx < half_dim; idx += 3) {
                    dst_row[idx] = src_row[idx];
                }
            }
        }
    }
}

void MRoPE::Compute(const int64_t* position_ids,
                     int64_t batch_size, int64_t seq_len,
                     float* cos_out, float* sin_out) const {
    int64_t half_dim = head_dim_ / 2;
    int64_t plane_size = batch_size * seq_len * half_dim;

    // Compute freqs for all 3 dims: freqs_3d[d, b, s, i] = inv_freq[i] * position_ids[d, b, s]
    std::vector<float> freqs_3d(3 * plane_size);

    for (int d = 0; d < 3; d++) {
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t s = 0; s < seq_len; s++) {
                float pos = static_cast<float>(position_ids[d * batch_size * seq_len + b * seq_len + s]);
                float* row = freqs_3d.data() + d * plane_size + (b * seq_len + s) * half_dim;
                for (int32_t i = 0; i < half_dim; i++) {
                    row[i] = inv_freq_[i] * pos;
                }
            }
        }
    }

    // Apply interleaved MRoPE
    std::vector<float> interleaved(plane_size);
    ApplyInterleaved(freqs_3d.data(), batch_size, seq_len, interleaved.data());

    // Concat(freqs, freqs) then cos/sin -> output (B*S, head_dim)
    int64_t total = batch_size * seq_len;
    for (int64_t t = 0; t < total; t++) {
        const float* row = interleaved.data() + t * half_dim;
        float* c = cos_out + t * head_dim_;
        float* s_out = sin_out + t * head_dim_;
        for (int32_t i = 0; i < half_dim; i++) {
            float val = row[i];
            c[i] = std::cos(val);
            c[i + half_dim] = std::cos(val);
            s_out[i] = std::sin(val);
            s_out[i + half_dim] = std::sin(val);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// VisionRotaryEmbedding
// ═════════════════════════════════════════════════════════════════════

VisionRotaryEmbedding::VisionRotaryEmbedding(int32_t dim) : dim_(dim) {
    // inv_freq[i] = 1.0 / (10000^(2i/dim)) for i in [0, dim/2)
    int32_t half = dim / 2;
    inv_freq_.resize(half);
    for (int32_t i = 0; i < half; i++) {
        inv_freq_[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / dim);
    }
}

std::vector<float> VisionRotaryEmbedding::ComputeFreqTable(int32_t max_hw) const {
    // Python: einsum("i,j->ij", t, inv_freq) -> (max_hw, dim//2)
    // inv_freq has dim//2 elements, so output has dim//2 columns
    int32_t half = dim_ / 2;
    std::vector<float> table(max_hw * half);
    for (int32_t p = 0; p < max_hw; p++) {
        for (int32_t i = 0; i < half; i++) {
            table[p * half + i] = static_cast<float>(p) * inv_freq_[i];
        }
    }
    return table;
}

int64_t VisionRotaryEmbedding::ComputeRoPE(const int64_t* grid_thw, int64_t num_images,
                                            int32_t merge_size,
                                            float* cos_out, float* sin_out) const {
    // Find max_hw across all images
    int32_t max_hw = 0;
    for (int64_t i = 0; i < num_images; i++) {
        max_hw = std::max(max_hw, static_cast<int32_t>(grid_thw[i * 3 + 1]));
        max_hw = std::max(max_hw, static_cast<int32_t>(grid_thw[i * 3 + 2]));
    }

    // Compute freq table and delegate to ComputeVisionRotPosEmb
    auto freq_table = ComputeFreqTable(max_hw);
    return ComputeVisionRotPosEmb(grid_thw, num_images, *this, merge_size,
                                  freq_table, cos_out, sin_out);
}

int64_t ComputeVisionRotPosEmb(const int64_t* grid_thw, int64_t num_images,
                                const VisionRotaryEmbedding& rotary_emb,
                                int32_t merge_size,
                                const std::vector<float>& freq_table,
                                float* out_cos, float* out_sin) {
    int32_t dim = rotary_emb.Dim();
    int32_t half = dim / 2;
    // freq_table shape: (max_hw, half) — matches Python einsum output
    // rope = Concat(row_freq, col_freq) -> (half*2,) = (dim,)
    // emb = Concat(rope, rope) -> (dim*2,)
    // cos/sin per token: dim*2 elements = vis_hd

    // First pass: compute total tokens
    int64_t total_tokens = 0;
    for (int64_t i = 0; i < num_images; i++) {
        int64_t t = grid_thw[i * 3 + 0];
        int64_t h = grid_thw[i * 3 + 1];
        int64_t w = grid_thw[i * 3 + 2];
        total_tokens += t * h * w;
    }

    // Second pass: compute position IDs and fill cos/sin
    int64_t offset = 0;
    for (int64_t img = 0; img < num_images; img++) {
        int64_t t = grid_thw[img * 3 + 0];
        int64_t h = grid_thw[img * 3 + 1];
        int64_t w = grid_thw[img * 3 + 2];
        int64_t merged_h = h / merge_size;
        int64_t merged_w = w / merge_size;

        // Compute row_idx and col_idx for each token (matching Python logic)
        for (int64_t ti = 0; ti < t; ti++) {
            for (int64_t br = 0; br < merged_h; br++) {
                for (int64_t bc = 0; bc < merged_w; bc++) {
                    for (int64_t ir = 0; ir < merge_size; ir++) {
                        for (int64_t ic = 0; ic < merge_size; ic++) {
                            int64_t row_idx = br * merge_size + ir;
                            int64_t col_idx = bc * merge_size + ic;

                            // Look up freq_table for row and col
                            // freq_table is (max_hw, half), stride = half
                            const float* row_freq = freq_table.data() + row_idx * half;
                            const float* col_freq = freq_table.data() + col_idx * half;

                            // Python: rope = Concat(row_freq, col_freq) -> (half*2,)
                            //         emb = Concat(rope, rope) -> (half*4,)
                            //         cos_out = Cos(emb), sin_out = Sin(emb)
                            // Output per token: half*4 = dim*2 elements
                            int32_t out_dim = half * 4;  // = dim * 2
                            float* cos_row = out_cos + offset * out_dim;
                            float* sin_row = out_sin + offset * out_dim;
                            // First half: [row, col] -> half*2 elements
                            for (int32_t d = 0; d < half; d++) {
                                cos_row[d] = std::cos(row_freq[d]);
                                sin_row[d] = std::sin(row_freq[d]);
                            }
                            for (int32_t d = 0; d < half; d++) {
                                cos_row[half + d] = std::cos(col_freq[d]);
                                sin_row[half + d] = std::sin(col_freq[d]);
                            }
                            // Second half: duplicate [row, col]
                            std::memcpy(cos_row + half * 2, cos_row, half * 2 * sizeof(float));
                            std::memcpy(sin_row + half * 2, sin_row, half * 2 * sizeof(float));

                            offset++;
                        }
                    }
                }
            }
        }
    }

    return total_tokens;
}

// ═════════════════════════════════════════════════════════════════════
// GetRopeIndex
// ═════════════════════════════════════════════════════════════════════

void GetRopeIndex(const int64_t* input_ids,
                  int64_t batch_size, int64_t seq_len,
                  const int64_t* image_grid_thw, int64_t num_images,
                  int64_t image_token_id, int64_t vision_start_token_id,
                  int64_t spatial_merge_size,
                  int64_t* position_ids_out) {
    // position_ids_out: (3, B, S) int64, stored as contiguous [3][B][S]

    // Count image segments: look for vision_start_token_id followed by image_token_id.
    // Matches Python: vision_start_indices = argwhere(ids == vision_start_token_id)
    //                 vision_tokens = ids[vision_start_indices + 1]
    //                 image_nums = (vision_tokens == image_token_id).sum()
    int64_t image_nums = 0;
    if (image_grid_thw != nullptr && num_images > 0) {
        for (int64_t b = 0; b < batch_size; b++) {
            const int64_t* ids = input_ids + b * seq_len;
            for (int64_t s = 0; s < seq_len - 1; s++) {
                if (ids[s] == vision_start_token_id &&
                    ids[s + 1] == image_token_id) {
                    image_nums++;
                    if (image_nums >= num_images) break;
                }
            }
            if (image_nums >= num_images) break;
        }
    }

    // No image segments detected: use sequential positions for all tokens.
    // This matches Python's fallback for inputs without vision_start_token_id.
    if (image_nums == 0) {
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t s = 0; s < seq_len; s++) {
                int64_t pos = s;
                position_ids_out[0 * batch_size * seq_len + b * seq_len + s] = pos;
                position_ids_out[1 * batch_size * seq_len + b * seq_len + s] = pos;
                position_ids_out[2 * batch_size * seq_len + b * seq_len + s] = pos;
            }
        }
        return;
    }

    // Process batches with image segments. For each batch:
    //   1. Find image segments (marked by image_token_id after vision_start_token_id)
    //   2. Text before each image: sequential positions
    //   3. Image tokens: 2D grid positions (T,H,W dims)
    //   4. Text after last image: sequential positions
    for (int64_t b = 0; b < batch_size; b++) {
        const int64_t* ids = input_ids + b * seq_len;
        int64_t* pid = position_ids_out + b * seq_len;  // points to (3, S) for this batch
        int64_t stride_bs = batch_size * seq_len;  // stride between T/H/W dims

        int64_t image_index = 0;
        int64_t st = 0;          // current position in the sequence
        int64_t pos_counter = 0; // next position index for text tokens

        for (int64_t iter = 0; iter < image_nums; iter++) {
            // Find first image_token_id at or after st (this is the start of image segment)
            int64_t ed = st;
            while (ed < seq_len && ids[ed] != image_token_id) {
                ed++;
            }
            if (ed >= seq_len) break;

            // Text tokens before this image segment: all 3 dims get sequential values
            int64_t text_len = ed - st;
            for (int64_t t = 0; t < text_len; t++) {
                int64_t idx = st + t;
                int64_t val = pos_counter + t;
                pid[0 * stride_bs + idx] = val;
                pid[1 * stride_bs + idx] = val;
                pid[2 * stride_bs + idx] = val;
            }
            pos_counter += text_len;

            // Image tokens: 2D grid positions
            if (image_index < num_images) {
                int64_t t_dim = image_grid_thw[image_index * 3 + 0];
                int64_t h_dim = image_grid_thw[image_index * 3 + 1] / spatial_merge_size;
                int64_t w_dim = image_grid_thw[image_index * 3 + 2] / spatial_merge_size;

                int64_t img_offset = 0;
                for (int64_t ti = 0; ti < t_dim; ti++) {
                    for (int64_t hi = 0; hi < h_dim; hi++) {
                        for (int64_t wi = 0; wi < w_dim; wi++) {
                            int64_t idx = ed + img_offset;
                            if (idx >= seq_len) break;
                            pid[0 * stride_bs + idx] = pos_counter + ti;
                            pid[1 * stride_bs + idx] = pos_counter + hi;
                            pid[2 * stride_bs + idx] = pos_counter + wi;
                            img_offset++;
                        }
                    }
                }

                st = ed + t_dim * h_dim * w_dim;
                pos_counter += std::max({t_dim, h_dim, w_dim});
                image_index++;
            } else {
                st = ed;
            }
        }

        // Remaining text after last image segment
        if (st < seq_len) {
            int64_t text_len = seq_len - st;
            for (int64_t t = 0; t < text_len; t++) {
                int64_t idx = st + t;
                int64_t val = pos_counter + t;
                pid[0 * stride_bs + idx] = val;
                pid[1 * stride_bs + idx] = val;
                pid[2 * stride_bs + idx] = val;
            }
        }
    }
}

} // namespace components
} // namespace atb_llm
