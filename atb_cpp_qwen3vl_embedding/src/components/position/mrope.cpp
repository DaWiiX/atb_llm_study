#include "components/position/mrope.h"
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
    // freq_table[pos, i] = pos * inv_freq[i]
    // Shape: (max_hw, dim)
    int32_t half = dim_ / 2;
    std::vector<float> table(max_hw * dim_);
    for (int32_t p = 0; p < max_hw; p++) {
        for (int32_t i = 0; i < half; i++) {
            table[p * dim_ + i] = static_cast<float>(p) * inv_freq_[i];
        }
    }
    return table;
}

int64_t ComputeVisionRotPosEmb(const int64_t* grid_thw, int64_t num_images,
                                const VisionRotaryEmbedding& rotary_emb,
                                int32_t merge_size,
                                const std::vector<float>& freq_table,
                                float* out_cos, float* out_sin) {
    int32_t dim = rotary_emb.Dim();
    // freq_table shape: (max_hw, dim)
    // We need to compute position IDs (row_idx, col_idx) for each token,
    // then look up freq_table and produce cos/sin.

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
                            const float* row_freq = freq_table.data() + row_idx * dim;
                            const float* col_freq = freq_table.data() + col_idx * dim;

                            // Concatenate: [row_freq..., col_freq...] -> (dim*2,)
                            // Then cos/sin. But output is (total_tokens, dim*2) in Python.
                            // Actually in Python: freq_table[pos_ids].flatten(1) -> (total, dim*2)
                            // pos_ids has shape (total, 2) with (row_idx, col_idx)
                            // freq_table[pos_ids] -> (total, 2, dim) -> flatten -> (total, 2*dim)
                            float* cos_row = out_cos + offset * dim * 2;
                            float* sin_row = out_sin + offset * dim * 2;
                            for (int32_t d = 0; d < dim; d++) {
                                // row component
                                cos_row[d] = std::cos(row_freq[d]);
                                sin_row[d] = std::sin(row_freq[d]);
                            }
                            for (int32_t d = 0; d < dim; d++) {
                                // col component
                                cos_row[dim + d] = std::cos(col_freq[d]);
                                sin_row[dim + d] = std::sin(col_freq[d]);
                            }

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
                  int64_t image_token_id, int64_t spatial_merge_size,
                  int64_t* position_ids_out) {
    // position_ids_out: (3, B, S) int64
    // For text-only case (no images): sequential positions for all 3 dims
    if (image_grid_thw == nullptr || num_images == 0) {
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t s = 0; s < seq_len; s++) {
                int64_t pos = s;
                // All 3 dims get the same sequential position
                position_ids_out[0 * batch_size * seq_len + b * seq_len + s] = pos;
                position_ids_out[1 * batch_size * seq_len + b * seq_len + s] = pos;
                position_ids_out[2 * batch_size * seq_len + b * seq_len + s] = pos;
            }
        }
        return;
    }

    // With images: visual tokens get 2D grid positions, text tokens sequential
    // Simplified implementation for batch_size=1 (common for embedding models)
    int64_t image_index = 0;
    int64_t st = 0;
    int64_t pos_counter = 0;

    const int64_t* ids = input_ids;  // batch 0

    // Count image tokens
    int64_t image_count = 0;
    for (int64_t s = 0; s < seq_len; s++) {
        if (ids[s] == image_token_id) image_count++;
    }

    // Process text segments and image regions
    int64_t remaining_images = image_count;

    for (int64_t s = 0; s < seq_len && remaining_images > 0; s++) {
        if (ids[s] == image_token_id) {
            // Text tokens before this image token
            int64_t text_len = s - st;
            for (int64_t t = 0; t < text_len; t++) {
                int64_t idx = st + t;
                for (int d = 0; d < 3; d++) {
                    position_ids_out[d * batch_size * seq_len + idx] = pos_counter + t;
                }
            }
            pos_counter += text_len;
            st = s;

            // Image tokens: 2D grid positions
            if (image_index < num_images) {
                int64_t t_dim = image_grid_thw[image_index * 3 + 0];
                int64_t h_dim = image_grid_thw[image_index * 3 + 1] / spatial_merge_size;
                int64_t w_dim = image_grid_thw[image_index * 3 + 2] / spatial_merge_size;

                int64_t img_tokens = t_dim * h_dim * w_dim;
                int64_t img_offset = 0;
                for (int64_t ti = 0; ti < t_dim && st + img_offset < seq_len; ti++) {
                    for (int64_t hi = 0; hi < h_dim && st + img_offset < seq_len; hi++) {
                        for (int64_t wi = 0; wi < w_dim && st + img_offset < seq_len; wi++) {
                            int64_t idx = st + img_offset;
                            position_ids_out[0 * batch_size * seq_len + idx] = pos_counter + ti;
                            position_ids_out[1 * batch_size * seq_len + idx] = pos_counter + hi;
                            position_ids_out[2 * batch_size * seq_len + idx] = pos_counter + wi;
                            img_offset++;
                        }
                    }
                }

                st += img_offset;
                pos_counter += std::max({t_dim, h_dim, w_dim});
                image_index++;
            }
            remaining_images--;
        }
    }

    // Remaining text tokens after last image
    if (st < seq_len) {
        int64_t text_len = seq_len - st;
        for (int64_t t = 0; t < text_len; t++) {
            int64_t idx = st + t;
            for (int d = 0; d < 3; d++) {
                position_ids_out[d * batch_size * seq_len + idx] = pos_counter + t;
            }
        }
    }
}

} // namespace components
} // namespace atb_llm
