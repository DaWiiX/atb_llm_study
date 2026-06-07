#pragma once
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace components {

/// MRoPE (Multi-dimensional Rotary Position Embedding) for text model.
///
/// This is CPU-side computation (no ATB graph).
/// Takes 3D position_ids (T, H, W) and produces cos/sin values
/// for the ATB RopeOperation.
///
/// Equivalent to Python engine_utils.py: TextRotaryEmbedding.
class MRoPE {
public:
    /// Constructor.
    /// @param head_dim      Dimension per attention head (128)
    /// @param rope_theta    Base frequency (5,000,000 for Qwen3VL)
    /// @param mrope_section Section sizes for T, H, W dims (e.g. {24, 20, 20})
    MRoPE(int32_t head_dim, float rope_theta,
          const std::vector<int32_t>& mrope_section);

    /// Compute MRoPE cos/sin from 3D position IDs.
    ///
    /// @param position_ids  Flattened 3D position IDs, layout [3][B][S] as contiguous int64.
    ///                      Dim 0 = T, H, W (3 elements), Dim 1 = batch, Dim 2 = sequence.
    /// @param batch_size    Batch size B
    /// @param seq_len       Sequence length S
    /// @param cos_out       Output: cos values, shape (B*S, head_dim), float32, pre-allocated
    /// @param sin_out       Output: sin values, shape (B*S, head_dim), float32, pre-allocated
    void Compute(const int64_t* position_ids,
                 int64_t batch_size, int64_t seq_len,
                 float* cos_out, float* sin_out) const;

    int32_t HeadDim() const { return head_dim_; }
    const std::vector<int32_t>& MropeSection() const { return mrope_section_; }

private:
    int32_t head_dim_;
    float rope_theta_;
    std::vector<int32_t> mrope_section_;
    std::vector<float> inv_freq_;  // head_dim/2 elements

    /// Apply interleaved MRoPE: reorganize chunked [T...H...W] -> interleaved.
    void ApplyInterleaved(const float* freqs_3d,
                          int64_t batch_size, int64_t seq_len,
                          float* out) const;
};

/// Vision Rotary Embedding for vision encoder.
///
/// Pre-computed 2D rotary frequency table.
/// Equivalent to Python engine_utils.py: VisionRotaryEmbedding.
class VisionRotaryEmbedding {
public:
    /// Constructor.
    /// @param dim  head_dim // 2 = hidden_size // (2 * num_heads)
    explicit VisionRotaryEmbedding(int32_t dim);

    /// Generate frequency table for positions up to max_hw.
    /// @param max_hw  Maximum spatial dimension
    /// @return Flattened frequency table, shape (max_hw, dim), row-major
    std::vector<float> ComputeFreqTable(int32_t max_hw) const;

    /// Convenience: compute vision RoPE cos/sin for given grid.
    /// Equivalent to calling ComputeFreqTable + ComputeVisionRotPosEmb.
    /// @param grid_thw     Grid dimensions [t, h, w] per image, flattened (N*3)
    /// @param num_images    Number of images N
    /// @param merge_size    Spatial merge size
    /// @param cos_out       Output: cos values, shape (total_tokens, dim*2), pre-allocated
    /// @param sin_out       Output: sin values, shape (total_tokens, dim*2), pre-allocated
    /// @return total number of tokens
    int64_t ComputeRoPE(const int64_t* grid_thw, int64_t num_images,
                        int32_t merge_size,
                        float* cos_out, float* sin_out) const;

    int32_t Dim() const { return dim_; }

private:
    int32_t dim_;
    std::vector<float> inv_freq_;  // dim/2 elements
};

/// Compute vision 2D rotary position embeddings.
///
/// Equivalent to Python engine_utils.py: compute_rot_pos_emb.
///
/// @param grid_thw       Grid dimensions [t, h, w] for each image, flattened (N*3)
/// @param num_images     Number of images N
/// @param rotary_emb     VisionRotaryEmbedding instance
/// @param merge_size     Spatial merge size
/// @param freq_table     Pre-computed freq table from VisionRotaryEmbedding
/// @param out_cos        Output: cos values, shape (total_tokens, head_dim), pre-allocated
/// @param out_sin        Output: sin values, shape (total_tokens, head_dim), pre-allocated
/// @return total number of tokens
int64_t ComputeVisionRotPosEmb(const int64_t* grid_thw, int64_t num_images,
                                const VisionRotaryEmbedding& rotary_emb,
                                int32_t merge_size,
                                const std::vector<float>& freq_table,
                                float* out_cos, float* out_sin);

/// Get rope index for text + image inputs.
///
/// Equivalent to Python engine_utils.py: get_rope_index.
/// Computes 3D MRoPE position IDs for text tokens and image tokens.
///
/// @param input_ids          Token IDs (B, S)
/// @param batch_size         Batch size B
/// @param seq_len            Sequence length S
/// @param image_grid_thw     Image grid dimensions (N, 3) or nullptr
/// @param num_images         Number of images N
/// @param image_token_id     Image token ID (151655)
/// @param spatial_merge_size Spatial merge size (2)
/// @param position_ids_out   Output: (3, B, S) int64, pre-allocated
void GetRopeIndex(const int64_t* input_ids,
                  int64_t batch_size, int64_t seq_len,
                  const int64_t* image_grid_thw, int64_t num_images,
                  int64_t image_token_id, int64_t spatial_merge_size,
                  int64_t* position_ids_out);

} // namespace components
} // namespace atb_llm
