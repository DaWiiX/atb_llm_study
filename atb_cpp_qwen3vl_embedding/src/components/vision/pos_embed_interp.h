#pragma once
#include <cstdint>

namespace atb_llm {
namespace components {

/// Bilinear position embedding interpolation + spatial merge shuffling.
///
/// Equivalent to Python fast_pos_embed_interpolate.
/// This is a pure CPU computation -- no ATB graph or NPU dependency.
///
/// Takes the source position embeddings (e.g., from a (num_grid, num_grid) grid)
/// and interpolates them to the target grid dimensions given by grid_thw,
/// then applies spatial merge shuffling to produce the final position embeddings.
///
/// @param pos_embed_src  Source position embeddings on CPU, shape (num_grid*num_grid, hidden_size), fp16
/// @param hidden_size    Embedding dimension per position
/// @param num_grid       Grid dimension (sqrt of num_position_embeddings)
/// @param merge_size     Spatial merge size (typically 2 for Qwen3VL)
/// @param grid_thw       Grid dimensions [t, h, w] per image, flattened (N*3)
/// @param num_images     Number of images N
/// @param pos_out        Output buffer, shape (total_patches, hidden_size), fp16, pre-allocated
void ComputePosEmbedInterp(const uint16_t* pos_embed_src,
                           int32_t hidden_size, int32_t num_grid,
                           int32_t merge_size,
                           const int64_t* grid_thw, int64_t num_images,
                           uint16_t* pos_out);

}  // namespace components
}  // namespace atb_llm
