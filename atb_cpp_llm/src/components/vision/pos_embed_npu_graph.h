#pragma once

#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace components {

/// Vision PosEmbed Interpolation expressed as an NPU graph.
///
/// Replaces the CPU-only `ComputePosEmbedInterp` for the hot path.  The
/// trick is that bilinear interpolation + spatial-merge shuffle + temporal
/// repeat all collapse into a single per-row recipe that is independent of
/// `vis_hidden_size`:
///
///   out[i] = pos_embed_w[idx00[i]] * wt00[i]
///          + pos_embed_w[idx01[i]] * wt01[i]
///          + pos_embed_w[idx10[i]] * wt10[i]
///          + pos_embed_w[idx11[i]] * wt11[i]
///
/// The (idx, wt) tensors are cheap to build on the host (O(N) where
/// N = T*H*W per image) — we pay the (N * vis_hs) cost only on NPU.
///
/// ── Graph I/O ──
///
/// Inputs (8 + 1 = 9 tensors, all NPU resident):
///   pos_embed_w : (G*G, vis_hs)   fp16   — pre-loaded position embedding table
///   idx00       : (N,)            int32  — gather indices for top-left corners
///   idx01       : (N,)            int32  — gather indices for top-right corners
///   idx10       : (N,)            int32  — gather indices for bottom-left corners
///   idx11       : (N,)            int32  — gather indices for bottom-right corners
///   wt00        : (N, 1)          fp16   — interp weights for top-left (broadcasts over vis_hs)
///   wt01        : (N, 1)          fp16
///   wt10        : (N, 1)          fp16
///   wt11        : (N, 1)          fp16
///
/// Output:
///   out         : (N, vis_hs)     fp16   — interpolated + shuffled + repeated positions
///
/// Note: `wt*` are passed as shape `(N, 1)`, NOT `(N,)`. The caller must
/// reshape on the host side (or ATB ElewiseMul will refuse to broadcast).
class PosEmbedInterpGraph {
public:
    /// Build the ATB graph. The returned op is shape-agnostic: it adapts
    /// to whatever N and vis_hs the runtime tensors carry.
    static Status Build(const std::string& name, OperationHandle& out);
};

/// Host-side Stage A: precompute the four (idx, wt) tensors for a given
/// `grid_thw` (concatenated images) and `num_grid`.
///
/// `idx_out[k]` / `wt_out[k]` (k in 0..3) are resized to length N where
///   N = sum over images of T*H*W   (after temporal repeat).
///
/// Output:
///   idx_out[0..3] : int32 vectors of length N
///   wt_out[0..3]  : fp16  vectors of length N  (caller must reshape to (N,1)
///                                                when uploading to NPU)
///
/// `merge_size` must evenly divide every (H, W).
void BuildPosEmbedIndicesAndWeights(
    const int64_t* grid_thw, int64_t num_images,
    int32_t num_grid, int32_t merge_size,
    std::vector<int32_t> idx_out[4],
    std::vector<uint16_t> wt_out[4]);

}  // namespace components
}  // namespace atb_llm
