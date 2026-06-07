#pragma once

#include "atb_llm/types.h"
#include "core/raii.h"
#include <cstdint>
#include <string>
#include <vector>

namespace atb_llm {
namespace components {

/// Vision RoPE cos/sin generation as an NPU graph.
///
/// Replaces the CPU `ComputeVisionRotPosEmb` for the hot path. Like
/// pos_embed_npu_graph, we factor compute into:
///
///   Stage A (CPU, O(N)):
///     Build flat (row_idx, col_idx) tensors (int32, length N=ΣT*H*W)
///     in the spatial-merge-shuffled + temporal-repeated order.
///     Also build the (max_hw, half) `freq_table` fp32 = i * inv_freq[j].
///
///   Stage B (NPU, O(N * vis_hd)):
///     row_freq = Gather(freq_table, row_idx, axis=0)        // (N, half) f32
///     col_freq = Gather(freq_table, col_idx, axis=0)        // (N, half) f32
///     rc       = Concat(row_freq, col_freq, axis=1)         // (N, 2*half) f32
///     emb      = Concat(rc,        rc,        axis=1)       // (N, 4*half = vis_hd) f32
///     cos_out  = Cos(emb)  → fp32 → caller casts to fp16
///     sin_out  = Sin(emb)
///
/// ── Graph I/O ──
///
/// Inputs (3 tensors):
///   freq_table : (max_hw, half)  fp16   — caller must convert fp32 → fp16
///   row_idx    : (N,)            int32
///   col_idx    : (N,)            int32
///
/// Outputs (2 tensors):
///   cos_out    : (N, vis_hd)     fp16   // vis_hd = half * 4
///   sin_out    : (N, vis_hd)     fp16
///
/// Note: ATB's Concat only supports fp16/bf16, so the freq_table input
/// and cos/sin outputs are fp16. The trig functions of values in
/// [0, max_hw * 1.0] fit comfortably in fp16's dynamic range.
class VisRopeGraph {
public:
    static Status Build(const std::string& name, OperationHandle& out);
};

/// Host-side Stage A: build (row_idx, col_idx) for the given grid_thw.
/// Lengths match: N = sum over images of T*H*W.
void BuildVisRopeIndices(const int64_t* grid_thw, int64_t num_images,
                          int32_t merge_size,
                          std::vector<int32_t>& row_idx_out,
                          std::vector<int32_t>& col_idx_out);

/// Build the (max_hw, half) fp32 frequency table.
///   freq_table[i][j] = i * inv_freq[j],   inv_freq[j] = 1 / 10000^(2j/dim)
/// where dim = 2 * half (mirrors VisionRotaryEmbedding's `dim`).
void BuildVisRopeFreqTable(int32_t max_hw, int32_t half,
                            std::vector<float>& freq_table_out);

/// Convenience: scan grid_thw to find max(h, w) across all images.
int32_t MaxGridHW(const int64_t* grid_thw, int64_t num_images);

}  // namespace components
}  // namespace atb_llm
