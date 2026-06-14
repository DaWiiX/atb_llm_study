#include "components/vision/vis_rope_npu_graph.h"
#include "core/graph_builder.h"
#include "ops/concat_op.h"
#include "ops/elewise_op.h"
#include "ops/gather_op.h"
#include "log/logger.h"

#include <algorithm>
#include <cmath>

namespace atb_llm {
namespace components {

// ═════════════════════════════════════════════════════════════════════
// Build — NPU graph
// ═════════════════════════════════════════════════════════════════════

Status VisRopeGraph::Build(const std::string& name, OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    atb::SVector<std::string> in_names  = {"freq_table", "row_idx", "col_idx"};
    atb::SVector<std::string> out_names = {"cos_out", "sin_out"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    // ── 2× Gather (axis=0): freq_table[row_idx] / freq_table[col_idx] ──
    s = builder->AddOp(ops::GatherOp::Create(/*axis=*/0, /*batch_dims=*/0),
               {"freq_table", "row_idx"}, {"row_freq"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::GatherOp::Create(0, 0),
               {"freq_table", "col_idx"}, {"col_freq"});
    if (s != STATUS_OK) return s;

    // ── 2× Concat (axis=1): build [row, col, row, col] over half-dim ──
    // ATB Concat takes exactly 2 inputs, so we stack two operations.
    s = builder->AddOp(ops::ConcatOp::Create(/*concat_dim=*/1),
               {"row_freq", "col_freq"}, {"rc"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::ConcatOp::Create(1),
               {"rc", "rc"}, {"emb"});
    if (s != STATUS_OK) return s;

    // ── Cos / Sin ───────────────────────────────────────────────
    s = builder->AddOp(ops::ElewiseOp::MakeCos(), {"emb"}, {"cos_out"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::ElewiseOp::MakeSin(), {"emb"}, {"sin_out"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) {
        LOG_ERROR("VisRopeGraph::Build: builder->Build() returned null");
        return ERROR_GRAPH_BUILD;
    }
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// BuildVisRopeIndices — host-side Stage A
//
// For each image (T, H, W):
//   row_idx[i] = (br*ms + ir)    where i = ((br * merged_w) + bc) * ms*ms + ir*ms + ic
//   col_idx[i] = (bc*ms + ic)
// then repeated T times.
//
// Equivalent to:
//   row = arange(H).view(merged_h, ms, 1, 1).expand(merged_h, ms, merged_w, ms)
//   col = arange(W).view(1, 1, merged_w, ms).expand(merged_h, ms, merged_w, ms)
//   row.permute(0, 2, 1, 3).reshape(-1)
//   col.permute(0, 2, 1, 3).reshape(-1)
//   .repeat(T)
// ═════════════════════════════════════════════════════════════════════

void BuildVisRopeIndices(const int64_t* grid_thw, int64_t num_images,
                          int32_t merge_size,
                          std::vector<int32_t>& row_idx_out,
                          std::vector<int32_t>& col_idx_out) {
    // Total length so we can reserve once
    int64_t total = 0;
    for (int64_t img = 0; img < num_images; img++) {
        total += grid_thw[img * 3 + 0] *
                 grid_thw[img * 3 + 1] *
                 grid_thw[img * 3 + 2];
    }
    row_idx_out.clear(); row_idx_out.reserve(total);
    col_idx_out.clear(); col_idx_out.reserve(total);

    for (int64_t img = 0; img < num_images; img++) {
        int64_t t = grid_thw[img * 3 + 0];
        int64_t h = grid_thw[img * 3 + 1];
        int64_t w = grid_thw[img * 3 + 2];
        int64_t merged_h = h / merge_size;
        int64_t merged_w = w / merge_size;

        // Build one (H*W) block in spatial-merge order.
        std::vector<int32_t> row_blk, col_blk;
        row_blk.reserve(h * w);
        col_blk.reserve(h * w);
        for (int64_t br = 0; br < merged_h; br++) {
            for (int64_t bc = 0; bc < merged_w; bc++) {
                for (int64_t ir = 0; ir < merge_size; ir++) {
                    for (int64_t ic = 0; ic < merge_size; ic++) {
                        row_blk.push_back(static_cast<int32_t>(br * merge_size + ir));
                        col_blk.push_back(static_cast<int32_t>(bc * merge_size + ic));
                    }
                }
            }
        }
        for (int64_t ti = 0; ti < t; ti++) {
            row_idx_out.insert(row_idx_out.end(), row_blk.begin(), row_blk.end());
            col_idx_out.insert(col_idx_out.end(), col_blk.begin(), col_blk.end());
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// BuildVisRopeFreqTable
//
// freq_table[i][j] = i * inv_freq[j]
// inv_freq[j] = 1 / 10000^(2j / dim),   dim = 2 * half
// ═════════════════════════════════════════════════════════════════════

void BuildVisRopeFreqTable(int32_t max_hw, int32_t half,
                            std::vector<float>& freq_table_out) {
    freq_table_out.assign(static_cast<size_t>(max_hw) * half, 0.0f);
    int32_t dim = 2 * half;
    std::vector<float> inv_freq(half);
    for (int32_t j = 0; j < half; j++) {
        inv_freq[j] = 1.0f / std::pow(10000.0f,
            static_cast<float>(2 * j) / static_cast<float>(dim));
    }
    for (int32_t i = 0; i < max_hw; i++) {
        for (int32_t j = 0; j < half; j++) {
            freq_table_out[static_cast<size_t>(i) * half + j] =
                static_cast<float>(i) * inv_freq[j];
        }
    }
}

int32_t MaxGridHW(const int64_t* grid_thw, int64_t num_images) {
    int32_t m = 0;
    for (int64_t img = 0; img < num_images; img++) {
        m = std::max<int32_t>(m, static_cast<int32_t>(grid_thw[img * 3 + 1]));
        m = std::max<int32_t>(m, static_cast<int32_t>(grid_thw[img * 3 + 2]));
    }
    return m;
}

}  // namespace components
}  // namespace atb_llm
