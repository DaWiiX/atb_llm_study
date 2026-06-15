#include "components/vision/pos_embed_npu_graph.h"
#include "core/graph_builder.h"
#include "ops/elewise_op.h"
#include "ops/gather_op.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace atb_llm {
namespace components {

// ═════════════════════════════════════════════════════════════════════
// Build — NPU graph that turns 4 (idx, wt) pairs into the final positions
// ═════════════════════════════════════════════════════════════════════

Status PosEmbedInterpGraph::Build(const std::string& name,
                                   OperationHandle& out) {
    std::unique_ptr<GraphBuilder> builder;
    Status s = GraphBuilder::Create(name, builder);
    if (s != STATUS_OK) return s;

    atb::SVector<std::string> in_names = {
        "pos_embed_w",
        "idx00", "idx01", "idx10", "idx11",
        "wt00",  "wt01",  "wt10",  "wt11"
    };
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init(name, nullptr, in_names, out_names);
    if (s != STATUS_OK) return s;

    // ── 4× Gather: idx_k → corner_k (N, vis_hs) ─────────────────
    s = builder->AddOp(ops::GatherOp::Create(/*axis=*/0, /*batch_dims=*/0),
               {"pos_embed_w", "idx00"}, {"v00"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::GatherOp::Create(0, 0),
               {"pos_embed_w", "idx01"}, {"v01"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::GatherOp::Create(0, 0),
               {"pos_embed_w", "idx10"}, {"v10"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::GatherOp::Create(0, 0),
               {"pos_embed_w", "idx11"}, {"v11"});
    if (s != STATUS_OK) return s;

    // ── 4× Mul with broadcast: wt_k(N,1) × v_k(N,vis_hs) → m_k(N,vis_hs) ──
    // ElewiseMul auto-broadcasts when one dim is 1; caller must shape wt
    // as (N, 1).
    s = builder->AddOp(ops::ElewiseOp::MakeMul(), {"v00", "wt00"}, {"m00"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::ElewiseOp::MakeMul(), {"v01", "wt01"}, {"m01"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::ElewiseOp::MakeMul(), {"v10", "wt10"}, {"m10"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::ElewiseOp::MakeMul(), {"v11", "wt11"}, {"m11"});
    if (s != STATUS_OK) return s;

    // ── 3× Add: pairwise reduction over the 4 corners ───────────
    s = builder->AddOp(ops::ElewiseOp::MakeAdd(), {"m00", "m01"}, {"sum_top"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::ElewiseOp::MakeAdd(), {"m10", "m11"}, {"sum_bot"});
    if (s != STATUS_OK) return s;
    s = builder->AddOp(ops::ElewiseOp::MakeAdd(), {"sum_top", "sum_bot"}, {"output"});
    if (s != STATUS_OK) return s;

    out = builder->Build();
    if (!out) {
        LOG_ERROR("PosEmbedInterpGraph::Build: builder->Build() returned null");
        return ERROR_GRAPH_BUILD;
    }
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// BuildPosEmbedIndicesAndWeights — host-side Stage A
//
// Mirrors gen_pos_embed_npu_reference.py:build_indices_and_weights exactly.
//
// For each image (T, H, W):
//   1. Compute bilinear interp tables (H*W entries) — corner indices into
//      the (G*G) embedding table + four weight tables.
//   2. Apply spatial-merge shuffle to those (H*W) entries:
//        view(H//ms, ms, W//ms, ms) -> permute(0,2,1,3) -> flatten
//   3. Repeat the shuffled (H*W) block T times along the leading dim.
// ═════════════════════════════════════════════════════════════════════

namespace {

// Clamp + cast helpers
inline int32_t Floor(float v) { return static_cast<int32_t>(std::floor(v)); }

}  // namespace

void BuildPosEmbedIndicesAndWeights(
        const int64_t* grid_thw, int64_t num_images,
        int32_t num_grid, int32_t merge_size,
        std::vector<int32_t> idx_out[4],
        std::vector<uint16_t> wt_out[4]) {
    // Compute total output length so we can reserve once
    int64_t total = 0;
    for (int64_t img = 0; img < num_images; img++) {
        int64_t t = grid_thw[img * 3 + 0];
        int64_t h = grid_thw[img * 3 + 1];
        int64_t w = grid_thw[img * 3 + 2];
        total += t * h * w;
    }
    for (int k = 0; k < 4; k++) {
        idx_out[k].clear();
        wt_out[k].clear();
        idx_out[k].reserve(total);
        wt_out[k].reserve(total);
    }

    for (int64_t img = 0; img < num_images; img++) {
        int64_t t = grid_thw[img * 3 + 0];
        int64_t h = grid_thw[img * 3 + 1];
        int64_t w = grid_thw[img * 3 + 2];

        // torch.linspace(0, num_grid-1, h)  →  step = (num_grid-1)/(h-1)
        // Special-case h==1: linspace returns [0.0]
        std::vector<float> h_pos(h), w_pos(w);
        for (int64_t i = 0; i < h; i++) {
            h_pos[i] = (h > 1)
                ? static_cast<float>(i) * static_cast<float>(num_grid - 1) /
                  static_cast<float>(h - 1)
                : 0.0f;
        }
        for (int64_t i = 0; i < w; i++) {
            w_pos[i] = (w > 1)
                ? static_cast<float>(i) * static_cast<float>(num_grid - 1) /
                  static_cast<float>(w - 1)
                : 0.0f;
        }

        // Per-axis floor / ceil / delta
        std::vector<int32_t> h_floor(h), w_floor(w), h_ceil(h), w_ceil(w);
        std::vector<float> dh(h), dw(w);
        for (int64_t i = 0; i < h; i++) {
            h_floor[i] = Floor(h_pos[i]);
            h_ceil[i]  = std::min(h_floor[i] + 1, num_grid - 1);
            dh[i] = h_pos[i] - static_cast<float>(h_floor[i]);
        }
        for (int64_t j = 0; j < w; j++) {
            w_floor[j] = Floor(w_pos[j]);
            w_ceil[j]  = std::min(w_floor[j] + 1, num_grid - 1);
            dw[j] = w_pos[j] - static_cast<float>(w_floor[j]);
        }

        int64_t merged_h = h / merge_size;
        int64_t merged_w = w / merge_size;

        // Spatial-merge shuffle: walk output rows in the order produced by
        // `view(merged_h, ms, merged_w, ms) -> permute(0,2,1,3)`.
        // That order is:  for br: for bc: for ir: for ic:  (row = br*ms+ir, col = bc*ms+ic)
        //
        // We emit `t` consecutive copies of the same shuffled block (matches
        // torch.repeat(t, 1) AFTER the shuffle in the reference).
        int64_t block_size = h * w;
        std::vector<int32_t> block_idx[4];
        std::vector<uint16_t> block_wt[4];
        for (int k = 0; k < 4; k++) {
            block_idx[k].reserve(block_size);
            block_wt[k].reserve(block_size);
        }

        for (int64_t br = 0; br < merged_h; br++) {
            for (int64_t bc = 0; bc < merged_w; bc++) {
                for (int64_t ir = 0; ir < merge_size; ir++) {
                    for (int64_t ic = 0; ic < merge_size; ic++) {
                        int64_t row = br * merge_size + ir;
                        int64_t col = bc * merge_size + ic;

                        int32_t r0 = h_floor[row];
                        int32_t r1 = h_ceil[row];
                        int32_t c0 = w_floor[col];
                        int32_t c1 = w_ceil[col];

                        float ddh = dh[row];
                        float ddw = dw[col];

                        block_idx[0].push_back(r0 * num_grid + c0);
                        block_idx[1].push_back(r0 * num_grid + c1);
                        block_idx[2].push_back(r1 * num_grid + c0);
                        block_idx[3].push_back(r1 * num_grid + c1);

                        block_wt[0].push_back(
                            Fp32ToFp16((1.0f - ddh) * (1.0f - ddw)));
                        block_wt[1].push_back(
                            Fp32ToFp16((1.0f - ddh) * ddw));
                        block_wt[2].push_back(
                            Fp32ToFp16(ddh * (1.0f - ddw)));
                        block_wt[3].push_back(
                            Fp32ToFp16(ddh * ddw));
                    }
                }
            }
        }

        // Repeat block T times — append each block t copies
        for (int64_t ti = 0; ti < t; ti++) {
            for (int k = 0; k < 4; k++) {
                idx_out[k].insert(idx_out[k].end(),
                                  block_idx[k].begin(), block_idx[k].end());
                wt_out[k].insert(wt_out[k].end(),
                                 block_wt[k].begin(), block_wt[k].end());
            }
        }
    }
}

}  // namespace components
}  // namespace atb_llm
