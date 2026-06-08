#include "components/common/deepstack_fusion.h"
#include "utils/float_utils.h"
#include "core/tensor_allocator.h"
#include "core/npu_tensor.h"
#include "ops/index_add_op.h"
#include "log/logger.h"
#include <cstring>
#include <algorithm>

namespace atb_llm {
namespace components {

DeepstackFusion::DeepstackFusion(const Config& cfg,
                                 OperationHandle deepstack_graph)
    : cfg_(cfg)
    , deepstack_graph_(std::move(deepstack_graph))
{
    merger_weights_.resize(cfg_.deepstack_visual_indexes.size());
}

// ═════════════════════════════════════════════════════════════════════
// BuildInjectOp — create the IndexAdd op once, reuse it for all injections.
// ═════════════════════════════════════════════════════════════════════
Status DeepstackFusion::BuildInjectOp() {
    if (inject_op_) return STATUS_OK;  // already built
    inject_op_ = ops::IndexAddOp::Create(/*axis=*/0);
    if (!inject_op_) {
        LOG_ERROR("DeepstackFusion::BuildInjectOp: failed to create IndexAdd op");
        return ERROR_GRAPH_BUILD;
    }
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// IsDeepstackLayer
// ═════════════════════════════════════════════════════════════════════

bool DeepstackFusion::IsDeepstackLayer(int32_t layer_idx,
                                       size_t& fusion_idx) const {
    auto it = std::find(cfg_.deepstack_visual_indexes.begin(),
                        cfg_.deepstack_visual_indexes.end(), layer_idx);
    if (it != cfg_.deepstack_visual_indexes.end()) {
        fusion_idx = static_cast<size_t>(
            std::distance(cfg_.deepstack_visual_indexes.begin(), it));
        return true;
    }
    return false;
}

// ═════════════════════════════════════════════════════════════════════
// ExtractFeatures
// ═════════════════════════════════════════════════════════════════════

Status DeepstackFusion::ExtractFeatures(
        NpuTensor& hidden_npu,
        int64_t total_tokens,
        int32_t layer_idx,
        size_t& fusion_idx,
        IRuntime* runtime,
        std::vector<NpuTensor>& ds_features) {
    if (!IsDeepstackLayer(layer_idx, fusion_idx)) {
        return STATUS_OK;  // Not a deepstack layer, nothing to extract
    }

    if (fusion_idx >= merger_weights_.size()) {
        LOG_ERROR("DeepstackFusion::ExtractFeatures: fusion_idx %zu out of range [0, %zu)",
                  fusion_idx, merger_weights_.size());
        return ERROR_INVALID_PARAM;
    }

    int64_t merge_size = cfg_.spatial_merge_size;
    int64_t merged_tokens = total_tokens / (merge_size * merge_size);

    const auto& mw = merger_weights_[fusion_idx];

    // Run the deepstack merger graph
    NpuTensor ds_out = AllocNpuFloat16({merged_tokens, cfg_.vis_out_hidden_size});

    atb::VariantPack ds_vp;
    ds_vp.inTensors = {
        *hidden_npu.Get(),
        mw.norm_weight, mw.norm_bias,
        mw.fc1_weight, mw.fc1_bias,
        mw.fc2_weight, mw.fc2_bias};
    ds_vp.outTensors = {*ds_out.Get()};

    // Execute the deepstack merger graph
    auto* ctx = runtime->GetContext();
    uint64_t ws_size = 0;
    atb::Status atb_s = deepstack_graph_.get()->Setup(ds_vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("DeepstackMerger graph Setup failed: %d", static_cast<int>(atb_s));
        return ERROR_GRAPH_BUILD;
    }

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
        ws_ptr = ws;
        if (ws_s != STATUS_OK || ws_ptr == nullptr) {
            LOG_ERROR("Failed to get workspace for DeepstackMerger: %zu bytes",
                      static_cast<size_t>(ws_size));
            return ERROR_NPU_MEMORY;
        }
    } else {
        auto [ws, ws_s] = runtime->GetWorkspace(1);
        if (ws_s == STATUS_OK && ws != nullptr) {
            ws_ptr = ws;
            ws_size = 1;
        }
    }

    atb_s = deepstack_graph_.get()->Execute(ds_vp, ws_ptr, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("DeepstackMerger %zu execution failed", fusion_idx);
        return ERROR_INFERENCE;
    }

    // Move the NPU-resident output tensor into the caller's collection.
    // The deepstack graph wrote ds_out on the same stream as subsequent ops
    // (IndexAdd in InjectFeatures), so stream FIFO ordering guarantees
    // ds_out is ready before it is read — no host sync needed.
    ds_features[fusion_idx] = std::move(ds_out);

    LOG_DEBUG("DeepstackMerger %zu: extracted %ld merged tokens, kept on NPU",
             fusion_idx, static_cast<long>(merged_tokens));
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// InjectFeatures — NPU IndexAdd-based implementation.
//
// Original (CPU-fallback) implementation made `positions.size()` round-trips
// through host memory (D→H copy, fp16→fp32→add→fp16, H→D copy).
// This version uploads (ds_feat, positions) once and runs a single NPU
// IndexAdd that adds ds_feat[i] into hidden[positions[i]] for all i in
// one kernel launch.
// ═════════════════════════════════════════════════════════════════════
void DeepstackFusion::InjectFeatures(NpuTensor& hidden_npu,
                                     const NpuTensor& ds_feat,
                                     const std::vector<int64_t>& positions,
                                     int64_t /*seq_len*/, int64_t hidden_size,
                                     int64_t feat_dim,
                                     IRuntime* runtime) {
    if (positions.empty() || !ds_feat) return;
    if (feat_dim != hidden_size) {
        // Deepstack features are expected to already be projected to the
        // text hidden_size by the deepstack merger MLP.
        LOG_ERROR("InjectFeatures: feat_dim=%ld != hidden_size=%ld",
                  static_cast<long>(feat_dim), static_cast<long>(hidden_size));
        return;
    }

    // Lazy-build op on first call (cheap idempotent).
    if (!inject_op_) {
        if (BuildInjectOp() != STATUS_OK) return;
    }

    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    int64_t n = static_cast<int64_t>(positions.size());

    // ── Upload positions as int32 indices ──
    // (positions upload is left as a future optimization; not in this patch.)
    std::vector<int32_t> idx_i32(n);
    for (int64_t i = 0; i < n; i++) {
        idx_i32[i] = static_cast<int32_t>(positions[i]);
    }
    NpuTensor idx_npu = AllocNpuInt32({n});
    if (!idx_npu) {
        LOG_ERROR("InjectFeatures: alloc idx_npu failed");
        return;
    }
    alloc->CopyToDevice(*idx_npu.Get(), idx_i32.data(),
                        n * sizeof(int32_t));

    // ── Upload alpha=1.0 (4th input to ATB IndexAdd: scalar multiplier on updates) ──
    NpuTensor alpha_npu = AllocNpuFloat16({1});
    if (!alpha_npu) {
        LOG_ERROR("InjectFeatures: alloc alpha_npu failed");
        return;
    }
    uint16_t alpha_fp16 = atb_llm::Fp32ToFp16(1.0f);
    alloc->CopyToDevice(*alpha_npu.Get(), &alpha_fp16, sizeof(uint16_t));

    // ── Run IndexAdd ─────────────────────────────────────────
    // hidden_npu is both input (var) and output. ATB IndexAdd writes the
    // result back into the SAME buffer as inTensor0, so we list
    // hidden_npu in both inTensors[0] and outTensors[0].
    // ds_feat is already an NPU-resident tensor (produced by the deepstack
    // merger graph in ExtractFeatures), so it is consumed directly without
    // any host→device copy.
    atb::VariantPack vp;
    vp.inTensors  = {*hidden_npu.Get(), *idx_npu.Get(), *ds_feat.Get(), *alpha_npu.Get()};
    vp.outTensors = {*hidden_npu.Get()};

    uint64_t ws_size = 0;
    atb::Status atb_s = inject_op_.get()->Setup(vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("InjectFeatures: IndexAdd Setup failed: %d",
                  static_cast<int>(atb_s));
        return;
    }
    uint8_t* ws_ptr = nullptr;
    auto [ws, ws_st] = runtime->GetWorkspace(ws_size > 0 ? ws_size : 1);
    if (ws_st == STATUS_OK) ws_ptr = ws;
    atb_s = inject_op_.get()->Execute(vp, ws_ptr, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("InjectFeatures: IndexAdd Execute failed: %d",
                  static_cast<int>(atb_s));
        return;
    }
    // Sync to enforce ordering w.r.t. the next decoder layer that reads
    // from hidden_npu.
    runtime->Synchronize();
}

// ═════════════════════════════════════════════════════════════════════
// SetMergerWeights
// ═════════════════════════════════════════════════════════════════════

void DeepstackFusion::SetMergerWeights(size_t idx,
                                       const DeepstackMergerWeights& w) {
    if (idx >= merger_weights_.size()) {
        LOG_ERROR("DeepstackFusion::SetMergerWeights: idx %zu out of range [0, %zu)",
                  idx, merger_weights_.size());
        return;
    }
    merger_weights_[idx] = w;
}

}  // namespace components
}  // namespace atb_llm
