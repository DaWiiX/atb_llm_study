#include "components/common/deepstack_fusion.h"
#include "utils/float_utils.h"
#include "core/tensor_allocator.h"
#include "core/npu_tensor.h"
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
        std::vector<std::vector<uint16_t>>& ds_features) {
    if (!IsDeepstackLayer(layer_idx, fusion_idx)) {
        return STATUS_OK;  // Not a deepstack layer, nothing to extract
    }

    if (fusion_idx >= merger_weights_.size()) {
        LOG_ERROR("DeepstackFusion::ExtractFeatures: fusion_idx %zu out of range [0, %zu)",
                  fusion_idx, merger_weights_.size());
        return ERROR_INVALID_PARAM;
    }

    auto* alloc = runtime->GetAllocator();
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

    Status sync_s = runtime->Synchronize();
    if (sync_s != STATUS_OK) {
        LOG_ERROR("Stream sync failed after DeepstackMerger %zu", fusion_idx);
        return sync_s;
    }

    // Copy deepstack features to host
    size_t ds_bytes = static_cast<size_t>(merged_tokens) *
                      cfg_.vis_out_hidden_size * sizeof(uint16_t);
    ds_features[fusion_idx].resize(merged_tokens * cfg_.vis_out_hidden_size);
    alloc->CopyToHost(ds_features[fusion_idx].data(), *ds_out.Get(), ds_bytes);

    LOG_INFO("DeepstackMerger %zu: extracted %ld merged tokens",
             fusion_idx, static_cast<long>(merged_tokens));
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// InjectFeatures
// ═════════════════════════════════════════════════════════════════════

void DeepstackFusion::InjectFeatures(NpuTensor& hidden_npu,
                                     const std::vector<uint16_t>& ds_feat,
                                     const std::vector<int64_t>& positions,
                                     int64_t seq_len, int64_t hidden_size,
                                     int64_t feat_dim,
                                     IRuntime* runtime) {
    auto* alloc = runtime->GetAllocator();
    size_t h_bytes = static_cast<size_t>(seq_len) * hidden_size * sizeof(uint16_t);
    std::vector<uint16_t> h_host(seq_len * hidden_size);
    alloc->CopyToHost(h_host.data(), *hidden_npu.Get(), h_bytes);
    int64_t ds_tokens = static_cast<int64_t>(ds_feat.size()) / feat_dim;
    for (int64_t t = 0; t < ds_tokens && t < static_cast<int64_t>(positions.size()); t++) {
        int64_t pos = positions[t];
        for (int64_t d = 0; d < feat_dim; d++) {
            float h = atb_llm::Fp16ToF32(h_host[pos * hidden_size + d]);
            float ds = atb_llm::Fp16ToF32(ds_feat[static_cast<size_t>(t) * feat_dim + d]);
            h_host[pos * hidden_size + d] = atb_llm::Fp32ToFp16(h + ds);
        }
    }
    alloc->CopyToDevice(*hidden_npu.Get(), h_host.data(), h_bytes);
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
