#pragma once
#include "components/common/cross_modal_fusion.h"
#include "core/raii.h"
#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace components {

/// Deepstack merger weight structure (per deepstack index)
struct DeepstackMergerWeights {
    atb::Tensor norm_weight;
    atb::Tensor norm_bias;
    atb::Tensor fc1_weight;
    atb::Tensor fc1_bias;
    atb::Tensor fc2_weight;
    atb::Tensor fc2_bias;
};

/// Qwen3VL-style Deepstack fusion implementation.
///
/// At specific intermediate vision block layers (e.g. [5, 11, 17]), extracts
/// hidden states, processes them through a merger MLP, and additively injects
/// into the image token positions of corresponding text decoder layers.
class DeepstackFusion : public ICrossModalFusion {
public:
    struct Config {
        int64_t vis_hidden_size = 1024;
        int64_t vis_out_hidden_size = 2048;
        int64_t spatial_merge_size = 2;
        std::vector<int32_t> deepstack_visual_indexes;  // e.g. {5, 11, 17}
    };

    DeepstackFusion(const Config& cfg, OperationHandle deepstack_graph);
    ~DeepstackFusion() override = default;

    /// Build the inject_features_op_ (IndexAdd). Called once at model load.
    /// Safe to call multiple times (idempotent).
    Status BuildInjectOp();

    size_t GetNumFusionLayers() const override {
        return cfg_.deepstack_visual_indexes.size();
    }

    /// Check whether a given vision layer should extract deepstack features.
    /// @return true if this layer is a deepstack layer, and sets fusion_idx
    bool IsDeepstackLayer(int32_t layer_idx, size_t& fusion_idx) const;

    /// Extract deepstack features from vision hidden states.
    /// The features are sliced from the NPU tensor and returned as NPU tensors
    /// (not copied to host), to be consumed later by InjectFeatures.
    /// @param hidden_npu    Current vision hidden states (NPU fp16)
    /// @param total_tokens  Number of tokens in the current vision hidden states
    /// @param layer_idx     Vision block layer index
    /// @param fusion_idx    Output: fusion feature index
    /// @param runtime       Runtime interface
    /// @param[out] ds_features  Output deepstack features (NPU fp16 tensors)
    Status ExtractFeatures(NpuTensor& hidden_npu,
                           int64_t total_tokens,
                           int32_t layer_idx,
                           size_t& fusion_idx,
                           IRuntime* runtime,
                           std::vector<NpuTensor>& ds_features) override;

    /// Inject deepstack features into text hidden states using IndexAdd.
    /// @param hidden_npu   Text hidden states (NPU fp16, modified in-place)
    /// @param ds_feat      Deepstack features (NPU fp16)
    /// @param positions    Image token positions in text sequence
    /// @param seq_len      Text sequence length
    /// @param hidden_size  Text hidden dimension
    /// @param feat_dim     Deepstack feature dimension
    /// @param runtime      Runtime interface
    void InjectFeatures(NpuTensor& hidden_npu,
                        const NpuTensor& ds_feat,
                        const std::vector<int64_t>& positions,
                        int64_t seq_len, int64_t hidden_size,
                        int64_t feat_dim,
                        IRuntime* runtime) override;

    /// Set merger weights for a given deepstack index (called during model Load)
    void SetMergerWeights(size_t idx, const DeepstackMergerWeights& w);

private:
    Config cfg_;
    OperationHandle deepstack_graph_;
    OperationHandle inject_op_;            // IndexAdd op for NPU-side injection
    std::vector<DeepstackMergerWeights> merger_weights_;
    // H4: cache idx/alpha device tensors across the 3 injections per forward.
    // alpha is the constant 1.0 multiplier; cached_idx_npu_ holds the int32
    // image-token positions, re-uploaded only when `positions` changes value.
    NpuTensor alpha_npu_;
    NpuTensor cached_idx_npu_;
    std::vector<int64_t> cached_positions_;
};

}  // namespace components
}  // namespace atb_llm
