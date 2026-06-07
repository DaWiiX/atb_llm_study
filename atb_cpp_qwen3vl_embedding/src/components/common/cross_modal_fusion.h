#pragma once
#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/npu_tensor.h"
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace components {

/// Cross-modal fusion interface -- defines the strategy for injecting
/// visual features into text hidden states.
///
/// Different VLM models use different fusion strategies:
///   - Qwen3VL: Deepstack (intermediate-layer vision features + additive injection)
///   - LLaVA: simple vision token concatenation (no fusion component needed)
///   - Others: cross-attention, adapter, etc.
class ICrossModalFusion {
public:
    virtual ~ICrossModalFusion() = default;

    /// Get the number of fusion layers (e.g. deepstack layer count)
    virtual size_t GetNumFusionLayers() const = 0;

    /// Execute the fusion graph to extract visual intermediate features to host.
    /// @param hidden_npu    Current vision hidden states (NPU)
    /// @param total_tokens  Number of tokens in the current vision hidden states
    /// @param layer_idx     Vision block layer index (used to decide whether to extract)
    /// @param fusion_idx    Output: fusion feature index
    /// @param runtime       Runtime interface
    /// @param[out] ds_features  Output deepstack features (host fp16)
    /// @return STATUS_OK on success
    virtual Status ExtractFeatures(NpuTensor& hidden_npu,
                                   int64_t total_tokens,
                                   int32_t layer_idx,
                                   size_t& fusion_idx,
                                   IRuntime* runtime,
                                   std::vector<std::vector<uint16_t>>& ds_features) = 0;

    /// Inject fusion features into text hidden states.
    /// @param hidden_npu   Text hidden states (NPU, modified in-place)
    /// @param ds_feat      Deepstack features (host fp16)
    /// @param positions    Image token positions in text sequence
    /// @param seq_len      Text sequence length
    /// @param hidden_size  Text hidden dimension
    /// @param feat_dim     Deepstack feature dimension
    /// @param runtime      Runtime interface
    virtual void InjectFeatures(NpuTensor& hidden_npu,
                                const std::vector<uint16_t>& ds_feat,
                                const std::vector<int64_t>& positions,
                                int64_t seq_len, int64_t hidden_size,
                                int64_t feat_dim,
                                IRuntime* runtime) = 0;
};

}  // namespace components
}  // namespace atb_llm
