#pragma once
#include "atb_llm/model.h"
#include "atb_llm/runtime.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include "components/position/mrope.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace atb_llm {
namespace adapters {

/// Qwen3VL Embedding Model -- implements IModel interface.
///
/// Full inference pipeline:
///   1. Text embedding (CPU lookup into fp16 table)
///   2. Vision inference (if image provided): patch_embed -> pos_embed -> 24 blocks -> merger
///   3. Vision token injection into embedding
///   4. MRoPE position encoding (CPU)
///   5. Text decoder 28-layer loop (NPU, split-graph)
///   6. FinalNorm (NPU)
///   7. Pooling (last non-padded token) + copy to host
///
/// Split-graph strategy:
///   - Vision: FirstLayer graph + Block graph (looped) + Merger graph + Deepstack graph
///   - Text: DecoderLayer graph (looped 28x) + FinalNorm graph
class Qwen3VLModel : public IModel {
public:
    Qwen3VLModel();
    ~Qwen3VLModel() override;

    // IModel interface
    Status Load(const std::string& model_dir, IRuntime* runtime) override;
    Status Forward(const InferRequest& request, InferResult& result) override;
    const char* GetName() const override {
        return "qwen3vl_embedding";
    }
    bool HasVision() const override {
        return true;
    }

private:
    // ── Config & weights ──────────────────────────────────
    Qwen3VLConfig config_;
    Qwen3VLWeights weights_;
    IRuntime* runtime_ = nullptr;

    // ── ATB graphs ────────────────────────────────────────
    OperationHandle vis_first_layer_graph_;
    OperationHandle vis_block_graph_;
    OperationHandle vis_merger_graph_;
    OperationHandle vis_deepstack_graph_;
    OperationHandle text_decoder_graph_;
    OperationHandle text_norm_graph_;

    int32_t cached_text_seq_len_ = 0;

    // ── Position encoding helpers ─────────────────────────
    std::unique_ptr<components::MRoPE> mrope_;
    std::unique_ptr<components::VisionRotaryEmbedding> vis_rope_;

    // ── Graph building ────────────────────────────────────
    Status BuildGraphs();
    Status EnsureTextGraph(int32_t seq_len);

    // ── Vision pipeline ───────────────────────────────────
    Status RunVision(const uint16_t* pixel_values, int64_t num_patches,
                     const int64_t* grid_thw, int64_t num_images,
                     uint16_t* vis_embeds_out, int64_t vis_embed_dim,
                     std::vector<std::vector<uint16_t>>& ds_features);

    // ── Input preparation ────────────────────────────────
    Status PrepareInputs(const InferRequest& request,
                         std::vector<uint16_t>& inputs_embeds,
                         int64_t& seq_len, int64_t& hidden_size,
                         int32_t& hd, int64_t& vis_embed_dim,
                         std::vector<float>& cos_f32,
                         std::vector<float>& sin_f32,
                         std::vector<float>& mask,
                         std::vector<std::vector<uint16_t>>& ds_features,
                         std::vector<int64_t>& image_token_positions);

    // ── Text pipeline ─────────────────────────────────────
    Status RunTextDecoder(uint16_t* hidden_states, int32_t seq_len,
                          const float* cos, const float* sin,
                          const float* mask,
                          const std::vector<std::vector<uint16_t>>& ds_features,
                          const std::vector<int64_t>& image_token_positions);

    // ── Pooling ────────────────────────────────────────────
    Status RunPooling(const uint16_t* hidden_states, int64_t seq_len,
                      int64_t hidden_size, InferResult& result);

    // ── Helpers ───────────────────────────────────────────
    Status ExecuteGraph(OperationHandle& graph,
                        atb::VariantPack& vp);

    void InjectDeepstack(NpuTensor& hidden_npu,
                         const std::vector<uint16_t>& ds_feat,
                         const std::vector<int64_t>& positions,
                         int64_t seq_len, int64_t hidden_size,
                         int64_t vis_embed_dim);

    /// Compute fast_pos_embed_interpolate on CPU.
    void ComputePosEmbedInterp(const int64_t* grid_thw, int64_t num_images,
                               uint16_t* pos_out);

    /// Compute vision RoPE cos/sin on CPU.
    void ComputeVisionRoPE(const int64_t* grid_thw, int64_t num_images,
                           float* cos_out, float* sin_out, int64_t& total_tokens);

    /// Embedding lookup on CPU (fp16 table -> fp16 output).
    void EmbeddingLookup(const int64_t* input_ids, int64_t seq_len,
                         uint16_t* output);

    /// Find image token positions in input_ids.
    std::vector<int64_t> FindImageTokenPositions(const int64_t* input_ids,
                                                 int64_t seq_len);
};

}  // namespace adapters
}  // namespace atb_llm
