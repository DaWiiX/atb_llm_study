#pragma once
#include "atb_llm/model.h"
#include "atb_llm/runtime.h"
#include "families/base_model.h"
#include "adapters/qwen3vl_embedding/qwen3vl_config.h"
#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include "components/common/mrope.h"
#include "components/common/deepstack_fusion.h"
#include "runners/text_runner.h"
#include "runners/vision_runner.h"
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
/// Split-graph strategy (delegated to Runners):
///   - Vision: VisionRunner owns FirstLayer + Block + Merger + Deepstack graphs
///   - Text: TextRunner owns DecoderLayer + FinalNorm graphs
///   - This adapter only orchestrates execution and cross-modal fusion
class Qwen3VLModel : public families::BaseModel {
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

    // ── Runners (own ATB graphs) ──────────────────────────
    std::unique_ptr<runners::TextRunner> text_runner_;
    std::unique_ptr<runners::VisionRunner> vision_runner_;

    // ── Cross-modal fusion ────────────────────────────────
    std::unique_ptr<components::DeepstackFusion> deepstack_fusion_;

    // ── Position encoding helpers ─────────────────────────
    std::unique_ptr<components::MRoPE> mrope_;
    std::unique_ptr<components::VisionRotaryEmbedding> vis_rope_;

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

    // ── Qwen3VL-specific helpers ──────────────────────────
    /// Compute fast_pos_embed_interpolate on CPU.
    void ComputePosEmbedInterp(const int64_t* grid_thw, int64_t num_images,
                               uint16_t* pos_out);

    /// Compute vision RoPE cos/sin on CPU.
    void ComputeVisionRoPE(const int64_t* grid_thw, int64_t num_images,
                           float* cos_out, float* sin_out, int64_t& total_tokens);
};

}  // namespace adapters
}  // namespace atb_llm
