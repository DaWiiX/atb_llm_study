#include "adapters/qwen3vl_embedding/qwen3vl_model.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "utils/float_utils.h"
#include "safetensors.hh"
#include "components/vision/vision_block_graph.h"
#include "components/vision/vision_merger_graph.h"
#include "components/vision/deepstack_graph.h"
#include "components/vision/patch_embed_graph.h"
#include "components/common/rms_norm_graph.h"
#include "components/text/decoder_layer_graph.h"
#include "runners/text_runner.h"
#include "runners/vision_runner.h"
#include "core/graph_builder.h"
#include "core/tensor_allocator.h"
#include "core/npu_tensor.h"
#include "ops/elewise_op.h"
#include "log/logger.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace atb_llm {
namespace adapters {

Qwen3VLModel::Qwen3VLModel() = default;
Qwen3VLModel::~Qwen3VLModel() = default;

// ═════════════════════════════════════════════════════════════════════
// Load
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::Load(const std::string& model_dir, IRuntime* runtime) {
    runtime_ = runtime;

    // 1. Load config
    Status s = LoadQwen3VLConfig(model_dir, config_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to load Qwen3VL config");
        return s;
    }

    // 2. Load weights (bf16 -> fp16, all to NPU)
    s = LoadQwen3VLWeights(model_dir, config_,
                           *runtime->GetWeightLoader(),
                           *runtime->GetAllocator(),
                           weights_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to load Qwen3VL weights");
        return s;
    }

    // 3. Initialize position encoding helpers
    mrope_ = std::make_unique<components::MRoPE>(
        config_.text_head_dim, config_.text_rope_theta, config_.text_mrope_section);
    vis_rope_ = std::make_unique<components::VisionRotaryEmbedding>(
        config_.vis_head_dim());

    // 4. Build ATB graphs
    s = BuildGraphs();
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build ATB graphs");
        return s;
    }

    LOG_INFO("Qwen3VLModel loaded successfully");
    return STATUS_OK;
}

Status Qwen3VLModel::BuildGraphs() {
    int32_t vis_nh = config_.vis_num_heads;
    int32_t vis_hd = config_.vis_head_dim();

    // Vision graphs
    Status s = runners::BuildVisionFirstLayer(
        [&]() {
            runners::VisionRunner::Config vc;
            vc.hidden_size = config_.vis_hidden_size;
            vc.num_heads = config_.vis_num_heads;
            vc.intermediate_size = config_.vis_intermediate_size;
            vc.depth = config_.vis_depth;
            vc.in_channels = config_.vis_in_channels;
            vc.temporal_patch_size = config_.vis_temporal_patch_size;
            vc.patch_size = config_.vis_patch_size;
            vc.spatial_merge_size = config_.vis_spatial_merge_size;
            vc.num_position_embeddings = config_.vis_num_position_embeddings;
            vc.deepstack_visual_indexes = config_.vis_deepstack_visual_indexes;
            vc.epsilon = config_.vis_epsilon;
            return vc;
        }(),
        vis_first_layer_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build VisionFirstLayer graph");
        return s;
    }

    s = components::VisionBlockGraph::Build(
        "VisionBlock", vis_nh, vis_hd, config_.vis_epsilon, vis_block_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build VisionBlock graph");
        return s;
    }

    s = components::VisionMergerGraph::Build(
        "VisionMerger", config_.vis_hidden_size, config_.vis_spatial_merge_size,
        false, config_.vis_epsilon, vis_merger_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build VisionMerger graph");
        return s;
    }

    s = components::DeepstackGraph::Build(
        "DeepstackMerger", config_.vis_hidden_size, config_.vis_spatial_merge_size,
        config_.vis_epsilon, vis_deepstack_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build DeepstackMerger graph");
        return s;
    }

    // Text norm graph
    s = components::RmsNormGraph::Build(
        "TextFinalNorm", config_.text_rms_norm_eps, text_norm_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build TextFinalNorm graph");
        return s;
    }

    LOG_INFO("All ATB graphs built successfully");
    return STATUS_OK;
}

Status Qwen3VLModel::EnsureTextGraph(int32_t seq_len) {
    if (cached_text_seq_len_ == seq_len && text_decoder_graph_) {
        return STATUS_OK;
    }

    Status s = components::text::TextDecoderLayerGraph::Build(
        "TextDecoderLayer",
        config_.text_num_heads, config_.text_num_kv_heads,
        config_.text_head_dim, seq_len,
        config_.text_rms_norm_eps, /*use_mask=*/true,
        text_decoder_graph_);
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build TextDecoderLayer graph for seq_len=%d", seq_len);
        return s;
    }

    cached_text_seq_len_ = seq_len;
    LOG_INFO("TextDecoderLayer graph built for seq_len=%d", seq_len);
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// Forward (orchestrator)
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::Forward(const InferRequest& request, InferResult& result) {
    std::vector<uint16_t> inputs_embeds;
    int64_t seq_len, hidden_size;
    int32_t hd;
    int64_t vis_embed_dim;
    std::vector<float> cos_f32, sin_f32, mask;
    std::vector<std::vector<uint16_t>> ds_features;
    std::vector<int64_t> image_token_positions;

    Status s = PrepareInputs(request, inputs_embeds, seq_len, hidden_size,
                             hd, vis_embed_dim, cos_f32, sin_f32, mask,
                             ds_features, image_token_positions);
    if (s != STATUS_OK) return s;

    s = RunTextDecoder(inputs_embeds.data(), static_cast<int32_t>(seq_len),
                       cos_f32.data(), sin_f32.data(), mask.data(),
                       ds_features, image_token_positions);
    if (s != STATUS_OK) return s;

    s = RunPooling(inputs_embeds.data(), seq_len, hidden_size, result);
    if (s != STATUS_OK) return s;

    runtime_->Synchronize();
    LOG_INFO("Forward completed: seq_len=%ld, output shape=(%ld)",
             static_cast<long>(seq_len), static_cast<long>(hidden_size));
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// PrepareInputs
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::PrepareInputs(const InferRequest& request,
                                   std::vector<uint16_t>& inputs_embeds,
                                   int64_t& seq_len, int64_t& hidden_size, int32_t& hd,
                                   int64_t& vis_embed_dim,
                                   std::vector<float>& cos_f32, std::vector<float>& sin_f32,
                                   std::vector<float>& mask,
                                   std::vector<std::vector<uint16_t>>& ds_features,
                                   std::vector<int64_t>& image_token_positions) {
    const int64_t* input_ids = request.text.input_ids;
    seq_len = request.text.seq_length;
    if (!input_ids || request.text.batch_size != 1) {
        LOG_ERROR("Only batch_size=1 is supported, got %ld",
                  static_cast<long>(request.text.batch_size));
        return ERROR_INVALID_PARAM;
    }
    if (seq_len <= 0) {
        LOG_ERROR("seq_length must be > 0, got %ld", static_cast<long>(seq_len));
        return ERROR_INVALID_PARAM;
    }
    hidden_size = config_.text_hidden_size;
    hd = config_.text_head_dim;
    vis_embed_dim = config_.vis_out_hidden_size;

    // Text embedding lookup (CPU, fp16)
    inputs_embeds.resize(seq_len * hidden_size);
    EmbeddingLookup(input_ids, seq_len, inputs_embeds.data());

    // Vision inference (if image provided)
    std::vector<uint16_t> vis_embeds_host;
    std::vector<int64_t> grid_thw_host;
    int64_t num_images = 0;
    bool has_image = (request.mode == InputMode::IMAGE_AND_TEXT ||
                      request.mode == InputMode::PREPROCESSED) &&
                     request.preprocessed.pixel_values != nullptr;
    if (has_image) {
        const uint16_t* pv = static_cast<const uint16_t*>(
            request.preprocessed.pixel_values);
        int64_t np = request.preprocessed.num_patches;
        const int64_t* gthw = request.preprocessed.grid_thw;
        if (!gthw && request.preprocessed.metadata) {
            gthw = static_cast<const int64_t*>(request.preprocessed.metadata);
        }
        grid_thw_host.assign(gthw, gthw + 3);
        num_images = 1;
        vis_embeds_host.resize(np * vis_embed_dim);
        ds_features.resize(config_.vis_deepstack_visual_indexes.size());
        Status s = RunVision(pv, np, gthw, num_images,
                             vis_embeds_host.data(), vis_embed_dim, ds_features);
        if (s != STATUS_OK) return s;
        image_token_positions = FindImageTokenPositions(input_ids, seq_len);
        if (image_token_positions.size() != static_cast<size_t>(np)) {
            LOG_ERROR("Image token count mismatch: pos=%zu, vis=%ld",
                      image_token_positions.size(), static_cast<long>(np));
            return ERROR_INVALID_PARAM;
        }
        for (int64_t i = 0; i < np; i++) {
            int64_t pos = image_token_positions[i];
            std::memcpy(inputs_embeds.data() + pos * hidden_size,
                        vis_embeds_host.data() + i * vis_embed_dim,
                        vis_embed_dim * sizeof(uint16_t));
        }
    }

    // Position encoding (CPU)
    std::vector<int64_t> position_ids(3 * seq_len);
    components::GetRopeIndex(input_ids, 1, seq_len,
                             grid_thw_host.empty() ? nullptr : grid_thw_host.data(),
                             num_images, config_.image_token_id,
                             config_.vis_spatial_merge_size, position_ids.data());
    cos_f32.resize(seq_len * hd);
    sin_f32.resize(seq_len * hd);
    mrope_->Compute(position_ids.data(), 1, seq_len,
                    cos_f32.data(), sin_f32.data());

    // Causal mask
    mask.resize(seq_len * seq_len);
    runners::MakeCausalMask(seq_len, mask.data());
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// RunTextDecoder
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::RunTextDecoder(uint16_t* hidden_states, int32_t seq_len,
                                    const float* cos, const float* sin, const float* mask,
                                    const std::vector<std::vector<uint16_t>>& ds_features,
                                    const std::vector<int64_t>& image_token_positions) {
    auto* alloc = runtime_->GetAllocator();
    int64_t hs = config_.text_hidden_size;
    int32_t hd = config_.text_head_dim;
    int64_t ved = config_.vis_out_hidden_size;
    int64_t n = seq_len;
    NpuTensor h_npu = AllocNpuFloat16({n, hs});
    alloc->CopyToDevice(*h_npu.Get(), hidden_states,
                        static_cast<size_t>(n) * hs * sizeof(uint16_t));
    int64_t rope_n = n * hd;
    std::vector<uint16_t> cos16(rope_n), sin16(rope_n), m16(n * n);
    for (int64_t i = 0; i < rope_n; i++) {
        cos16[i] = atb_llm::Fp32ToFp16(cos[i]);
        sin16[i] = atb_llm::Fp32ToFp16(sin[i]);
    }
    for (int64_t i = 0; i < n * n; i++)
        m16[i] = atb_llm::Fp32ToFp16(mask[i]);
    NpuTensor cos_npu = AllocNpuFloat16({n, hd});
    alloc->CopyToDevice(*cos_npu.Get(), cos16.data(), rope_n * sizeof(uint16_t));
    NpuTensor sin_npu = AllocNpuFloat16({n, hd});
    alloc->CopyToDevice(*sin_npu.Get(), sin16.data(), rope_n * sizeof(uint16_t));
    NpuTensor mask_npu = AllocNpuFloat16({n, n});
    alloc->CopyToDevice(*mask_npu.Get(), m16.data(), n * n * sizeof(uint16_t));
    atb::Tensor seqlen;
    int32_t sl = seq_len;
    seqlen.desc.dtype = ACL_INT32;
    seqlen.desc.format = ACL_FORMAT_ND;
    seqlen.desc.shape.dimNum = 1;
    seqlen.desc.shape.dims[0] = 1;
    seqlen.dataSize = sizeof(int32_t);
    seqlen.hostData = &sl;
    Status s = EnsureTextGraph(seq_len);
    if (s != STATUS_OK) return s;
    for (int32_t li = 0; li < config_.text_num_layers; li++) {
        const auto& lw = weights_.text_layers[li];
        NpuTensor h_out = AllocNpuFloat16({n, hs});
        atb::VariantPack vp;
        vp.inTensors = {*h_npu.Get(),
                        lw.q_weight, lw.k_weight, lw.v_weight, lw.o_weight,
                        lw.q_norm_weight, lw.k_norm_weight,
                        lw.gate_weight, lw.up_weight, lw.down_weight,
                        lw.input_ln_weight, lw.post_ln_weight,
                        *cos_npu.Get(), *sin_npu.Get(), *mask_npu.Get(), seqlen};
        vp.outTensors = {*h_out.Get()};
        s = ExecuteGraph(text_decoder_graph_, vp);
        if (s != STATUS_OK) {
            LOG_ERROR("TextDecoderLayer %d failed", li);
            return s;
        }
        h_npu = std::move(h_out);
        if (li < static_cast<int32_t>(ds_features.size()) &&
            !ds_features[li].empty() && !image_token_positions.empty()) {
            InjectDeepstack(h_npu, ds_features[li], image_token_positions, n, hs, ved);
        }
    }
    NpuTensor norm_out = AllocNpuFloat16({n, hs});
    atb::VariantPack nvp;
    nvp.inTensors = {*h_npu.Get(), weights_.text_norm_weight};
    nvp.outTensors = {*norm_out.Get()};
    s = ExecuteGraph(text_norm_graph_, nvp);
    if (s != STATUS_OK) {
        LOG_ERROR("TextFinalNorm failed");
        return s;
    }
    s = runtime_->Synchronize();
    if (s != STATUS_OK) return s;
    alloc->CopyToHost(hidden_states, *norm_out.Get(),
                      static_cast<size_t>(n) * hs * sizeof(uint16_t));
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// RunPooling
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::RunPooling(const uint16_t* hidden_states, int64_t seq_len,
                                int64_t hidden_size, InferResult& result) {
    int64_t last_pos = seq_len - 1;
    const uint16_t* last_token = hidden_states + last_pos * hidden_size;
    result.shape = {hidden_size};
    result.dtype = ACL_FLOAT16;
    result.data.resize(hidden_size * sizeof(uint16_t));
    if (config_.normalize) {
        float norm = 0.0f;
        for (int64_t i = 0; i < hidden_size; i++) {
            float v = atb_llm::Fp16ToF32(last_token[i]);
            norm += v * v;
        }
        norm = std::sqrt(norm + 1e-12f);
        uint16_t* out = reinterpret_cast<uint16_t*>(result.data.data());
        for (int64_t i = 0; i < hidden_size; i++)
            out[i] = atb_llm::Fp32ToFp16(atb_llm::Fp16ToF32(last_token[i]) / norm);
    } else {
        std::memcpy(result.data.data(), last_token, hidden_size * sizeof(uint16_t));
    }
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// InjectDeepstack
// ═════════════════════════════════════════════════════════════════════

void Qwen3VLModel::InjectDeepstack(NpuTensor& hidden_npu,
                                   const std::vector<uint16_t>& ds_feat,
                                   const std::vector<int64_t>& positions,
                                   int64_t seq_len, int64_t hidden_size, int64_t vis_embed_dim) {
    auto* alloc = runtime_->GetAllocator();
    size_t h_bytes = static_cast<size_t>(seq_len) * hidden_size * sizeof(uint16_t);
    std::vector<uint16_t> h_host(seq_len * hidden_size);
    alloc->CopyToHost(h_host.data(), *hidden_npu.Get(), h_bytes);
    int64_t ds_tokens = ds_feat.size() / vis_embed_dim;
    for (int64_t t = 0; t < ds_tokens && t < static_cast<int64_t>(positions.size()); t++) {
        int64_t pos = positions[t];
        for (int64_t d = 0; d < vis_embed_dim; d++) {
            float h = atb_llm::Fp16ToF32(h_host[pos * hidden_size + d]);
            float ds = atb_llm::Fp16ToF32(ds_feat[t * vis_embed_dim + d]);
            h_host[pos * hidden_size + d] = atb_llm::Fp32ToFp16(h + ds);
        }
    }
    alloc->CopyToDevice(*hidden_npu.Get(), h_host.data(), h_bytes);
}

// ═════════════════════════════════════════════════════════════════════
// Vision pipeline
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::RunVision(const uint16_t* pixel_values, int64_t num_patches,
                               const int64_t* grid_thw, int64_t num_images,
                               uint16_t* vis_embeds_out, int64_t vis_embed_dim,
                               std::vector<std::vector<uint16_t>>& ds_features) {
    auto* alloc = runtime_->GetAllocator();

    int64_t vis_hs = config_.vis_hidden_size;
    int32_t vis_hd = config_.vis_head_dim();
    int64_t merge_size = config_.vis_spatial_merge_size;

    // 1. Compute position embeddings (CPU)
    size_t pos_bytes = static_cast<size_t>(num_patches) * vis_hs * sizeof(uint16_t);
    std::vector<uint16_t> pos_embed_host(num_patches * vis_hs);
    ComputePosEmbedInterp(grid_thw, num_images, pos_embed_host.data());

    // 2. Compute vision RoPE (CPU)
    std::vector<float> cos_f32(num_patches * vis_hd * 2);
    std::vector<float> sin_f32(num_patches * vis_hd * 2);
    int64_t total_tokens;
    ComputeVisionRoPE(grid_thw, num_images, cos_f32.data(), sin_f32.data(), total_tokens);

    // Convert to fp16
    int64_t rope_dim = vis_hd * 2;
    std::vector<uint16_t> cos_fp16(total_tokens * rope_dim);
    std::vector<uint16_t> sin_fp16(total_tokens * rope_dim);
    for (int64_t i = 0; i < total_tokens * rope_dim; i++) {
        cos_fp16[i] = atb_llm::Fp32ToFp16(cos_f32[i]);
        sin_fp16[i] = atb_llm::Fp32ToFp16(sin_f32[i]);
    }

    // 3. Copy vision inputs to NPU
    int64_t patch_dim = static_cast<int64_t>(config_.vis_in_channels) *
                        config_.vis_temporal_patch_size *
                        config_.vis_patch_size * config_.vis_patch_size;
    int64_t flat_elements = num_patches * patch_dim;
    size_t pixel_bytes = static_cast<size_t>(flat_elements) * sizeof(uint16_t);

    // PatchEmbed graph expects 1D flat input: (N * patch_dim,)
    // Its Reshape divides by kernel_size to recover N
    NpuTensor pixels_npu = AllocNpuFloat16({flat_elements});
    alloc->CopyToDevice(*pixels_npu.Get(), pixel_values, pixel_bytes);

    NpuTensor pos_npu = AllocNpuFloat16({num_patches, vis_hs});
    alloc->CopyToDevice(*pos_npu.Get(), pos_embed_host.data(), pos_bytes);

    NpuTensor cos_npu = AllocNpuFloat16({total_tokens, rope_dim});
    alloc->CopyToDevice(*cos_npu.Get(), cos_fp16.data(), cos_fp16.size() * sizeof(uint16_t));

    NpuTensor sin_npu = AllocNpuFloat16({total_tokens, rope_dim});
    alloc->CopyToDevice(*sin_npu.Get(), sin_fp16.data(), sin_fp16.size() * sizeof(uint16_t));

    // Seqlen for vision (host-side int32, not NPU-allocated)
    atb::Tensor vis_seqlen;
    int32_t vis_seqlen_val = static_cast<int32_t>(total_tokens);
    vis_seqlen.desc.dtype = ACL_INT32;
    vis_seqlen.desc.format = ACL_FORMAT_ND;
    vis_seqlen.desc.shape.dimNum = 1;
    vis_seqlen.desc.shape.dims[0] = 1;
    vis_seqlen.dataSize = sizeof(int32_t);
    vis_seqlen.hostData = &vis_seqlen_val;

    // 4. Run FirstLayer (patch_embed + pos_embed + block 0)
    NpuTensor h_npu = AllocNpuFloat16({total_tokens, vis_hs});

    {
        const auto& bw = weights_.vis_blocks[0];
        atb::VariantPack vp;
        vp.inTensors = {
            *pixels_npu.Get(),
            weights_.vis_patch_embed.weight, weights_.vis_patch_embed.bias,
            *pos_npu.Get(), *cos_npu.Get(), *sin_npu.Get(), vis_seqlen,
            bw.qkv_weight, bw.qkv_bias, bw.proj_weight, bw.proj_bias,
            bw.fc1_weight, bw.fc1_bias, bw.fc2_weight, bw.fc2_bias,
            bw.n1_weight, bw.n1_bias, bw.n2_weight, bw.n2_bias};
        vp.outTensors = {*h_npu.Get()};

        Status s = ExecuteGraph(vis_first_layer_graph_, vp);
        if (s != STATUS_OK) {
            LOG_ERROR("VisionFirstLayer execution failed");
            return s;
        }
    }

    // 5. Run blocks 1..depth-1, collect deepstack features
    ds_features.resize(config_.vis_deepstack_visual_indexes.size());
    for (int32_t li = 1; li < config_.vis_depth; li++) {
        const auto& bw = weights_.vis_blocks[li];

        // Allocate fresh output tensor (ATB does NOT support in-place: input==output)
        NpuTensor h_out = AllocNpuFloat16({total_tokens, vis_hs});

        atb::VariantPack vp;
        vp.inTensors = {
            *h_npu.Get(),
            bw.qkv_weight, bw.qkv_bias, bw.proj_weight, bw.proj_bias,
            bw.fc1_weight, bw.fc1_bias, bw.fc2_weight, bw.fc2_bias,
            bw.n1_weight, bw.n1_bias, bw.n2_weight, bw.n2_bias,
            *cos_npu.Get(), *sin_npu.Get(), vis_seqlen};
        vp.outTensors = {*h_out.Get()};

        Status s = ExecuteGraph(vis_block_graph_, vp);
        if (s != STATUS_OK) {
            LOG_ERROR("VisionBlock %d execution failed", li);
            return s;
        }

        // Swap: old h_npu freed by move-assign, new h_out takes ownership
        h_npu = std::move(h_out);

        // Check if this layer produces deepstack features
        auto it = std::find(config_.vis_deepstack_visual_indexes.begin(),
                            config_.vis_deepstack_visual_indexes.end(), li);
        if (it != config_.vis_deepstack_visual_indexes.end()) {
            size_t ds_idx = std::distance(config_.vis_deepstack_visual_indexes.begin(), it);
            const auto& mw = weights_.deepstack_mergers[ds_idx];

            int64_t merged_tokens = total_tokens / (merge_size * merge_size);
            NpuTensor ds_out = AllocNpuFloat16({merged_tokens, config_.vis_out_hidden_size});

            atb::VariantPack ds_vp;
            ds_vp.inTensors = {
                *h_npu.Get(),
                mw.norm_weight, mw.norm_bias,
                mw.fc1_weight, mw.fc1_bias,
                mw.fc2_weight, mw.fc2_bias};
            ds_vp.outTensors = {*ds_out.Get()};

            s = ExecuteGraph(vis_deepstack_graph_, ds_vp);
            if (s != STATUS_OK) {
                LOG_ERROR("DeepstackMerger %zu execution failed", ds_idx);
                return s;
            }

            // Copy deepstack features to host
            size_t ds_bytes = static_cast<size_t>(merged_tokens) *
                              config_.vis_out_hidden_size * sizeof(uint16_t);
            ds_features[ds_idx].resize(merged_tokens * config_.vis_out_hidden_size);
            alloc->CopyToHost(ds_features[ds_idx].data(), *ds_out.Get(), ds_bytes);
            // ds_out auto-freed at end of block
        }
    }

    // 6. Run main merger
    {
        const auto& mw = weights_.merger;
        int64_t merged_tokens = total_tokens / (merge_size * merge_size);
        NpuTensor merged_out = AllocNpuFloat16({merged_tokens, config_.vis_out_hidden_size});

        atb::VariantPack vp;
        vp.inTensors = {
            *h_npu.Get(),
            mw.norm_weight, mw.norm_bias,
            mw.fc1_weight, mw.fc1_bias,
            mw.fc2_weight, mw.fc2_bias};
        vp.outTensors = {*merged_out.Get()};

        Status s = ExecuteGraph(vis_merger_graph_, vp);
        if (s != STATUS_OK) {
            LOG_ERROR("VisionMerger execution failed");
            return s;
        }

        // Copy merged vision embeddings to host
        size_t merged_bytes = static_cast<size_t>(merged_tokens) *
                              config_.vis_out_hidden_size * sizeof(uint16_t);
        alloc->CopyToHost(vis_embeds_out, *merged_out.Get(), merged_bytes);
        // merged_out auto-freed at end of block
    }

    // All NPU tensors are RAII-managed and freed automatically on scope exit.
    runtime_->Synchronize();

    LOG_INFO("Vision inference completed: %ld patches, %ld merged tokens",
             static_cast<long>(total_tokens),
             static_cast<long>(total_tokens / (merge_size * merge_size)));
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::ExecuteGraph(OperationHandle& graph,
                                  atb::VariantPack& vp) {
    if (!graph) return ERROR_GRAPH_BUILD;

    auto* ctx = runtime_->GetContext();

    uint64_t ws_size = 0;
    atb::Status atb_s = graph.get()->Setup(vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("Graph Setup failed: %d", static_cast<int>(atb_s));
        return ERROR_GRAPH_BUILD;
    }

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto [ws, ws_s] = runtime_->GetWorkspace(ws_size);
        ws_ptr = ws;
        if (ws_s != STATUS_OK) {
            LOG_ERROR("Failed to get workspace: %zu bytes", static_cast<size_t>(ws_size));
            return ws_s;
        }
        if (ws_ptr == nullptr) {
            LOG_ERROR("Workspace pointer is null despite size=%zu", static_cast<size_t>(ws_size));
            return ERROR_NPU_MEMORY;
        }
    } else {
        // GRAPH_LAUNCH_MODE requires non-null workspace device pointer
        auto [ws, ws_s] = runtime_->GetWorkspace(1);
        if (ws_s == STATUS_OK && ws != nullptr) {
            ws_ptr = ws;
            ws_size = 1;
        }
    }

    atb_s = graph.get()->Execute(vp, ws_ptr, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("Graph Execute failed: %d", static_cast<int>(atb_s));
        return ERROR_INFERENCE;
    }

    Status sync_s = runtime_->Synchronize();
    if (sync_s != STATUS_OK) {
        LOG_ERROR("Stream sync failed after Execute: %d", static_cast<int>(sync_s));
        return sync_s;
    }

    return STATUS_OK;
}

void Qwen3VLModel::EmbeddingLookup(const int64_t* input_ids, int64_t seq_len,
                                   uint16_t* output) {
    const uint16_t* embed_table = weights_.embed_weight_host.data();
    int64_t hs = config_.text_hidden_size;
    int64_t vocab_size = config_.text_vocab_size;

    for (int64_t s = 0; s < seq_len; s++) {
        int64_t token_id = input_ids[s];
        if (token_id < 0 || token_id >= vocab_size) {
            LOG_ERROR("EmbeddingLookup: token_id %ld out of range [0, %ld)",
                      static_cast<long>(token_id), static_cast<long>(vocab_size));
            // Fill with zeros for invalid token
            std::memset(output + s * hs, 0, hs * sizeof(uint16_t));
            continue;
        }
        std::memcpy(output + s * hs,
                    embed_table + token_id * hs,
                    hs * sizeof(uint16_t));
    }
}

std::vector<int64_t> Qwen3VLModel::FindImageTokenPositions(const int64_t* input_ids,
                                                           int64_t seq_len) {
    std::vector<int64_t> positions;
    for (int64_t s = 0; s < seq_len; s++) {
        if (input_ids[s] == config_.image_token_id) {
            positions.push_back(s);
        }
    }
    return positions;
}

void Qwen3VLModel::ComputePosEmbedInterp(const int64_t* grid_thw, int64_t num_images,
                                         uint16_t* pos_out) {
    // Fast position embedding interpolation
    // Equivalent to Python fast_pos_embed_interpolate
    auto* alloc = runtime_->GetAllocator();

    int64_t vis_hs = config_.vis_hidden_size;
    int32_t num_grid = config_.num_grid();
    int32_t merge_size = config_.vis_spatial_merge_size;

    // Copy pos_embed weight from NPU to CPU
    WeightInfo pe_info;
    runtime_->GetWeightLoader()->GetTensor("model.visual.pos_embed.weight", pe_info);
    size_t pe_elements = 1;
    for (auto d : pe_info.shape)
        pe_elements *= d;
    std::vector<uint16_t> pe_host(pe_elements);
    // pos_embed is already on NPU as fp16, we need to get it to CPU
    // Use the raw data from safetensors
    const uint8_t* pe_data = runtime_->GetWeightLoader()->GetTensorData(
        "model.visual.pos_embed.weight");
    auto st_dtype = static_cast<safetensors::dtype>(pe_info.dtype);
    if (st_dtype == safetensors::kBFLOAT16) {
        atb_llm::Bf16ToFp16Buffer(reinterpret_cast<const uint16_t*>(pe_data),
                                  pe_host.data(), pe_elements);
    } else if (st_dtype == safetensors::kFLOAT16) {
        std::memcpy(pe_host.data(), pe_data, pe_elements * sizeof(uint16_t));
    }

    // Interpolate for each image
    int64_t out_offset = 0;
    for (int64_t img = 0; img < num_images; img++) {
        int64_t t = grid_thw[img * 3 + 0];
        int64_t h = grid_thw[img * 3 + 1];
        int64_t w = grid_thw[img * 3 + 2];

        // Bilinear interpolation from (num_grid, num_grid) to (h, w)
        std::vector<float> interpolated(h * w * vis_hs, 0.0f);

        for (int64_t hi = 0; hi < h; hi++) {
            for (int64_t wi = 0; wi < w; wi++) {
                float fy = static_cast<float>(hi) * (num_grid - 1) / (h - 1);
                float fx = static_cast<float>(wi) * (num_grid - 1) / (w - 1);

                int32_t y0 = static_cast<int32_t>(fy);
                int32_t x0 = static_cast<int32_t>(fx);
                int32_t y1 = std::min(y0 + 1, num_grid - 1);
                int32_t x1 = std::min(x0 + 1, num_grid - 1);

                float dy = fy - y0;
                float dx = fx - x0;

                int64_t idx00 = y0 * num_grid + x0;
                int64_t idx01 = y0 * num_grid + x1;
                int64_t idx10 = y1 * num_grid + x0;
                int64_t idx11 = y1 * num_grid + x1;

                float* out_row = interpolated.data() + (hi * w + wi) * vis_hs;
                for (int64_t d = 0; d < vis_hs; d++) {
                    // Read fp16 values and convert to fp32
                    float v00 = atb_llm::Fp16ToF32(pe_host[idx00 * vis_hs + d]);
                    float v01 = atb_llm::Fp16ToF32(pe_host[idx01 * vis_hs + d]);
                    float v10 = atb_llm::Fp16ToF32(pe_host[idx10 * vis_hs + d]);
                    float v11 = atb_llm::Fp16ToF32(pe_host[idx11 * vis_hs + d]);

                    out_row[d] = v00 * (1 - dy) * (1 - dx) + v01 * (1 - dy) * dx +
                                 v10 * dy * (1 - dx) + v11 * dy * dx;
                }
            }
        }

        // Apply spatial merge shuffling and convert to fp16
        // Python: pos_embed.view(t, h//ms, ms, w//ms, ms, -1).permute(0,1,3,2,4,5).flatten(0,4)
        int64_t merged_h = h / merge_size;
        int64_t merged_w = w / merge_size;

        int64_t patch_idx = 0;
        for (int64_t br = 0; br < merged_h; br++) {
            for (int64_t bc = 0; bc < merged_w; bc++) {
                for (int64_t ir = 0; ir < merge_size; ir++) {
                    for (int64_t ic = 0; ic < merge_size; ic++) {
                        int64_t row = br * merge_size + ir;
                        int64_t col = bc * merge_size + ic;
                        const float* src = interpolated.data() + (row * w + col) * vis_hs;
                        uint16_t* dst = pos_out + (out_offset + patch_idx) * vis_hs;
                        for (int64_t d = 0; d < vis_hs; d++) {
                            dst[d] = atb_llm::Bf16ToFp16(
                                static_cast<uint16_t>(
                                    atb_llm::FloatToUint32(src[d]) >> 16));
                        }
                        patch_idx++;
                    }
                }
            }
        }
        // Repeat for temporal dimension
        for (int64_t ti = 1; ti < t; ti++) {
            std::memcpy(pos_out + (out_offset + ti * merged_h * merged_w * merge_size * merge_size) * vis_hs,
                        pos_out + out_offset * vis_hs,
                        merged_h * merged_w * merge_size * merge_size * vis_hs * sizeof(uint16_t));
        }
        out_offset += t * h * w;
    }
}

void Qwen3VLModel::ComputeVisionRoPE(const int64_t* grid_thw, int64_t num_images,
                                     float* cos_out, float* sin_out,
                                     int64_t& total_tokens) {
    int32_t dim = config_.vis_head_dim();
    int32_t merge_size = config_.vis_spatial_merge_size;

    // Find max_hw
    int32_t max_hw = 0;
    for (int64_t i = 0; i < num_images; i++) {
        max_hw = std::max(max_hw, static_cast<int32_t>(grid_thw[i * 3 + 1]));
        max_hw = std::max(max_hw, static_cast<int32_t>(grid_thw[i * 3 + 2]));
    }

    // Compute freq table
    auto freq_table = vis_rope_->ComputeFreqTable(max_hw);

    // Compute vision rot pos emb
    total_tokens = components::ComputeVisionRotPosEmb(
        grid_thw, num_images, *vis_rope_, merge_size,
        freq_table, cos_out, sin_out);
}

}  // namespace adapters
}  // namespace atb_llm
