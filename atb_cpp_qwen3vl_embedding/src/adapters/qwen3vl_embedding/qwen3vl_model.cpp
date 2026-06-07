#include "adapters/qwen3vl_embedding/qwen3vl_model.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "utils/float_utils.h"
#include "safetensors.hh"
#include "core/tensor_allocator.h"
#include "core/npu_tensor.h"
#include "io/weight_helpers.h"
#include "log/logger.h"
#include <chrono>
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

    // 2b. Cache pos_embed on host (needed for CPU-side bilinear interpolation)
    {
        int64_t num_pos = config_.vis_num_position_embeddings;
        int64_t hs = config_.vis_hidden_size;
        vis_pos_embed_host_.resize(num_pos * hs);
        s = io::CopyWeightToFp16Host(*runtime->GetWeightLoader(),
                                     "model.visual.pos_embed.weight",
                                     vis_pos_embed_host_.data(),
                                     vis_pos_embed_host_.size() * sizeof(uint16_t));
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to cache vision pos_embed to host");
            return s;
        }
    }

    // 3. Initialize position encoding helpers
    mrope_ = std::make_unique<components::MRoPE>(
        config_.text_head_dim, config_.text_rope_theta, config_.text_mrope_section);
    vis_rope_ = std::make_unique<components::VisionRotaryEmbedding>(
        config_.vis_head_dim());

    // 4. Create VisionRunner and build vision graphs
    runners::VisionRunner::Config vis_cfg;
    vis_cfg.hidden_size = config_.vis_hidden_size;
    vis_cfg.num_heads = config_.vis_num_heads;
    vis_cfg.intermediate_size = config_.vis_intermediate_size;
    vis_cfg.depth = config_.vis_depth;
    vis_cfg.in_channels = config_.vis_in_channels;
    vis_cfg.temporal_patch_size = config_.vis_temporal_patch_size;
    vis_cfg.patch_size = config_.vis_patch_size;
    vis_cfg.spatial_merge_size = config_.vis_spatial_merge_size;
    vis_cfg.num_position_embeddings = config_.vis_num_position_embeddings;
    vis_cfg.deepstack_visual_indexes = config_.vis_deepstack_visual_indexes;
    vis_cfg.epsilon = config_.vis_epsilon;
    vision_runner_ = std::make_unique<runners::VisionRunner>(vis_cfg);

    s = vision_runner_->Build();
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build VisionRunner graphs");
        return s;
    }

    // 5. Setup deepstack fusion using vision runner's deepstack graph
    components::DeepstackFusion::Config ds_cfg;
    ds_cfg.vis_hidden_size = config_.vis_hidden_size;
    ds_cfg.vis_out_hidden_size = config_.vis_out_hidden_size;
    ds_cfg.spatial_merge_size = config_.vis_spatial_merge_size;
    ds_cfg.deepstack_visual_indexes = config_.vis_deepstack_visual_indexes;
    deepstack_fusion_ = std::make_unique<components::DeepstackFusion>(
        ds_cfg, std::move(vision_runner_->GetDeepstackGraph()));

    // Set merger weights from loaded weights
    for (size_t i = 0; i < weights_.deepstack_mergers.size(); i++) {
        const auto& mw = weights_.deepstack_mergers[i];
        components::DeepstackMergerWeights ds_w;
        ds_w.norm_weight = mw.norm_weight;
        ds_w.norm_bias = mw.norm_bias;
        ds_w.fc1_weight = mw.fc1_weight;
        ds_w.fc1_bias = mw.fc1_bias;
        ds_w.fc2_weight = mw.fc2_weight;
        ds_w.fc2_bias = mw.fc2_bias;
        deepstack_fusion_->SetMergerWeights(i, ds_w);
    }

    // 6. Create TextRunner (graph built lazily on first Forward via EnsureBuilt)
    runners::TextRunner::Config text_cfg;
    text_cfg.num_heads = config_.text_num_heads;
    text_cfg.num_kv_heads = config_.text_num_kv_heads;
    text_cfg.head_dim = config_.text_head_dim;
    text_cfg.intermediate_size = config_.text_intermediate_size;
    text_cfg.num_layers = config_.text_num_layers;
    text_cfg.epsilon = config_.text_rms_norm_eps;
    text_cfg.use_qk_norm = true;
    text_cfg.rotary_dim = 2;
    text_cfg.use_mask = true;
    text_runner_ = std::make_unique<runners::TextRunner>(text_cfg);

    LOG_INFO("Qwen3VLModel loaded successfully");
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

    s = BaseModel::RunPooling(inputs_embeds.data(), seq_len, hidden_size,
                              config_.normalize,
                              families::BaseModel::PoolingStrategy::LAST_TOKEN,
                              result);
    if (s != STATUS_OK) return s;

    runtime_->Synchronize();
    LOG_INFO("Forward completed: seq_len=%ld, output shape=(%ld)",
             static_cast<long>(seq_len), static_cast<long>(hidden_size));
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// ForwardWithTiming -- per-stage benchmark instrumentation
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::ForwardWithTiming(const InferRequest& request,
                                         InferResult& result,
                                         StageTimings& timings) {
    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_prev = t_start;

    // ── Validate input ───────────────────────────────────────
    const int64_t* input_ids = request.text.input_ids;
    int64_t seq_len = request.text.seq_length;
    if (!input_ids || request.text.batch_size != 1 || seq_len <= 0) {
        return ERROR_INVALID_PARAM;
    }

    int64_t hidden_size = config_.text_hidden_size;
    int32_t hd = config_.text_head_dim;
    int64_t vis_embed_dim = config_.vis_out_hidden_size;
    auto* alloc = runtime_->GetAllocator();

    // ── Stage: Text Embedding ────────────────────────────────
    std::vector<uint16_t> inputs_embeds(seq_len * hidden_size);
    EmbeddingLookup(input_ids, seq_len,
                    weights_.embed_weight_host.data(),
                    hidden_size, config_.text_vocab_size,
                    inputs_embeds.data());

    auto t_text_embed = std::chrono::high_resolution_clock::now();
    timings.text_embed_ms = std::chrono::duration<double, std::milli>(
        t_text_embed - t_prev).count();
    t_prev = t_text_embed;

    // ── Vision pipeline (if image provided) ──────────────────
    std::vector<uint16_t> vis_embeds_host;
    std::vector<int64_t> grid_thw_host;
    std::vector<std::vector<uint16_t>> ds_features;
    std::vector<int64_t> image_token_positions;
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

        int64_t vis_hs = config_.vis_hidden_size;
        int32_t vis_hd = config_.vis_head_dim();
        int64_t merge_size = config_.vis_spatial_merge_size;
        int64_t patch_dim = static_cast<int64_t>(config_.vis_in_channels) *
                            config_.vis_temporal_patch_size *
                            config_.vis_patch_size * config_.vis_patch_size;
        int64_t flat_elements = np * patch_dim;

        // ── Stage: Vision Pos Embed + RoPE (CPU) ─────────────
        std::vector<uint16_t> pos_embed_host(np * vis_hs);
        components::ComputePosEmbedInterp(vis_pos_embed_host_.data(),
                                          config_.vis_hidden_size, config_.num_grid(),
                                          merge_size, grid_thw_host.data(), num_images,
                                          pos_embed_host.data());

        int rope_dim = vis_hd * 2;
        std::vector<float> vis_cos_f32(np * rope_dim);
        std::vector<float> vis_sin_f32(np * rope_dim);
        int64_t total_tokens = vis_rope_->ComputeRoPE(
            grid_thw_host.data(), num_images, merge_size,
            vis_cos_f32.data(), vis_sin_f32.data());

        std::vector<uint16_t> cos_fp16(total_tokens * rope_dim);
        std::vector<uint16_t> sin_fp16(total_tokens * rope_dim);
        for (int64_t i = 0; i < total_tokens * rope_dim; i++) {
            cos_fp16[i] = atb_llm::Fp32ToFp16(vis_cos_f32[i]);
            sin_fp16[i] = atb_llm::Fp32ToFp16(vis_sin_f32[i]);
        }

        auto t_vis_pos = std::chrono::high_resolution_clock::now();
        timings.vision_pos_ms = std::chrono::duration<double, std::milli>(
            t_vis_pos - t_prev).count();
        t_prev = t_vis_pos;

        // ── Stage: Vision Model (NPU) ────────────────────────
        size_t pixel_bytes = static_cast<size_t>(flat_elements) * sizeof(uint16_t);
        size_t pos_bytes = static_cast<size_t>(np) * vis_hs * sizeof(uint16_t);

        NpuTensor pixels_npu = AllocNpuFloat16({flat_elements});
        alloc->CopyToDevice(*pixels_npu.Get(), pv, pixel_bytes);
        NpuTensor pos_npu = AllocNpuFloat16({np, vis_hs});
        alloc->CopyToDevice(*pos_npu.Get(), pos_embed_host.data(), pos_bytes);
        NpuTensor cos_npu = AllocNpuFloat16({total_tokens, static_cast<int64_t>(rope_dim)});
        alloc->CopyToDevice(*cos_npu.Get(), cos_fp16.data(),
                            cos_fp16.size() * sizeof(uint16_t));
        NpuTensor sin_npu = AllocNpuFloat16({total_tokens, static_cast<int64_t>(rope_dim)});
        alloc->CopyToDevice(*sin_npu.Get(), sin_fp16.data(),
                            sin_fp16.size() * sizeof(uint16_t));

        atb::Tensor vis_seqlen;
        int32_t vis_seqlen_val = static_cast<int32_t>(total_tokens);
        vis_seqlen.desc.dtype = ACL_INT32;
        vis_seqlen.desc.format = ACL_FORMAT_ND;
        vis_seqlen.desc.shape.dimNum = 1;
        vis_seqlen.desc.shape.dims[0] = 1;
        vis_seqlen.dataSize = sizeof(int32_t);
        vis_seqlen.hostData = &vis_seqlen_val;

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
            Status s = ExecuteGraph(vision_runner_->GetFirstLayerGraph(), vp);
            if (s != STATUS_OK) return s;
        }

        ds_features.resize(config_.vis_deepstack_visual_indexes.size());
        for (int32_t li = 1; li < config_.vis_depth; li++) {
            const auto& bw = weights_.vis_blocks[li];
            NpuTensor h_out = AllocNpuFloat16({total_tokens, vis_hs});
            atb::VariantPack vp;
            vp.inTensors = {
                *h_npu.Get(),
                bw.qkv_weight, bw.qkv_bias, bw.proj_weight, bw.proj_bias,
                bw.fc1_weight, bw.fc1_bias, bw.fc2_weight, bw.fc2_bias,
                bw.n1_weight, bw.n1_bias, bw.n2_weight, bw.n2_bias,
                *cos_npu.Get(), *sin_npu.Get(), vis_seqlen};
            vp.outTensors = {*h_out.Get()};
            Status s = ExecuteGraph(vision_runner_->GetBlockGraph(), vp);
            if (s != STATUS_OK) return s;
            h_npu = std::move(h_out);

            size_t fusion_idx = 0;
            if (deepstack_fusion_->IsDeepstackLayer(li, fusion_idx)) {
                Status ds_s = deepstack_fusion_->ExtractFeatures(
                    h_npu, total_tokens, li, fusion_idx, runtime_, ds_features);
                if (ds_s != STATUS_OK) return ds_s;
            }
        }

        // Merger
        int64_t merged_tokens = total_tokens / (merge_size * merge_size);
        vis_embeds_host.resize(merged_tokens * vis_embed_dim);
        {
            const auto& mw = weights_.merger;
            NpuTensor merged_out = AllocNpuFloat16({merged_tokens, vis_embed_dim});
            atb::VariantPack vp;
            vp.inTensors = {
                *h_npu.Get(),
                mw.norm_weight, mw.norm_bias,
                mw.fc1_weight, mw.fc1_bias,
                mw.fc2_weight, mw.fc2_bias};
            vp.outTensors = {*merged_out.Get()};
            Status s = ExecuteGraph(vision_runner_->GetMergerGraph(), vp);
            if (s != STATUS_OK) return s;

            size_t merged_bytes = static_cast<size_t>(merged_tokens) *
                                  vis_embed_dim * sizeof(uint16_t);
            alloc->CopyToHost(vis_embeds_host.data(), *merged_out.Get(), merged_bytes);
        }

        runtime_->Synchronize();
        auto t_vis_model = std::chrono::high_resolution_clock::now();
        timings.vision_model_ms = std::chrono::duration<double, std::milli>(
            t_vis_model - t_prev).count();
        t_prev = t_vis_model;

        // Inject vision tokens into text embeddings
        image_token_positions =
            FindImageTokenPositions(input_ids, seq_len, config_.image_token_id);
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

    // ── Stage: Position IDs (MRoPE) ──────────────────────────
    std::vector<int64_t> position_ids(3 * seq_len);
    components::GetRopeIndex(input_ids, 1, seq_len,
                             grid_thw_host.empty() ? nullptr : grid_thw_host.data(),
                             num_images, config_.image_token_id,
                             config_.vis_spatial_merge_size, position_ids.data());
    std::vector<float> cos_f32(seq_len * hd);
    std::vector<float> sin_f32(seq_len * hd);
    mrope_->Compute(position_ids.data(), 1, seq_len,
                    cos_f32.data(), sin_f32.data());
    std::vector<float> mask(seq_len * seq_len);
    runners::MakeCausalMask(seq_len, mask.data());

    auto t_pos_ids = std::chrono::high_resolution_clock::now();
    timings.position_ids_ms = std::chrono::duration<double, std::milli>(
        t_pos_ids - t_prev).count();
    t_prev = t_pos_ids;

    // ── Stage: Text Model (NPU) ──────────────────────────────
    {
        int64_t n = seq_len;
        NpuTensor h_npu = AllocNpuFloat16({n, hidden_size});
        alloc->CopyToDevice(*h_npu.Get(), inputs_embeds.data(),
                            static_cast<size_t>(n) * hidden_size * sizeof(uint16_t));
        int64_t rope_n = n * hd;
        std::vector<uint16_t> cos16(rope_n), sin16(rope_n), m16(n * n);
        for (int64_t i = 0; i < rope_n; i++) {
            cos16[i] = atb_llm::Fp32ToFp16(cos_f32[i]);
            sin16[i] = atb_llm::Fp32ToFp16(sin_f32[i]);
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
        Status s = text_runner_->EnsureBuilt(seq_len);
        if (s != STATUS_OK) return s;
        for (int32_t li = 0; li < config_.text_num_layers; li++) {
            const auto& lw = weights_.text_layers[li];
            NpuTensor h_out = AllocNpuFloat16({n, hidden_size});
            atb::VariantPack vp;
            vp.inTensors = {*h_npu.Get(),
                            lw.q_weight, lw.k_weight, lw.v_weight, lw.o_weight,
                            lw.q_norm_weight, lw.k_norm_weight,
                            lw.gate_weight, lw.up_weight, lw.down_weight,
                            lw.input_ln_weight, lw.post_ln_weight,
                            *cos_npu.Get(), *sin_npu.Get(), *mask_npu.Get(), seqlen};
            vp.outTensors = {*h_out.Get()};
            s = ExecuteGraph(text_runner_->GetLayerGraph(), vp);
            if (s != STATUS_OK) return s;
            h_npu = std::move(h_out);
            if (li < static_cast<int32_t>(ds_features.size()) &&
                !ds_features[li].empty() && !image_token_positions.empty()) {
                deepstack_fusion_->InjectFeatures(
                    h_npu, ds_features[li], image_token_positions,
                    n, hidden_size, vis_embed_dim, runtime_);
            }
        }
        NpuTensor norm_out = AllocNpuFloat16({n, hidden_size});
        atb::VariantPack nvp;
        nvp.inTensors = {*h_npu.Get(), weights_.text_norm_weight};
        nvp.outTensors = {*norm_out.Get()};
        s = ExecuteGraph(text_runner_->GetNormGraph(), nvp);
        if (s != STATUS_OK) return s;
        runtime_->Synchronize();
        alloc->CopyToHost(inputs_embeds.data(), *norm_out.Get(),
                          static_cast<size_t>(n) * hidden_size * sizeof(uint16_t));
    }

    auto t_text_model = std::chrono::high_resolution_clock::now();
    timings.text_model_ms = std::chrono::duration<double, std::milli>(
        t_text_model - t_prev).count();
    t_prev = t_text_model;

    // ── Stage: Pooling ───────────────────────────────────────
    Status s = BaseModel::RunPooling(inputs_embeds.data(), seq_len, hidden_size,
                                     config_.normalize,
                                     families::BaseModel::PoolingStrategy::LAST_TOKEN,
                                     result);
    if (s != STATUS_OK) return s;
    runtime_->Synchronize();

    auto t_end = std::chrono::high_resolution_clock::now();
    timings.pooling_ms = std::chrono::duration<double, std::milli>(
        t_end - t_prev).count();
    timings.e2e_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

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
    BaseModel::EmbeddingLookup(input_ids, seq_len,
                               weights_.embed_weight_host.data(),
                               config_.text_hidden_size, config_.text_vocab_size,
                               inputs_embeds.data());

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
        image_token_positions = BaseModel::FindImageTokenPositions(input_ids, seq_len, config_.image_token_id);
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
    Status s = text_runner_->EnsureBuilt(seq_len);
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
        s = ExecuteGraph(text_runner_->GetLayerGraph(), vp);
        if (s != STATUS_OK) {
            LOG_ERROR("TextDecoderLayer %d failed", li);
            return s;
        }
        h_npu = std::move(h_out);
        if (li < static_cast<int32_t>(ds_features.size()) &&
            !ds_features[li].empty() && !image_token_positions.empty()) {
            deepstack_fusion_->InjectFeatures(h_npu, ds_features[li], image_token_positions, n, hs, ved, runtime_);
        }
    }
    NpuTensor norm_out = AllocNpuFloat16({n, hs});
    atb::VariantPack nvp;
    nvp.inTensors = {*h_npu.Get(), weights_.text_norm_weight};
    nvp.outTensors = {*norm_out.Get()};
    s = ExecuteGraph(text_runner_->GetNormGraph(), nvp);
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

    // 1. Compute position embeddings (CPU) via extracted component
    size_t pos_bytes = static_cast<size_t>(num_patches) * vis_hs * sizeof(uint16_t);
    std::vector<uint16_t> pos_embed_host(num_patches * vis_hs);
    components::ComputePosEmbedInterp(vis_pos_embed_host_.data(),
                                      config_.vis_hidden_size, config_.num_grid(),
                                      config_.vis_spatial_merge_size,
                                      grid_thw, num_images,
                                      pos_embed_host.data());

    // 2. Compute vision RoPE (CPU) via VisionRotaryEmbedding component
    std::vector<float> cos_f32(num_patches * vis_hd * 2);
    std::vector<float> sin_f32(num_patches * vis_hd * 2);
    int64_t total_tokens = vis_rope_->ComputeRoPE(
        grid_thw, num_images, config_.vis_spatial_merge_size,
        cos_f32.data(), sin_f32.data());

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

        Status s = ExecuteGraph(vision_runner_->GetFirstLayerGraph(), vp);
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

        Status s = ExecuteGraph(vision_runner_->GetBlockGraph(), vp);
        if (s != STATUS_OK) {
            LOG_ERROR("VisionBlock %d execution failed", li);
            return s;
        }

        // Swap: old h_npu freed by move-assign, new h_out takes ownership
        h_npu = std::move(h_out);

        // Check if this layer produces deepstack features
        size_t fusion_idx = 0;
        if (deepstack_fusion_->IsDeepstackLayer(li, fusion_idx)) {
            Status ds_s = deepstack_fusion_->ExtractFeatures(
                h_npu, total_tokens, li, fusion_idx, runtime_, ds_features);
            if (ds_s != STATUS_OK) {
                LOG_ERROR("DeepstackMerger %zu execution failed", fusion_idx);
                return ds_s;
            }
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

        Status s = ExecuteGraph(vision_runner_->GetMergerGraph(), vp);
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

}  // namespace adapters
}  // namespace atb_llm
