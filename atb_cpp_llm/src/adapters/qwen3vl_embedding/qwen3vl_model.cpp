#include "adapters/qwen3vl_embedding/qwen3vl_model.h"
#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "utils/float_utils.h"
#include "safetensors.hh"
#include "core/tensor_allocator.h"
#include "core/npu_tensor.h"
#include "core/debug_dump.h"
#include "io/weight_helpers.h"
#include "log/logger.h"
#include "util/cpp11_compat.h"
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdlib>

namespace atb_llm {
namespace adapters {

namespace {
// G5 experiment: when ATB_SKIP_TIMING_SYNCS=1, skip the stage-boundary
// Synchronize calls that exist purely for per-stage timing accuracy.
// These syncs are NOT needed for correctness — they only ensure the
// NPU timer captures true GPU execution time rather than launch time.
//
// The D2H-before-CopyToHost syncs (vision merger, text FinalNorm) and
// the deepstack InjectFeatures sync are NEVER skipped — those are
// correctness-critical.
bool SkipTimingSyncs() {
    const char* env = getenv("ATB_SKIP_TIMING_SYNCS");
    return env != nullptr;
}
}  // namespace

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

    // 2c. 310P GQA→MHA weight expansion.
    //     310P SelfAttention may not support GQA (kv_head_num < head_num).
    //     Expand K/V/K-norm weights so the graph uses MHA (same as vision path).
    if (Is310P() && config_.text_num_kv_heads < config_.text_num_heads) {
        int32_t nh = config_.text_num_heads;
        int32_t kv_nh = config_.text_num_kv_heads;
        int32_t hd = config_.text_head_dim;
        int32_t ratio = nh / kv_nh;
        LOG_INFO("310P: Expanding GQA→MHA: nh=%d kv_nh=%d ratio=%d hd=%d",
                 nh, kv_nh, ratio, hd);

        auto* alloc = runtime_->GetAllocator();
        for (int32_t i = 0; i < config_.text_num_layers; i++) {
            auto& lw = weights_.text_layers[i];

            // k_weight: (kv_nh*hd, hidden) → (nh*hd, hidden)
            {
                int64_t old_rows = lw.k_weight.desc.shape.dims[0];  // kv_nh * hd
                int64_t hidden = lw.k_weight.desc.shape.dims[1];
                size_t old_elems = static_cast<size_t>(old_rows * hidden);

                std::vector<uint16_t> k_old(old_elems);
                s = alloc->CopyToHost(k_old.data(), lw.k_weight,
                                      old_elems * sizeof(uint16_t));
                if (s != STATUS_OK) return s;

                std::vector<uint16_t> k_new(nh * hd * hidden);
                for (int32_t h = 0; h < nh; h++) {
                    int32_t src_h = h / ratio;
                    std::memcpy(k_new.data() + h * hd * hidden,
                                k_old.data() + src_h * hd * hidden,
                                hd * hidden * sizeof(uint16_t));
                }

                s = alloc->AllocFloat16(lw.k_weight, {nh * hd, hidden});
                if (s != STATUS_OK) return s;
                s = alloc->CopyToDevice(lw.k_weight, k_new.data(),
                                        k_new.size() * sizeof(uint16_t));
                if (s != STATUS_OK) return s;
            }

            // v_weight: (kv_nh*hd, hidden) → (nh*hd, hidden)
            {
                int64_t old_rows = lw.v_weight.desc.shape.dims[0];
                int64_t hidden = lw.v_weight.desc.shape.dims[1];
                size_t old_elems = static_cast<size_t>(old_rows * hidden);

                std::vector<uint16_t> v_old(old_elems);
                s = alloc->CopyToHost(v_old.data(), lw.v_weight,
                                      old_elems * sizeof(uint16_t));
                if (s != STATUS_OK) return s;

                std::vector<uint16_t> v_new(nh * hd * hidden);
                for (int32_t h = 0; h < nh; h++) {
                    int32_t src_h = h / ratio;
                    std::memcpy(v_new.data() + h * hd * hidden,
                                v_old.data() + src_h * hd * hidden,
                                hd * hidden * sizeof(uint16_t));
                }

                s = alloc->AllocFloat16(lw.v_weight, {nh * hd, hidden});
                if (s != STATUS_OK) return s;
                s = alloc->CopyToDevice(lw.v_weight, v_new.data(),
                                        v_new.size() * sizeof(uint16_t));
                if (s != STATUS_OK) return s;
            }

            // k_norm_weight: (hd,) or (kv_nh*hd,) → (nh*hd,)
            {
                int64_t kn_dim0 = lw.k_norm_weight.desc.shape.dims[0];
                int64_t kn_elems = kn_dim0;
                // If per-kv-group (kv_nh*hd,), expand to (nh*hd,)
                if (kn_elems == kv_nh * hd) {
                    std::vector<uint16_t> kn_old(kn_elems);
                    s = alloc->CopyToHost(kn_old.data(), lw.k_norm_weight,
                                          kn_elems * sizeof(uint16_t));
                    if (s != STATUS_OK) return s;

                    std::vector<uint16_t> kn_new(nh * hd);
                    for (int32_t h = 0; h < nh; h++) {
                        int32_t src_h = h / ratio;
                        std::memcpy(kn_new.data() + h * hd,
                                    kn_old.data() + src_h * hd,
                                    hd * sizeof(uint16_t));
                    }

                    s = alloc->AllocFloat16(lw.k_norm_weight, {nh * hd});
                    if (s != STATUS_OK) return s;
                    s = alloc->CopyToDevice(lw.k_norm_weight, kn_new.data(),
                                            kn_new.size() * sizeof(uint16_t));
                    if (s != STATUS_OK) return s;
                }
                // else: per-head (hd,) — already correct for MHA, no expansion needed
            }
        }

        // Update effective head count so graph uses MHA
        config_.text_num_kv_heads = config_.text_num_heads;
    }

    // 2d. Upload pos_embed to NPU + build the interp graph (one-time cost).
    // The graph is shape-agnostic; runtime tensors carry the actual N.
    {
        int64_t num_pos = config_.vis_num_position_embeddings;
        int64_t hs = config_.vis_hidden_size;
        s = runtime->GetAllocator()->AllocFloat16(vis_pos_embed_npu_, {num_pos, hs});
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to alloc pos_embed NPU tensor");
            return s;
        }
        s = runtime->GetAllocator()->CopyToDevice(
            vis_pos_embed_npu_, vis_pos_embed_host_.data(),
            vis_pos_embed_host_.size() * sizeof(uint16_t));
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to copy pos_embed to NPU");
            return s;
        }

        s = components::PosEmbedInterpGraph::Build(
            "PosEmbedInterp", pos_embed_interp_graph_);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to build PosEmbedInterpGraph");
            return s;
        }
    }

    // 2d. Build Vision RoPE graph (cos/sin generation).
    {
        s = components::VisRopeGraph::Build("VisRope", vis_rope_graph_);
        if (s != STATUS_OK) {
            LOG_ERROR("Failed to build VisRopeGraph");
            return s;
        }
    }

    // 3. Initialize position encoding helpers
    mrope_ = atb_llm::make_unique<components::MRoPE>(
        config_.text_head_dim, config_.text_rope_theta, config_.text_mrope_section);
    // Python: VisionRotaryEmbedding(dim=hd_v // 2) — half the head dim
    vis_rope_ = atb_llm::make_unique<components::VisionRotaryEmbedding>(
        config_.vis_head_dim() / 2);

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
    vision_runner_ = atb_llm::make_unique<runners::VisionRunner>(vis_cfg);

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
    deepstack_fusion_ = atb_llm::make_unique<components::DeepstackFusion>(
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

    // Build the NPU IndexAdd op used for cross-modal injection.
    s = deepstack_fusion_->BuildInjectOp();
    if (s != STATUS_OK) {
        LOG_ERROR("Failed to build Deepstack inject op");
        return s;
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
    text_runner_ = atb_llm::make_unique<runners::TextRunner>(text_cfg);

    LOG_INFO("Qwen3VLModel loaded successfully");
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// Forward — thin wrapper over ForwardWithTiming.
//
// Phase17 P1: the legacy fp32-cos/sin/mask path is removed.  All
// inference flows through the single timing-aware implementation,
// which uses direct fp16 generation + NPU cache for mask/cos/sin.
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::Forward(const InferRequest& request, InferResult& result) {
    StageTimings ignored;
    return ForwardWithTiming(request, result, ignored);
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
    std::vector<NpuTensor> ds_features;
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

        // ── Stage: Vision Pos Embed + RoPE ─────────────────────
        // PosEmbed: computed directly on NPU via the interp graph.
        NpuTensor pos_npu = AllocNpuFloat16({np, vis_hs});
        Status pe_s = ComputePosEmbedNpu(grid_thw_host.data(), num_images,
                                          np, *pos_npu.Get());
        if (pe_s != STATUS_OK) return pe_s;

        // VisionRoPE cos/sin: computed on NPU.
        // total_tokens = np (= sum over images of T*H*W)
        NpuTensor cos_npu = AllocNpuFloat16({np, static_cast<int64_t>(vis_hd)});
        NpuTensor sin_npu = AllocNpuFloat16({np, static_cast<int64_t>(vis_hd)});
        int64_t total_tokens = ComputeVisionRopeNpu(
            grid_thw_host.data(), num_images, *cos_npu.Get(), *sin_npu.Get());
        if (total_tokens < 0 || total_tokens != np) {
            LOG_ERROR("Vision RoPE total_tokens=%ld != np=%ld",
                      static_cast<long>(total_tokens), static_cast<long>(np));
            return ERROR_INFERENCE;
        }

        // DEBUG: Save RoPE cos/sin for comparison with Python
        debug::DumpNpuFp16(runtime_, *cos_npu.Get(), total_tokens * vis_hd,
                           "/tmp/cpp_rope_cos.bin");
        debug::DumpNpuFp16(runtime_, *sin_npu.Get(), total_tokens * vis_hd,
                           "/tmp/cpp_rope_sin.bin");

        // Synchronize to capture true GPU time for pos_embed + RoPE.
        // Without this, the NPU time leaks into vision_model_ms.
        // G5: skippable — timing-only, not correctness-critical.
        if (!SkipTimingSyncs()) {
            runtime_->Synchronize();
        }
        auto t_vis_pos = std::chrono::high_resolution_clock::now();
        timings.vision_pos_ms = std::chrono::duration<double, std::milli>(
            t_vis_pos - t_prev).count();
        t_prev = t_vis_pos;

        // ── Stage: Vision Model (NPU) ────────────────────────
        // pos_npu, cos_npu, sin_npu were already populated above.
        size_t pixel_bytes = static_cast<size_t>(flat_elements) * sizeof(uint16_t);

        NpuTensor pixels_npu = AllocNpuFloat16({flat_elements});
        alloc->CopyToDevice(*pixels_npu.Get(), pv, pixel_bytes);

        atb::Tensor vis_seqlen;
        int32_t vis_seqlen_val = static_cast<int32_t>(total_tokens);
        vis_seqlen.desc.dtype = ACL_INT32;
        vis_seqlen.desc.format = ACL_FORMAT_ND;
        vis_seqlen.desc.shape.dimNum = 1;
        vis_seqlen.desc.shape.dims[0] = 1;
        vis_seqlen.dataSize = sizeof(int32_t);
        vis_seqlen.hostData = &vis_seqlen_val;

        // ── Vision sub-stage: FirstLayer (patch_embed + pos_embed + block 0) ──
        auto t_v_first = std::chrono::high_resolution_clock::now();
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

        // DEBUG: Save first layer output
        debug::DumpNpuFp16(runtime_, *h_npu.Get(), total_tokens * vis_hs,
                           "/tmp/cpp_first_layer_out.bin");

        // ── Vision sub-stage: Blocks 1..(depth-1) + Deepstack extraction ──
        auto t_v_blocks = std::chrono::high_resolution_clock::now();
        double v_first_ms = std::chrono::duration<double, std::milli>(
            t_v_blocks - t_v_first).count();

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

            // DEBUG: Save block output at deepstack layers and last block
            if (debug::VisionDumpEnabled()) {
                bool is_ds = false;
                for (auto di : config_.vis_deepstack_visual_indexes) {
                    if (li == di) { is_ds = true; break; }
                }
                if (is_ds || li == config_.vis_depth - 1) {
                    char path[256];
                    std::snprintf(path, sizeof(path), "/tmp/cpp_block_%d_out.bin", li);
                    debug::DumpNpuFp16(runtime_, *h_npu.Get(),
                                       total_tokens * vis_hs, path);
                }
            }

            size_t fusion_idx = 0;
            if (deepstack_fusion_->IsDeepstackLayer(li, fusion_idx)) {
                Status ds_s = deepstack_fusion_->ExtractFeatures(
                    h_npu, total_tokens, li, fusion_idx, runtime_, ds_features);
                if (ds_s != STATUS_OK) return ds_s;
            }
        }

        // ── Vision sub-stage: Merger + D2H ──
        auto t_v_merger = std::chrono::high_resolution_clock::now();
        double v_blocks_ms = std::chrono::duration<double, std::milli>(
            t_v_merger - t_v_blocks).count();

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
        double v_merger_ms = std::chrono::duration<double, std::milli>(
            t_vis_model - t_v_merger).count();
        timings.vision_model_ms = std::chrono::duration<double, std::milli>(
            t_vis_model - t_prev).count();
        t_prev = t_vis_model;

        LOG_INFO("  [Vision detail] FirstLayer: %.2f ms | Blocks(1..%d): %.2f ms | Merger+D2H: %.2f ms",
                 v_first_ms, config_.vis_depth - 1, v_blocks_ms, v_merger_ms);

        // Inject vision tokens into text embeddings
        image_token_positions =
            FindImageTokenPositions(input_ids, seq_len, config_.image_token_id);
        // merged_tokens already computed above (from merger output)
        if (image_token_positions.size() != static_cast<size_t>(merged_tokens)) {
            LOG_ERROR("Image token count mismatch: pos=%zu, merged_vis=%ld (patches=%ld)",
                      image_token_positions.size(), static_cast<long>(merged_tokens),
                      static_cast<long>(np));
            return ERROR_INVALID_PARAM;
        }
        for (size_t i = 0; i < image_token_positions.size(); i++) {
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
                             config_.vision_start_token_id,
                             config_.vis_spatial_merge_size, position_ids.data());

    // DEBUG: Save inputs_embeds and position_ids for comparison with Python
    debug::DumpHostFp16(inputs_embeds.data(), seq_len * hidden_size,
                        "/tmp/cpp_inputs_embeds.bin");
    debug::DumpHostInt64(position_ids.data(), 3 * seq_len,
                         "/tmp/cpp_position_ids.bin");
    // ── P0: MRoPE cos/sin + Causal mask — direct fp16 + NPU cache ──
    // When seq_len is unchanged across calls, reuse the cached NPU tensors
    // to avoid regeneration + H2D upload (saves ~212ms at S=4096).
    bool cache_hit = (seq_len == cached_text_seq_len_ && cached_mask_npu_);
    if (!cache_hit) {
        // Generate cos/sin directly in fp16 (skip fp32 intermediate)
        std::vector<uint16_t> cos16(seq_len * hd);
        std::vector<uint16_t> sin16(seq_len * hd);
        mrope_->ComputeFp16(position_ids.data(), 1, seq_len,
                            cos16.data(), sin16.data());
        // Generate causal mask directly in fp16 (skip fp32 intermediate)
        std::vector<uint16_t> m16(static_cast<size_t>(seq_len) * seq_len);
        runners::MakeCausalMaskFp16(seq_len, m16.data());
        // Upload to NPU and cache
        cached_cos_npu_ = AllocNpuFloat16({seq_len, hd});
        alloc->CopyToDevice(*cached_cos_npu_.Get(), cos16.data(),
                            static_cast<size_t>(seq_len) * hd * sizeof(uint16_t));
        cached_sin_npu_ = AllocNpuFloat16({seq_len, hd});
        alloc->CopyToDevice(*cached_sin_npu_.Get(), sin16.data(),
                            static_cast<size_t>(seq_len) * hd * sizeof(uint16_t));
        cached_mask_npu_ = AllocNpuFloat16({seq_len, seq_len});
        alloc->CopyToDevice(*cached_mask_npu_.Get(), m16.data(),
                            static_cast<size_t>(seq_len) * seq_len * sizeof(uint16_t));
        cached_text_seq_len_ = seq_len;
    }

    auto t_pos_ids = std::chrono::high_resolution_clock::now();
    timings.position_ids_ms = std::chrono::duration<double, std::milli>(
        t_pos_ids - t_prev).count();
    t_prev = t_pos_ids;

    // ── Stage: Text Model (NPU) ──────────────────────────────
    double t_h2d_ms = 0, t_layers_ms = 0, t_norm_ms = 0;
    std::chrono::high_resolution_clock::time_point t_t_norm;
    {
        int64_t n = seq_len;
        NpuTensor h_npu = AllocNpuFloat16({n, hidden_size});
        alloc->CopyToDevice(*h_npu.Get(), inputs_embeds.data(),
                            static_cast<size_t>(n) * hidden_size * sizeof(uint16_t));
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

        // ── Text sub-stage: 28 Decoder Layers ──
        auto t_t_layers = std::chrono::high_resolution_clock::now();
        t_h2d_ms = std::chrono::duration<double, std::milli>(
            t_t_layers - t_prev).count();

        for (int32_t li = 0; li < config_.text_num_layers; li++) {
            const auto& lw = weights_.text_layers[li];
            NpuTensor h_out = AllocNpuFloat16({n, hidden_size});
            atb::VariantPack vp;
            vp.inTensors = {*h_npu.Get(),
                            lw.q_weight, lw.k_weight, lw.v_weight, lw.o_weight,
                            lw.q_norm_weight, lw.k_norm_weight,
                            lw.gate_weight, lw.up_weight, lw.down_weight,
                            lw.input_ln_weight, lw.post_ln_weight,
                            *cached_cos_npu_.Get(), *cached_sin_npu_.Get(),
                            *cached_mask_npu_.Get(), seqlen};
            vp.outTensors = {*h_out.Get()};
            s = ExecuteGraph(text_runner_->GetLayerGraph(), vp);
            if (s != STATUS_OK) return s;
            h_npu = std::move(h_out);
            if (li < static_cast<int32_t>(ds_features.size()) &&
                ds_features[li] && !image_token_positions.empty()) {
                deepstack_fusion_->InjectFeatures(
                    h_npu, ds_features[li], image_token_positions,
                    n, hidden_size, vis_embed_dim, runtime_);
            }
        }

        // ── Text sub-stage: FinalNorm + D2H ──
        t_t_norm = std::chrono::high_resolution_clock::now();
        t_layers_ms = std::chrono::duration<double, std::milli>(
            t_t_norm - t_t_layers).count();

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
    t_norm_ms = std::chrono::duration<double, std::milli>(
        t_text_model - t_t_norm).count();
    timings.text_model_ms = std::chrono::duration<double, std::milli>(
        t_text_model - t_prev).count();
    t_prev = t_text_model;

    LOG_INFO("  [Text detail] H2D prep: %.2f ms | %d layers: %.2f ms (avg %.3f/layer) | Norm+D2H: %.2f ms",
             t_h2d_ms, config_.text_num_layers, t_layers_ms,
             t_layers_ms / config_.text_num_layers, t_norm_ms);

    // ── Stage: Pooling ───────────────────────────────────────
    Status s = BaseModel::RunPooling(inputs_embeds.data(), seq_len, hidden_size,
                                     config_.normalize,
                                     request.text.attention_mask
                                        ? families::BaseModel::PoolingStrategy::LAST_TOKEN_BY_MASK
                                        : families::BaseModel::PoolingStrategy::LAST_TOKEN,
                                     result,
                                     request.text.attention_mask);
    if (s != STATUS_OK) return s;
    // G5: timing-only sync for pooling stage — skippable.
    if (!SkipTimingSyncs()) {
        runtime_->Synchronize();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    timings.pooling_ms = std::chrono::duration<double, std::milli>(
        t_end - t_prev).count();
    timings.e2e_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    return STATUS_OK;
}


// ═════════════════════════════════════════════════════════════════════
// Vision pipeline
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::RunVision(const uint16_t* pixel_values, int64_t num_patches,
                               const int64_t* grid_thw, int64_t num_images,
                               uint16_t* vis_embeds_out, int64_t vis_embed_dim,
                               std::vector<NpuTensor>& ds_features) {
    (void)vis_embed_dim;  // caller-side bookkeeping only; output stride is derived from config_
    auto* alloc = runtime_->GetAllocator();

    int64_t vis_hs = config_.vis_hidden_size;
    int32_t vis_hd = config_.vis_head_dim();
    int64_t merge_size = config_.vis_spatial_merge_size;

    // 1. Compute position embeddings on NPU via the interp graph.
    NpuTensor pos_npu = AllocNpuFloat16({num_patches, vis_hs});
    Status pe_s = ComputePosEmbedNpu(grid_thw, num_images,
                                      num_patches, *pos_npu.Get());
    if (pe_s != STATUS_OK) return pe_s;

    // 2. Compute vision RoPE cos/sin on NPU.
    // VisionRoPE outputs (total_tokens, vis_hd) — dim*2 = half*4 = vis_hd
    NpuTensor cos_npu = AllocNpuFloat16({num_patches, static_cast<int64_t>(vis_hd)});
    NpuTensor sin_npu = AllocNpuFloat16({num_patches, static_cast<int64_t>(vis_hd)});
    int64_t total_tokens = ComputeVisionRopeNpu(
        grid_thw, num_images, *cos_npu.Get(), *sin_npu.Get());
    if (total_tokens < 0 || total_tokens != num_patches) {
        LOG_ERROR("Vision RoPE total_tokens=%ld != num_patches=%ld",
                  static_cast<long>(total_tokens), static_cast<long>(num_patches));
        return ERROR_INFERENCE;
    }

    // DEBUG: Save RoPE cos/sin for comparison with Python
    debug::DumpNpuFp16(runtime_, *cos_npu.Get(), total_tokens * vis_hd,
                       "/tmp/cpp_rope_cos.bin");
    debug::DumpNpuFp16(runtime_, *sin_npu.Get(), total_tokens * vis_hd,
                       "/tmp/cpp_rope_sin.bin");

    // 3. Copy vision inputs to NPU (pos_npu/cos_npu/sin_npu already done)
    int64_t patch_dim = static_cast<int64_t>(config_.vis_in_channels) *
                        config_.vis_temporal_patch_size *
                        config_.vis_patch_size * config_.vis_patch_size;
    int64_t flat_elements = num_patches * patch_dim;
    size_t pixel_bytes = static_cast<size_t>(flat_elements) * sizeof(uint16_t);

    // PatchEmbed graph expects 1D flat input: (N * patch_dim,)
    // Its Reshape divides by kernel_size to recover N
    NpuTensor pixels_npu = AllocNpuFloat16({flat_elements});
    alloc->CopyToDevice(*pixels_npu.Get(), pixel_values, pixel_bytes);

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

    // DEBUG: Save first layer output
    debug::DumpNpuFp16(runtime_, *h_npu.Get(), total_tokens * vis_hs,
                       "/tmp/cpp_first_layer_out.bin");

    // CONTROLLED FEED: When ATB_DEBUG_VISION=2, overwrite C++ first_layer
    // output with Python's first_layer output to isolate whether block
    // computation differs or error is amplified from input difference.
    debug::MaybeFeedNpuFp16(runtime_, *h_npu.Get(), total_tokens * vis_hs,
                            "/tmp/diag_first_layer_out.bin");

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

        // DEBUG: Save block 1 output for early-divergence diagnosis
        if (li == 1) {
            debug::DumpNpuFp16(runtime_, *h_npu.Get(), total_tokens * vis_hs,
                               "/tmp/cpp_block_1_out.bin");
        }

        // DEBUG: Save block output at deepstack layers and last block
        if (debug::VisionDumpEnabled()) {
            bool is_ds = false;
            for (auto di : config_.vis_deepstack_visual_indexes) {
                if (li == di) { is_ds = true; break; }
            }
            if (is_ds || li == config_.vis_depth - 1) {
                char path[256];
                std::snprintf(path, sizeof(path), "/tmp/cpp_block_%d_out.bin", li);
                debug::DumpNpuFp16(runtime_, *h_npu.Get(),
                                   total_tokens * vis_hs, path);
            }
        }

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

        // DEBUG: Save merged vision output for comparison with Python
        debug::DumpHostFp16(vis_embeds_out,
                            merged_tokens * config_.vis_out_hidden_size,
                            "/tmp/cpp_vision_merged.bin");
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
// ComputePosEmbedNpu — host Stage A + NPU Stage B replacement for the
// old CPU `ComputePosEmbedInterp`.
//
// Caller owns @p out_npu (must be pre-allocated to {expected_n, vis_hs} fp16).
// We allocate the 8 transient (idx, wt) tensors locally; they free on scope
// exit via the central TensorAllocator tracker.
// ═════════════════════════════════════════════════════════════════════

Status Qwen3VLModel::ComputePosEmbedNpu(const int64_t* grid_thw, int64_t num_images,
                                         int64_t expected_n,
                                         atb::Tensor& out_npu) {
    if (!pos_embed_interp_graph_) {
        LOG_ERROR("ComputePosEmbedNpu: graph not built");
        return ERROR_GRAPH_BUILD;
    }
    if (!grid_thw || num_images <= 0) {
        LOG_ERROR("ComputePosEmbedNpu: invalid grid_thw/num_images");
        return ERROR_INVALID_PARAM;
    }

    int32_t num_grid = config_.num_grid();
    int32_t merge_size = static_cast<int32_t>(config_.vis_spatial_merge_size);

    // ── Stage A: host-side index/weight construction ──
    std::vector<int32_t> idx_host[4];
    std::vector<uint16_t> wt_host[4];
    components::BuildPosEmbedIndicesAndWeights(
        grid_thw, num_images, num_grid, merge_size, idx_host, wt_host);

    int64_t n = static_cast<int64_t>(idx_host[0].size());
    if (n != expected_n) {
        LOG_ERROR("ComputePosEmbedNpu: built N=%ld != expected=%ld",
                  static_cast<long>(n), static_cast<long>(expected_n));
        return ERROR_INVALID_PARAM;
    }

    // ── Stage B: NPU graph ──
    auto* alloc = runtime_->GetAllocator();

    NpuTensor idx_npu[4], wt_npu[4];
    for (int k = 0; k < 4; k++) {
        idx_npu[k] = AllocNpuInt32({n});
        if (!idx_npu[k]) {
            LOG_ERROR("ComputePosEmbedNpu: alloc idx%d failed", k);
            return ERROR_NPU_MEMORY;
        }
        // wt shape (N, 1) so ElewiseMul broadcasts over vis_hs
        wt_npu[k] = AllocNpuFloat16({n, 1});
        if (!wt_npu[k]) {
            LOG_ERROR("ComputePosEmbedNpu: alloc wt%d failed", k);
            return ERROR_NPU_MEMORY;
        }

        alloc->CopyToDevice(*idx_npu[k].Get(), idx_host[k].data(),
                            n * sizeof(int32_t));
        alloc->CopyToDevice(*wt_npu[k].Get(), wt_host[k].data(),
                            n * sizeof(uint16_t));
    }

    atb::VariantPack vp;
    vp.inTensors = {
        vis_pos_embed_npu_,
        *idx_npu[0].Get(), *idx_npu[1].Get(), *idx_npu[2].Get(), *idx_npu[3].Get(),
        *wt_npu[0].Get(),  *wt_npu[1].Get(),  *wt_npu[2].Get(),  *wt_npu[3].Get()};
    vp.outTensors = {out_npu};

    Status s = ExecuteGraph(pos_embed_interp_graph_, vp);
    if (s != STATUS_OK) {
        LOG_ERROR("ComputePosEmbedNpu: graph execute failed");
        return s;
    }
    return STATUS_OK;
}

// ═════════════════════════════════════════════════════════════════════
// ComputeVisionRopeNpu — host Stage A + NPU Stage B replacement for the
// CPU `VisionRotaryEmbedding::ComputeRoPE`.
//
// Caller owns @p cos_npu / @p sin_npu; both must be pre-allocated to
// {total_tokens, vis_head_dim} fp16.
//
// Returns the total number of tokens written (= sum over images of T*H*W),
// or -1 on error.
// ═════════════════════════════════════════════════════════════════════

int64_t Qwen3VLModel::ComputeVisionRopeNpu(const int64_t* grid_thw,
                                            int64_t num_images,
                                            atb::Tensor& cos_npu,
                                            atb::Tensor& sin_npu) {
    if (!vis_rope_graph_ || !grid_thw || num_images <= 0) {
        LOG_ERROR("ComputeVisionRopeNpu: invalid args or graph not built");
        return -1;
    }

    int32_t merge_size = static_cast<int32_t>(config_.vis_spatial_merge_size);
    int32_t vis_hd = config_.vis_head_dim();         // 64
    int32_t half = vis_hd / 4;                       // 16  (mirrors C++ mrope.cpp:140)
    int32_t max_hw = components::MaxGridHW(grid_thw, num_images);

    // ── Stage A: host ──
    std::vector<int32_t> row_idx_host, col_idx_host;
    components::BuildVisRopeIndices(grid_thw, num_images, merge_size,
                                     row_idx_host, col_idx_host);
    int64_t n = static_cast<int64_t>(row_idx_host.size());

    std::vector<float> freq_table_f32;
    components::BuildVisRopeFreqTable(max_hw, half, freq_table_f32);

    // Convert freq_table to fp16 (ATB Concat doesn't support fp32)
    std::vector<uint16_t> freq_table_fp16(freq_table_f32.size());
    for (size_t i = 0; i < freq_table_f32.size(); i++) {
        freq_table_fp16[i] = Fp32ToFp16(freq_table_f32[i]);
    }

    // ── Stage B: NPU ──
    auto* alloc = runtime_->GetAllocator();
    NpuTensor ft_npu  = AllocNpuFloat16({max_hw, half});
    NpuTensor row_npu = AllocNpuInt32({n});
    NpuTensor col_npu = AllocNpuInt32({n});
    if (!ft_npu || !row_npu || !col_npu) {
        LOG_ERROR("ComputeVisionRopeNpu: tensor alloc failed");
        return -1;
    }
    alloc->CopyToDevice(*ft_npu.Get(), freq_table_fp16.data(),
                        freq_table_fp16.size() * sizeof(uint16_t));
    alloc->CopyToDevice(*row_npu.Get(), row_idx_host.data(),
                        n * sizeof(int32_t));
    alloc->CopyToDevice(*col_npu.Get(), col_idx_host.data(),
                        n * sizeof(int32_t));

    atb::VariantPack vp;
    vp.inTensors  = {*ft_npu.Get(), *row_npu.Get(), *col_npu.Get()};
    vp.outTensors = {cos_npu, sin_npu};

    Status s = ExecuteGraph(vis_rope_graph_, vp);
    if (s != STATUS_OK) {
        LOG_ERROR("ComputeVisionRopeNpu: graph execute failed");
        return -1;
    }
    return n;
}

}  // namespace adapters
}  // namespace atb_llm
