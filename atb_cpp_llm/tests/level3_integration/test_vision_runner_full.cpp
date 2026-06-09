/**
 * Level 3 Runner integration test: VisionRunner full pipeline.
 *
 * Verifies that VisionRunner can execute the complete vision encoder
 * pipeline end-to-end with small dimensions:
 *   pixel_values
 *     -> first_layer (patch_embed + pos_embed + block 0)
 *     -> loop block 1 .. depth-1
 *     -> merger
 *     -> vision embeddings
 *
 * This is an INTEGRATION TEST — it verifies that:
 *   - All 4 graphs (first_layer, block, merger, deepstack) Build successfully
 *   - Each graph can Setup + Execute on NPU without ATB errors
 *   - Multi-block loop works (output of block i feeds block i+1)
 *   - Deepstack feature extraction at configured layer indices works
 *   - Output tensors are non-zero and have the expected shape
 *
 * It does NOT compare against a Python reference — that is the responsibility
 * of test_accuracy.cpp / test_vision_stages.cpp. Here we only verify pipeline
 * plumbing.
 *
 * Run: ./test_vision_runner_full
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "runners/vision_runner.h"
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

namespace {

// fp16 constants
constexpr uint16_t FP16_ONE       = 0x3C00;  // 1.0
constexpr uint16_t FP16_ZERO      = 0x0000;  // 0.0
constexpr uint16_t FP16_QUARTER   = 0x3400;  // 0.25
constexpr uint16_t FP16_SMALL     = 0x2C00;  // ~0.0625
constexpr uint16_t FP16_VERYSMALL = 0x1C00;  // ~0.004 (for weights to keep values bounded)

inline atb_llm::Status fill_fp16(atb_llm::TensorAllocator* alloc,
                                 atb::Tensor& t, uint16_t val) {
    std::vector<uint16_t> data(t.dataSize / sizeof(uint16_t), val);
    return alloc->CopyToDevice(t, data.data(), data.size() * sizeof(uint16_t));
}

// Run a graph: setup + (workspace) + execute + sync.
inline atb::Status run_graph(atb_llm::OperationHandle& op,
                              atb::VariantPack& vp,
                              atb_llm::IRuntime* runtime) {
    auto* ctx = runtime->GetContext();
    uint64_t ws_size = 0;
    atb::Status s = op.get()->Setup(vp, ws_size, ctx);
    if (s != atb::NO_ERROR) return s;
    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
        ws_ptr = ws;
    }
    s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
    if (s != atb::NO_ERROR) return s;
    runtime->Synchronize();
    return atb::NO_ERROR;
}

inline bool any_nonzero_fp16(const std::vector<uint16_t>& v) {
    for (auto x : v) if (x != 0) return true;
    return false;
}

}  // namespace

// ══════════════════════════════════════════════════════════════════════
// Test 1: Small-dimension full vision pipeline
//   - depth=2 (block 0 inside first_layer + block 1 in loop)
//   - merger collapses spatial_merge^2 patches into 1 token
//   - Verifies first_layer -> block loop -> merger plumbing works
// ══════════════════════════════════════════════════════════════════════
TEST_CASE("VisionRunner Full Pipeline (small)") {
    LOG_INFO("=== Test: VisionRunner full pipeline ===");

    // ── 1. Build runner ─────────────────────────────────────
    atb_llm::runners::VisionRunner::Config cfg;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.intermediate_size = 128;
    cfg.depth = 2;                  // block 0 in first_layer + 1 looped block
    cfg.in_channels = 3;
    cfg.temporal_patch_size = 2;
    cfg.patch_size = 14;
    cfg.spatial_merge_size = 2;
    cfg.num_position_embeddings = 2304;
    cfg.deepstack_visual_indexes = {1};   // extract from block 1 (last block in loop)
    cfg.epsilon = 1e-6f;

    atb_llm::runners::VisionRunner runner(cfg);
    atb_llm::Status s = runner.Build();
    REQUIRE(IS_OK(s));
    REQUIRE(runner.GetFirstLayerGraph().get() != nullptr);
    REQUIRE(runner.GetBlockGraph().get() != nullptr);
    REQUIRE(runner.GetMergerGraph().get() != nullptr);
    REQUIRE(runner.GetDeepstackGraph().get() != nullptr);

    // ── 2. Runtime ──────────────────────────────────────────
    auto runtime = atb_llm::CreateRuntime(0, 5LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();

    // ── 3. Dimensions ───────────────────────────────────────
    // Pick num_patches = merge^2 * K so merger produces K output tokens.
    const int32_t nh = cfg.num_heads;
    const int32_t hd = cfg.hidden_size / nh;            // 16
    const int32_t hs = cfg.hidden_size;                 // 64
    const int32_t merge = cfg.spatial_merge_size;       // 2
    const int64_t merge_sq = static_cast<int64_t>(merge) * merge;  // 4
    const int64_t merged_tokens = 2;                    // arbitrary
    const int64_t num_patches = merged_tokens * merge_sq;  // 8

    const int64_t kernel = static_cast<int64_t>(cfg.in_channels) *
                            cfg.temporal_patch_size *
                            cfg.patch_size * cfg.patch_size;  // 3*2*14*14 = 1176
    const int64_t flat_elements = num_patches * kernel;

    // Merger fc2 output dim — we control via weight shape.
    const int64_t merged_out_dim = 32;

    // ── 4. Allocate vision-side inputs ───────────────────────
    atb::Tensor pixels, pe_w, pe_b, pos, cos_t, sin_t;
    alloc->AllocFloat16(pixels, {flat_elements});
    alloc->AllocFloat16(pe_w, {hs, kernel});
    alloc->AllocFloat16(pe_b, {hs});
    alloc->AllocFloat16(pos, {num_patches, hs});
    alloc->AllocFloat16(cos_t, {num_patches, hd});
    alloc->AllocFloat16(sin_t, {num_patches, hd});

    // seqlen int32 (hostData)
    atb::Tensor seqlen_t;
    int32_t seqlen_val = static_cast<int32_t>(num_patches);
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    // ── 5. Allocate per-block weights for 2 blocks ──────────
    struct VisBlockW {
        atb::Tensor qkv_w, qkv_b, proj_w, proj_b;
        atb::Tensor fc1_w, fc1_b, fc2_w, fc2_b;
        atb::Tensor n1_w, n1_b, n2_w, n2_b;
    };
    std::vector<VisBlockW> blocks(cfg.depth);
    for (int32_t i = 0; i < cfg.depth; i++) {
        auto& b = blocks[i];
        alloc->AllocFloat16(b.qkv_w,  {3 * hs, hs});
        alloc->AllocFloat16(b.qkv_b,  {3 * hs});
        alloc->AllocFloat16(b.proj_w, {hs, hs});
        alloc->AllocFloat16(b.proj_b, {hs});
        alloc->AllocFloat16(b.fc1_w,  {cfg.intermediate_size, hs});
        alloc->AllocFloat16(b.fc1_b,  {cfg.intermediate_size});
        alloc->AllocFloat16(b.fc2_w,  {hs, cfg.intermediate_size});
        alloc->AllocFloat16(b.fc2_b,  {hs});
        alloc->AllocFloat16(b.n1_w,   {hs});
        alloc->AllocFloat16(b.n1_b,   {hs});
        alloc->AllocFloat16(b.n2_w,   {hs});
        alloc->AllocFloat16(b.n2_b,   {hs});
    }

    // ── 6. Allocate merger weights ───────────────────────────
    const int32_t mer_in = hs * merge * merge;          // 256
    atb::Tensor mn_w, mn_b, mf1_w, mf1_b, mf2_w, mf2_b;
    alloc->AllocFloat16(mn_w,  {hs});
    alloc->AllocFloat16(mn_b,  {hs});
    alloc->AllocFloat16(mf1_w, {mer_in, mer_in});       // fc1: in=mer_in, out=mer_in
    alloc->AllocFloat16(mf1_b, {mer_in});
    alloc->AllocFloat16(mf2_w, {merged_out_dim, mer_in});  // fc2: in=mer_in, out=out_dim
    alloc->AllocFloat16(mf2_b, {merged_out_dim});

    // Deepstack merger weights: norm is applied AFTER reshape to (N/m^2, mer_in),
    // so n_w/n_b are sized {mer_in} rather than {hs}.
    atb::Tensor dn_w, dn_b, df1_w, df1_b, df2_w, df2_b;
    alloc->AllocFloat16(dn_w,  {mer_in});
    alloc->AllocFloat16(dn_b,  {mer_in});
    alloc->AllocFloat16(df1_w, {mer_in, mer_in});
    alloc->AllocFloat16(df1_b, {mer_in});
    alloc->AllocFloat16(df2_w, {merged_out_dim, mer_in});
    alloc->AllocFloat16(df2_b, {merged_out_dim});

    // ── 7. Fill device buffers ──────────────────────────────
    // Inputs: small magnitude so we don't blow up after attention/MLP cascades.
    fill_fp16(alloc, pixels, FP16_SMALL);
    fill_fp16(alloc, pe_w,   FP16_VERYSMALL);
    fill_fp16(alloc, pe_b,   FP16_ZERO);
    fill_fp16(alloc, pos,    FP16_SMALL);
    fill_fp16(alloc, cos_t,  FP16_ONE);   // cos≈1, sin≈0 -> identity rotation
    fill_fp16(alloc, sin_t,  FP16_ZERO);

    for (auto& b : blocks) {
        fill_fp16(alloc, b.qkv_w,  FP16_VERYSMALL);
        fill_fp16(alloc, b.qkv_b,  FP16_ZERO);
        fill_fp16(alloc, b.proj_w, FP16_VERYSMALL);
        fill_fp16(alloc, b.proj_b, FP16_ZERO);
        fill_fp16(alloc, b.fc1_w,  FP16_VERYSMALL);
        fill_fp16(alloc, b.fc1_b,  FP16_ZERO);
        fill_fp16(alloc, b.fc2_w,  FP16_VERYSMALL);
        fill_fp16(alloc, b.fc2_b,  FP16_ZERO);
        fill_fp16(alloc, b.n1_w,   FP16_ONE);
        fill_fp16(alloc, b.n1_b,   FP16_ZERO);
        fill_fp16(alloc, b.n2_w,   FP16_ONE);
        fill_fp16(alloc, b.n2_b,   FP16_ZERO);
    }

    fill_fp16(alloc, mn_w,  FP16_ONE);
    fill_fp16(alloc, mn_b,  FP16_ZERO);
    fill_fp16(alloc, mf1_w, FP16_VERYSMALL);
    fill_fp16(alloc, mf1_b, FP16_ZERO);
    fill_fp16(alloc, mf2_w, FP16_VERYSMALL);
    fill_fp16(alloc, mf2_b, FP16_SMALL);   // non-zero bias so merger output is non-zero

    fill_fp16(alloc, dn_w,  FP16_ONE);
    fill_fp16(alloc, dn_b,  FP16_ZERO);
    fill_fp16(alloc, df1_w, FP16_VERYSMALL);
    fill_fp16(alloc, df1_b, FP16_ZERO);
    fill_fp16(alloc, df2_w, FP16_VERYSMALL);
    fill_fp16(alloc, df2_b, FP16_SMALL);   // non-zero bias so deepstack output is non-zero

    // ── 8. Run FirstLayer (patch_embed + pos + block 0) ─────
    atb::Tensor h0;
    alloc->AllocFloat16(h0, {num_patches, hs});

    {
        const auto& b = blocks[0];
        atb::VariantPack vp;
        vp.inTensors = {
            pixels, pe_w, pe_b, pos, cos_t, sin_t, seqlen_t,
            b.qkv_w, b.qkv_b, b.proj_w, b.proj_b,
            b.fc1_w, b.fc1_b, b.fc2_w, b.fc2_b,
            b.n1_w, b.n1_b, b.n2_w, b.n2_b};
        vp.outTensors = {h0};

        atb::Status as = run_graph(runner.GetFirstLayerGraph(), vp, runtime.get());
        CHECK(as == atb::NO_ERROR);
    }

    // Verify first_layer output is non-zero
    {
        std::vector<uint16_t> host(num_patches * hs);
        alloc->CopyToHost(host.data(), h0, host.size() * sizeof(uint16_t));
        CHECK(any_nonzero_fp16(host));
    }

    // ── 9. Loop blocks 1..depth-1, collect deepstack at index 1 ─
    std::vector<std::vector<uint16_t>> ds_features(cfg.deepstack_visual_indexes.size());
    atb::Tensor h_in = h0;
    atb::Tensor h_out;
    alloc->AllocFloat16(h_out, {num_patches, hs});

    for (int32_t li = 1; li < cfg.depth; li++) {
        const auto& b = blocks[li];
        atb::VariantPack vp;
        vp.inTensors = {
            h_in,
            b.qkv_w, b.qkv_b, b.proj_w, b.proj_b,
            b.fc1_w, b.fc1_b, b.fc2_w, b.fc2_b,
            b.n1_w, b.n1_b, b.n2_w, b.n2_b,
            cos_t, sin_t, seqlen_t};
        vp.outTensors = {h_out};

        atb::Status as = run_graph(runner.GetBlockGraph(), vp, runtime.get());
        CHECK(as == atb::NO_ERROR);

        // Check this block produces deepstack features
        for (size_t di = 0; di < cfg.deepstack_visual_indexes.size(); di++) {
            if (cfg.deepstack_visual_indexes[di] == li) {
                // Run deepstack merger on h_out
                atb::Tensor ds_out;
                const int64_t ds_tokens = num_patches / merge_sq;
                alloc->AllocFloat16(ds_out, {ds_tokens, merged_out_dim});

                atb::VariantPack ds_vp;
                ds_vp.inTensors = {h_out, dn_w, dn_b, df1_w, df1_b, df2_w, df2_b};
                ds_vp.outTensors = {ds_out};

                atb::Status ds_s = run_graph(
                    runner.GetDeepstackGraph(), ds_vp, runtime.get());
                CHECK(ds_s == atb::NO_ERROR);

                ds_features[di].resize(ds_tokens * merged_out_dim);
                alloc->CopyToHost(ds_features[di].data(), ds_out,
                                   ds_features[di].size() * sizeof(uint16_t));
                CHECK(any_nonzero_fp16(ds_features[di]));
            }
        }

        // Next iteration: h_out becomes input. For depth=2, this loop runs once
        // and we use h_out directly for the merger below.
        h_in = h_out;
    }

    // ── 10. Run main Merger ──────────────────────────────────
    atb::Tensor merged_out;
    alloc->AllocFloat16(merged_out, {merged_tokens, merged_out_dim});

    {
        atb::VariantPack vp;
        vp.inTensors = {h_in, mn_w, mn_b, mf1_w, mf1_b, mf2_w, mf2_b};
        vp.outTensors = {merged_out};

        atb::Status as = run_graph(runner.GetMergerGraph(), vp, runtime.get());
        CHECK(as == atb::NO_ERROR);
    }

    // ── 11. Verify final merger output ───────────────────────
    {
        std::vector<uint16_t> host(merged_tokens * merged_out_dim);
        alloc->CopyToHost(host.data(), merged_out, host.size() * sizeof(uint16_t));
        CHECK(any_nonzero_fp16(host));
    }

    // ── 12. Verify deepstack was extracted ───────────────────
    CHECK(ds_features.size() == 1u);
    CHECK(!ds_features[0].empty());

    LOG_INFO("VisionRunner full pipeline test done");
}

// ══════════════════════════════════════════════════════════════════════
// Test 2: Multiple deepstack extraction layers
//   - depth=3, deepstack_visual_indexes={1, 2}
//   - Verifies both deepstack hooks fire and produce non-zero features
// ══════════════════════════════════════════════════════════════════════
TEST_CASE("VisionRunner Multi-Deepstack") {
    LOG_INFO("=== Test: VisionRunner multi-deepstack ===");

    atb_llm::runners::VisionRunner::Config cfg;
    cfg.hidden_size = 64;
    cfg.num_heads = 4;
    cfg.intermediate_size = 128;
    cfg.depth = 3;                          // block 0 + blocks 1, 2
    cfg.in_channels = 3;
    cfg.temporal_patch_size = 2;
    cfg.patch_size = 14;
    cfg.spatial_merge_size = 2;
    cfg.deepstack_visual_indexes = {1, 2};  // extract at both looped blocks
    cfg.epsilon = 1e-6f;

    atb_llm::runners::VisionRunner runner(cfg);
    REQUIRE(IS_OK(runner.Build()));

    auto runtime = atb_llm::CreateRuntime(0, 5LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();

    const int32_t nh = cfg.num_heads;
    const int32_t hd = cfg.hidden_size / nh;
    const int32_t hs = cfg.hidden_size;
    const int32_t merge = cfg.spatial_merge_size;
    const int64_t merge_sq = static_cast<int64_t>(merge) * merge;
    const int64_t merged_tokens = 1;
    const int64_t num_patches = merged_tokens * merge_sq;
    const int64_t kernel = static_cast<int64_t>(cfg.in_channels) *
                            cfg.temporal_patch_size *
                            cfg.patch_size * cfg.patch_size;
    const int64_t merged_out_dim = 32;
    const int32_t mer_in = hs * merge * merge;

    // ── Allocate & fill (compact) ────────────────────────────
    atb::Tensor pixels, pe_w, pe_b, pos, cos_t, sin_t;
    alloc->AllocFloat16(pixels, {num_patches * kernel});
    alloc->AllocFloat16(pe_w,   {hs, kernel});
    alloc->AllocFloat16(pe_b,   {hs});
    alloc->AllocFloat16(pos,    {num_patches, hs});
    alloc->AllocFloat16(cos_t,  {num_patches, hd});
    alloc->AllocFloat16(sin_t,  {num_patches, hd});

    atb::Tensor seqlen_t;
    int32_t seqlen_val = static_cast<int32_t>(num_patches);
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    struct VisBlockW {
        atb::Tensor qkv_w, qkv_b, proj_w, proj_b;
        atb::Tensor fc1_w, fc1_b, fc2_w, fc2_b;
        atb::Tensor n1_w, n1_b, n2_w, n2_b;
    };
    std::vector<VisBlockW> blocks(cfg.depth);
    for (int32_t i = 0; i < cfg.depth; i++) {
        auto& b = blocks[i];
        alloc->AllocFloat16(b.qkv_w,  {3 * hs, hs});
        alloc->AllocFloat16(b.qkv_b,  {3 * hs});
        alloc->AllocFloat16(b.proj_w, {hs, hs});
        alloc->AllocFloat16(b.proj_b, {hs});
        alloc->AllocFloat16(b.fc1_w,  {cfg.intermediate_size, hs});
        alloc->AllocFloat16(b.fc1_b,  {cfg.intermediate_size});
        alloc->AllocFloat16(b.fc2_w,  {hs, cfg.intermediate_size});
        alloc->AllocFloat16(b.fc2_b,  {hs});
        alloc->AllocFloat16(b.n1_w,   {hs});
        alloc->AllocFloat16(b.n1_b,   {hs});
        alloc->AllocFloat16(b.n2_w,   {hs});
        alloc->AllocFloat16(b.n2_b,   {hs});
    }

    atb::Tensor mn_w, mn_b, mf1_w, mf1_b, mf2_w, mf2_b;
    alloc->AllocFloat16(mn_w,  {hs});
    alloc->AllocFloat16(mn_b,  {hs});
    alloc->AllocFloat16(mf1_w, {mer_in, mer_in});
    alloc->AllocFloat16(mf1_b, {mer_in});
    alloc->AllocFloat16(mf2_w, {merged_out_dim, mer_in});
    alloc->AllocFloat16(mf2_b, {merged_out_dim});

    // Deepstack merger weights: norm runs after reshape to (N/m^2, mer_in).
    atb::Tensor dn_w, dn_b, df1_w, df1_b, df2_w, df2_b;
    alloc->AllocFloat16(dn_w,  {mer_in});
    alloc->AllocFloat16(dn_b,  {mer_in});
    alloc->AllocFloat16(df1_w, {mer_in, mer_in});
    alloc->AllocFloat16(df1_b, {mer_in});
    alloc->AllocFloat16(df2_w, {merged_out_dim, mer_in});
    alloc->AllocFloat16(df2_b, {merged_out_dim});

    fill_fp16(alloc, pixels, FP16_SMALL);
    fill_fp16(alloc, pe_w,   FP16_VERYSMALL);
    fill_fp16(alloc, pe_b,   FP16_ZERO);
    fill_fp16(alloc, pos,    FP16_SMALL);
    fill_fp16(alloc, cos_t,  FP16_ONE);
    fill_fp16(alloc, sin_t,  FP16_ZERO);

    for (auto& b : blocks) {
        fill_fp16(alloc, b.qkv_w,  FP16_VERYSMALL);
        fill_fp16(alloc, b.qkv_b,  FP16_ZERO);
        fill_fp16(alloc, b.proj_w, FP16_VERYSMALL);
        fill_fp16(alloc, b.proj_b, FP16_ZERO);
        fill_fp16(alloc, b.fc1_w,  FP16_VERYSMALL);
        fill_fp16(alloc, b.fc1_b,  FP16_ZERO);
        fill_fp16(alloc, b.fc2_w,  FP16_VERYSMALL);
        fill_fp16(alloc, b.fc2_b,  FP16_ZERO);
        fill_fp16(alloc, b.n1_w,   FP16_ONE);
        fill_fp16(alloc, b.n1_b,   FP16_ZERO);
        fill_fp16(alloc, b.n2_w,   FP16_ONE);
        fill_fp16(alloc, b.n2_b,   FP16_ZERO);
    }
    fill_fp16(alloc, mn_w,  FP16_ONE);
    fill_fp16(alloc, mn_b,  FP16_ZERO);
    fill_fp16(alloc, mf1_w, FP16_VERYSMALL);
    fill_fp16(alloc, mf1_b, FP16_ZERO);
    fill_fp16(alloc, mf2_w, FP16_VERYSMALL);
    fill_fp16(alloc, mf2_b, FP16_SMALL);

    fill_fp16(alloc, dn_w,  FP16_ONE);
    fill_fp16(alloc, dn_b,  FP16_ZERO);
    fill_fp16(alloc, df1_w, FP16_VERYSMALL);
    fill_fp16(alloc, df1_b, FP16_ZERO);
    fill_fp16(alloc, df2_w, FP16_VERYSMALL);
    fill_fp16(alloc, df2_b, FP16_SMALL);

    // ── FirstLayer ──────────────────────────────────────────
    atb::Tensor h_state;
    alloc->AllocFloat16(h_state, {num_patches, hs});
    {
        const auto& b = blocks[0];
        atb::VariantPack vp;
        vp.inTensors = {
            pixels, pe_w, pe_b, pos, cos_t, sin_t, seqlen_t,
            b.qkv_w, b.qkv_b, b.proj_w, b.proj_b,
            b.fc1_w, b.fc1_b, b.fc2_w, b.fc2_b,
            b.n1_w, b.n1_b, b.n2_w, b.n2_b};
        vp.outTensors = {h_state};
        CHECK(run_graph(runner.GetFirstLayerGraph(), vp, runtime.get()) == atb::NO_ERROR);
    }

    // ── Block loop with deepstack extraction ─────────────────
    std::vector<std::vector<uint16_t>> ds_features(cfg.deepstack_visual_indexes.size());

    for (int32_t li = 1; li < cfg.depth; li++) {
        atb::Tensor h_next;
        alloc->AllocFloat16(h_next, {num_patches, hs});

        const auto& b = blocks[li];
        atb::VariantPack vp;
        vp.inTensors = {
            h_state,
            b.qkv_w, b.qkv_b, b.proj_w, b.proj_b,
            b.fc1_w, b.fc1_b, b.fc2_w, b.fc2_b,
            b.n1_w, b.n1_b, b.n2_w, b.n2_b,
            cos_t, sin_t, seqlen_t};
        vp.outTensors = {h_next};
        CHECK(run_graph(runner.GetBlockGraph(), vp, runtime.get()) == atb::NO_ERROR);

        // Deepstack extraction at this layer
        for (size_t di = 0; di < cfg.deepstack_visual_indexes.size(); di++) {
            if (cfg.deepstack_visual_indexes[di] == li) {
                atb::Tensor ds_out;
                const int64_t ds_tokens = num_patches / merge_sq;
                alloc->AllocFloat16(ds_out, {ds_tokens, merged_out_dim});

                atb::VariantPack ds_vp;
                ds_vp.inTensors = {h_next, dn_w, dn_b, df1_w, df1_b, df2_w, df2_b};
                ds_vp.outTensors = {ds_out};
                CHECK(run_graph(runner.GetDeepstackGraph(), ds_vp, runtime.get())
                      == atb::NO_ERROR);

                ds_features[di].resize(ds_tokens * merged_out_dim);
                alloc->CopyToHost(ds_features[di].data(), ds_out,
                                   ds_features[di].size() * sizeof(uint16_t));
            }
        }

        h_state = h_next;
    }

    // Both deepstack layers should have produced non-zero features
    CHECK(ds_features.size() == 2u);
    CHECK(!ds_features[0].empty());
    CHECK(!ds_features[1].empty());
    CHECK(any_nonzero_fp16(ds_features[0]));
    CHECK(any_nonzero_fp16(ds_features[1]));

    LOG_INFO("VisionRunner multi-deepstack test done");
}
