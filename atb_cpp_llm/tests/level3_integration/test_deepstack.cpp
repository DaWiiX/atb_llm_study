/**
 * Level 3 integration test: DeepstackFusion component.
 *
 * Covers Qwen3VL cross-modal fusion:
 *   - IsDeepstackLayer(): vision-layer -> fusion-index lookup
 *   - GetNumFusionLayers(): config-driven count
 *   - SetMergerWeights(): per-fusion-index weight registration
 *   - InjectFeatures(): additive scatter to text-hidden positions on NPU
 *
 * ExtractFeatures() is intentionally NOT covered here. It requires a
 * fully wired deepstack merger graph with matching weight set; that path
 * is exercised end-to-end via test_e2e.cpp and the per-stage precision
 * suite (test_vision_block_precision.cpp / compare_stages.py).
 *
 * Run: ./test_deepstack
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include "core/tensor_allocator.h"
#include "components/common/deepstack_fusion.h"
#include "components/vision/deepstack_graph.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include <cmath>
#include <cstdint>
#include <vector>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

using atb_llm::components::DeepstackFusion;
using atb_llm::components::DeepstackMergerWeights;

// Build a deepstack merger graph used as a placeholder for the
// DeepstackFusion constructor. Never executed in this file.
static atb_llm::OperationHandle BuildPlaceholderGraph() {
    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::DeepstackGraph::Build(
        "PlaceholderDeepstack", 1280, 2, 1e-6f, op);
    REQUIRE(IS_OK(s));
    REQUIRE(op.get() != nullptr);
    return op;
}

// ══════════════════════════════════════════════════════════
// Test 1: IsDeepstackLayer basic mapping
// ══════════════════════════════════════════════════════════
TEST_CASE("DeepstackFusion::IsDeepstackLayer basic") {
    LOG_INFO("=== Test: DeepstackFusion::IsDeepstackLayer basic ===");

    DeepstackFusion::Config cfg;
    cfg.vis_hidden_size = 1024;
    cfg.vis_out_hidden_size = 2048;
    cfg.spatial_merge_size = 2;
    cfg.deepstack_visual_indexes = {5, 11, 17};

    auto placeholder = BuildPlaceholderGraph();
    DeepstackFusion fusion(cfg, std::move(placeholder));

    CHECK(fusion.GetNumFusionLayers() == 3);

    size_t idx = 999;
    CHECK(fusion.IsDeepstackLayer(5, idx));
    CHECK(idx == 0);

    idx = 999;
    CHECK(fusion.IsDeepstackLayer(11, idx));
    CHECK(idx == 1);

    idx = 999;
    CHECK(fusion.IsDeepstackLayer(17, idx));
    CHECK(idx == 2);

    // Non-deepstack layers must return false. Contract leaves idx
    // unspecified on miss, so we do not assert idx in those cases.
    CHECK_FALSE(fusion.IsDeepstackLayer(0, idx));
    CHECK_FALSE(fusion.IsDeepstackLayer(6, idx));
    CHECK_FALSE(fusion.IsDeepstackLayer(24, idx));
    CHECK_FALSE(fusion.IsDeepstackLayer(-1, idx));
}

// ══════════════════════════════════════════════════════════
// Test 2: IsDeepstackLayer with empty configuration
// ══════════════════════════════════════════════════════════
TEST_CASE("DeepstackFusion::IsDeepstackLayer empty config") {
    LOG_INFO("=== Test: DeepstackFusion::IsDeepstackLayer empty ===");

    DeepstackFusion::Config cfg;
    cfg.vis_hidden_size = 1024;
    cfg.vis_out_hidden_size = 2048;
    cfg.spatial_merge_size = 2;
    cfg.deepstack_visual_indexes = {};

    auto placeholder = BuildPlaceholderGraph();
    DeepstackFusion fusion(cfg, std::move(placeholder));

    CHECK(fusion.GetNumFusionLayers() == 0);

    size_t idx = 0;
    for (int32_t layer = 0; layer < 32; ++layer) {
        CHECK_FALSE(fusion.IsDeepstackLayer(layer, idx));
    }
}

// ══════════════════════════════════════════════════════════
// Test 3: SetMergerWeights + GetNumFusionLayers
// ══════════════════════════════════════════════════════════
TEST_CASE("DeepstackFusion::SetMergerWeights accepts valid indices") {
    LOG_INFO("=== Test: DeepstackFusion::SetMergerWeights ===");

    DeepstackFusion::Config cfg;
    cfg.vis_hidden_size = 1024;
    cfg.vis_out_hidden_size = 2048;
    cfg.spatial_merge_size = 2;
    cfg.deepstack_visual_indexes = {5, 11, 17};

    auto placeholder = BuildPlaceholderGraph();
    DeepstackFusion fusion(cfg, std::move(placeholder));

    CHECK(fusion.GetNumFusionLayers() == 3);

    // Three default-initialized weight bundles, one per deepstack layer.
    // We are not executing the graph here — only validating registration.
    DeepstackMergerWeights w0{};
    DeepstackMergerWeights w1{};
    DeepstackMergerWeights w2{};
    fusion.SetMergerWeights(0, w0);
    fusion.SetMergerWeights(1, w1);
    fusion.SetMergerWeights(2, w2);

    // Out-of-range indices must be silently rejected (logged in impl).
    fusion.SetMergerWeights(99, w0);
    fusion.SetMergerWeights(static_cast<size_t>(-1), w0);

    // Count is fixed by the config, not by weight registration calls.
    CHECK(fusion.GetNumFusionLayers() == 3);
}

// ══════════════════════════════════════════════════════════
// Test 4: InjectFeatures additively updates only target positions (NPU)
// ══════════════════════════════════════════════════════════
TEST_CASE("DeepstackFusion::InjectFeatures position scatter (NPU)") {
    LOG_INFO("=== Test: DeepstackFusion::InjectFeatures position scatter ===");

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();

    // deepstack output dim == text hidden dim (typical Qwen3VL case).
    DeepstackFusion::Config cfg;
    cfg.vis_hidden_size = 16;
    cfg.vis_out_hidden_size = 16;
    cfg.spatial_merge_size = 2;
    cfg.deepstack_visual_indexes = {5};

    auto placeholder = BuildPlaceholderGraph();
    DeepstackFusion fusion(cfg, std::move(placeholder));

    const int64_t seq_len = 10;
    const int64_t hidden_size = 16;
    const int64_t feat_dim = hidden_size;

    // Allocate hidden_states on NPU via RAII NpuTensor.
    atb_llm::NpuTensor hidden = atb_llm::AllocNpuFloat16({seq_len, hidden_size});
    REQUIRE(static_cast<bool>(hidden));

    // Initialize every element to 1.0 (fp16).
    const uint16_t fp16_one = atb_llm::Fp32ToFp16(1.0f);
    std::vector<uint16_t> host_init(seq_len * hidden_size, fp16_one);
    REQUIRE(IS_OK(alloc->CopyToDevice(*hidden.Get(), host_init.data(),
                                      host_init.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(runtime->Synchronize()));

    // Build a flat (ds_tokens, feat_dim) host buffer of 0.5 (fp16) values.
    const std::vector<int64_t> positions = {2, 5, 8};
    const int64_t ds_tokens = static_cast<int64_t>(positions.size());
    const uint16_t fp16_half = atb_llm::Fp32ToFp16(0.5f);
    std::vector<uint16_t> ds_feat(ds_tokens * feat_dim, fp16_half);

    // Run scatter-add. Each row at positions[t] becomes 1.0 + 0.5 = 1.5;
    // every other row stays at 1.0.
    fusion.InjectFeatures(hidden, ds_feat, positions,
                          seq_len, hidden_size, feat_dim, runtime.get());
    REQUIRE(IS_OK(runtime->Synchronize()));

    // Read back and verify per-row.
    std::vector<uint16_t> host_out(seq_len * hidden_size, 0);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out.data(), *hidden.Get(),
                                    host_out.size() * sizeof(uint16_t))));

    auto is_target = [&](int64_t r) {
        for (auto p : positions) if (p == r) return true;
        return false;
    };

    for (int64_t r = 0; r < seq_len; ++r) {
        float expected = is_target(r) ? 1.5f : 1.0f;
        for (int64_t c = 0; c < hidden_size; ++c) {
            float got = atb_llm::Fp16ToF32(host_out[r * hidden_size + c]);
            // fp16 round-trip of {1.0, 0.5, 1.5} is exact; allow tiny slack
            // in case the NPU path introduces denormal-rounding artefacts.
            CHECK(std::fabs(got - expected) < 1e-3f);
        }
    }
}
