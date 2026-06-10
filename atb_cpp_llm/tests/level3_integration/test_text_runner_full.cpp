/**
 * Level 3 Runner integration test: TextRunner full pipeline.
 *
 * Verifies that TextRunner can execute the complete text decoder pipeline
 * end-to-end with small dimensions:
 *   hidden_states
 *     -> decoder layer 0
 *     -> decoder layer 1
 *     -> ... loop num_layers times
 *     -> final_norm
 *     -> hidden_states
 *
 * This is an INTEGRATION TEST — it verifies that:
 *   - Both graphs (layer + norm) Build successfully via EnsureBuilt
 *   - The shared decoder layer graph can be Setup + Executed N times in a loop
 *     with per-layer weights, with output of layer i feeding layer i+1
 *   - The final_norm graph runs after the layer loop
 *   - Output is non-zero and shape-correct
 *   - GQA (different num_kv_heads vs num_heads) works end-to-end
 *
 * Does NOT compare against a Python reference. Plumbing only.
 *
 * Run: ./test_text_runner_full
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "runners/text_runner.h"
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

namespace {

constexpr uint16_t FP16_ONE       = 0x3C00;  // 1.0
constexpr uint16_t FP16_ZERO      = 0x0000;  // 0.0
constexpr uint16_t FP16_QUARTER   = 0x3400;  // 0.25
constexpr uint16_t FP16_VERYSMALL = 0x1C00;  // ~0.004

inline atb_llm::Status fill_fp16(atb_llm::TensorAllocator* alloc,
                                 atb::Tensor& t, uint16_t val) {
    std::vector<uint16_t> data(t.dataSize / sizeof(uint16_t), val);
    return alloc->CopyToDevice(t, data.data(), data.size() * sizeof(uint16_t));
}

inline atb::Status run_graph(atb_llm::OperationHandle& op,
                              atb::VariantPack& vp,
                              atb_llm::IRuntime* runtime) {
    auto* ctx = runtime->GetContext();
    uint64_t ws_size = 0;
    atb::Status s = op.get()->Setup(vp, ws_size, ctx);
    if (s != atb::NO_ERROR) return s;
    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
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

// Per-layer text decoder weight bundle.
struct TextLayerW {
    atb::Tensor q_w, k_w, v_w, o_w;
    atb::Tensor qn_w, kn_w;
    atb::Tensor gate_w, up_w, down_w;
    atb::Tensor iln_w, pln_w;
};

void AllocLayer(atb_llm::TensorAllocator* alloc, TextLayerW& w,
                int32_t hidden, int32_t kv_hidden, int32_t hd,
                int32_t intermediate) {
    alloc->AllocFloat16(w.q_w,    {hidden, hidden});
    alloc->AllocFloat16(w.k_w,    {kv_hidden, hidden});
    alloc->AllocFloat16(w.v_w,    {kv_hidden, hidden});
    alloc->AllocFloat16(w.o_w,    {hidden, hidden});
    alloc->AllocFloat16(w.qn_w,   {hd});
    alloc->AllocFloat16(w.kn_w,   {hd});
    alloc->AllocFloat16(w.gate_w, {intermediate, hidden});
    alloc->AllocFloat16(w.up_w,   {intermediate, hidden});
    alloc->AllocFloat16(w.down_w, {hidden, intermediate});
    alloc->AllocFloat16(w.iln_w,  {hidden});
    alloc->AllocFloat16(w.pln_w,  {hidden});
}

void FillLayer(atb_llm::TensorAllocator* alloc, TextLayerW& w) {
    // Keep projection weights small to bound activation magnitudes across layers.
    fill_fp16(alloc, w.q_w,    FP16_VERYSMALL);
    fill_fp16(alloc, w.k_w,    FP16_VERYSMALL);
    fill_fp16(alloc, w.v_w,    FP16_VERYSMALL);
    fill_fp16(alloc, w.o_w,    FP16_VERYSMALL);
    fill_fp16(alloc, w.qn_w,   FP16_ONE);
    fill_fp16(alloc, w.kn_w,   FP16_ONE);
    fill_fp16(alloc, w.gate_w, FP16_VERYSMALL);
    fill_fp16(alloc, w.up_w,   FP16_VERYSMALL);
    fill_fp16(alloc, w.down_w, FP16_VERYSMALL);
    fill_fp16(alloc, w.iln_w,  FP16_ONE);
    fill_fp16(alloc, w.pln_w,  FP16_ONE);
}

}  // namespace

// ══════════════════════════════════════════════════════════════════════
// Test 1: Multi-layer loop with shared graph
//   - 2 layers, shared decoder layer graph reused
//   - Output of layer 0 feeds layer 1, then final_norm
// ══════════════════════════════════════════════════════════════════════
TEST_CASE("TextRunner Full Pipeline (multi-layer loop)") {
    LOG_INFO("=== Test: TextRunner multi-layer loop ===");

    // ── 1. Build runner ─────────────────────────────────────
    atb_llm::runners::TextRunner::Config cfg;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 32;
    cfg.intermediate_size = 64;
    cfg.num_layers = 2;        // only 2 layers — looped manually
    cfg.epsilon = 1e-6f;

    atb_llm::runners::TextRunner runner(cfg);
    const int32_t seq_len = 4;
    REQUIRE(IS_OK(runner.EnsureBuilt(seq_len)));
    REQUIRE(runner.GetLayerGraph().get() != nullptr);
    REQUIRE(runner.GetNormGraph().get() != nullptr);

    // ── 2. Runtime ──────────────────────────────────────────
    auto runtime = atb_llm::CreateRuntime(0, 5LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();

    const int32_t nh = cfg.num_heads;
    const int32_t kvh = cfg.num_kv_heads;
    const int32_t hd = cfg.head_dim;
    const int32_t hidden = nh * hd;
    const int32_t kv_hidden = kvh * hd;
    const int32_t intermediate = cfg.intermediate_size;

    // ── 3. Allocate hidden_in / per-layer weights / cos/sin/mask ─
    atb::Tensor hidden_in;
    alloc->AllocFloat16(hidden_in, {seq_len, hidden});

    std::vector<TextLayerW> layers(cfg.num_layers);
    for (auto& w : layers)
        AllocLayer(alloc, w, hidden, kv_hidden, hd, intermediate);

    atb::Tensor cos_t, sin_t, mask_t;
    alloc->AllocFloat16(cos_t,  {seq_len, hd});
    alloc->AllocFloat16(sin_t,  {seq_len, hd});
    alloc->AllocFloat16(mask_t, {seq_len, seq_len});

    // seqlen tensor (int32, host)
    atb::Tensor seqlen_t;
    int32_t seqlen_val = seq_len;
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    // ── 4. Fill device buffers ──────────────────────────────
    fill_fp16(alloc, hidden_in, FP16_QUARTER);
    for (auto& w : layers) FillLayer(alloc, w);
    fill_fp16(alloc, cos_t,  FP16_ONE);   // identity rotation
    fill_fp16(alloc, sin_t,  FP16_ZERO);
    fill_fp16(alloc, mask_t, FP16_ZERO);  // no masking

    // ── 5. Loop layers ──────────────────────────────────────
    atb::Tensor h_in = hidden_in;
    atb::Tensor h_out;
    alloc->AllocFloat16(h_out, {seq_len, hidden});

    for (int32_t li = 0; li < cfg.num_layers; li++) {
        const auto& w = layers[li];
        atb::VariantPack vp;
        vp.inTensors = {
            h_in,
            w.q_w, w.k_w, w.v_w, w.o_w,
            w.qn_w, w.kn_w,
            w.gate_w, w.up_w, w.down_w,
            w.iln_w, w.pln_w,
            cos_t, sin_t, mask_t, seqlen_t};
        vp.outTensors = {h_out};

        atb::Status as = run_graph(runner.GetLayerGraph(), vp, runtime.get());
        CHECK(as == atb::NO_ERROR);

        // Verify intermediate output is non-zero
        std::vector<uint16_t> host(seq_len * hidden);
        alloc->CopyToHost(host.data(), h_out, host.size() * sizeof(uint16_t));
        CHECK(any_nonzero_fp16(host));

        // Feed forward: alloc new output tensor for next iter (ATB forbids in-place)
        if (li + 1 < cfg.num_layers) {
            atb::Tensor h_next;
            alloc->AllocFloat16(h_next, {seq_len, hidden});
            h_in = h_out;
            h_out = h_next;
        }
    }

    // ── 6. Final norm ───────────────────────────────────────
    atb::Tensor norm_w, norm_out;
    alloc->AllocFloat16(norm_w,   {hidden});
    alloc->AllocFloat16(norm_out, {seq_len, hidden});
    fill_fp16(alloc, norm_w, FP16_ONE);

    {
        atb::VariantPack nvp;
        nvp.inTensors = {h_out, norm_w};
        nvp.outTensors = {norm_out};

        atb::Status as = run_graph(runner.GetNormGraph(), nvp, runtime.get());
        CHECK(as == atb::NO_ERROR);
    }

    // ── 7. Verify final output ──────────────────────────────
    std::vector<uint16_t> final_host(seq_len * hidden);
    alloc->CopyToHost(final_host.data(), norm_out,
                       final_host.size() * sizeof(uint16_t));
    CHECK(any_nonzero_fp16(final_host));

    LOG_INFO("TextRunner multi-layer loop test done");
}

// ══════════════════════════════════════════════════════════════════════
// Test 2: GQA end-to-end
//   - num_heads=12, num_kv_heads=4 (3:1 grouping)
//   - Verifies the shared layer graph handles GQA correctly through the
//     full multi-layer loop + final norm
// ══════════════════════════════════════════════════════════════════════
TEST_CASE("TextRunner GQA Full Pipeline") {
    LOG_INFO("=== Test: TextRunner GQA full pipeline ===");

    atb_llm::runners::TextRunner::Config cfg;
    cfg.num_heads = 12;
    cfg.num_kv_heads = 4;       // GQA: 3 query heads per KV head
    cfg.head_dim = 32;
    cfg.intermediate_size = 64;
    cfg.num_layers = 2;
    cfg.epsilon = 1e-6f;

    atb_llm::runners::TextRunner runner(cfg);
    const int32_t seq_len = 4;
    REQUIRE(IS_OK(runner.EnsureBuilt(seq_len)));
    REQUIRE(runner.GetLayerGraph().get() != nullptr);

    auto runtime = atb_llm::CreateRuntime(0, 5LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();

    const int32_t nh = cfg.num_heads;
    const int32_t kvh = cfg.num_kv_heads;
    const int32_t hd = cfg.head_dim;
    const int32_t hidden = nh * hd;          // 12 * 32 = 384
    const int32_t kv_hidden = kvh * hd;      // 4  * 32 = 128
    const int32_t intermediate = cfg.intermediate_size;

    // Allocations
    atb::Tensor hidden_in;
    alloc->AllocFloat16(hidden_in, {seq_len, hidden});

    std::vector<TextLayerW> layers(cfg.num_layers);
    for (auto& w : layers)
        AllocLayer(alloc, w, hidden, kv_hidden, hd, intermediate);

    atb::Tensor cos_t, sin_t, mask_t;
    alloc->AllocFloat16(cos_t,  {seq_len, hd});
    alloc->AllocFloat16(sin_t,  {seq_len, hd});
    alloc->AllocFloat16(mask_t, {seq_len, seq_len});

    atb::Tensor seqlen_t;
    int32_t seqlen_val = seq_len;
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    // Fill
    fill_fp16(alloc, hidden_in, FP16_QUARTER);
    for (auto& w : layers) FillLayer(alloc, w);
    fill_fp16(alloc, cos_t,  FP16_ONE);
    fill_fp16(alloc, sin_t,  FP16_ZERO);
    fill_fp16(alloc, mask_t, FP16_ZERO);

    // Loop layers
    atb::Tensor h_in = hidden_in;
    atb::Tensor h_out;
    alloc->AllocFloat16(h_out, {seq_len, hidden});

    for (int32_t li = 0; li < cfg.num_layers; li++) {
        const auto& w = layers[li];
        atb::VariantPack vp;
        vp.inTensors = {
            h_in,
            w.q_w, w.k_w, w.v_w, w.o_w,
            w.qn_w, w.kn_w,
            w.gate_w, w.up_w, w.down_w,
            w.iln_w, w.pln_w,
            cos_t, sin_t, mask_t, seqlen_t};
        vp.outTensors = {h_out};

        CHECK(run_graph(runner.GetLayerGraph(), vp, runtime.get()) == atb::NO_ERROR);

        if (li + 1 < cfg.num_layers) {
            atb::Tensor h_next;
            alloc->AllocFloat16(h_next, {seq_len, hidden});
            h_in = h_out;
            h_out = h_next;
        }
    }

    // Final norm
    atb::Tensor norm_w, norm_out;
    alloc->AllocFloat16(norm_w,   {hidden});
    alloc->AllocFloat16(norm_out, {seq_len, hidden});
    fill_fp16(alloc, norm_w, FP16_ONE);

    {
        atb::VariantPack nvp;
        nvp.inTensors = {h_out, norm_w};
        nvp.outTensors = {norm_out};
        CHECK(run_graph(runner.GetNormGraph(), nvp, runtime.get()) == atb::NO_ERROR);
    }

    // Verify output
    std::vector<uint16_t> final_host(seq_len * hidden);
    alloc->CopyToHost(final_host.data(), norm_out,
                       final_host.size() * sizeof(uint16_t));
    CHECK(any_nonzero_fp16(final_host));

    LOG_INFO("TextRunner GQA full pipeline test done");
}
