/**
 * Phase 3 tests: Text Model runner.
 *
 * Tests:
 *   1. TextModel graph build (decoder layer + final norm)
 *   2. Causal mask generation
 *   3. NPU execution of decoder layer (small dimensions)
 *
 * Run: ./test_text_model
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/graph_builder.h"
#include "core/context_manager.h"
#include "util/cpp11_compat.h"
#include "core/tensor_allocator.h"
#include "runners/text_runner.h"
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

// ══════════════════════════════════════════════════════════
// Test: TextModel build
// ══════════════════════════════════════════════════════════
TEST_CASE("TextModel Build") {
    LOG_INFO("=== Test: TextModel Build ===");

    atb_llm::runners::TextRunner::Config cfg;
    cfg.num_heads = 12;
    cfg.num_kv_heads = 12;
    cfg.head_dim = 64;
    cfg.intermediate_size = 256;
    cfg.num_layers = 28;
    cfg.epsilon = 1e-6f;

    atb_llm::runners::TextRunner model(cfg);
    atb_llm::Status s = model.EnsureBuilt(8);  // seq_len=8
    CHECK(IS_OK(s));

    CHECK(model.GetLayerGraph().get() != nullptr);
    CHECK(model.GetNormGraph().get() != nullptr);

    if (model.GetLayerGraph()) {
        // 16 inputs: hidden + 4 proj + 2 qk_norm + 3 mlp + 2 ln + cos + sin + mask + seqlen
        CHECK(model.GetLayerGraph().get()->GetInputNum() == 16);
    }

    if (model.GetNormGraph()) {
        CHECK(model.GetNormGraph().get()->GetInputNum() == 2);
    }

    LOG_INFO("TextModel Build test done");
}

// ══════════════════════════════════════════════════════════
// Test: Causal mask
// ══════════════════════════════════════════════════════════
TEST_CASE("Causal Mask") {
    LOG_INFO("=== Test: Causal Mask ===");

    int32_t seq_len = 4;
    std::vector<float> mask(seq_len * seq_len);
    atb_llm::runners::MakeCausalMask(seq_len, mask.data());

    // Upper triangle should be -65504, lower triangle + diagonal should be 0
    bool correct = true;
    for (int32_t i = 0; i < seq_len; i++) {
        for (int32_t j = 0; j < seq_len; j++) {
            float expected = (j > i) ? -65504.0f : 0.0f;
            if (std::fabs(mask[i * seq_len + j] - expected) > 0.01f) {
                correct = false;
                LOG_ERROR("Mask mismatch at [%d][%d]: got %f, expected %f",
                          i, j, mask[i * seq_len + j], expected);
            }
        }
    }
    CHECK(correct);

    LOG_INFO("Causal Mask test done");
}

// ══════════════════════════════════════════════════════════
// Test: TextModel NPU execution (small dimensions)
// ══════════════════════════════════════════════════════════
TEST_CASE("TextModel Execute") {
    LOG_INFO("=== Test: TextModel Execute on NPU ===");

    auto runtime = atb_llm::CreateRuntime(0, 5LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    // Small dimensions for testing
    atb_llm::runners::TextRunner::Config cfg;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 32;
    cfg.intermediate_size = 64;
    cfg.num_layers = 2;
    cfg.epsilon = 1e-6f;

    int32_t seq_len = 4;
    int32_t nh = cfg.num_heads;
    int32_t hd = cfg.head_dim;
    int32_t hidden = nh * hd;

    atb_llm::runners::TextRunner model(cfg);
    atb_llm::Status s = model.EnsureBuilt(seq_len);
    CHECK(IS_OK(s));

    REQUIRE(model.GetLayerGraph().get() != nullptr);

    // Allocate inputs for decoder layer
    // Inputs: hidden_states, q_w, k_w, v_w, o_w, qn_w, kn_w,
    //         gate_w, up_w, down_w, input_ln_w, post_ln_w, cos, sin, mask, seqlen
    atb::Tensor hidden_t, q_w, k_w, v_w, o_w, qn_w, kn_w;
    atb::Tensor gate_w, up_w, down_w, iln_w, pln_w;
    atb::Tensor cos_t, sin_t, mask_t, seqlen_t, output_t;

    alloc->AllocFloat16(hidden_t, {seq_len, hidden});
    alloc->AllocFloat16(q_w, {hidden, hidden});
    alloc->AllocFloat16(k_w, {hidden, hidden});
    alloc->AllocFloat16(v_w, {hidden, hidden});
    alloc->AllocFloat16(o_w, {hidden, hidden});
    alloc->AllocFloat16(qn_w, {hd});
    alloc->AllocFloat16(kn_w, {hd});
    alloc->AllocFloat16(gate_w, {cfg.intermediate_size, hidden});
    alloc->AllocFloat16(up_w, {cfg.intermediate_size, hidden});
    alloc->AllocFloat16(down_w, {hidden, cfg.intermediate_size});
    alloc->AllocFloat16(iln_w, {hidden});
    alloc->AllocFloat16(pln_w, {hidden});
    alloc->AllocFloat16(cos_t, {seq_len, hd});
    alloc->AllocFloat16(sin_t, {seq_len, hd});
    alloc->AllocFloat16(mask_t, {seq_len, seq_len});
    alloc->AllocInt64(seqlen_t, {1});  // int32 tensor for seqlen (AllocInt64 allocates enough)
    alloc->AllocFloat16(output_t, {seq_len, hidden});

    // Fill with small values
    auto fill_fp16 = [&](atb::Tensor& t, uint16_t val) {
        std::vector<uint16_t> data(t.dataSize / sizeof(uint16_t), val);
        alloc->CopyToDevice(t, data.data(), data.size() * sizeof(uint16_t));
    };

    fill_fp16(hidden_t, 0x3400);  // 0.25
    fill_fp16(q_w, 0x3C00);       // 1.0
    fill_fp16(k_w, 0x3C00);
    fill_fp16(v_w, 0x3C00);
    fill_fp16(o_w, 0x3C00);
    fill_fp16(qn_w, 0x3C00);
    fill_fp16(kn_w, 0x3C00);
    fill_fp16(gate_w, 0x3C00);
    fill_fp16(up_w, 0x3C00);
    fill_fp16(down_w, 0x3C00);
    fill_fp16(iln_w, 0x3C00);
    fill_fp16(pln_w, 0x3C00);
    fill_fp16(cos_t, 0x3C00);   // cos ~1.0
    fill_fp16(sin_t, 0x0000);   // sin ~0.0
    fill_fp16(mask_t, 0x0000);  // all attend

    // seqlen: int32 value = seq_len * batch_size
    // Use hostData for the seqlen tensor (small, single int32)
    int32_t seqlen_val = seq_len;  // batch=1
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    atb::VariantPack vp;
    vp.inTensors = {
        hidden_t, q_w, k_w, v_w, o_w, qn_w, kn_w,
        gate_w, up_w, down_w, iln_w, pln_w,
        cos_t, sin_t, mask_t, seqlen_t};
    vp.outTensors = {output_t};

    uint64_t ws_size = 0;
    atb::Status atb_s = model.GetLayerGraph().get()->Setup(vp, ws_size, ctx);
    CHECK(atb_s == atb::NO_ERROR);

    if (atb_s == atb::NO_ERROR) {
        uint8_t* ws_ptr = nullptr;
        if (ws_size > 0) {
            auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
            ws_ptr = ws;
        }
        atb_s = model.GetLayerGraph().get()->Execute(vp, ws_ptr, ws_size, ctx);
        CHECK(atb_s == atb::NO_ERROR);
    }

    runtime->Synchronize();

    // Verify output is non-zero
    std::vector<uint16_t> host_out(seq_len * hidden);
    alloc->CopyToHost(host_out.data(), output_t, host_out.size() * sizeof(uint16_t));
    bool non_zero = false;
    for (auto v : host_out) {
        if (v != 0) {
            non_zero = true;
            break;
        }
    }
    CHECK(non_zero);

    // Test FinalNorm
    {
        atb::Tensor norm_input, norm_weight, norm_output;
        alloc->AllocFloat16(norm_input, {seq_len, hidden});
        alloc->AllocFloat16(norm_weight, {hidden});
        alloc->AllocFloat16(norm_output, {seq_len, hidden});

        fill_fp16(norm_input, 0x3C00);
        fill_fp16(norm_weight, 0x3C00);

        atb::VariantPack norm_vp;
        norm_vp.inTensors = {norm_input, norm_weight};
        norm_vp.outTensors = {norm_output};

        ws_size = 0;
        atb_s = model.GetNormGraph().get()->Setup(norm_vp, ws_size, ctx);
        CHECK(atb_s == atb::NO_ERROR);

        if (atb_s == atb::NO_ERROR) {
            uint8_t* ws_ptr = nullptr;
            if (ws_size > 0) {
                auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
                ws_ptr = ws;
            }
            atb_s = model.GetNormGraph().get()->Execute(norm_vp, ws_ptr, ws_size, ctx);
            CHECK(atb_s == atb::NO_ERROR);
        }

        runtime->Synchronize();

        std::vector<uint16_t> norm_out(seq_len * hidden);
        alloc->CopyToHost(norm_out.data(), norm_output, norm_out.size() * sizeof(uint16_t));
        non_zero = false;
        for (auto v : norm_out) {
            if (v != 0) {
                non_zero = true;
                break;
            }
        }
        CHECK(non_zero);
    }

    LOG_INFO("TextModel Execute test done");
}

// ══════════════════════════════════════════════════════════
// Test: TextModel with GQA
//
// SKIP on 310P: SelfAttention GQA (kv_head_num < head_num) is not
// supported on 310P hardware.  Production inference uses GQA→MHA weight
// expansion in Qwen3VLModel::Load() instead.
// ══════════════════════════════════════════════════════════
TEST_CASE("TextModel GQA") {
    if (atb_llm::Is310P()) {
        MESSAGE("Skipping TextModel GQA test on 310P (GQA→MHA expansion handles this at engine layer)");
        return;
    }
    LOG_INFO("=== Test: TextModel GQA ===");

    atb_llm::runners::TextRunner::Config cfg;
    cfg.num_heads = 12;
    cfg.num_kv_heads = 2;  // GQA: 12 query heads, 2 KV heads
    cfg.head_dim = 64;
    cfg.intermediate_size = 256;
    cfg.num_layers = 28;

    atb_llm::runners::TextRunner model(cfg);
    atb_llm::Status s = model.EnsureBuilt(8);
    CHECK(IS_OK(s));

    if (model.GetLayerGraph()) {
        CHECK(model.GetLayerGraph().get()->GetInputNum() == 16);
    }

    LOG_INFO("TextModel GQA test done");
}
