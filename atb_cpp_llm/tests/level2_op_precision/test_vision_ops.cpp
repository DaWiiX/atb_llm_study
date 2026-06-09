/**
 * Phase 3 tests: Vision Ops, Vision Components, and Vision Model.
 *
 * Tests:
 *   1. New Op creation (LayerNormOp, SetValueOp, GatherOp, ReduceOp)
 *   2. Vision component graph builds (VisionAttention, VisionMLP, VisionBlock,
 *      PatchEmbed, VisionMerger, Deepstack)
 *   3. Vision Model graph builds
 *   4. NPU execution tests for vision components
 *
 * Run: ./test_vision_ops
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/graph_builder.h"
#include "core/context_manager.h"
#include "core/tensor_allocator.h"

// New Ops
#include "ops/layer_norm_op.h"
#include "ops/set_value_op.h"
#include "ops/gather_op.h"
#include "ops/reduce_op.h"

// Existing Ops
#include "ops/linear_op.h"
#include "ops/activation_op.h"
#include "ops/elewise_op.h"
#include "ops/split_op.h"
#include "ops/rope_op.h"
#include "ops/self_attention_op.h"

// Vision Components
#include "components/vision/vision_attention_graph.h"
#include "components/vision/vision_mlp_graph.h"
#include "components/vision/vision_block_graph.h"
#include "components/vision/patch_embed_graph.h"
#include "components/vision/vision_merger_graph.h"
#include "components/vision/deepstack_graph.h"

// Vision Model
#include "runners/vision_runner.h"

// Position
#include "components/common/mrope.h"

// Runtime
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

// ══════════════════════════════════════════════════════════
// Test: New Op creation
// ══════════════════════════════════════════════════════════
TEST_CASE("New Op Creation") {
    LOG_INFO("=== Test: New Op Creation (Phase 3) ===");

    // LayerNorm
    {
        auto op = atb_llm::ops::LayerNormOp::Create();
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::LayerNormOp::Create(1e-5f, -1, -1);
        CHECK(op.get() != nullptr);
    }

    // SetValue
    {
        auto op = atb_llm::ops::SetValueOp::Create({0, 0}, {1, 10});
        CHECK(op.get() != nullptr);
    }

    // Gather
    {
        auto op = atb_llm::ops::GatherOp::Create(0, 0);
        CHECK(op.get() != nullptr);
    }

    // Reduce
    {
        auto op = atb_llm::ops::ReduceOp::Create(
            atb_llm::ops::ReduceOp::ReduceType::MAX, {0});
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::ReduceOp::Create(
            atb_llm::ops::ReduceOp::ReduceType::SUM, {1});
        CHECK(op.get() != nullptr);
    }

    LOG_INFO("New op creation tests done");
}

// ══════════════════════════════════════════════════════════
// Test: Vision Component graph builds
// ══════════════════════════════════════════════════════════
TEST_CASE("VisionAttentionGraph") {
    LOG_INFO("=== Test: VisionAttentionGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::VisionAttentionGraph::Build(
        "TestVisAttn", 16, 80, op);
    CHECK(IS_OK(s));
    CHECK(op.get() != nullptr);

    if (op) {
        // 8 inputs: hidden, qkv_w, qkv_b, proj_w, proj_b, c, s, seq
        CHECK(op.get()->GetInputNum() == 8);
        CHECK(op.get()->GetOutputNum() == 1);
    }

    LOG_INFO("VisionAttentionGraph test done");
}

TEST_CASE("VisionMlpGraph") {
    LOG_INFO("=== Test: VisionMlpGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::VisionMlpGraph::Build("TestVisMLP", op);
    CHECK(IS_OK(s));
    CHECK(op.get() != nullptr);

    if (op) {
        // 5 inputs: hidden, fc1_w, fc1_b, fc2_w, fc2_b
        CHECK(op.get()->GetInputNum() == 5);
        CHECK(op.get()->GetOutputNum() == 1);
    }

    LOG_INFO("VisionMlpGraph test done");
}

TEST_CASE("VisionBlockGraph") {
    LOG_INFO("=== Test: VisionBlockGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::VisionBlockGraph::Build(
        "TestVisBlock", 16, 80, 1e-6f, op);
    CHECK(IS_OK(s));
    CHECK(op.get() != nullptr);

    if (op) {
        // 16 inputs: hidden + 4 attn + 4 mlp + 4 norm + cos + sin + seqlen
        CHECK(op.get()->GetInputNum() == 16);
        CHECK(op.get()->GetOutputNum() == 1);
    }

    LOG_INFO("VisionBlockGraph test done");
}

TEST_CASE("PatchEmbedGraph") {
    LOG_INFO("=== Test: PatchEmbedGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::PatchEmbedGraph::Build(
        "TestPatchEmbed", 3, 2, 14, 1280, op);
    CHECK(IS_OK(s));
    CHECK(op.get() != nullptr);

    if (op) {
        // 3 inputs: pixels, w, b
        CHECK(op.get()->GetInputNum() == 3);
        CHECK(op.get()->GetOutputNum() == 1);
    }

    LOG_INFO("PatchEmbedGraph test done");
}

TEST_CASE("VisionMergerGraph") {
    LOG_INFO("=== Test: VisionMergerGraph ===");

    // Main merger
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::VisionMergerGraph::Build(
            "TestMerger", 1280, 2, false, 1e-6f, op);
        CHECK(IS_OK(s));
        CHECK(op.get() != nullptr);
        if (op) {
            CHECK(op.get()->GetInputNum() == 7);
        }
    }

    // Deepstack merger
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::VisionMergerGraph::Build(
            "TestDSMerger", 1280, 2, true, 1e-6f, op);
        CHECK(IS_OK(s));
        CHECK(op.get() != nullptr);
    }

    LOG_INFO("VisionMergerGraph test done");
}

TEST_CASE("DeepstackGraph") {
    LOG_INFO("=== Test: DeepstackGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::DeepstackGraph::Build(
        "TestDeepstack", 1280, 2, 1e-6f, op);
    CHECK(IS_OK(s));
    CHECK(op.get() != nullptr);

    if (op) {
        CHECK(op.get()->GetInputNum() == 7);
    }

    LOG_INFO("DeepstackGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: Vision Model graph builds
// ══════════════════════════════════════════════════════════
TEST_CASE("VisionModel") {
    LOG_INFO("=== Test: VisionModel ===");

    atb_llm::runners::VisionRunner::Config cfg;
    cfg.hidden_size = 1280;
    cfg.num_heads = 16;
    cfg.depth = 24;
    cfg.in_channels = 3;
    cfg.temporal_patch_size = 2;
    cfg.patch_size = 14;
    cfg.spatial_merge_size = 2;

    atb_llm::runners::VisionRunner model(cfg);
    atb_llm::Status s = model.Build();
    CHECK(IS_OK(s));

    CHECK(model.GetFirstLayerGraph().get() != nullptr);
    CHECK(model.GetBlockGraph().get() != nullptr);
    CHECK(model.GetMergerGraph().get() != nullptr);
    CHECK(model.GetDeepstackGraph().get() != nullptr);

    // Verify first layer input count
    if (model.GetFirstLayerGraph()) {
        // 19 inputs: pixels, pe_w, pe_b, pos, c, s, seq, + 12 block weights
        CHECK(model.GetFirstLayerGraph().get()->GetInputNum() == 19);
    }

    LOG_INFO("VisionModel test done");
}

// ══════════════════════════════════════════════════════════
// Test: MRoPE (CPU)
// ══════════════════════════════════════════════════════════
TEST_CASE("MRoPE") {
    LOG_INFO("=== Test: MRoPE ===");

    // Text MRoPE
    {
        atb_llm::components::MRoPE mrope(128, 5000000.0f, {24, 20, 20});

        // position_ids: (3, 1, 4) -- batch=1, seq=4
        int64_t position_ids[3 * 1 * 4];
        for (int d = 0; d < 3; d++) {
            for (int s = 0; s < 4; s++) {
                position_ids[d * 4 + s] = s;
            }
        }

        std::vector<float> cos_out(4 * 128);
        std::vector<float> sin_out(4 * 128);
        mrope.Compute(position_ids, 1, 4, cos_out.data(), sin_out.data());

        // Verify cos/sin are not all zeros and cos(0) = 1
        bool has_nonzero = false;
        for (auto v : cos_out) {
            if (v != 0.0f) {
                has_nonzero = true;
                break;
            }
        }
        CHECK(has_nonzero);

        // cos(0) should be ~1.0
        CHECK(std::fabs(cos_out[0] - 1.0f) < 0.01f);

        // sin(0) should be ~0.0
        CHECK(std::fabs(sin_out[0]) < 0.01f);
    }

    // VisionRotaryEmbedding
    {
        atb_llm::components::VisionRotaryEmbedding vre(80);
        auto table = vre.ComputeFreqTable(48);
        // ComputeFreqTable returns (max_hw, dim/2) columns (matches Python)
        // dim=80 → half=40 columns per row, 48 rows → 48*40 elements
        CHECK(!table.empty());
        CHECK(table.size() == static_cast<size_t>(48 * 40));

        // freq_table[0] (pos=0): first dim/2=40 elements should be 0
        bool all_zero = true;
        for (int d = 0; d < 40; d++) {
            if (table[d] != 0.0f) {
                all_zero = false;
                break;
            }
        }
        CHECK(all_zero);
    }

    // GetRopeIndex (text-only)
    {
        int64_t input_ids[] = {1, 2, 3, 4, 5};
        int64_t pos_ids[3 * 1 * 5];
        atb_llm::components::GetRopeIndex(
            input_ids, 1, 5, nullptr, 0, 151655, 151652, 2, pos_ids);

        // Text-only: all 3 dims should be sequential 0,1,2,3,4
        bool correct = true;
        for (int d = 0; d < 3; d++) {
            for (int s = 0; s < 5; s++) {
                if (pos_ids[d * 5 + s] != s) {
                    correct = false;
                    break;
                }
            }
        }
        CHECK(correct);
    }

    LOG_INFO("MRoPE test done");
}

// ══════════════════════════════════════════════════════════
// Test: Vision component NPU execution
// ══════════════════════════════════════════════════════════
TEST_CASE("Vision Component Execute") {
    LOG_INFO("=== Test: Vision Component Execute on NPU ===");

    auto runtime = atb_llm::CreateRuntime(0, 5LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    // Test VisionMLP execute
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::VisionMlpGraph::Build("ExecVisMLP", op);
        CHECK(IS_OK(s));
        CHECK(op.get() != nullptr);

        if (op) {
            int64_t seq = 4, hidden = 64, inter = 128;
            atb::Tensor input, fc1_w, fc1_b, fc2_w, fc2_b, output;
            alloc->AllocFloat16(input, {seq, hidden});
            alloc->AllocFloat16(fc1_w, {inter, hidden});
            alloc->AllocFloat16(fc1_b, {inter});
            alloc->AllocFloat16(fc2_w, {hidden, inter});
            alloc->AllocFloat16(fc2_b, {hidden});
            alloc->AllocFloat16(output, {seq, hidden});

            std::vector<uint16_t> ones(seq * hidden, 0x3400);  // 0.25
            std::vector<uint16_t> w_fc1(inter * hidden, 0x3C00);
            std::vector<uint16_t> b_fc1(inter, 0x3C00);
            std::vector<uint16_t> w_fc2(hidden * inter, 0x3C00);
            std::vector<uint16_t> b_fc2(hidden, 0x3C00);

            alloc->CopyToDevice(input, ones.data(), ones.size() * sizeof(uint16_t));
            alloc->CopyToDevice(fc1_w, w_fc1.data(), w_fc1.size() * sizeof(uint16_t));
            alloc->CopyToDevice(fc1_b, b_fc1.data(), b_fc1.size() * sizeof(uint16_t));
            alloc->CopyToDevice(fc2_w, w_fc2.data(), w_fc2.size() * sizeof(uint16_t));
            alloc->CopyToDevice(fc2_b, b_fc2.data(), b_fc2.size() * sizeof(uint16_t));

            atb::VariantPack vp;
            vp.inTensors = {input, fc1_w, fc1_b, fc2_w, fc2_b};
            vp.outTensors = {output};

            uint64_t ws_size = 0;
            atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
            CHECK(atb_s == atb::NO_ERROR);

            if (atb_s == atb::NO_ERROR) {
                uint8_t* ws_ptr = nullptr;
                if (ws_size > 0) {
                    auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                    ws_ptr = ws;
                }
                atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
                CHECK(atb_s == atb::NO_ERROR);
            }

            runtime->Synchronize();

            // Verify output is non-zero
            std::vector<uint16_t> host_out(seq * hidden);
            alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
            bool non_zero = false;
            for (auto v : host_out) {
                if (v != 0) {
                    non_zero = true;
                    break;
                }
            }
            CHECK(non_zero);
        }
    }

    // Test PatchEmbed execute
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::PatchEmbedGraph::Build(
            "ExecPatchEmbed", 3, 2, 14, 64, op);
        CHECK(IS_OK(s));

        if (op) {
            int64_t kernel_size = 3 * 2 * 14 * 14;  // 1176
            int64_t num_patches = 4;
            int64_t embed_dim = 64;

            atb::Tensor pixels, w, b, output;
            alloc->AllocFloat16(pixels, {num_patches * kernel_size});
            alloc->AllocFloat16(w, {embed_dim, kernel_size});
            alloc->AllocFloat16(b, {embed_dim});
            alloc->AllocFloat16(output, {num_patches, embed_dim});

            std::vector<uint16_t> pv_data(num_patches * kernel_size, 0x3400);
            std::vector<uint16_t> w_data(embed_dim * kernel_size, 0x3C00);
            std::vector<uint16_t> b_data(embed_dim, 0x3C00);

            alloc->CopyToDevice(pixels, pv_data.data(), pv_data.size() * sizeof(uint16_t));
            alloc->CopyToDevice(w, w_data.data(), w_data.size() * sizeof(uint16_t));
            alloc->CopyToDevice(b, b_data.data(), b_data.size() * sizeof(uint16_t));

            atb::VariantPack vp;
            vp.inTensors = {pixels, w, b};
            vp.outTensors = {output};

            uint64_t ws_size = 0;
            atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
            CHECK(atb_s == atb::NO_ERROR);

            if (atb_s == atb::NO_ERROR) {
                uint8_t* ws_ptr = nullptr;
                if (ws_size > 0) {
                    auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                    ws_ptr = ws;
                }
                atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
                CHECK(atb_s == atb::NO_ERROR);
            }

            runtime->Synchronize();

            std::vector<uint16_t> host_out(num_patches * embed_dim);
            alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
            bool non_zero = false;
            for (auto v : host_out) {
                if (v != 0) {
                    non_zero = true;
                    break;
                }
            }
            CHECK(non_zero);
        }
    }

    LOG_INFO("Vision Execute tests done");
}

// ══════════════════════════════════════════════════════════
// Test: LayerNorm NPU execution
// ══════════════════════════════════════════════════════════
TEST_CASE("LayerNorm Execute") {
    LOG_INFO("=== Test: LayerNorm Execute on NPU ===");

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    atb_llm::OperationHandle op = atb_llm::ops::LayerNormOp::Create(1e-5f);
    CHECK(op.get() != nullptr);

    if (op) {
        int64_t seq = 4, hidden = 64;
        atb::Tensor input, gamma, beta, output;
        alloc->AllocFloat16(input, {seq, hidden});
        alloc->AllocFloat16(gamma, {hidden});
        alloc->AllocFloat16(beta, {hidden});
        alloc->AllocFloat16(output, {seq, hidden});

        std::vector<uint16_t> ones(seq * hidden, 0x3C00);  // 1.0
        std::vector<uint16_t> gamma_data(hidden, 0x3C00);
        std::vector<uint16_t> beta_data(hidden, 0x3C00);

        alloc->CopyToDevice(input, ones.data(), ones.size() * sizeof(uint16_t));
        alloc->CopyToDevice(gamma, gamma_data.data(), gamma_data.size() * sizeof(uint16_t));
        alloc->CopyToDevice(beta, beta_data.data(), beta_data.size() * sizeof(uint16_t));

        atb::VariantPack vp;
        vp.inTensors = {input, gamma, beta};
        vp.outTensors = {output};

        uint64_t ws_size = 0;
        atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
        CHECK(atb_s == atb::NO_ERROR);

        if (atb_s == atb::NO_ERROR) {
            uint8_t* ws_ptr = nullptr;
            if (ws_size > 0) {
                auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                ws_ptr = ws;
            }
            atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
            CHECK(atb_s == atb::NO_ERROR);
        }

        runtime->Synchronize();

        std::vector<uint16_t> host_out(seq * hidden);
        alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
        bool non_zero = false;
        for (auto v : host_out) {
            if (v != 0) {
                non_zero = true;
                break;
            }
        }
        CHECK(non_zero);
    }

    LOG_INFO("LayerNorm Execute test done");
}
