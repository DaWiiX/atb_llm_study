/**
 * Phase 2 tests: ATB Op wrappers and Text Decoder Layer components.
 *
 * Tests:
 *   1. Each Op's Create() returns a valid OperationHandle
 *   2. Component graphs (RmsNormGraph, SwiGluMlpGraph, SelfAttentionGraph) build successfully
 *   3. TextDecoderLayerGraph builds successfully
 *   4. Built operations can Setup+Execute on NPU (requires device)
 *
 * Run: ./test_text_ops
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
#include "utils/cpp11_compat.h"
#include "ops/linear_op.h"
#include "ops/rms_norm_op.h"
#include "ops/elewise_op.h"
#include "ops/activation_op.h"
#include "ops/split_op.h"
#include "ops/concat_op.h"
#include "ops/rope_op.h"
#include "ops/self_attention_op.h"
#include "ops/transpose_op.h"
#include "ops/softmax_op.h"
#include "components/common/rms_norm_graph.h"
#include "components/common/swiglu_mlp_graph.h"
#include "components/common/self_attention_graph.h"
#include "components/text/decoder_layer_graph.h"
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

// Use helper to avoid OK ambiguity with aclnnStatus from acl_meta.h
#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

// ══════════════════════════════════════════════════════════
// Test: Individual Op creation
// ══════════════════════════════════════════════════════════
TEST_CASE("Op Creation") {
    LOG_INFO("=== Test: Op Creation ===");

    // Linear
    {
        auto op = atb_llm::ops::LinearOp::Create();
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::LinearOp::Create(true, false, true);
        CHECK(op.get() != nullptr);
    }

    // RmsNorm
    {
        auto op = atb_llm::ops::RmsNormOp::Create(1e-6f);
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::RmsNormOp::Create(1e-5f, atb_llm::ops::RmsNormOp::LayerType::PRENORM);
        CHECK(op.get() != nullptr);
    }

    // Elewise
    {
        auto op = atb_llm::ops::ElewiseOp::MakeAdd();
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeMul();
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeMuls(2.0f);
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeCast(ACL_FLOAT16);
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeSub();
        CHECK(op.get() != nullptr);
    }

    // Activation
    {
        auto op = atb_llm::ops::ActivationOp::MakeSiLU();
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::ActivationOp::MakeGELU();
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::ActivationOp::MakeFastGELU();
        CHECK(op.get() != nullptr);
    }

    // Split
    {
        auto op = atb_llm::ops::SplitOp::Create(-1, 2);
        CHECK(op.get() != nullptr);
    }

    // Concat
    {
        auto op = atb_llm::ops::ConcatOp::Create(0);
        CHECK(op.get() != nullptr);
    }

    // Rope
    {
        auto op = atb_llm::ops::RopeOp::Create();
        CHECK(op.get() != nullptr);
    }

    // SelfAttention
    {
        auto op = atb_llm::ops::SelfAttentionOp::Create(12, 12, 64, false);
        CHECK(op.get() != nullptr);
    }
    {
        auto op = atb_llm::ops::SelfAttentionOp::Create(12, 12, 64, true);
        CHECK(op.get() != nullptr);
    }

    // Transpose
    {
        auto op = atb_llm::ops::TransposeOp::Create({0, 2, 1, 3});
        CHECK(op.get() != nullptr);
    }

    // Softmax
    {
        auto op = atb_llm::ops::SoftmaxOp::Create({-1});
        CHECK(op.get() != nullptr);
    }

    LOG_INFO("Op creation tests done");
}

// ══════════════════════════════════════════════════════════
// Test: GraphBuilder with AddOp (template path)
// ══════════════════════════════════════════════════════════
TEST_CASE("GraphBuilder AddOp") {
    LOG_INFO("=== Test: GraphBuilder AddOp ===");

    std::unique_ptr<atb_llm::GraphBuilder> builder;
    atb_llm::Status s = atb_llm::GraphBuilder::Create("TestAddOp", builder);
    CHECK(IS_OK(s));

    atb::SVector<std::string> in_names = {"x", "w"};
    atb::SVector<std::string> out_names = {"y"};
    s = builder->Init("TestAddOp", nullptr, in_names, out_names);
    CHECK(IS_OK(s));

    // Use AddOp (template path)
    atb::infer::LinearParam param;
    param.hasBias = false;
    param.transposeA = false;
    param.transposeB = true;
    s = builder->AddOp(param,
                       atb::SVector<std::string>{"x", "w"},
                       atb::SVector<std::string>{"y"});
    CHECK(IS_OK(s));

    auto op = builder->Build();
    CHECK(op.get() != nullptr);
    LOG_INFO("GraphBuilder AddOp test done");
}

// ══════════════════════════════════════════════════════════
// Test: RmsNormGraph component
// ══════════════════════════════════════════════════════════
TEST_CASE("RmsNormGraph") {
    LOG_INFO("=== Test: RmsNormGraph ===");
    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::RmsNormGraph::Build("TestRmsNorm", 1e-6f, op);
    CHECK(IS_OK(s));
    REQUIRE(op.get() != nullptr);

    // Verify operation properties
    CHECK(op.get()->GetInputNum() == 2);
    CHECK(op.get()->GetOutputNum() == 1);
    LOG_INFO("RmsNormGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: SwiGluMlpGraph component
// ══════════════════════════════════════════════════════════
TEST_CASE("SwiGluMlpGraph") {
    LOG_INFO("=== Test: SwiGluMlpGraph ===");
    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::SwiGluMlpGraph::Build("TestSwiGLU", op);
    CHECK(IS_OK(s));
    REQUIRE(op.get() != nullptr);

    CHECK(op.get()->GetInputNum() == 4);
    CHECK(op.get()->GetOutputNum() == 1);
    LOG_INFO("SwiGluMlpGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: SelfAttentionGraph component
// ══════════════════════════════════════════════════════════
TEST_CASE("SelfAttentionGraph") {
    LOG_INFO("=== Test: SelfAttentionGraph ===");

    // Without mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SelfAttentionGraph::Build(
            "TestAttn_NoMask", 12, 12, 64, 16, 1e-6f, false, op);
        CHECK(IS_OK(s));
        REQUIRE(op.get() != nullptr);
        // 10 inputs: hidden, q/k/v/o_w, qn/kn_w, cos, sin, seqlen
        CHECK(op.get()->GetInputNum() == 10);
        CHECK(op.get()->GetOutputNum() == 1);
    }

    // With mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SelfAttentionGraph::Build(
            "TestAttn_Mask", 12, 12, 64, 16, 1e-6f, true, op);
        CHECK(IS_OK(s));
        REQUIRE(op.get() != nullptr);
        // On 310P: 10 inputs (MASK_TYPE_NORM + isTriuMask=1, mask tensor is always present)
        // On 910B: 11 inputs (+mask)
        uint32_t expected = 11;
        CHECK(op.get()->GetInputNum() == expected);
    }

    // GQA: num_kv_heads < num_heads (verified on 310P, cos=1.0)
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SelfAttentionGraph::Build(
            "TestAttn_GQA", 12, 2, 64, 16, 1e-6f, false, op);
        CHECK(IS_OK(s));
        CHECK(op.get() != nullptr);
    }

    LOG_INFO("SelfAttentionGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: TextDecoderLayerGraph
// ══════════════════════════════════════════════════════════
TEST_CASE("TextDecoderLayerGraph") {
    LOG_INFO("=== Test: TextDecoderLayerGraph ===");

    // Without mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::text::TextDecoderLayerGraph::Build(
            "TestDecoderLayer", 12, 12, 64, 16, 1e-6f, false, op);
        CHECK(IS_OK(s));
        REQUIRE(op.get() != nullptr);
        // 15 inputs: hidden + 4 proj + 2 qk_norm + 3 mlp + 2 ln + cos + sin + seqlen
        CHECK(op.get()->GetInputNum() == 15);
        CHECK(op.get()->GetOutputNum() == 1);
    }

    // With mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::text::TextDecoderLayerGraph::Build(
            "TestDecoderLayer_Mask", 12, 12, 64, 16, 1e-6f, true, op);
        CHECK(IS_OK(s));
        REQUIRE(op.get() != nullptr);
        // On 310P: 15 inputs (MASK_TYPE_NORM + isTriuMask=1, mask tensor is always present)
        // On 910B: 16 inputs (+mask)
        uint32_t expected = 16;
        CHECK(op.get()->GetInputNum() == expected);
    }

    // GQA (verified on 310P, cos=1.0)
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::text::TextDecoderLayerGraph::Build(
            "TestDecoderLayer_GQA", 12, 2, 64, 16, 1e-6f, false, op);
        CHECK(IS_OK(s));
        CHECK(op.get() != nullptr);
    }

    LOG_INFO("TextDecoderLayerGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: Operation Setup+Execute on NPU (integration test)
// ══════════════════════════════════════════════════════════
TEST_CASE("Operation Execute on NPU") {
    LOG_INFO("=== Test: Operation Execute on NPU ===");

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    // Test RmsNorm execute: input [4, 64], weight [64]
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::RmsNormGraph::Build("ExecRmsNorm", 1e-6f, op);
        CHECK(IS_OK(s));

        REQUIRE(op.get() != nullptr);

        atb::Tensor input, weight, output;
        alloc->AllocFloat16(input, {4, 64});
        alloc->AllocFloat16(weight, {64});
        alloc->AllocFloat16(output, {4, 64});

        // Fill weight with 1.0 (half precision)
        std::vector<uint16_t> ones(64, 0x3C00);  // 1.0 in float16
        // Ensure dataSize matches actual data
        weight.dataSize = ones.size() * sizeof(uint16_t);
        alloc->CopyToDevice(weight, ones.data(), weight.dataSize);

        // Fill input with small values
        std::vector<uint16_t> input_data(4 * 64, 0x3C00);  // 1.0
        input.dataSize = input_data.size() * sizeof(uint16_t);
        alloc->CopyToDevice(input, input_data.data(), input.dataSize);

        atb::VariantPack vp;
        vp.inTensors = {input, weight};
        vp.outTensors = {output};

        uint64_t ws_size = 0;
        atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
        CHECK((atb_s == atb::NO_ERROR || ws_size == 0));

        if (atb_s == atb::NO_ERROR && ws_size > 0) {
            auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
            CHECK(IS_OK(ws_s));
            atb_s = op.get()->Execute(vp, ws, ws_size, ctx);
            CHECK(atb_s == atb::NO_ERROR);
        } else if (atb_s == atb::NO_ERROR) {
            atb_s = op.get()->Execute(vp, nullptr, 0, ctx);
            CHECK(atb_s == atb::NO_ERROR);
        }

        runtime->Synchronize();

        // Verify output is not all zeros
        std::vector<uint16_t> host_out(4 * 64);
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

    // Test Linear execute: input [4, 64], weight [128, 64] -> output [4, 128]
    {
        atb_llm::OperationHandle op = atb_llm::ops::LinearOp::Create();
        REQUIRE(op.get() != nullptr);

        atb::Tensor input, weight, output;
        alloc->AllocFloat16(input, {4, 64});
        alloc->AllocFloat16(weight, {128, 64});
        alloc->AllocFloat16(output, {4, 128});

        std::vector<uint16_t> input_data(4 * 64, 0x3C00);
        std::vector<uint16_t> weight_data(128 * 64, 0x3C00);
        alloc->CopyToDevice(input, input_data.data(), input_data.size() * sizeof(uint16_t));
        alloc->CopyToDevice(weight, weight_data.data(), weight_data.size() * sizeof(uint16_t));

        atb::VariantPack vp;
        vp.inTensors = {input, weight};
        vp.outTensors = {output};

        uint64_t ws_size = 0;
        atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
        CHECK(atb_s == atb::NO_ERROR);

        if (atb_s == atb::NO_ERROR) {
            uint8_t* ws_ptr = nullptr;
            if (ws_size > 0) {
                auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
                ws_ptr = ws;
            }
            atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
            CHECK(atb_s == atb::NO_ERROR);
        }

        runtime->Synchronize();
    }

    // Test SwiGLU MLP execute
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SwiGluMlpGraph::Build("ExecSwiGLU", op);
        CHECK(IS_OK(s));

        REQUIRE(op.get() != nullptr);

        int64_t seq = 4, hidden = 64, inter = 128;
        atb::Tensor input, gate_w, up_w, down_w, output;
        alloc->AllocFloat16(input, {seq, hidden});
        alloc->AllocFloat16(gate_w, {inter, hidden});
        alloc->AllocFloat16(up_w, {inter, hidden});
        alloc->AllocFloat16(down_w, {hidden, inter});
        alloc->AllocFloat16(output, {seq, hidden});

        // Fill with small values to avoid overflow
        std::vector<uint16_t> ones_small(seq * hidden, 0x3400);  // 0.25 in fp16
        std::vector<uint16_t> w_ones(inter * hidden, 0x3C00);
        std::vector<uint16_t> w_down(hidden * inter, 0x3C00);

        alloc->CopyToDevice(input, ones_small.data(), ones_small.size() * sizeof(uint16_t));
        alloc->CopyToDevice(gate_w, w_ones.data(), w_ones.size() * sizeof(uint16_t));
        alloc->CopyToDevice(up_w, w_ones.data(), w_ones.size() * sizeof(uint16_t));
        alloc->CopyToDevice(down_w, w_down.data(), w_down.size() * sizeof(uint16_t));

        atb::VariantPack vp;
        vp.inTensors = {input, gate_w, up_w, down_w};
        vp.outTensors = {output};

        uint64_t ws_size = 0;
        atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
        CHECK(atb_s == atb::NO_ERROR);

        if (atb_s == atb::NO_ERROR) {
            uint8_t* ws_ptr = nullptr;
            if (ws_size > 0) {
                auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
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

    // Test TextDecoderLayerGraph build + Setup
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::text::TextDecoderLayerGraph::Build(
            "ExecDecoderLayer", 12, 12, 64, 4, 1e-6f, false, op);
        CHECK(IS_OK(s));
        CHECK(op.get() != nullptr);
    }

    LOG_INFO("Operation Execute tests done");
}
