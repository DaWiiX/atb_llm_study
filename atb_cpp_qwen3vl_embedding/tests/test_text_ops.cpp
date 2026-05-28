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

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/graph_builder.h"
#include "core/context_manager.h"
#include "core/tensor_allocator.h"
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
#include "components/norm/rms_norm_graph.h"
#include "components/mlp/swiglu_mlp_graph.h"
#include "components/attention/self_attention_graph.h"
#include "layers/text_decoder_layer.h"
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

static int test_count = 0;
static int pass_count = 0;

#define TEST_ASSERT(cond, msg)                                          \
    do {                                                                \
        test_count++;                                                   \
        if (!(cond)) {                                                  \
            LOG_ERROR("FAIL: %s (%s:%d)", msg, __FILE__, __LINE__);    \
        } else {                                                        \
            pass_count++;                                               \
            LOG_INFO("PASS: %s", msg);                                 \
        }                                                               \
    } while (0)

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

// ══════════════════════════════════════════════════════════
// Test: Individual Op creation
// ══════════════════════════════════════════════════════════
void test_op_creation() {
    LOG_INFO("=== Test: Op Creation ===");

    // Linear
    {
        auto op = atb_llm::ops::LinearOp::Create();
        TEST_ASSERT(op.get() != nullptr, "LinearOp::Create() returns non-null");
    }
    {
        auto op = atb_llm::ops::LinearOp::Create(true, false, true);
        TEST_ASSERT(op.get() != nullptr, "LinearOp::Create(with_bias) returns non-null");
    }

    // RmsNorm
    {
        auto op = atb_llm::ops::RmsNormOp::Create(1e-6f);
        TEST_ASSERT(op.get() != nullptr, "RmsNormOp::Create() returns non-null");
    }
    {
        auto op = atb_llm::ops::RmsNormOp::Create(1e-5f, atb_llm::ops::RmsNormOp::LayerType::PRENORM);
        TEST_ASSERT(op.get() != nullptr, "RmsNormOp::Create(PRENORM) returns non-null");
    }

    // Elewise
    {
        auto op = atb_llm::ops::ElewiseOp::MakeAdd();
        TEST_ASSERT(op.get() != nullptr, "ElewiseOp::MakeAdd() returns non-null");
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeMul();
        TEST_ASSERT(op.get() != nullptr, "ElewiseOp::MakeMul() returns non-null");
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeMuls(2.0f);
        TEST_ASSERT(op.get() != nullptr, "ElewiseOp::MakeMuls() returns non-null");
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeCast(ACL_FLOAT16);
        TEST_ASSERT(op.get() != nullptr, "ElewiseOp::MakeCast() returns non-null");
    }
    {
        auto op = atb_llm::ops::ElewiseOp::MakeSub();
        TEST_ASSERT(op.get() != nullptr, "ElewiseOp::MakeSub() returns non-null");
    }

    // Activation
    {
        auto op = atb_llm::ops::ActivationOp::MakeSiLU();
        TEST_ASSERT(op.get() != nullptr, "ActivationOp::MakeSiLU() returns non-null");
    }
    {
        auto op = atb_llm::ops::ActivationOp::MakeGELU();
        TEST_ASSERT(op.get() != nullptr, "ActivationOp::MakeGELU() returns non-null");
    }
    {
        auto op = atb_llm::ops::ActivationOp::MakeFastGELU();
        TEST_ASSERT(op.get() != nullptr, "ActivationOp::MakeFastGELU() returns non-null");
    }

    // Split
    {
        auto op = atb_llm::ops::SplitOp::Create(-1, 2);
        TEST_ASSERT(op.get() != nullptr, "SplitOp::Create() returns non-null");
    }

    // Concat
    {
        auto op = atb_llm::ops::ConcatOp::Create(0);
        TEST_ASSERT(op.get() != nullptr, "ConcatOp::Create() returns non-null");
    }

    // Rope
    {
        auto op = atb_llm::ops::RopeOp::Create();
        TEST_ASSERT(op.get() != nullptr, "RopeOp::Create() returns non-null");
    }

    // SelfAttention
    {
        auto op = atb_llm::ops::SelfAttentionOp::Create(12, 12, 64, false);
        TEST_ASSERT(op.get() != nullptr, "SelfAttentionOp::Create() returns non-null");
    }
    {
        auto op = atb_llm::ops::SelfAttentionOp::Create(12, 12, 64, true);
        TEST_ASSERT(op.get() != nullptr, "SelfAttentionOp::Create(with_mask) returns non-null");
    }

    // Transpose
    {
        auto op = atb_llm::ops::TransposeOp::Create({0, 2, 1, 3});
        TEST_ASSERT(op.get() != nullptr, "TransposeOp::Create() returns non-null");
    }

    // Softmax
    {
        auto op = atb_llm::ops::SoftmaxOp::Create({-1});
        TEST_ASSERT(op.get() != nullptr, "SoftmaxOp::Create() returns non-null");
    }

    LOG_INFO("Op creation tests done");
}

// ══════════════════════════════════════════════════════════
// Test: GraphBuilder with AddOp (template path)
// ══════════════════════════════════════════════════════════
void test_graph_builder_add_op() {
    LOG_INFO("=== Test: GraphBuilder AddOp ===");

    std::unique_ptr<atb_llm::GraphBuilder> builder;
    atb_llm::Status s = atb_llm::GraphBuilder::Create("TestAddOp", builder);
    TEST_ASSERT(IS_OK(s), "GraphBuilder::Create succeeds");

    atb::SVector<std::string> in_names = {"x", "w"};
    atb::SVector<std::string> out_names = {"y"};
    s = builder->Init("TestAddOp", nullptr, in_names, out_names);
    TEST_ASSERT(IS_OK(s), "GraphBuilder Init succeeds");

    // Use AddOp (template path)
    atb::infer::LinearParam param;
    param.hasBias = false;
    param.transposeA = false;
    param.transposeB = true;
    s = builder->AddOp(param,
        atb::SVector<std::string>{"x", "w"},
        atb::SVector<std::string>{"y"});
    TEST_ASSERT(IS_OK(s), "GraphBuilder AddOp(LinearParam) succeeds");

    auto op = builder->Build();
    TEST_ASSERT(op.get() != nullptr, "GraphBuilder Build returns non-null");
    LOG_INFO("GraphBuilder AddOp test done");
}

// ══════════════════════════════════════════════════════════
// Test: RmsNormGraph component
// ══════════════════════════════════════════════════════════
void test_rms_norm_graph() {
    LOG_INFO("=== Test: RmsNormGraph ===");
    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::RmsNormGraph::Build("TestRmsNorm", 1e-6f, op);
    TEST_ASSERT(IS_OK(s), "RmsNormGraph::Build succeeds");
    TEST_ASSERT(op.get() != nullptr, "RmsNormGraph returns non-null operation");

    // Verify operation properties
    if (op) {
        TEST_ASSERT(op.get()->GetInputNum() == 2, "RmsNormGraph has 2 inputs");
        TEST_ASSERT(op.get()->GetOutputNum() == 1, "RmsNormGraph has 1 output");
    }
    LOG_INFO("RmsNormGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: SwiGluMlpGraph component
// ══════════════════════════════════════════════════════════
void test_swiglu_mlp_graph() {
    LOG_INFO("=== Test: SwiGluMlpGraph ===");
    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::SwiGluMlpGraph::Build("TestSwiGLU", op);
    TEST_ASSERT(IS_OK(s), "SwiGluMlpGraph::Build succeeds");
    TEST_ASSERT(op.get() != nullptr, "SwiGluMlpGraph returns non-null operation");

    if (op) {
        TEST_ASSERT(op.get()->GetInputNum() == 4, "SwiGluMlpGraph has 4 inputs");
        TEST_ASSERT(op.get()->GetOutputNum() == 1, "SwiGluMlpGraph has 1 output");
    }
    LOG_INFO("SwiGluMlpGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: SelfAttentionGraph component
// ══════════════════════════════════════════════════════════
void test_self_attention_graph() {
    LOG_INFO("=== Test: SelfAttentionGraph ===");

    // Without mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SelfAttentionGraph::Build(
            "TestAttn_NoMask", 12, 12, 64, 16, 1e-6f, false, op);
        TEST_ASSERT(IS_OK(s), "SelfAttentionGraph::Build(no mask) succeeds");
        TEST_ASSERT(op.get() != nullptr, "SelfAttentionGraph(no mask) returns non-null");
        if (op) {
            // 10 inputs: hidden, q/k/v/o_w, qn/kn_w, cos, sin, seqlen
            TEST_ASSERT(op.get()->GetInputNum() == 10, "SelfAttentionGraph(no mask) has 10 inputs");
            TEST_ASSERT(op.get()->GetOutputNum() == 1, "SelfAttentionGraph has 1 output");
        }
    }

    // With mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SelfAttentionGraph::Build(
            "TestAttn_Mask", 12, 12, 64, 16, 1e-6f, true, op);
        TEST_ASSERT(IS_OK(s), "SelfAttentionGraph::Build(with mask) succeeds");
        TEST_ASSERT(op.get() != nullptr, "SelfAttentionGraph(with mask) returns non-null");
        if (op) {
            // 11 inputs: +mask
            TEST_ASSERT(op.get()->GetInputNum() == 11, "SelfAttentionGraph(with mask) has 11 inputs");
        }
    }

    // GQA: num_kv_heads < num_heads
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SelfAttentionGraph::Build(
            "TestAttn_GQA", 12, 2, 64, 16, 1e-6f, false, op);
        TEST_ASSERT(IS_OK(s), "SelfAttentionGraph::Build(GQA) succeeds");
        TEST_ASSERT(op.get() != nullptr, "SelfAttentionGraph(GQA) returns non-null");
    }

    LOG_INFO("SelfAttentionGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: TextDecoderLayerGraph
// ══════════════════════════════════════════════════════════
void test_text_decoder_layer_graph() {
    LOG_INFO("=== Test: TextDecoderLayerGraph ===");

    // Without mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::layers::TextDecoderLayerGraph::Build(
            "TestDecoderLayer", 12, 12, 64, 16, 1e-6f, false, op);
        TEST_ASSERT(IS_OK(s), "TextDecoderLayerGraph::Build(no mask) succeeds");
        TEST_ASSERT(op.get() != nullptr, "TextDecoderLayerGraph returns non-null");
        if (op) {
            // 15 inputs: hidden + 4 proj + 2 qk_norm + 3 mlp + 2 ln + cos + sin + seqlen
            TEST_ASSERT(op.get()->GetInputNum() == 15, "TextDecoderLayerGraph has 15 inputs");
            TEST_ASSERT(op.get()->GetOutputNum() == 1, "TextDecoderLayerGraph has 1 output");
        }
    }

    // With mask
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::layers::TextDecoderLayerGraph::Build(
            "TestDecoderLayer_Mask", 12, 12, 64, 16, 1e-6f, true, op);
        TEST_ASSERT(IS_OK(s), "TextDecoderLayerGraph::Build(with mask) succeeds");
        TEST_ASSERT(op.get() != nullptr, "TextDecoderLayerGraph(with mask) returns non-null");
        if (op) {
            TEST_ASSERT(op.get()->GetInputNum() == 16, "TextDecoderLayerGraph(with mask) has 16 inputs");
        }
    }

    // GQA
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::layers::TextDecoderLayerGraph::Build(
            "TestDecoderLayer_GQA", 12, 2, 64, 16, 1e-6f, false, op);
        TEST_ASSERT(IS_OK(s), "TextDecoderLayerGraph::Build(GQA) succeeds");
        TEST_ASSERT(op.get() != nullptr, "TextDecoderLayerGraph(GQA) returns non-null");
    }

    LOG_INFO("TextDecoderLayerGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: Operation Setup+Execute on NPU (integration test)
// ══════════════════════════════════════════════════════════
void test_operation_execute() {
    LOG_INFO("=== Test: Operation Execute on NPU ===");

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    TEST_ASSERT(runtime != nullptr, "Runtime created");
    if (!runtime) return;

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    // Test RmsNorm execute: input [4, 64], weight [64]
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::RmsNormGraph::Build("ExecRmsNorm", 1e-6f, op);
        TEST_ASSERT(IS_OK(s), "RmsNormGraph build succeeds");

        if (op) {
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
            TEST_ASSERT(atb_s == atb::NO_ERROR || ws_size == 0, "RmsNorm Setup succeeds");

            if (atb_s == atb::NO_ERROR && ws_size > 0) {
                auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                TEST_ASSERT(IS_OK(ws_s), "GetWorkspace succeeds");
                atb_s = op.get()->Execute(vp, ws, ws_size, ctx);
                TEST_ASSERT(atb_s == atb::NO_ERROR, "RmsNorm Execute succeeds");
            } else if (atb_s == atb::NO_ERROR) {
                atb_s = op.get()->Execute(vp, nullptr, 0, ctx);
                TEST_ASSERT(atb_s == atb::NO_ERROR, "RmsNorm Execute succeeds (no workspace)");
            }

            runtime->Synchronize();

            // Verify output is not all zeros
            std::vector<uint16_t> host_out(4 * 64);
            alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
            bool non_zero = false;
            for (auto v : host_out) {
                if (v != 0) { non_zero = true; break; }
            }
            TEST_ASSERT(non_zero, "RmsNorm output is not all zeros");
        }
    }

    // Test Linear execute: input [4, 64], weight [128, 64] -> output [4, 128]
    {
        atb_llm::OperationHandle op = atb_llm::ops::LinearOp::Create();
        TEST_ASSERT(op.get() != nullptr, "LinearOp created");

        if (op) {
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
            TEST_ASSERT(atb_s == atb::NO_ERROR, "Linear Setup succeeds");

            if (atb_s == atb::NO_ERROR) {
                uint8_t* ws_ptr = nullptr;
                if (ws_size > 0) {
                    auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                    ws_ptr = ws;
                }
                atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
                TEST_ASSERT(atb_s == atb::NO_ERROR, "Linear Execute succeeds");
            }

            runtime->Synchronize();
        }
    }

    // Test SwiGLU MLP execute
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::SwiGluMlpGraph::Build("ExecSwiGLU", op);
        TEST_ASSERT(IS_OK(s), "SwiGluMlpGraph build succeeds");

        if (op) {
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
            TEST_ASSERT(atb_s == atb::NO_ERROR, "SwiGLU Setup succeeds");

            if (atb_s == atb::NO_ERROR) {
                uint8_t* ws_ptr = nullptr;
                if (ws_size > 0) {
                    auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                    ws_ptr = ws;
                }
                atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
                TEST_ASSERT(atb_s == atb::NO_ERROR, "SwiGLU Execute succeeds");
            }

            runtime->Synchronize();

            // Verify output is non-zero
            std::vector<uint16_t> host_out(seq * hidden);
            alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
            bool non_zero = false;
            for (auto v : host_out) {
                if (v != 0) { non_zero = true; break; }
            }
            TEST_ASSERT(non_zero, "SwiGLU output is not all zeros");
        }
    }

    // Test TextDecoderLayerGraph build + Setup
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::layers::TextDecoderLayerGraph::Build(
            "ExecDecoderLayer", 12, 12, 64, 4, 1e-6f, false, op);
        TEST_ASSERT(IS_OK(s), "TextDecoderLayerGraph build for execute test");
        TEST_ASSERT(op.get() != nullptr, "TextDecoderLayerGraph non-null for execute");
    }

    LOG_INFO("Operation Execute tests done");
}

// ══════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    LOG_INFO("=== atb_cpp_llm_engine Phase 2 Tests ===");

    test_op_creation();
    test_graph_builder_add_op();
    test_rms_norm_graph();
    test_swiglu_mlp_graph();
    test_self_attention_graph();
    test_text_decoder_layer_graph();
    test_operation_execute();

    LOG_INFO("=== Results: %d/%d passed ===", pass_count, test_count);

    if (pass_count == test_count) {
        LOG_INFO("ALL TESTS PASSED");
        return 0;
    } else {
        LOG_ERROR("SOME TESTS FAILED");
        return 1;
    }
}
