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
#include "models/vision_model.h"

// Position
#include "components/position/mrope.h"

// Runtime
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
// Test: New Op creation
// ══════════════════════════════════════════════════════════
void test_new_op_creation() {
    LOG_INFO("=== Test: New Op Creation (Phase 3) ===");

    // LayerNorm
    {
        auto op = atb_llm::ops::LayerNormOp::Create();
        TEST_ASSERT(op.get() != nullptr, "LayerNormOp::Create() returns non-null");
    }
    {
        auto op = atb_llm::ops::LayerNormOp::Create(1e-5f, -1, -1);
        TEST_ASSERT(op.get() != nullptr, "LayerNormOp::Create(with_params) returns non-null");
    }

    // SetValue
    {
        auto op = atb_llm::ops::SetValueOp::Create({0, 0}, {1, 10});
        TEST_ASSERT(op.get() != nullptr, "SetValueOp::Create() returns non-null");
    }

    // Gather
    {
        auto op = atb_llm::ops::GatherOp::Create(0, 0);
        TEST_ASSERT(op.get() != nullptr, "GatherOp::Create() returns non-null");
    }

    // Reduce
    {
        auto op = atb_llm::ops::ReduceOp::Create(
            atb_llm::ops::ReduceOp::ReduceType::MAX, {0});
        TEST_ASSERT(op.get() != nullptr, "ReduceOp::Create(MAX) returns non-null");
    }
    {
        auto op = atb_llm::ops::ReduceOp::Create(
            atb_llm::ops::ReduceOp::ReduceType::SUM, {1});
        TEST_ASSERT(op.get() != nullptr, "ReduceOp::Create(SUM) returns non-null");
    }

    LOG_INFO("New op creation tests done");
}

// ══════════════════════════════════════════════════════════
// Test: Vision Component graph builds
// ══════════════════════════════════════════════════════════
void test_vision_attention_graph() {
    LOG_INFO("=== Test: VisionAttentionGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::VisionAttentionGraph::Build(
        "TestVisAttn", 16, 80, op);
    TEST_ASSERT(IS_OK(s), "VisionAttentionGraph::Build succeeds");
    TEST_ASSERT(op.get() != nullptr, "VisionAttentionGraph returns non-null");

    if (op) {
        // 8 inputs: hidden, qkv_w, qkv_b, proj_w, proj_b, c, s, seq
        TEST_ASSERT(op.get()->GetInputNum() == 8,
                    "VisionAttentionGraph has 8 inputs");
        TEST_ASSERT(op.get()->GetOutputNum() == 1,
                    "VisionAttentionGraph has 1 output");
    }

    LOG_INFO("VisionAttentionGraph test done");
}

void test_vision_mlp_graph() {
    LOG_INFO("=== Test: VisionMlpGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::VisionMlpGraph::Build("TestVisMLP", op);
    TEST_ASSERT(IS_OK(s), "VisionMlpGraph::Build succeeds");
    TEST_ASSERT(op.get() != nullptr, "VisionMlpGraph returns non-null");

    if (op) {
        // 5 inputs: hidden, fc1_w, fc1_b, fc2_w, fc2_b
        TEST_ASSERT(op.get()->GetInputNum() == 5,
                    "VisionMlpGraph has 5 inputs");
        TEST_ASSERT(op.get()->GetOutputNum() == 1,
                    "VisionMlpGraph has 1 output");
    }

    LOG_INFO("VisionMlpGraph test done");
}

void test_vision_block_graph() {
    LOG_INFO("=== Test: VisionBlockGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::VisionBlockGraph::Build(
        "TestVisBlock", 16, 80, 1e-6f, op);
    TEST_ASSERT(IS_OK(s), "VisionBlockGraph::Build succeeds");
    TEST_ASSERT(op.get() != nullptr, "VisionBlockGraph returns non-null");

    if (op) {
        // 16 inputs: hidden + 4 attn + 4 mlp + 4 norm + cos + sin + seqlen
        TEST_ASSERT(op.get()->GetInputNum() == 16,
                    "VisionBlockGraph has 16 inputs");
        TEST_ASSERT(op.get()->GetOutputNum() == 1,
                    "VisionBlockGraph has 1 output");
    }

    LOG_INFO("VisionBlockGraph test done");
}

void test_patch_embed_graph() {
    LOG_INFO("=== Test: PatchEmbedGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::PatchEmbedGraph::Build(
        "TestPatchEmbed", 3, 2, 14, 1280, op);
    TEST_ASSERT(IS_OK(s), "PatchEmbedGraph::Build succeeds");
    TEST_ASSERT(op.get() != nullptr, "PatchEmbedGraph returns non-null");

    if (op) {
        // 3 inputs: pixels, w, b
        TEST_ASSERT(op.get()->GetInputNum() == 3,
                    "PatchEmbedGraph has 3 inputs");
        TEST_ASSERT(op.get()->GetOutputNum() == 1,
                    "PatchEmbedGraph has 1 output");
    }

    LOG_INFO("PatchEmbedGraph test done");
}

void test_vision_merger_graph() {
    LOG_INFO("=== Test: VisionMergerGraph ===");

    // Main merger
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::VisionMergerGraph::Build(
            "TestMerger", 1280, 2, false, 1e-6f, op);
        TEST_ASSERT(IS_OK(s), "VisionMergerGraph::Build(main) succeeds");
        TEST_ASSERT(op.get() != nullptr, "VisionMergerGraph(main) returns non-null");
        if (op) {
            TEST_ASSERT(op.get()->GetInputNum() == 7,
                        "VisionMergerGraph has 7 inputs");
        }
    }

    // Deepstack merger
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::VisionMergerGraph::Build(
            "TestDSMerger", 1280, 2, true, 1e-6f, op);
        TEST_ASSERT(IS_OK(s), "VisionMergerGraph::Build(deepstack) succeeds");
        TEST_ASSERT(op.get() != nullptr, "VisionMergerGraph(deepstack) returns non-null");
    }

    LOG_INFO("VisionMergerGraph test done");
}

void test_deepstack_graph() {
    LOG_INFO("=== Test: DeepstackGraph ===");

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::DeepstackGraph::Build(
        "TestDeepstack", 1280, 2, 1e-6f, op);
    TEST_ASSERT(IS_OK(s), "DeepstackGraph::Build succeeds");
    TEST_ASSERT(op.get() != nullptr, "DeepstackGraph returns non-null");

    if (op) {
        TEST_ASSERT(op.get()->GetInputNum() == 7,
                    "DeepstackGraph has 7 inputs");
    }

    LOG_INFO("DeepstackGraph test done");
}

// ══════════════════════════════════════════════════════════
// Test: Vision Model graph builds
// ══════════════════════════════════════════════════════════
void test_vision_model() {
    LOG_INFO("=== Test: VisionModel ===");

    atb_llm::models::VisionModel::Config cfg;
    cfg.hidden_size = 1280;
    cfg.num_heads = 16;
    cfg.depth = 24;
    cfg.in_channels = 3;
    cfg.temporal_patch_size = 2;
    cfg.patch_size = 14;
    cfg.spatial_merge_size = 2;

    atb_llm::models::VisionModel model(cfg);
    atb_llm::Status s = model.Build();
    TEST_ASSERT(IS_OK(s), "VisionModel::Build succeeds");

    TEST_ASSERT(model.GetFirstLayerGraph().get() != nullptr,
                "VisionModel first_layer_graph is non-null");
    TEST_ASSERT(model.GetBlockGraph().get() != nullptr,
                "VisionModel block_graph is non-null");
    TEST_ASSERT(model.GetMergerGraph().get() != nullptr,
                "VisionModel merger_graph is non-null");
    TEST_ASSERT(model.GetDeepstackGraph().get() != nullptr,
                "VisionModel deepstack_graph is non-null");

    // Verify first layer input count
    if (model.GetFirstLayerGraph()) {
        // 19 inputs: pixels, pe_w, pe_b, pos, c, s, seq, + 12 block weights
        TEST_ASSERT(model.GetFirstLayerGraph().get()->GetInputNum() == 19,
                    "VisionFirstLayer has 19 inputs");
    }

    LOG_INFO("VisionModel test done");
}

// ══════════════════════════════════════════════════════════
// Test: MRoPE (CPU)
// ══════════════════════════════════════════════════════════
void test_mrope() {
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
            if (v != 0.0f) { has_nonzero = true; break; }
        }
        TEST_ASSERT(has_nonzero, "MRoPE cos output is not all zeros");

        // cos(0) should be ~1.0
        TEST_ASSERT(std::fabs(cos_out[0] - 1.0f) < 0.01f,
                    "MRoPE cos(0) is approximately 1.0");

        // sin(0) should be ~0.0
        TEST_ASSERT(std::fabs(sin_out[0]) < 0.01f,
                    "MRoPE sin(0) is approximately 0.0");
    }

    // VisionRotaryEmbedding
    {
        atb_llm::components::VisionRotaryEmbedding vre(80);
        auto table = vre.ComputeFreqTable(48);
        TEST_ASSERT(!table.empty(), "VisionRotaryEmbedding freq table is not empty");
        TEST_ASSERT(table.size() == static_cast<size_t>(48 * 80),
                    "VisionRotaryEmbedding freq table size is 48*80");

        // freq_table[0] should be all zeros (pos=0)
        bool all_zero = true;
        for (int d = 0; d < 80; d++) {
            if (table[d] != 0.0f) { all_zero = false; break; }
        }
        TEST_ASSERT(all_zero, "VisionRotaryEmbedding freq_table[0] is all zeros");
    }

    // GetRopeIndex (text-only)
    {
        int64_t input_ids[] = {1, 2, 3, 4, 5};
        int64_t pos_ids[3 * 1 * 5];
        atb_llm::components::GetRopeIndex(
            input_ids, 1, 5, nullptr, 0, 151655, 2, pos_ids);

        // Text-only: all 3 dims should be sequential 0,1,2,3,4
        bool correct = true;
        for (int d = 0; d < 3; d++) {
            for (int s = 0; s < 5; s++) {
                if (pos_ids[d * 5 + s] != s) { correct = false; break; }
            }
        }
        TEST_ASSERT(correct, "GetRopeIndex(text-only) returns sequential positions");
    }

    LOG_INFO("MRoPE test done");
}

// ══════════════════════════════════════════════════════════
// Test: Vision component NPU execution
// ══════════════════════════════════════════════════════════
void test_vision_execute() {
    LOG_INFO("=== Test: Vision Component Execute on NPU ===");

    auto runtime = atb_llm::CreateRuntime(0, 5LL * 1024 * 1024 * 1024);
    TEST_ASSERT(runtime != nullptr, "Runtime created for vision tests");
    if (!runtime) return;

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    // Test VisionMLP execute
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::VisionMlpGraph::Build("ExecVisMLP", op);
        TEST_ASSERT(IS_OK(s), "VisionMlpGraph build for execute");
        TEST_ASSERT(op.get() != nullptr, "VisionMlpGraph non-null for execute");

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
            TEST_ASSERT(atb_s == atb::NO_ERROR, "VisionMLP Setup succeeds");

            if (atb_s == atb::NO_ERROR) {
                uint8_t* ws_ptr = nullptr;
                if (ws_size > 0) {
                    auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                    ws_ptr = ws;
                }
                atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
                TEST_ASSERT(atb_s == atb::NO_ERROR, "VisionMLP Execute succeeds");
            }

            runtime->Synchronize();

            // Verify output is non-zero
            std::vector<uint16_t> host_out(seq * hidden);
            alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
            bool non_zero = false;
            for (auto v : host_out) {
                if (v != 0) { non_zero = true; break; }
            }
            TEST_ASSERT(non_zero, "VisionMLP output is not all zeros");
        }
    }

    // Test PatchEmbed execute
    {
        atb_llm::OperationHandle op;
        atb_llm::Status s = atb_llm::components::PatchEmbedGraph::Build(
            "ExecPatchEmbed", 3, 2, 14, 64, op);
        TEST_ASSERT(IS_OK(s), "PatchEmbedGraph build for execute");

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
            TEST_ASSERT(atb_s == atb::NO_ERROR, "PatchEmbed Setup succeeds");

            if (atb_s == atb::NO_ERROR) {
                uint8_t* ws_ptr = nullptr;
                if (ws_size > 0) {
                    auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                    ws_ptr = ws;
                }
                atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
                TEST_ASSERT(atb_s == atb::NO_ERROR, "PatchEmbed Execute succeeds");
            }

            runtime->Synchronize();

            std::vector<uint16_t> host_out(num_patches * embed_dim);
            alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
            bool non_zero = false;
            for (auto v : host_out) {
                if (v != 0) { non_zero = true; break; }
            }
            TEST_ASSERT(non_zero, "PatchEmbed output is not all zeros");
        }
    }

    LOG_INFO("Vision Execute tests done");
}

// ══════════════════════════════════════════════════════════
// Test: LayerNorm NPU execution
// ══════════════════════════════════════════════════════════
void test_layer_norm_execute() {
    LOG_INFO("=== Test: LayerNorm Execute on NPU ===");

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    TEST_ASSERT(runtime != nullptr, "Runtime created for LayerNorm test");
    if (!runtime) return;

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    atb_llm::OperationHandle op = atb_llm::ops::LayerNormOp::Create(1e-5f);
    TEST_ASSERT(op.get() != nullptr, "LayerNormOp created");

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
        TEST_ASSERT(atb_s == atb::NO_ERROR, "LayerNorm Setup succeeds");

        if (atb_s == atb::NO_ERROR) {
            uint8_t* ws_ptr = nullptr;
            if (ws_size > 0) {
                auto [ws, ws_s] = runtime->GetWorkspace(ws_size);
                ws_ptr = ws;
            }
            atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
            TEST_ASSERT(atb_s == atb::NO_ERROR, "LayerNorm Execute succeeds");
        }

        runtime->Synchronize();

        std::vector<uint16_t> host_out(seq * hidden);
        alloc->CopyToHost(host_out.data(), output, host_out.size() * sizeof(uint16_t));
        bool non_zero = false;
        for (auto v : host_out) {
            if (v != 0) { non_zero = true; break; }
        }
        TEST_ASSERT(non_zero, "LayerNorm output is not all zeros");
    }

    LOG_INFO("LayerNorm Execute test done");
}

// ══════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    LOG_INFO("=== atb_cpp_llm_engine Phase 3 Tests ===");

    // Op creation
    test_new_op_creation();

    // Vision component graph builds
    test_vision_attention_graph();
    test_vision_mlp_graph();
    test_vision_block_graph();
    test_patch_embed_graph();
    test_vision_merger_graph();
    test_deepstack_graph();

    // Vision Model
    test_vision_model();

    // MRoPE (CPU)
    test_mrope();

    // NPU execution tests
    test_layer_norm_execute();
    test_vision_execute();

    LOG_INFO("=== Results: %d/%d passed ===", pass_count, test_count);

    if (pass_count == test_count) {
        LOG_INFO("ALL TESTS PASSED");
        return 0;
    } else {
        LOG_ERROR("SOME TESTS FAILED");
        return 1;
    }
}
