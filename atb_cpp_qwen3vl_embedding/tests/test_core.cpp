/**
 * Minimal test for Phase 1 core framework.
 * Tests: ContextManager, TensorAllocator, GraphBuilder, BufferPool, JsonConfig.
 *
 * Run: ./test_core
 * Requires: NPU device + ATB/ACL runtime.
 */

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/context_manager.h"
#include "core/tensor_allocator.h"
#include "core/graph_builder.h"
#include "core/buffer_pool.h"
#include "io/json_config.h"
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>

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

// Use helper to avoid OK ambiguity with aclnnStatus from acl_meta.h
#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

// ── Test: ContextManager ─────────────────────────────────
void test_context_manager() {
    LOG_INFO("=== Test: ContextManager ===");
    std::unique_ptr<atb_llm::ContextManager> mgr;
    atb_llm::Status s = atb_llm::ContextManager::Create(0, mgr);
    TEST_ASSERT(IS_OK(s), "ContextManager::Create succeeds");
    TEST_ASSERT(mgr != nullptr, "ContextManager created");
    TEST_ASSERT(mgr->GetContext() != nullptr, "Context created");
    TEST_ASSERT(mgr->GetStream() != nullptr, "Stream created");
    TEST_ASSERT(IS_OK(mgr->Synchronize()), "Synchronize succeeds");
    TEST_ASSERT(mgr->GetDeviceId() == 0, "Device ID is 0");
}

// ── Test: TensorAllocator ────────────────────────────────
void test_tensor_allocator() {
    LOG_INFO("=== Test: TensorAllocator ===");
    std::unique_ptr<atb_llm::ContextManager> mgr;
    atb_llm::ContextManager::Create(0, mgr);
    atb_llm::TensorAllocator alloc(mgr->GetContext(), mgr->GetStream());

    // Allocate float16 tensor [2, 3]
    atb::Tensor t1;
    atb_llm::Status s = alloc.AllocFloat16(t1, {2, 3});
    TEST_ASSERT(IS_OK(s), "AllocFloat16 succeeds");
    TEST_ASSERT(t1.deviceData != nullptr, "Float16 tensor has device data");
    TEST_ASSERT(t1.desc.dtype == ACL_FLOAT16, "Float16 dtype correct");
    TEST_ASSERT(t1.desc.shape.dimNum == 2, "Float16 shape ndim=2");
    TEST_ASSERT(t1.desc.shape.dims[0] == 2, "Float16 shape[0]=2");
    TEST_ASSERT(t1.desc.shape.dims[1] == 3, "Float16 shape[1]=3");

    // Allocate float32 tensor [4]
    atb::Tensor t2;
    s = alloc.AllocFloat32(t2, {4});
    TEST_ASSERT(IS_OK(s), "AllocFloat32 succeeds");
    TEST_ASSERT(t2.desc.dtype == ACL_FLOAT, "Float32 dtype correct");

    // Host -> Device -> Host roundtrip
    float host_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_dst[4] = {0};
    s = alloc.CopyToDevice(t2, host_src, sizeof(host_src));
    TEST_ASSERT(IS_OK(s), "CopyToDevice succeeds");
    s = alloc.CopyToHost(host_dst, t2, sizeof(host_dst));
    TEST_ASSERT(IS_OK(s), "CopyToHost succeeds");
    TEST_ASSERT(host_dst[0] == 1.0f && host_dst[3] == 4.0f, "H2D roundtrip data correct");

    TEST_ASSERT(alloc.NumTracked() == 2, "2 tensors tracked");
    alloc.FreeAll();
    TEST_ASSERT(alloc.NumTracked() == 0, "All tensors freed");
}

// ── Test: GraphBuilder ───────────────────────────────────
void test_graph_builder() {
    LOG_INFO("=== Test: GraphBuilder ===");
    std::unique_ptr<atb_llm::ContextManager> mgr;
    atb_llm::ContextManager::Create(0, mgr);

    // Create a minimal graph with no operations (just passthrough)
    std::unique_ptr<atb_llm::GraphBuilder> builder;
    atb_llm::Status s = atb_llm::GraphBuilder::Create("TestGraph", builder);
    TEST_ASSERT(IS_OK(s), "GraphBuilder::Create succeeds");
    TEST_ASSERT(builder != nullptr, "GraphBuilder created");
    TEST_ASSERT(builder->GetBuilder() != nullptr, "GraphBuilder has underlying builder");

    // Initialize with 1 input, 1 output, no infer shape func
    atb::SVector<std::string> in_names = {"input"};
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init("TestOp", nullptr, in_names, out_names);
    TEST_ASSERT(IS_OK(s), "GraphBuilder Init succeeds");

    // Build returns an OperationHandle
    atb_llm::OperationHandle op = builder->Build();
    LOG_INFO("GraphBuilder Build: op=%p", static_cast<void*>(op.get()));
}

// ── Test: BufferPool ─────────────────────────────────────
void test_buffer_pool() {
    LOG_INFO("=== Test: BufferPool ===");
    atb_llm::BufferPool pool;

    // Set initial size
    atb_llm::Status s = pool.SetBufferSize(1024 * 1024);  // 1MB
    TEST_ASSERT(IS_OK(s), "SetBufferSize 1MB succeeds");
    TEST_ASSERT(pool.GetSize() >= 1024 * 1024, "Buffer size >= 1MB");

    // Get workspace within existing buffer
    auto [ws, ws_s] = pool.GetWorkspace(512);
    TEST_ASSERT(IS_OK(ws_s), "GetWorkspace(512) succeeds");
    TEST_ASSERT(ws != nullptr, "GetWorkspace(512) returns non-null");

    // Get workspace larger than current (triggers grow)
    auto [ws2, ws2_s] = pool.GetWorkspace(2 * 1024 * 1024);  // 2MB
    TEST_ASSERT(IS_OK(ws2_s), "GetWorkspace(2MB) succeeds");
    TEST_ASSERT(ws2 != nullptr, "GetWorkspace(2MB) grows and returns non-null");
    TEST_ASSERT(pool.GetSize() >= 2 * 1024 * 1024, "Buffer grew to >= 2MB");

    pool.Free();
    TEST_ASSERT(pool.GetSize() == 0, "Buffer freed");
}

// ── Test: JsonConfig ─────────────────────────────────────
void test_json_config() {
    LOG_INFO("=== Test: JsonConfig ===");

    const char* json_str = R"({
        "model_type": "qwen3vl_embedding",
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
        "use_silu": true,
        "head_dims": [64, 128],
        "vision_config": {
            "depth": 24,
            "hidden_size": 1280
        }
    })";

    atb_llm::JsonConfig cfg = atb_llm::JsonConfig::Parse(json_str);
    TEST_ASSERT(cfg.IsValid(), "JsonConfig parsed successfully");

    TEST_ASSERT(cfg.GetString("model_type") == "qwen3vl_embedding", "GetString works");
    TEST_ASSERT(cfg.GetInt("hidden_size") == 1536, "GetInt works");
    TEST_ASSERT(cfg.GetInt("nonexistent", 42) == 42, "GetInt default works");
    TEST_ASSERT(cfg.GetBool("use_silu") == true, "GetBool works");
    TEST_ASSERT(cfg.HasKey("hidden_size") == true, "HasKey true");
    TEST_ASSERT(cfg.HasKey("nonexistent") == false, "HasKey false");

    std::vector<int> head_dims = cfg.GetIntArray("head_dims");
    TEST_ASSERT(head_dims.size() == 2, "GetIntArray size=2");
    TEST_ASSERT(head_dims[0] == 64 && head_dims[1] == 128, "GetIntArray values correct");

    atb_llm::JsonConfig sub = cfg.GetSubConfig("vision_config");
    TEST_ASSERT(sub.IsValid(), "GetSubConfig valid");
    TEST_ASSERT(sub.GetInt("depth") == 24, "SubConfig GetInt works");
}

// ── Test: RAII (OperationHandle) ─────────────────────────
void test_raii_handles() {
    LOG_INFO("=== Test: RAII Handles ===");

    // OperationHandle with null
    {
        atb_llm::OperationHandle h;
        TEST_ASSERT(!h, "Null OperationHandle is falsy");
        TEST_ASSERT(h.get() == nullptr, "Null OperationHandle get() == nullptr");
    }

    // OperationHandle move
    {
        atb_llm::OperationHandle h1;
        atb_llm::OperationHandle h2(std::move(h1));
        TEST_ASSERT(!h1, "Moved-from handle is null");
        TEST_ASSERT(!h2, "Moved-to handle is null (was already null)");
    }

    LOG_INFO("RAII handle tests passed");
}

// ── Test: CreateRuntime ──────────────────────────────────
void test_create_runtime() {
    LOG_INFO("=== Test: CreateRuntime ===");
    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);  // 2GB buffer
    TEST_ASSERT(runtime != nullptr, "Runtime created");
    TEST_ASSERT(runtime->GetContext() != nullptr, "Runtime context non-null");
    TEST_ASSERT(runtime->GetStream() != nullptr, "Runtime stream non-null");
    TEST_ASSERT(runtime->GetAllocator() != nullptr, "Runtime allocator non-null");
    TEST_ASSERT(runtime->GetWeightLoader() != nullptr, "Runtime weight loader non-null");

    // Test Synchronize
    atb_llm::Status s = runtime->Synchronize();
    TEST_ASSERT(IS_OK(s), "Runtime Synchronize succeeds");

    // Test GraphBuilder::Create
    std::unique_ptr<atb_llm::GraphBuilder> gb;
    atb_llm::Status gb_s = atb_llm::GraphBuilder::Create("TestBuilder", gb);
    TEST_ASSERT(IS_OK(gb_s), "GraphBuilder::Create succeeds");
    TEST_ASSERT(gb != nullptr, "GraphBuilder created");

    // Test GetWorkspace
    auto [ws, ws_status] = runtime->GetWorkspace(1024);
    TEST_ASSERT(IS_OK(ws_status), "GetWorkspace succeeds");
    TEST_ASSERT(ws != nullptr, "GetWorkspace returns non-null");
}

// ── Main ─────────────────────────────────────────────────
int main(int argc, char** argv) {
    LOG_INFO("=== atb_cpp_llm_engine Phase 1 Tests ===");

    test_context_manager();
    test_tensor_allocator();
    test_graph_builder();
    test_buffer_pool();
    test_json_config();
    test_raii_handles();
    test_create_runtime();

    LOG_INFO("=== Results: %d/%d passed ===", pass_count, test_count);

    if (pass_count == test_count) {
        LOG_INFO("ALL TESTS PASSED");
        return 0;
    } else {
        LOG_ERROR("SOME TESTS FAILED");
        return 1;
    }
}
