/**
 * Minimal test for Phase 1 core framework.
 * Tests: ContextManager, TensorAllocator, GraphBuilder, BufferPool, JsonConfig.
 *
 * Run: ./test_core
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/context_manager.h"
#include "core/tensor_allocator.h"
#include "core/npu_tensor.h"
#include "core/graph_builder.h"
#include "core/buffer_pool.h"
#include "io/json_config.h"
#include "engine/runtime_impl.h"
#include "log/logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Use helper to avoid OK ambiguity with aclnnStatus from acl_meta.h
#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

// ── Test: ContextManager ─────────────────────────────────
TEST_CASE("ContextManager") {
    LOG_INFO("=== Test: ContextManager ===");
    std::unique_ptr<atb_llm::ContextManager> mgr;
    atb_llm::Status s = atb_llm::ContextManager::Create(0, mgr);
    CHECK(IS_OK(s));
    CHECK(mgr != nullptr);
    CHECK(mgr->GetContext() != nullptr);
    CHECK(mgr->GetStream() != nullptr);
    CHECK(IS_OK(mgr->Synchronize()));
    CHECK(mgr->GetDeviceId() == 0);
}

// ── Test: TensorAllocator ────────────────────────────────
TEST_CASE("TensorAllocator") {
    LOG_INFO("=== Test: TensorAllocator ===");
    std::unique_ptr<atb_llm::ContextManager> mgr;
    atb_llm::ContextManager::Create(0, mgr);
    atb_llm::TensorAllocator alloc(mgr->GetContext(), mgr->GetStream());

    // Allocate float16 tensor [2, 3]
    atb::Tensor t1;
    atb_llm::Status s = alloc.AllocFloat16(t1, {2, 3});
    CHECK(IS_OK(s));
    CHECK(t1.deviceData != nullptr);
    CHECK(t1.desc.dtype == ACL_FLOAT16);
    CHECK(t1.desc.shape.dimNum == 2);
    CHECK(t1.desc.shape.dims[0] == 2);
    CHECK(t1.desc.shape.dims[1] == 3);

    // Allocate float32 tensor [4]
    atb::Tensor t2;
    s = alloc.AllocFloat32(t2, {4});
    CHECK(IS_OK(s));
    CHECK(t2.desc.dtype == ACL_FLOAT);

    // Host -> Device -> Host roundtrip
    float host_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_dst[4] = {0};
    s = alloc.CopyToDevice(t2, host_src, sizeof(host_src));
    CHECK(IS_OK(s));
    s = alloc.CopyToHost(host_dst, t2, sizeof(host_dst));
    CHECK(IS_OK(s));
    CHECK(host_dst[0] == 1.0f);
    CHECK(host_dst[3] == 4.0f);

    CHECK(alloc.NumTracked() == 2);
    alloc.FreeAll();
    CHECK(alloc.NumTracked() == 0);
}

// ── Test: GraphBuilder ───────────────────────────────────
TEST_CASE("GraphBuilder") {
    LOG_INFO("=== Test: GraphBuilder ===");
    std::unique_ptr<atb_llm::ContextManager> mgr;
    atb_llm::ContextManager::Create(0, mgr);

    // Create a minimal graph with no operations (just passthrough)
    std::unique_ptr<atb_llm::GraphBuilder> builder;
    atb_llm::Status s = atb_llm::GraphBuilder::Create("TestGraph", builder);
    CHECK(IS_OK(s));
    CHECK(builder != nullptr);
    CHECK(builder->GetBuilder() != nullptr);

    // Initialize with 1 input, 1 output, no infer shape func
    atb::SVector<std::string> in_names = {"input"};
    atb::SVector<std::string> out_names = {"output"};
    s = builder->Init("TestOp", nullptr, in_names, out_names);
    CHECK(IS_OK(s));

    // Build returns an OperationHandle
    atb_llm::OperationHandle op = builder->Build();
    LOG_INFO("GraphBuilder Build: op=%p", static_cast<void*>(op.get()));
}

// ── Test: BufferPool ─────────────────────────────────────
TEST_CASE("BufferPool") {
    LOG_INFO("=== Test: BufferPool ===");
    atb_llm::BufferPool pool;

    // Set initial size
    atb_llm::Status s = pool.SetBufferSize(1024 * 1024);  // 1MB
    CHECK(IS_OK(s));
    CHECK(pool.GetSize() >= 1024 * 1024);

    // Get workspace within existing buffer
    auto __atb_pair_ws = pool.GetWorkspace(512); auto& ws = __atb_pair_ws.first; auto& ws_s = __atb_pair_ws.second;
    CHECK(IS_OK(ws_s));
    CHECK(ws != nullptr);

    // Get workspace larger than current (triggers grow)
    auto __atb_pair_ws2 = pool.GetWorkspace(2 * 1024 * 1024); auto& ws2 = __atb_pair_ws2.first; auto& ws2_s = __atb_pair_ws2.second;  // 2MB
    CHECK(IS_OK(ws2_s));
    CHECK(ws2 != nullptr);
    CHECK(pool.GetSize() >= 2 * 1024 * 1024);

    pool.Free();
    CHECK(pool.GetSize() == 0);
}

// ── Test: JsonConfig ─────────────────────────────────────
TEST_CASE("JsonConfig") {
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
    CHECK(cfg.IsValid());

    CHECK(cfg.GetString("model_type") == "qwen3vl_embedding");
    CHECK(cfg.GetInt("hidden_size") == 1536);
    CHECK(cfg.GetInt("nonexistent", 42) == 42);
    CHECK(cfg.GetBool("use_silu") == true);
    CHECK(cfg.HasKey("hidden_size") == true);
    CHECK(cfg.HasKey("nonexistent") == false);

    std::vector<int> head_dims = cfg.GetIntArray("head_dims");
    CHECK(head_dims.size() == 2);
    CHECK(head_dims[0] == 64);
    CHECK(head_dims[1] == 128);

    atb_llm::JsonConfig sub = cfg.GetSubConfig("vision_config");
    CHECK(sub.IsValid());
    CHECK(sub.GetInt("depth") == 24);
}

// ── Test: RAII (OperationHandle) ─────────────────────────
TEST_CASE("RAII Handles") {
    LOG_INFO("=== Test: RAII Handles ===");

    // OperationHandle with null
    {
        atb_llm::OperationHandle h;
        CHECK(!h);
        CHECK(h.get() == nullptr);
    }

    // OperationHandle move
    {
        atb_llm::OperationHandle h1;
        atb_llm::OperationHandle h2(std::move(h1));
        CHECK(!h1);
        CHECK(!h2);
    }

    LOG_INFO("RAII handle tests passed");
}

// ── Test: CreateRuntime ──────────────────────────────────
TEST_CASE("CreateRuntime") {
    LOG_INFO("=== Test: CreateRuntime ===");
    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);  // 2GB buffer
    REQUIRE(runtime != nullptr);
    CHECK(runtime->GetContext() != nullptr);
    CHECK(runtime->GetStream() != nullptr);
    CHECK(runtime->GetAllocator() != nullptr);
    CHECK(runtime->GetWeightLoader() != nullptr);

    // Test Synchronize
    atb_llm::Status s = runtime->Synchronize();
    CHECK(IS_OK(s));

    // Test GraphBuilder::Create
    std::unique_ptr<atb_llm::GraphBuilder> gb;
    atb_llm::Status gb_s = atb_llm::GraphBuilder::Create("TestBuilder", gb);
    CHECK(IS_OK(gb_s));
    CHECK(gb != nullptr);

    // Test GetWorkspace
    auto __atb_pair_ws = runtime->GetWorkspace(1024); auto& ws = __atb_pair_ws.first; auto& ws_status = __atb_pair_ws.second;
    CHECK(IS_OK(ws_status));
    CHECK(ws != nullptr);
}

// ── Test: CreateRuntime with invalid device_id ───────────
TEST_CASE("CreateRuntime invalid device_id") {
    LOG_INFO("=== Test: CreateRuntime invalid device_id ===");
    auto runtime = atb_llm::CreateRuntime(999, 0);
    CHECK(runtime == nullptr);
}

// ── Test: NpuTensor RAII ─────────────────────────────────
TEST_CASE("NpuTensor RAII") {
    LOG_INFO("=== Test: NpuTensor RAII ===");

    // Default construction: null tensor
    {
        atb_llm::NpuTensor t;
        CHECK(!t);
        CHECK(t.Get() == nullptr);
    }

    // Allocate float16 [2, 3]
    {
        atb_llm::NpuTensor t = atb_llm::AllocNpuFloat16({2, 3});
        CHECK(static_cast<bool>(t));
        CHECK(t.Get() != nullptr);
        CHECK(t.Get()->deviceData != nullptr);
        CHECK(t.Get()->desc.dtype == ACL_FLOAT16);
        CHECK(t.Get()->desc.shape.dimNum == 2);
        CHECK(t.Get()->desc.shape.dims[0] == 2);
        CHECK(t.Get()->desc.shape.dims[1] == 3);
        // Destructor frees here
    }

    // Allocate float32 [4]
    {
        atb_llm::NpuTensor t = atb_llm::AllocNpuFloat32({4});
        CHECK(static_cast<bool>(t));
        CHECK(t.Get()->desc.dtype == ACL_FLOAT);
    }

    // Allocate int64 [8]
    {
        atb_llm::NpuTensor t = atb_llm::AllocNpuInt64({8});
        CHECK(static_cast<bool>(t));
        CHECK(t.Get()->desc.dtype == ACL_INT64);
    }

    // Move construction
    {
        atb_llm::NpuTensor t1 = atb_llm::AllocNpuFloat16({4, 4});
        void* raw_ptr = t1.Get()->deviceData;
        CHECK(t1.Get() != nullptr);

        atb_llm::NpuTensor t2(std::move(t1));
        CHECK(!t1);
        CHECK(t1.Get() == nullptr);
        CHECK(static_cast<bool>(t2));
        CHECK(t2.Get()->deviceData == raw_ptr);
    }

    // Move assignment
    {
        atb_llm::NpuTensor t1 = atb_llm::AllocNpuFloat32({16});
        atb_llm::NpuTensor t2;
        t2 = std::move(t1);
        CHECK(!t1);
        CHECK(static_cast<bool>(t2));
    }

    // Release
    {
        atb_llm::NpuTensor t = atb_llm::AllocNpuFloat16({2, 2});
        atb::Tensor* released = t.Release();
        CHECK(released != nullptr);
        CHECK(released->deviceData != nullptr);
        CHECK(!t);
        CHECK(t.Get() == nullptr);
        // Clean up the released tensor manually
        aclrtFree(released->deviceData);
    }

    // Invalid shape: empty
    {
        atb_llm::NpuTensor t = atb_llm::AllocNpuFloat16({});
        CHECK(!t);
    }

    // Invalid shape: zero dimension
    {
        atb_llm::NpuTensor t = atb_llm::AllocNpuFloat16({0, 3});
        CHECK(!t);
    }

    // Move-assign to self (no-op)
    {
        atb_llm::NpuTensor t = atb_llm::AllocNpuFloat16({2});
        atb::Tensor* ptr_before = t.Get();
        (void)ptr_before;
        t = std::move(t);  // NOLINT: intentional self-move
        // Self-move should not crash; state is valid but unspecified
        CHECK(true);
    }
}
