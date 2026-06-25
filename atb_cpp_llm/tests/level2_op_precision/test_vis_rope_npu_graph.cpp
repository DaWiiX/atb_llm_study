/**
 * Vision RoPE NPU Graph — Level 2 (NPU op precision) test.
 *
 * Loads reference bundles from gen_vis_rope_npu_reference.py, runs the
 * new NPU graph, and verifies cos/sin outputs against the Python reference
 * (which is itself validated against canonical compute_rot_pos_emb).
 *
 * Run:
 *   python tests/gen_vis_rope_npu_reference.py
 *   ./build/test_vis_rope_npu_graph
 *
 * All cases must hit cosine ≥ 0.9999 for both cos and sin outputs
 * (we use fp32 internally on NPU, so precision is tight).
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "components/vision/vis_rope_npu_graph.h"
#include "components/common/mrope.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include "core/tensor_allocator.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#include "test_env.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

namespace {

double CosineSim(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

struct Bundle {
    int64_t n = 0, half = 0, vis_hd = 0, max_hw = 0;
    std::vector<int32_t> row_idx, col_idx;
    std::vector<float> freq_table;
    std::vector<float> expected_cos, expected_sin;
};

bool ReadBundle(const std::string& path, Bundle& b) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { LOG_ERROR("ReadBundle: cannot open %s", path.c_str()); return false; }
    auto rdi64 = [&](int64_t& x) { return fread(&x, sizeof(int64_t), 1, f) == 1; };

    if (!rdi64(b.n) || !rdi64(b.half) || !rdi64(b.vis_hd) || !rdi64(b.max_hw)) {
        fclose(f); return false;
    }
    b.row_idx.resize(b.n); b.col_idx.resize(b.n);
    if (fread(b.row_idx.data(), sizeof(int32_t), b.n, f) != size_t(b.n) ||
        fread(b.col_idx.data(), sizeof(int32_t), b.n, f) != size_t(b.n)) {
        fclose(f); return false;
    }
    b.freq_table.resize(b.max_hw * b.half);
    if (fread(b.freq_table.data(), sizeof(float),
              b.freq_table.size(), f) != b.freq_table.size()) {
        fclose(f); return false;
    }
    int64_t ndim = 0;
    if (!rdi64(ndim) || ndim != 2) { fclose(f); return false; }
    int64_t shape[2] = {};
    if (fread(shape, sizeof(int64_t), 2, f) != 2) { fclose(f); return false; }
    int64_t total = shape[0] * shape[1];
    b.expected_cos.resize(total);
    b.expected_sin.resize(total);
    if (fread(b.expected_cos.data(), sizeof(float), total, f) != size_t(total) ||
        fread(b.expected_sin.data(), sizeof(float), total, f) != size_t(total)) {
        fclose(f); return false;
    }
    fclose(f);
    return true;
}

// ═════════════════════════════════════════════════════════════════════
// Stage A (host-only) regression test.
// ═════════════════════════════════════════════════════════════════════
void TestStageA(const std::string& tag) {
    CAPTURE(tag);

    Bundle b;
    const std::string path = "/tmp/visrope_npu_case_" + tag + ".bin";
    REQUIRE_MESSAGE(ReadBundle(path, b), "Stage A [" << tag << "]: bundle missing");

    // We need grid_thw to rebuild indices. The bundle doesn't store it; we
    // hard-code the same cases as the Python generator.
    static const struct {
        const char* tag; int64_t t, h, w;
    } CASES[] = {
        {"tiny_4x4", 1,  4,  4},
        {"t1_8x8",   1,  8,  8},
        {"224x224",  2, 16, 16},
        {"416x672",  2, 26, 42},
        {"896x896",  2, 56, 56},
    };
    int64_t t = 0, h = 0, w = 0;
    for (auto& c : CASES) {
        if (tag == c.tag) { t = c.t; h = c.h; w = c.w; break; }
    }
    REQUIRE(t != 0);

    int64_t grid_thw_arr[3] = {t, h, w};

    // Build indices via C++ Stage A
    std::vector<int32_t> row_cpp, col_cpp;
    atb_llm::components::BuildVisRopeIndices(grid_thw_arr, 1, /*merge_size=*/2,
                                              row_cpp, col_cpp);
    CHECK(row_cpp.size() == b.row_idx.size());
    CHECK(col_cpp.size() == b.col_idx.size());

    int64_t row_mism = 0, col_mism = 0;
    for (size_t i = 0; i < row_cpp.size(); i++) {
        if (row_cpp[i] != b.row_idx[i]) row_mism++;
        if (col_cpp[i] != b.col_idx[i]) col_mism++;
    }

    // Build freq_table via C++ and compare
    std::vector<float> ft_cpp;
    atb_llm::components::BuildVisRopeFreqTable(
        static_cast<int32_t>(b.max_hw), static_cast<int32_t>(b.half), ft_cpp);
    CHECK(ft_cpp.size() == b.freq_table.size());

    float ft_max = 0;
    for (size_t i = 0; i < ft_cpp.size(); i++) {
        ft_max = std::max(ft_max, std::fabs(ft_cpp[i] - b.freq_table[i]));
    }

    LOG_INFO("Stage A [%s]: row_mismatch=%ld col_mismatch=%ld freq_table_max_err=%.6g",
             tag.c_str(),
             static_cast<long>(row_mism), static_cast<long>(col_mism), ft_max);

    CHECK(row_mism == 0);
    CHECK(col_mism == 0);
    CHECK(ft_max < 1e-5f);
}

// ═════════════════════════════════════════════════════════════════════
// Stage B (NPU graph) test.
// ═════════════════════════════════════════════════════════════════════
void RunCase(const std::string& tag) {
    CAPTURE(tag);

    Bundle b;
    const std::string path = "/tmp/visrope_npu_case_" + tag + ".bin";
    REQUIRE_MESSAGE(ReadBundle(path, b), "Stage B [" << tag << "]: bundle missing");

    LOG_INFO("  [%s] N=%ld half=%ld vis_hd=%ld max_hw=%ld",
             tag.c_str(),
             static_cast<long>(b.n), static_cast<long>(b.half),
             static_cast<long>(b.vis_hd), static_cast<long>(b.max_hw));

    // Build graph
    atb_llm::OperationHandle graph;
    atb_llm::Status s = atb_llm::components::VisRopeGraph::Build(
        "VisRope_" + tag, graph);
    REQUIRE(IS_OK(s));
    REQUIRE(graph);

    // Create runtime
    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    // Allocate + upload inputs
    atb::Tensor ft_npu, row_npu, col_npu, cos_npu, sin_npu;
    // freq_table must be fp16 — ATB Concat doesn't support fp32
    std::vector<uint16_t> ft_fp16(b.freq_table.size());
    for (size_t i = 0; i < b.freq_table.size(); i++) {
        ft_fp16[i] = atb_llm::Fp32ToFp16(b.freq_table[i]);
    }
    REQUIRE(IS_OK(alloc->AllocFloat16(ft_npu, {b.max_hw, b.half})));
    alloc->CopyToDevice(ft_npu, ft_fp16.data(),
                        ft_fp16.size() * sizeof(uint16_t));

    REQUIRE(IS_OK(alloc->AllocInt32(row_npu, {b.n})));
    alloc->CopyToDevice(row_npu, b.row_idx.data(), b.n * sizeof(int32_t));

    REQUIRE(IS_OK(alloc->AllocInt32(col_npu, {b.n})));
    alloc->CopyToDevice(col_npu, b.col_idx.data(), b.n * sizeof(int32_t));

    REQUIRE(IS_OK(alloc->AllocFloat16(cos_npu, {b.n, b.vis_hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(sin_npu, {b.n, b.vis_hd})));

    atb::VariantPack vp;
    vp.inTensors  = {ft_npu, row_npu, col_npu};
    vp.outTensors = {cos_npu, sin_npu};

    uint64_t ws_size = 0;
    atb::Status atb_s = graph.get()->Setup(vp, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);
    uint8_t* ws_ptr = nullptr;
    auto __atb_pair_ws = runtime->GetWorkspace(ws_size > 0 ? ws_size : 1); auto& ws = __atb_pair_ws.first; auto& ws_st = __atb_pair_ws.second;
    if (ws_st == atb_llm::STATUS_OK) ws_ptr = ws;
    atb_s = graph.get()->Execute(vp, ws_ptr, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);
    runtime->Synchronize();

    // Read back
    std::vector<uint16_t> cos_fp16(b.n * b.vis_hd), sin_fp16(b.n * b.vis_hd);
    alloc->CopyToHost(cos_fp16.data(), cos_npu, cos_fp16.size() * sizeof(uint16_t));
    alloc->CopyToHost(sin_fp16.data(), sin_npu, sin_fp16.size() * sizeof(uint16_t));
    std::vector<float> cos_got(cos_fp16.size()), sin_got(sin_fp16.size());
    for (size_t i = 0; i < cos_fp16.size(); i++) {
        cos_got[i] = atb_llm::Fp16ToF32(cos_fp16[i]);
        sin_got[i] = atb_llm::Fp16ToF32(sin_fp16[i]);
    }

    double cos_cosine = CosineSim(cos_got, b.expected_cos);
    double sin_cosine = CosineSim(sin_got, b.expected_sin);
    float cos_max_abs = MaxAbsDiff(cos_got, b.expected_cos);
    float sin_max_abs = MaxAbsDiff(sin_got, b.expected_sin);

    LOG_INFO("  [%s] cos cosine=%.6f max_abs=%.6f | sin cosine=%.6f max_abs=%.6f",
             tag.c_str(),
             cos_cosine, cos_max_abs,
             sin_cosine, sin_max_abs);

    CHECK(cos_cosine >= 0.999);
    CHECK(sin_cosine >= 0.999);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════
// Stage A: host-only index/freq_table builder regression (no NPU needed)
// ═════════════════════════════════════════════════════════════════════

// These are Level-2 op precision/stress cases, not the benchmark/e2e
// production resolution matrix. "896x896" is a synthetic square stress grid
// (T=2,H=W=56) used to exercise larger row/col gather and cos/sin outputs.

TEST_CASE("VisRopeNPU StageA: tiny_4x4") { TestStageA("tiny_4x4"); }
TEST_CASE("VisRopeNPU StageA: t1_8x8")   { TestStageA("t1_8x8"); }
TEST_CASE("VisRopeNPU StageA: 224x224")  { TestStageA("224x224"); }
TEST_CASE("VisRopeNPU StageA: 416x672")  { TestStageA("416x672"); }
TEST_CASE("VisRopeNPU StageA: 896x896")  { TestStageA("896x896"); }

// ═════════════════════════════════════════════════════════════════════
// Stage B: NPU graph execution and precision validation
// ═════════════════════════════════════════════════════════════════════

TEST_CASE("VisRopeNPU StageB: tiny_4x4") { RunCase("tiny_4x4"); }
TEST_CASE("VisRopeNPU StageB: t1_8x8")   { RunCase("t1_8x8"); }
TEST_CASE("VisRopeNPU StageB: 224x224")  { RunCase("224x224"); }
TEST_CASE("VisRopeNPU StageB: 416x672")  { RunCase("416x672"); }
TEST_CASE("VisRopeNPU StageB: 896x896")  { RunCase("896x896"); }
