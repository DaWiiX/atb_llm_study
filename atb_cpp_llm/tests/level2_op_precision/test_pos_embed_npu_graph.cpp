/**
 * Vision PosEmbed NPU Graph — Level 2 (NPU op precision) test.
 *
 * Loads reference bundles produced by gen_pos_embed_npu_reference.py,
 * runs the new NPU graph end-to-end, and validates output against the
 * Python (Stage A + Stage B) reference, which is itself validated against
 * the canonical fast_pos_embed_interpolate.
 *
 * Prerequisite:
 *   python tests/gen_pos_embed_npu_reference.py
 *
 * Cases (each must hit cosine ≥ 0.999):
 *   - tiny_4x4    (1 frame, 4x4 grid)
 *   - t1_8x8      (1 frame, 8x8 grid)
 *   - 224x224     (2 frames, 16x16 grid)
 *   - 416x672     (2 frames, 26x42 grid)
 *   - 896x896     (2 frames, 56x56 grid)
 *
 * Run:  ./test_pos_embed_npu_graph
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "components/vision/pos_embed_npu_graph.h"
#include "components/vision/pos_embed_interp.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "core/context_manager.h"
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
#include <algorithm>
#include <memory>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

namespace {

// ─── Cosine similarity ────────────────────────────────────────
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

// ─── Binary readers (match writers in gen_pos_embed_npu_reference.py) ──
bool ReadAll(FILE* f, void* dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

// Read the pos_embed_w table:  [ndim:i64, shape:i64*ndim, fp16-bytes]
bool ReadPosEmbedTable(const std::string& path,
                       std::vector<uint16_t>& data,
                       int64_t& num_rows, int64_t& vis_hs) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERROR("ReadPosEmbedTable: cannot open %s", path.c_str());
        return false;
    }
    int64_t ndim = 0;
    if (!ReadAll(f, &ndim, sizeof(int64_t)) || ndim != 2) {
        LOG_ERROR("ReadPosEmbedTable: bad ndim=%ld", static_cast<long>(ndim));
        fclose(f); return false;
    }
    int64_t shape[2] = {};
    if (!ReadAll(f, shape, sizeof(int64_t) * 2)) {
        fclose(f); return false;
    }
    num_rows = shape[0];
    vis_hs   = shape[1];
    data.resize(num_rows * vis_hs);
    if (!ReadAll(f, data.data(), sizeof(uint16_t) * data.size())) {
        fclose(f); return false;
    }
    fclose(f);
    return true;
}

// Bundle layout (see gen_pos_embed_npu_reference.py):
//   i64 num_grid, vis_hs, merge_size, num_images
//   i64 grid_thw[num_images*3]
//   i64 idx_count
//   i32 idx00..idx11 [idx_count] each
//   fp16 wt00..wt11 [idx_count] each
//   i64 expected_ndim
//   i64 expected_shape[expected_ndim]
//   fp16 expected[ ... ]
struct Bundle {
    int64_t num_grid = 0;
    int64_t vis_hs = 0;
    int64_t merge_size = 0;
    int64_t num_images = 0;
    std::vector<int64_t> grid_thw;       // num_images * 3
    int64_t idx_count = 0;
    std::vector<int32_t> idx[4];
    std::vector<uint16_t> wt[4];
    std::vector<int64_t> expected_shape;
    std::vector<uint16_t> expected;       // fp16
};

bool ReadBundle(const std::string& path, Bundle& b) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { LOG_ERROR("ReadBundle: cannot open %s", path.c_str()); return false; }
    auto rd_i64 = [&](int64_t& v) { return ReadAll(f, &v, sizeof(int64_t)); };

    if (!rd_i64(b.num_grid)   ||
        !rd_i64(b.vis_hs)     ||
        !rd_i64(b.merge_size) ||
        !rd_i64(b.num_images)) {
        fclose(f); return false;
    }
    b.grid_thw.resize(b.num_images * 3);
    if (!ReadAll(f, b.grid_thw.data(), sizeof(int64_t) * b.grid_thw.size())) {
        fclose(f); return false;
    }
    if (!rd_i64(b.idx_count)) { fclose(f); return false; }
    for (int k = 0; k < 4; k++) {
        b.idx[k].resize(b.idx_count);
        if (!ReadAll(f, b.idx[k].data(), sizeof(int32_t) * b.idx_count)) {
            fclose(f); return false;
        }
    }
    for (int k = 0; k < 4; k++) {
        b.wt[k].resize(b.idx_count);
        if (!ReadAll(f, b.wt[k].data(), sizeof(uint16_t) * b.idx_count)) {
            fclose(f); return false;
        }
    }
    int64_t ndim = 0;
    if (!rd_i64(ndim) || ndim < 1) { fclose(f); return false; }
    b.expected_shape.resize(ndim);
    if (!ReadAll(f, b.expected_shape.data(), sizeof(int64_t) * ndim)) {
        fclose(f); return false;
    }
    int64_t total = 1;
    for (auto d : b.expected_shape) total *= d;
    b.expected.resize(total);
    if (!ReadAll(f, b.expected.data(), sizeof(uint16_t) * total)) {
        fclose(f); return false;
    }
    fclose(f);
    return true;
}

// ─── Convert fp16 vector → f32 vector ───
std::vector<float> Fp16VecToF32(const std::vector<uint16_t>& src) {
    std::vector<float> out(src.size());
    for (size_t i = 0; i < src.size(); i++) out[i] = atb_llm::Fp16ToF32(src[i]);
    return out;
}

// ═════════════════════════════════════════════════════════════════════
// Stage A (host-only) regression test — compares our C++ index/weight
// builder against the values baked into the bundle (which Python produced).
// This catches Stage A bugs WITHOUT needing the NPU at all.
// ═════════════════════════════════════════════════════════════════════
void TestStageA(const std::string& tag) {
    CAPTURE(tag);

    Bundle b;
    const std::string path = "/tmp/posembed_npu_case_" + tag + ".bin";
    REQUIRE_MESSAGE(ReadBundle(path, b), "Stage A [" << tag << "]: bundle missing");

    std::vector<int32_t> idx_cpp[4];
    std::vector<uint16_t> wt_cpp[4];
    atb_llm::components::BuildPosEmbedIndicesAndWeights(
        b.grid_thw.data(), b.num_images,
        static_cast<int32_t>(b.num_grid),
        static_cast<int32_t>(b.merge_size),
        idx_cpp, wt_cpp);

    // Compare lengths
    for (int k = 0; k < 4; k++) {
        CHECK(idx_cpp[k].size() == b.idx[k].size());
        CHECK(wt_cpp[k].size()  == b.wt[k].size());
    }

    // Compare values
    int64_t idx_mism = 0, wt_mism = 0;
    float wt_max_diff = 0.0f;
    for (int k = 0; k < 4; k++) {
        for (size_t i = 0; i < idx_cpp[k].size(); i++) {
            if (idx_cpp[k][i] != b.idx[k][i]) idx_mism++;
        }
        for (size_t i = 0; i < wt_cpp[k].size(); i++) {
            float a = atb_llm::Fp16ToF32(wt_cpp[k][i]);
            float c = atb_llm::Fp16ToF32(b.wt[k][i]);
            float d = std::fabs(a - c);
            if (d > 1e-4f) wt_mism++;
            wt_max_diff = std::max(wt_max_diff, d);
        }
    }
    LOG_INFO("Stage A [%s]: idx_mismatch=%ld wt_mismatch=%ld wt_max_diff=%.6f",
             tag.c_str(),
             static_cast<long>(idx_mism), static_cast<long>(wt_mism), wt_max_diff);

    // Indices must be 100% exact. Weights allow a few ULPs of fp16 rounding noise.
    CHECK(idx_mism == 0);
    CHECK(wt_max_diff < 1e-3f);
}

// ═════════════════════════════════════════════════════════════════════
// Stage B (NPU graph) — build graph, copy inputs, execute, compare with expected.
// ═════════════════════════════════════════════════════════════════════
void RunCase(const std::string& tag) {
    CAPTURE(tag);

    // ── Load pos_embed table ────────────────────────────────────
    std::vector<uint16_t> pe_data;
    int64_t pe_rows = 0, vis_hs = 0;
    if (!ReadPosEmbedTable("/tmp/posembed_npu_pos_embed_w.bin",
                           pe_data, pe_rows, vis_hs)) {
        FAIL_CHECK("pos_embed table missing — "
                   "run `python tests/gen_pos_embed_npu_reference.py` first");
        return;
    }
    LOG_INFO("pos_embed table: rows=%ld vis_hs=%ld",
             static_cast<long>(pe_rows), static_cast<long>(vis_hs));

    // ── Load case bundle ────────────────────────────────────────
    Bundle b;
    const std::string bundle_path = "/tmp/posembed_npu_case_" + tag + ".bin";
    if (!ReadBundle(bundle_path, b)) {
        FAIL_CHECK("Stage B [" << tag << "]: bundle missing " << bundle_path);
        return;
    }
    REQUIRE(b.vis_hs == vis_hs);

    int64_t N = b.idx_count;

    LOG_INFO("  [%s] N=%ld vis_hs=%ld expected=(%ld,%ld)",
             tag.c_str(),
             static_cast<long>(N), static_cast<long>(vis_hs),
             static_cast<long>(b.expected_shape[0]),
             static_cast<long>(b.expected_shape[1]));

    // ── Build NPU graph ─────────────────────────────────────────
    atb_llm::OperationHandle graph;
    atb_llm::Status s = atb_llm::components::PosEmbedInterpGraph::Build(
        "PosEmbedInterp_" + tag, graph);
    REQUIRE(IS_OK(s));
    REQUIRE(graph);

    // ── Create runtime ──────────────────────────────────────────
    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    // ── Allocate + upload tensors ───────────────────────────────
    atb::Tensor pe_npu, idx_npu[4], wt_npu[4], out_npu;

    REQUIRE(IS_OK(alloc->AllocFloat16(pe_npu, {pe_rows, vis_hs})));
    alloc->CopyToDevice(pe_npu, pe_data.data(),
                        pe_data.size() * sizeof(uint16_t));

    for (int k = 0; k < 4; k++) {
        REQUIRE(IS_OK(alloc->AllocInt32(idx_npu[k], {N})));
        alloc->CopyToDevice(idx_npu[k], b.idx[k].data(),
                            N * sizeof(int32_t));
        // wt: shape (N, 1) so ElewiseMul broadcasts over vis_hs
        REQUIRE(IS_OK(alloc->AllocFloat16(wt_npu[k], {N, 1})));
        alloc->CopyToDevice(wt_npu[k], b.wt[k].data(),
                            N * sizeof(uint16_t));
    }

    REQUIRE(IS_OK(alloc->AllocFloat16(out_npu, {N, vis_hs})));

    // ── Execute ──────────────────────────────────────────────────
    atb::VariantPack vp;
    vp.inTensors = {pe_npu,
                    idx_npu[0], idx_npu[1], idx_npu[2], idx_npu[3],
                    wt_npu[0],  wt_npu[1],  wt_npu[2],  wt_npu[3]};
    vp.outTensors = {out_npu};

    uint64_t ws_size = 0;
    atb::Status atb_s = graph.get()->Setup(vp, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);

    uint8_t* ws_ptr = nullptr;
    auto __atb_pair_ws = runtime->GetWorkspace(ws_size > 0 ? ws_size : 1); auto& ws = __atb_pair_ws.first; auto& ws_st = __atb_pair_ws.second;
    if (ws_st == atb_llm::STATUS_OK) ws_ptr = ws;

    atb_s = graph.get()->Execute(vp, ws_ptr, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);
    runtime->Synchronize();

    // ── Read back and compare ────────────────────────────────────
    std::vector<uint16_t> out_fp16(N * vis_hs);
    alloc->CopyToHost(out_fp16.data(), out_npu,
                      out_fp16.size() * sizeof(uint16_t));

    auto got = Fp16VecToF32(out_fp16);
    auto exp = Fp16VecToF32(b.expected);

    double cosine  = CosineSim(got, exp);
    float  max_abs = MaxAbsDiff(got, exp);

    LOG_INFO("  [%s] cosine=%.6f  max_abs=%.5f",
             tag.c_str(), cosine, max_abs);

    CHECK(cosine >= 0.999);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════
// Stage A: host-only index/weight builder regression (no NPU needed)
// ═════════════════════════════════════════════════════════════════════

TEST_CASE("PosEmbedNPU StageA: tiny_4x4") { TestStageA("tiny_4x4"); }
TEST_CASE("PosEmbedNPU StageA: t1_8x8")   { TestStageA("t1_8x8"); }
TEST_CASE("PosEmbedNPU StageA: 224x224")  { TestStageA("224x224"); }
TEST_CASE("PosEmbedNPU StageA: 416x672")  { TestStageA("416x672"); }
TEST_CASE("PosEmbedNPU StageA: 896x896")  { TestStageA("896x896"); }

// ═════════════════════════════════════════════════════════════════════
// Stage B: NPU graph execution and precision validation
// ═════════════════════════════════════════════════════════════════════

TEST_CASE("PosEmbedNPU StageB: tiny_4x4") { RunCase("tiny_4x4"); }
TEST_CASE("PosEmbedNPU StageB: t1_8x8")   { RunCase("t1_8x8"); }
TEST_CASE("PosEmbedNPU StageB: 224x224")  { RunCase("224x224"); }
TEST_CASE("PosEmbedNPU StageB: 416x672")  { RunCase("416x672"); }
TEST_CASE("PosEmbedNPU StageB: 896x896")  { RunCase("896x896"); }
