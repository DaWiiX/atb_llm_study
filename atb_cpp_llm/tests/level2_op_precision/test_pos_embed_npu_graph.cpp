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

}  // namespace

// ═════════════════════════════════════════════════════════════════════
// Run one case: build graph, copy inputs, execute, compare with expected.
// ═════════════════════════════════════════════════════════════════════
struct CaseResult {
    std::string tag;
    bool ok = false;
    double cosine = 0.0;
    float max_abs = 0.0f;
    int64_t n = 0;
    int64_t hs = 0;
    std::string reason;
};

static CaseResult RunCase(const std::string& tag,
                          const std::vector<uint16_t>& pos_embed_data,
                          int64_t num_pos_rows, int64_t vis_hs,
                          atb_llm::IRuntime* runtime) {
    CaseResult res; res.tag = tag;

    Bundle b;
    const std::string bundle_path = "/tmp/posembed_npu_case_" + tag + ".bin";
    if (!ReadBundle(bundle_path, b)) {
        res.reason = "missing bundle " + bundle_path;
        return res;
    }
    if (b.vis_hs != vis_hs) {
        res.reason = "bundle vis_hs mismatch";
        return res;
    }

    res.n = b.idx_count;
    res.hs = vis_hs;

    LOG_INFO("  [%s] N=%ld vis_hs=%ld expected=(%ld,%ld)",
             tag.c_str(),
             static_cast<long>(b.idx_count), static_cast<long>(vis_hs),
             static_cast<long>(b.expected_shape[0]),
             static_cast<long>(b.expected_shape[1]));

    // ── Build NPU graph ─────────────────────────────────────────
    atb_llm::OperationHandle graph;
    atb_llm::Status s = atb_llm::components::PosEmbedInterpGraph::Build(
        "PosEmbedInterp_" + tag, graph);
    if (!IS_OK(s) || !graph) {
        res.reason = "graph build failed";
        return res;
    }

    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();

    // ── Allocate + upload tensors ───────────────────────────────
    atb::Tensor pe_npu, idx_npu[4], wt_npu[4], out_npu;

    s = alloc->AllocFloat16(pe_npu, {num_pos_rows, vis_hs});
    if (!IS_OK(s)) { res.reason = "alloc pe_npu"; return res; }
    alloc->CopyToDevice(pe_npu, pos_embed_data.data(),
                        pos_embed_data.size() * sizeof(uint16_t));

    for (int k = 0; k < 4; k++) {
        s = alloc->AllocInt32(idx_npu[k], {b.idx_count});
        if (!IS_OK(s)) { res.reason = "alloc idx"; return res; }
        alloc->CopyToDevice(idx_npu[k], b.idx[k].data(),
                            b.idx_count * sizeof(int32_t));
        // wt: shape (N, 1) so ElewiseMul broadcasts over vis_hs
        s = alloc->AllocFloat16(wt_npu[k], {b.idx_count, 1});
        if (!IS_OK(s)) { res.reason = "alloc wt"; return res; }
        alloc->CopyToDevice(wt_npu[k], b.wt[k].data(),
                            b.idx_count * sizeof(uint16_t));
    }

    s = alloc->AllocFloat16(out_npu, {b.idx_count, vis_hs});
    if (!IS_OK(s)) { res.reason = "alloc out"; return res; }

    // ── Execute ──────────────────────────────────────────────────
    atb::VariantPack vp;
    vp.inTensors = {pe_npu,
                    idx_npu[0], idx_npu[1], idx_npu[2], idx_npu[3],
                    wt_npu[0],  wt_npu[1],  wt_npu[2],  wt_npu[3]};
    vp.outTensors = {out_npu};

    uint64_t ws_size = 0;
    atb::Status atb_s = graph.get()->Setup(vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        res.reason = "Setup failed: " + std::to_string(atb_s);
        return res;
    }

    uint8_t* ws_ptr = nullptr;
    auto [ws, ws_st] = runtime->GetWorkspace(ws_size > 0 ? ws_size : 1);
    if (ws_st == atb_llm::STATUS_OK) ws_ptr = ws;

    atb_s = graph.get()->Execute(vp, ws_ptr, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        res.reason = "Execute failed: " + std::to_string(atb_s);
        return res;
    }
    runtime->Synchronize();

    // ── Read back and compare ────────────────────────────────────
    std::vector<uint16_t> out_fp16(b.idx_count * vis_hs);
    alloc->CopyToHost(out_fp16.data(), out_npu,
                      out_fp16.size() * sizeof(uint16_t));

    auto got = Fp16VecToF32(out_fp16);
    auto exp = Fp16VecToF32(b.expected);

    res.cosine  = CosineSim(got, exp);
    res.max_abs = MaxAbsDiff(got, exp);
    res.ok      = (res.cosine >= 0.999);

    LOG_INFO("  [%s] cosine=%.6f  max_abs=%.5f  %s",
             tag.c_str(), res.cosine, res.max_abs,
             res.ok ? "PASS" : "FAIL");
    return res;
}

// ═════════════════════════════════════════════════════════════════════
// Host-only Stage A regression test — compares our C++ index/weight
// builder against the values baked into the bundle (which Python produced).
// This catches Stage A bugs WITHOUT needing the NPU at all.
// ═════════════════════════════════════════════════════════════════════
static bool TestStageA(const std::string& tag) {
    Bundle b;
    if (!ReadBundle("/tmp/posembed_npu_case_" + tag + ".bin", b)) {
        LOG_ERROR("Stage A [%s]: bundle missing", tag.c_str());
        return false;
    }

    std::vector<int32_t> idx_cpp[4];
    std::vector<uint16_t> wt_cpp[4];
    atb_llm::components::BuildPosEmbedIndicesAndWeights(
        b.grid_thw.data(), b.num_images,
        static_cast<int32_t>(b.num_grid),
        static_cast<int32_t>(b.merge_size),
        idx_cpp, wt_cpp);

    // Compare lengths
    for (int k = 0; k < 4; k++) {
        if (idx_cpp[k].size() != b.idx[k].size() ||
            wt_cpp[k].size()  != b.wt[k].size()) {
            LOG_ERROR("Stage A [%s] k=%d size mismatch: cpp=%zu/%zu py=%zu/%zu",
                      tag.c_str(), k,
                      idx_cpp[k].size(), wt_cpp[k].size(),
                      b.idx[k].size(),    b.wt[k].size());
            return false;
        }
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
    return (idx_mism == 0) && (wt_max_diff < 1e-3f);
}

int main() {
    LOG_INFO("=== Vision PosEmbed NPU Graph Test ===");

    // ── Stage A regression on a couple of cases ──
    int passed_a = 0, total_a = 0;
    for (auto& tag : {std::string("tiny_4x4"), std::string("t1_8x8"),
                      std::string("224x224"), std::string("416x672"),
                      std::string("896x896")}) {
        total_a++;
        if (TestStageA(tag)) passed_a++;
    }
    LOG_INFO("Stage A: %d/%d cases passed", passed_a, total_a);

    // ── Stage B (NPU) — need a runtime ──
    // Load pos_embed table once
    std::vector<uint16_t> pe_data;
    int64_t pe_rows = 0, vis_hs = 0;
    if (!ReadPosEmbedTable("/tmp/posembed_npu_pos_embed_w.bin",
                            pe_data, pe_rows, vis_hs)) {
        LOG_ERROR("Failed to read /tmp/posembed_npu_pos_embed_w.bin — "
                  "run `python tests/gen_pos_embed_npu_reference.py` first");
        return 1;
    }
    LOG_INFO("pos_embed table: rows=%ld vis_hs=%ld",
             static_cast<long>(pe_rows), static_cast<long>(vis_hs));

    // Create runtime — only needs (device_id, buffer_size); model dir is
    // not used because we load pos_embed table directly from /tmp.
    std::unique_ptr<atb_llm::IRuntime> runtime;
    auto rt_st = atb_llm::RuntimeImpl::Create(
        /*device_id=*/0,
        /*buffer_size=*/2LL * 1024 * 1024 * 1024,
        runtime);
    if (rt_st != atb_llm::STATUS_OK || !runtime) {
        LOG_ERROR("Runtime create failed");
        return 1;
    }

    int passed_b = 0, total_b = 0;
    for (auto& tag : {std::string("tiny_4x4"), std::string("t1_8x8"),
                      std::string("224x224"), std::string("416x672"),
                      std::string("896x896")}) {
        total_b++;
        auto r = RunCase(tag, pe_data, pe_rows, vis_hs, runtime.get());
        if (r.ok) passed_b++;
        if (!r.ok && !r.reason.empty()) {
            LOG_ERROR("  [%s] error: %s", tag.c_str(), r.reason.c_str());
        }
    }

    LOG_INFO("──────────────────────────────────────────────────");
    LOG_INFO("Stage A (host-only):  %d/%d", passed_a, total_a);
    LOG_INFO("Stage B (NPU graph):  %d/%d", passed_b, total_b);

    return (passed_a == total_a && passed_b == total_b) ? 0 : 1;
}
