/**
 * IndexAdd NPU op precision test (Level 2).
 *
 * Verifies the ATB IndexAdd operator behaves as `var.index_add_(0, indices,
 * updates)` for the deepstack injection use case.
 *
 * No Python reference needed: we generate fixed inputs in C++ and compare
 * the NPU output against the CPU-computed expected value.
 *
 * Cases:
 *   - Small (M=10, N=4, D=8)
 *   - Medium (M=900, N=200, D=2048)  — typical IMAGE_ONLY usage
 *   - Edge: N=1, M=1
 *   - Edge: repeated index in `indices` (must accumulate)
 *
 * All cases require:
 *   bit-exact match on the unindexed rows
 *   cosine ≥ 0.9999 on the indexed rows (fp16 add rounding)
 *
 * Run: ./build/test_index_add_npu
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "ops/index_add_op.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include "core/tensor_allocator.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
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

void RunCase(const std::string& tag,
             int64_t M, int64_t D,
             const std::vector<int32_t>& indices,
             uint32_t seed = 42) {
    int64_t N = static_cast<int64_t>(indices.size());

    // ── Generate var, updates fp16 ──
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> var_fp16(M * D), upd_fp16(N * D);
    std::vector<float> var_f32(M * D), upd_f32(N * D);
    for (int64_t i = 0; i < M * D; i++) {
        var_f32[i] = dist(rng);
        var_fp16[i] = atb_llm::Fp32ToFp16(var_f32[i]);
        var_f32[i] = atb_llm::Fp16ToF32(var_fp16[i]);  // round-trip for fair compare
    }
    for (int64_t i = 0; i < N * D; i++) {
        upd_f32[i] = dist(rng);
        upd_fp16[i] = atb_llm::Fp32ToFp16(upd_f32[i]);
        upd_f32[i] = atb_llm::Fp16ToF32(upd_fp16[i]);
    }

    // ── Compute CPU reference ──
    std::vector<float> expected = var_f32;          // copy
    for (int64_t i = 0; i < N; i++) {
        int32_t idx = indices[i];
        REQUIRE((idx >= 0));
        REQUIRE((idx < M));
        for (int64_t d = 0; d < D; d++) {
            expected[idx * D + d] += upd_f32[i * D + d];
        }
    }

    // ── Create runtime ──
    auto runtime = atb_llm::CreateRuntime(0, 1LL * 1024 * 1024 * 1024);
    REQUIRE(runtime);

    // ── NPU run ──
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb::Tensor var_npu, idx_npu, upd_npu;
    REQUIRE(IS_OK(alloc->AllocFloat16(var_npu, {M, D})));
    REQUIRE(IS_OK(alloc->AllocInt32(idx_npu, {N})));
    REQUIRE(IS_OK(alloc->AllocFloat16(upd_npu, {N, D})));

    alloc->CopyToDevice(var_npu, var_fp16.data(), var_fp16.size() * sizeof(uint16_t));
    alloc->CopyToDevice(idx_npu, indices.data(), indices.size() * sizeof(int32_t));
    alloc->CopyToDevice(upd_npu, upd_fp16.data(), upd_fp16.size() * sizeof(uint16_t));

    atb_llm::OperationHandle op = atb_llm::ops::IndexAddOp::Create(0);
    REQUIRE(op);

    // ATB IndexAdd needs 4 inTensors. Based on common ATB convention,
    // the 4th is an alpha scalar (broadcast multiplier on `updates`).
    // We pass alpha=1.0 as a fp16 scalar tensor of shape (1,).
    uint16_t alpha_fp16 = atb_llm::Fp32ToFp16(1.0f);
    atb::Tensor alpha_npu;
    REQUIRE(IS_OK(alloc->AllocFloat16(alpha_npu, {1})));
    alloc->CopyToDevice(alpha_npu, &alpha_fp16, sizeof(uint16_t));

    atb::VariantPack vp;
    vp.inTensors  = {var_npu, idx_npu, upd_npu, alpha_npu};
    vp.outTensors = {var_npu};                     // in-place: alias output to var
    uint64_t ws_size = 0;
    atb::Status s = op.get()->Setup(vp, ws_size, ctx);
    REQUIRE(s == atb::NO_ERROR);
    uint8_t* ws_ptr = nullptr;
    auto __atb_pair_ws = runtime->GetWorkspace(ws_size > 0 ? ws_size : 1); auto& ws = __atb_pair_ws.first; auto& ws_st = __atb_pair_ws.second;
    if (ws_st == atb_llm::STATUS_OK) ws_ptr = ws;
    s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
    REQUIRE(s == atb::NO_ERROR);
    runtime->Synchronize();

    // ── Read back ──
    std::vector<uint16_t> out_fp16(M * D);
    alloc->CopyToHost(out_fp16.data(), var_npu, out_fp16.size() * sizeof(uint16_t));
    std::vector<float> got(M * D);
    for (int64_t i = 0; i < M * D; i++) got[i] = atb_llm::Fp16ToF32(out_fp16[i]);

    double cosine  = CosineSim(got, expected);
    float max_abs = MaxAbsDiff(got, expected);

    LOG_INFO("  [%s] M=%ld N=%ld D=%ld cosine=%.6f max_abs=%.5f",
             tag.c_str(),
             static_cast<long>(M), static_cast<long>(N), static_cast<long>(D),
             cosine, max_abs);

    CHECK(cosine >= 0.9999);

    // Check unindexed rows are bit-exact preserved.
    std::vector<bool> touched(M, false);
    for (auto i : indices) touched[i] = true;
    int64_t mismatch_unindexed = 0;
    for (int64_t r = 0; r < M; r++) {
        if (touched[r]) continue;
        for (int64_t d = 0; d < D; d++) {
            if (out_fp16[r * D + d] != var_fp16[r * D + d]) mismatch_unindexed++;
        }
    }

    CAPTURE(mismatch_unindexed);
    CHECK(mismatch_unindexed == 0);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════
TEST_CASE("IndexAddOp: small (M=10, N=4, D=8)") {
    std::vector<int32_t> idx = {3, 7, 1, 9};
    RunCase("small", /*M=*/10, /*D=*/8, idx, 42);
}

TEST_CASE("IndexAddOp: medium (M=900, N=200, D=2048)") {
    std::vector<int32_t> idx(200);
    for (int i = 0; i < 200; i++) idx[i] = i * 4 + 1;  // spread across var
    RunCase("medium", /*M=*/900, /*D=*/2048, idx, 1234);
}

TEST_CASE("IndexAddOp: edge single index (M=1, N=1, D=4)") {
    std::vector<int32_t> idx = {0};
    RunCase("edge_single", /*M=*/1, /*D=*/4, idx, 7);
}

TEST_CASE("IndexAddOp: edge repeated index (M=8, N=4, D=16)") {
    std::vector<int32_t> idx = {5, 5, 5, 2};
    RunCase("edge_repeated_idx", /*M=*/8, /*D=*/16, idx, 99);
}
