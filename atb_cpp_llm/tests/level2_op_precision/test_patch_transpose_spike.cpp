/**
 * P10-B optimization #1 gate spike: ATB TransposeOp on 7-D patch permute.
 *
 * PURPOSE (architect decision gate, NOT a permanent regression test yet):
 *   The patch-extraction permute currently runs on CPU after a D2H copy — the
 *   root cause of the 1440x2560 regression (D2H 7.5MB + engine H2D 7.5MB).
 *   Moving it to NPU requires an ATB Transpose that handles the patch permute.
 *
 *   Python patch logic (qwen3vl_preprocess.cpp:244-246):
 *     frames.reshape(grid_t, tp, C, merged_h, ms, ps, merged_w, ms, ps)
 *           .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)   # 9-D, exceeds ATB MAX_DIM=8
 *           .reshape(grid_t*grid_h*grid_w, C*tp*ps*ps)
 *
 *   grid_t=1 is squeezed, and tp replication is handled separately, leaving a
 *   7-D permute within ATB's dimNum<=8 limit:
 *     in  (C, merged_h, ms_h, ps_h, merged_w, ms_w, ps_w)  dims [0..6]
 *     out (merged_h, merged_w, ms_h, ms_w, C, ps_h, ps_w)
 *     perm [1, 4, 2, 5, 0, 3, 6]
 *
 *   This spike answers: does ATB TransposeOp support a 7-D permute, and is the
 *   output bit-exact vs a CPU reference? A transpose is pure data movement (no
 *   arithmetic), so bit-exactness is the strongest possible gate; cos>=0.999 is
 *   the task's requested metric and is reported alongside.
 *
 *   Decision: pass -> #1 feasible, engineer the NPU patch pipeline (tp
 *   replication via AsStrided stride=0 or Elewise broadcast, TBD).  fail -> #1
 *   needs a different approach (multi-step small permutes or aclnn).
 *
 * Cases:
 *   - small: {3,2,2,4,2,2,4}  (C=3 merged=2 ms=2 ps=4, 768 elems) — quick
 *   - prod : {3,2,2,16,2,2,16} (ps=16, the real patch_size, 49152 elems)
 *
 * Run: ./test_patch_transpose_spike
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/transpose_op.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

namespace {

/// CPU reference permute.  Computes out[out_flat] = in[in_flat] where
/// in_idx[perm[k]] = out_idx[k]  (standard np.transpose(a, perm) semantics).
/// Returns the permuted fp16 buffer; also fills out_shape.
std::vector<uint16_t> CpuPermute(const std::vector<uint16_t>& in,
                                 const std::vector<int64_t>& in_shape,
                                 const std::vector<int32_t>& perm,
                                 std::vector<int64_t>& out_shape) {
    int ndim = static_cast<int>(in_shape.size());
    REQUIRE(perm.size() == static_cast<size_t>(ndim));

    // input strides (row-major)
    std::vector<int64_t> in_stride(ndim);
    in_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
    }

    int64_t total = 1;
    for (auto d : in_shape) total *= d;
    REQUIRE(static_cast<int64_t>(in.size()) == total);

    // output shape = in_shape[perm[i]]
    out_shape.resize(ndim);
    for (int i = 0; i < ndim; ++i) out_shape[i] = in_shape[perm[i]];

    // output strides (row-major)
    std::vector<int64_t> out_stride(ndim);
    out_stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }

    std::vector<uint16_t> out(total);
    for (int64_t out_flat = 0; out_flat < total; ++out_flat) {
        // decompose out_flat -> out multi-index, map to in multi-index
        int64_t rem = out_flat;
        int64_t in_idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < ndim; ++i) {
            int64_t o = rem / out_stride[i];
            rem -= o * out_stride[i];
            in_idx[perm[i]] = o;  // out dim i <- in dim perm[i]
        }
        int64_t in_flat = 0;
        for (int i = 0; i < ndim; ++i) in_flat += in_idx[i] * in_stride[i];
        out[out_flat] = in[in_flat];
    }
    return out;
}

float CosineSimFp16(const std::vector<uint16_t>& a,
                    const std::vector<uint16_t>& b) {
    int64_t n = static_cast<int64_t>(a.size());
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; ++i) {
        float x = atb_llm::Fp16ToF32(a[i]);
        float y = atb_llm::Fp16ToF32(b[i]);
        dot += static_cast<double>(x) * y;
        na  += static_cast<double>(x) * x;
        nb  += static_cast<double>(y) * y;
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

/// Run ATB TransposeOp on a 7-D fp16 input and return the NPU output.
std::vector<uint16_t> RunNpuTranspose(atb_llm::IRuntime* runtime,
                                      const std::vector<int64_t>& in_shape,
                                      const std::vector<int32_t>& perm,
                                      const std::vector<uint16_t>& in,
                                      const std::vector<int64_t>& out_shape) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::TransposeOp::Create(perm);
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_out;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in,  in_shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_out, out_shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_in, in.data(), in.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors  = {t_in};
    vp.outTensors = {t_out};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_w = runtime->GetWorkspace(ws_size);
        auto& w  = __atb_pair_w.first;
        auto& st = __atb_pair_w.second;
        REQUIRE(IS_OK(st));
        ws = w;
    }
    REQUIRE(op.get()->Execute(vp, ws, ws_size, ctx) == atb::NO_ERROR);
    runtime->Synchronize();

    int64_t total = 1;
    for (auto d : out_shape) total *= d;
    std::vector<uint16_t> host(total);
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), t_out, host.size() * sizeof(uint16_t))));
    return host;
}

/// One gate case: generate random fp16 input, compute CPU ref, run NPU, compare.
/// Returns (cos, bit_exact).  Reports first few mismatches if not bit-exact.
struct CaseResult { float cos; bool bit_exact; int64_t mismatches; };

CaseResult RunCase(atb_llm::IRuntime* runtime,
                   const std::vector<int64_t>& in_shape,
                   const std::vector<int32_t>& perm,
                   const std::string& label) {
    int64_t total = 1;
    for (auto d : in_shape) total *= d;

    // Random fp16 input with distinct values (fixed seed -> reproducible).
    // Distinct values make cosine sensitive to a wrong permutation; if all
    // values were equal, cos=1.0 trivially even for a scrambled permute.
    std::srand(42);
    std::vector<uint16_t> in(total);
    for (int64_t i = 0; i < total; ++i) {
        float v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);  // [0,1)
        in[i] = atb_llm::Fp32ToFp16(v);
    }

    std::vector<int64_t> out_shape;
    std::vector<uint16_t> ref = CpuPermute(in, in_shape, perm, out_shape);

    std::vector<uint16_t> got = RunNpuTranspose(runtime, in_shape, perm, in, out_shape);
    REQUIRE(got.size() == ref.size());

    float cos = CosineSimFp16(got, ref);
    int64_t mismatches = 0;
    for (int64_t i = 0; i < total; ++i) {
        if (got[i] != ref[i]) {
            if (mismatches < 5) {
                LOG_ERROR("  [%s] mismatch @%lld: got=0x%04x (%g) vs ref=0x%04x (%g)",
                          label.c_str(), static_cast<long long>(i),
                          got[i], atb_llm::Fp16ToF32(got[i]),
                          ref[i], atb_llm::Fp16ToF32(ref[i]));
            }
            ++mismatches;
        }
    }
    bool bit_exact = (mismatches == 0);

    LOG_INFO("  [%s] in_shape={%s} perm={%s} total=%lld | cos=%.6f bit_exact=%d mismatches=%lld",
             label.c_str(),
             (std::to_string(in_shape[0]) + "," + std::to_string(in_shape[1]) + "," +
              std::to_string(in_shape[2]) + "," + std::to_string(in_shape[3]) + "," +
              std::to_string(in_shape[4]) + "," + std::to_string(in_shape[5]) + "," +
              std::to_string(in_shape[6])).c_str(),
             (std::to_string(perm[0]) + "," + std::to_string(perm[1]) + "," +
              std::to_string(perm[2]) + "," + std::to_string(perm[3]) + "," +
              std::to_string(perm[4]) + "," + std::to_string(perm[5]) + "," +
              std::to_string(perm[6])).c_str(),
             static_cast<long long>(total), cos,
             static_cast<int>(bit_exact), static_cast<long long>(mismatches));
    return {cos, bit_exact, mismatches};
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Gate: ATB TransposeOp on 7-D patch permute (perm=[1,4,2,5,0,3,6]).
//
// Input  (C, merged_h, ms_h, ps_h, merged_w, ms_w, ps_w)
// Output (merged_h, merged_w, ms_h, ms_w, C, ps_h, ps_w)
//
// This is the grid_t=1-squeezed, tp-stripped core of the Python 9-D permute
// (0,3,6,4,7,2,1,5,8).  7-D is within ATB's dimNum<=8 (MAX_DIM=8) limit.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("patch-transpose-spike: 7-D patch permute bit-exact vs CPU") {
    LOG_INFO("=== P10-B #1 gate spike: ATB TransposeOp 7-D patch permute ===");

    auto runtime = atb_llm::CreateRuntime(0, 1LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    const std::vector<int32_t> perm = {1, 4, 2, 5, 0, 3, 6};

    // Case 1: small (C=3, merged=2, ms=2, ps=4) — 768 elems, quick sanity.
    CaseResult r_small = RunCase(runtime.get(),
                                 {3, 2, 2, 4, 2, 2, 4}, perm, "small");

    // Case 2: production patch_size=16 (C=3, merged=2, ms=2, ps=16) — 49152 elems.
    CaseResult r_prod = RunCase(runtime.get(),
                                {3, 2, 2, 16, 2, 2, 16}, perm, "prod_ps16");

    LOG_INFO("========================================================");
    LOG_INFO("[SPIKE GATE] small: cos=%.6f bit_exact=%d | prod_ps16: cos=%.6f bit_exact=%d",
             r_small.cos, static_cast<int>(r_small.bit_exact),
             r_prod.cos,  static_cast<int>(r_prod.bit_exact));
    LOG_INFO("  cos>=0.999 & bit_exact -> #1 feasible (engineer NPU patch pipeline)");
    LOG_INFO("  fail                    -> #1 needs different approach (multi-step/aclnn)");
    LOG_INFO("========================================================");

    // Gate: task's requested metric is cos>=0.999.  A transpose is pure data
    // movement, so we additionally require bit-exactness — anything less is a
    // bug, not a precision tradeoff.
    CHECK(r_small.cos >= 0.999f);
    CHECK(r_prod.cos  >= 0.999f);
    CHECK(r_small.bit_exact);
    CHECK(r_prod.bit_exact);
}
