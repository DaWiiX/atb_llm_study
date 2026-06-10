/**
 * Level 2 precision test: LinearOp vs PyTorch CPU reference.
 *
 * Tests three parameter combinations:
 *   1. typical:  no bias, transpose_b=True (default)   [16, 2048] x [2048, 2048]
 *   2. small_bias: has_bias=True, transpose_b=True     [4, 64] x [32, 64] + bias
 *   3. no_transpose: no bias, transpose_b=False        [4, 64] x [64, 32]
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_linear
 *
 * Run: ./test_linear_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/linear_op.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#define IS_OK(s) ((s) == atb_llm::STATUS_OK)

namespace {

struct ArrayF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;

    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { LOG_ERROR("Cannot open %s", path.c_str()); return false; }
        int64_t ndim = 0;
        fread(&ndim, sizeof(int64_t), 1, f);
        shape.resize(ndim);
        int64_t total = 1;
        for (int64_t i = 0; i < ndim; i++) {
            fread(&shape[i], sizeof(int64_t), 1, f);
            total *= shape[i];
        }
        data.resize(total);
        fread(data.data(), sizeof(float), total, f);
        fclose(f);
        return true;
    }
};

float CosineSim(const float* a, const float* b, int64_t n) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < n; i++) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

std::vector<uint16_t> F32ToFp16Buffer(const std::vector<float>& src) {
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = atb_llm::Fp32ToFp16(src[i]);
    return dst;
}

std::vector<float> Fp16ToF32Buffer(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = atb_llm::Fp16ToF32(src[i]);
    return dst;
}

/// Run a Linear case with the given parameters.
/// @param has_bias    Whether to feed an extra bias input
/// @param transpose_b ATB Linear transpose_b flag
float RunCase(atb_llm::IRuntime* runtime,
              const ArrayF32& x_ref,
              const ArrayF32& w_ref,
              const ArrayF32& expected,
              bool has_bias,
              const ArrayF32* b_ref = nullptr) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    REQUIRE(x_ref.shape.size() == 2);
    REQUIRE(w_ref.shape.size() == 2);
    int64_t S = x_ref.shape[0];
    int64_t K = x_ref.shape[1];
    int64_t N = w_ref.shape[0];  // [N, K] with transpose_b=True; [K, N] with False

    auto x_fp16 = F32ToFp16Buffer(x_ref.data);
    auto w_fp16 = F32ToFp16Buffer(w_ref.data);

    atb::Tensor input, weight, bias, output;
    REQUIRE(IS_OK(alloc->AllocFloat16(input,  {S, K})));
    REQUIRE(IS_OK(alloc->AllocFloat16(weight, w_ref.shape)));

    int64_t out_dim = N;  // transpose_b=True → output [S, N]
    if (!has_bias) {
        REQUIRE(IS_OK(alloc->AllocFloat16(output, {S, out_dim})));
    }

    REQUIRE(IS_OK(alloc->CopyToDevice(input,  x_fp16.data(), x_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(weight, w_fp16.data(), w_fp16.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    if (has_bias) {
        REQUIRE(b_ref != nullptr);
        REQUIRE(static_cast<int64_t>(b_ref->data.size()) == out_dim);
        auto b_fp16 = F32ToFp16Buffer(b_ref->data);
        REQUIRE(IS_OK(alloc->AllocFloat16(bias, {out_dim})));
        REQUIRE(IS_OK(alloc->AllocFloat16(output, {S, out_dim})));
        REQUIRE(IS_OK(alloc->CopyToDevice(bias, b_fp16.data(), b_fp16.size() * sizeof(uint16_t))));
        vp.inTensors = {input, weight, bias};
    } else {
        vp.inTensors = {input, weight};
    }
    vp.outTensors = {output};

    auto op = atb_llm::ops::LinearOp::Create(has_bias, /*transpose_a=*/false,
                                              /*transpose_b=*/true);
    REQUIRE(op.get() != nullptr);

    uint64_t ws_size = 0;
    atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_st = __atb_pair_ws.second;
        REQUIRE(IS_OK(ws_st));
        ws_ptr = ws;
    }
    atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> host_out_fp16(S * out_dim);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out_fp16.data(), output,
                                    host_out_fp16.size() * sizeof(uint16_t))));
    auto host_out_f32 = Fp16ToF32Buffer(host_out_fp16);

    REQUIRE(static_cast<int64_t>(expected.data.size()) == S * out_dim);
    return CosineSim(host_out_f32.data(), expected.data.data(), S * out_dim);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: typical — no bias, transpose_b=True, 2048→2048
// ═════════════════════════════════════════════════════════════════
TEST_CASE("LinearOp precision: typical (S=16, K=2048, N=2048, no bias)") {
    LOG_INFO("=== Linear precision: typical ===");

    ArrayF32 x, w, expected;
    REQUIRE(x.Load("/tmp/cpu_op_linear_typical_input.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_linear_typical_weight.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_linear_typical_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 4LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunCase(runtime.get(), x, w, expected,
                       /*has_bias=*/false);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: small with bias
// ═════════════════════════════════════════════════════════════════
TEST_CASE("LinearOp precision: small_bias (S=4, K=64, N=32, has_bias)") {
    LOG_INFO("=== Linear precision: small_bias ===");

    ArrayF32 x, w, b, expected;
    REQUIRE(x.Load("/tmp/cpu_op_linear_small_bias_input.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_linear_small_bias_weight.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_linear_small_bias_bias.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_linear_small_bias_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunCase(runtime.get(), x, w, expected,
                       /*has_bias=*/true, &b);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: no_transpose variant (transpose_b=False)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("LinearOp precision: no_transpose (S=4, K=64, N=32, transpose_b=False)") {
    LOG_INFO("=== Linear precision: no_transpose ===");

    ArrayF32 x, w, expected;
    REQUIRE(x.Load("/tmp/cpu_op_linear_no_transpose_input.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_linear_no_transpose_weight.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_linear_no_transpose_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    REQUIRE(x.shape.size() == 2);
    REQUIRE(w.shape.size() == 2);
    int64_t S = x.shape[0];
    int64_t K = x.shape[1];   // 64
    int64_t N = w.shape[1];   // 32  (w is [K, N] when transpose_b=False)

    auto x_fp16 = F32ToFp16Buffer(x.data);
    auto w_fp16 = F32ToFp16Buffer(w.data);

    atb::Tensor input, weight, output;
    REQUIRE(IS_OK(alloc->AllocFloat16(input,  {S, K})));
    REQUIRE(IS_OK(alloc->AllocFloat16(weight, w.shape)));  // [K, N]
    REQUIRE(IS_OK(alloc->AllocFloat16(output, {S, N})));

    REQUIRE(IS_OK(alloc->CopyToDevice(input,  x_fp16.data(), x_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(weight, w_fp16.data(), w_fp16.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors  = {input, weight};
    vp.outTensors = {output};

    // transpose_b=False
    auto op = atb_llm::ops::LinearOp::Create(/*has_bias=*/false,
                                              /*transpose_a=*/false,
                                              /*transpose_b=*/false);
    REQUIRE(op.get() != nullptr);

    uint64_t ws_size = 0;
    atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_st = __atb_pair_ws.second;
        REQUIRE(IS_OK(ws_st));
        ws_ptr = ws;
    }
    atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> host_out_fp16(S * N);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out_fp16.data(), output,
                                    host_out_fp16.size() * sizeof(uint16_t))));
    auto host_out_f32 = Fp16ToF32Buffer(host_out_fp16);

    REQUIRE(static_cast<int64_t>(expected.data.size()) == S * N);
    float cs = CosineSim(host_out_f32.data(), expected.data.data(), S * N);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}