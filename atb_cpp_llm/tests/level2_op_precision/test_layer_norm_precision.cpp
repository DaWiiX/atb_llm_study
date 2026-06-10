/**
 * Level 2 precision test: LayerNormOp vs PyTorch CPU reference.
 *
 * The ATB LayerNorm wrapper exposes `begin_norm_axis` and `begin_params_axis`.
 * For 2D input [S, H] both axis=1 and axis=-1 normalize over the trailing H
 * dimension (per-token norm), which is the only mode used by Qwen3VL Vision.
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_layer_norm
 *
 * Run: ./test_layer_norm_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/layer_norm_op.h"
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

// Run a LayerNorm case with the given axis parameters. Returns cosine.
float RunCase(atb_llm::IRuntime* runtime,
              int32_t begin_norm_axis,
              int32_t begin_params_axis,
              const ArrayF32& x_ref,
              const ArrayF32& g_ref,
              const ArrayF32& b_ref,
              const ArrayF32& expected) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    REQUIRE(x_ref.shape.size() == 2);
    int64_t S = x_ref.shape[0];
    int64_t H = x_ref.shape[1];

    auto x_fp16 = F32ToFp16Buffer(x_ref.data);
    auto g_fp16 = F32ToFp16Buffer(g_ref.data);
    auto b_fp16 = F32ToFp16Buffer(b_ref.data);

    atb::Tensor input, gamma, beta, output;
    REQUIRE(IS_OK(alloc->AllocFloat16(input,  {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(gamma,  {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(beta,   {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(output, {S, H})));

    REQUIRE(IS_OK(alloc->CopyToDevice(input, x_fp16.data(), x_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(gamma, g_fp16.data(), g_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(beta,  b_fp16.data(), b_fp16.size() * sizeof(uint16_t))));

    auto op = atb_llm::ops::LayerNormOp::Create(1e-5f, begin_norm_axis, begin_params_axis);
    REQUIRE(op.get() != nullptr);

    atb::VariantPack vp;
    vp.inTensors  = {input, gamma, beta};
    vp.outTensors = {output};

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

    std::vector<uint16_t> host_out_fp16(S * H);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out_fp16.data(), output,
                                    host_out_fp16.size() * sizeof(uint16_t))));
    auto host_out_f32 = Fp16ToF32Buffer(host_out_fp16);

    REQUIRE(static_cast<int64_t>(expected.data.size()) == S * H);
    return CosineSim(host_out_f32.data(), expected.data.data(), S * H);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: typical Vision hidden=1280, axis=1 (default)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("LayerNormOp precision: typical (S=16, H=1280, axis=1)") {
    LOG_INFO("=== LayerNorm precision: typical, axis=1 ===");

    ArrayF32 x, g, b, expected;
    REQUIRE(x.Load("/tmp/cpu_op_layer_norm_typical_input.bin"));
    REQUIRE(g.Load("/tmp/cpu_op_layer_norm_typical_gamma.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_layer_norm_typical_beta.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_layer_norm_typical_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunCase(runtime.get(), /*begin_norm_axis=*/1, /*begin_params_axis=*/1,
                       x, g, b, expected);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: small for debug, axis=1
// ═════════════════════════════════════════════════════════════════
TEST_CASE("LayerNormOp precision: small (S=4, H=64, axis=1)") {
    LOG_INFO("=== LayerNorm precision: small, axis=1 ===");

    ArrayF32 x, g, b, expected;
    REQUIRE(x.Load("/tmp/cpu_op_layer_norm_small_input.bin"));
    REQUIRE(g.Load("/tmp/cpu_op_layer_norm_small_gamma.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_layer_norm_small_beta.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_layer_norm_small_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunCase(runtime.get(), 1, 1, x, g, b, expected);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: axis=-1 variant — for a 2D input this normalizes over the
// last dim, which is mathematically identical to axis=1. We verify
// both that the op constructs with axis=-1 and that the precision
// holds against the same per-row reference.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("LayerNormOp precision: axis=-1 variant") {
    LOG_INFO("=== LayerNorm precision: axis=-1 ===");

    ArrayF32 x, g, b, expected;
    REQUIRE(x.Load("/tmp/cpu_op_layer_norm_axis_minus1_input.bin"));
    REQUIRE(g.Load("/tmp/cpu_op_layer_norm_axis_minus1_gamma.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_layer_norm_axis_minus1_beta.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_layer_norm_axis_minus1_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunCase(runtime.get(), /*begin_norm_axis=*/-1, /*begin_params_axis=*/-1,
                       x, g, b, expected);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
