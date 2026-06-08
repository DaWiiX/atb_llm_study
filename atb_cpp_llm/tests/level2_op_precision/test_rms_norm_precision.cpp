/**
 * Level 2 precision test: RmsNormOp vs PyTorch CPU reference.
 *
 * Compares NPU fp16 RmsNorm output to a Python fp32 reference via
 * cosine similarity. Each case runs one of NORM / PRENORM / POSTNORM
 * and validates the primary output (and resOut for PRENORM).
 *
 * Tensor I/O per ATB RmsNormOperation doc:
 *   NORM:     inTensors=[x, gamma]             outTensors=[output]
 *   PRENORM:  inTensors=[x, residual, gamma]   outTensors=[output, resOut]
 *   POSTNORM: inTensors=[x, residual, gamma]   outTensors=[output]
 *
 * PRENORM math:  resOut = x + residual;  output = RmsNorm(resOut) * gamma
 * POSTNORM math: output = RmsNorm(x + residual) * gamma
 *
 * Note: the current C++ op (rms_norm_op.cpp) only sets normParam.epsilon
 * and leaves preNormParam.epsilon / postNormParam.epsilon at the ATB
 * default of 1e-5, so PRENORM/POSTNORM references use 1e-5.
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_rms_norm
 *
 * Run: ./test_rms_norm_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/rms_norm_op.h"
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
        if (!f) {
            LOG_ERROR("Cannot open %s", path.c_str());
            return false;
        }
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

// ── NORM variant runner ───────────────────────────────────────────
// Execute one RmsNorm NORM case. Returns cosine similarity vs `expected`.
float RunNormCase(atb_llm::IRuntime* runtime,
                  const ArrayF32& x_ref,
                  const ArrayF32& w_ref,
                  const ArrayF32& expected) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    REQUIRE(x_ref.shape.size() == 2);
    int64_t S = x_ref.shape[0];
    int64_t H = x_ref.shape[1];

    auto x_fp16 = F32ToFp16Buffer(x_ref.data);
    auto w_fp16 = F32ToFp16Buffer(w_ref.data);

    atb::Tensor input, weight, output;
    REQUIRE(IS_OK(alloc->AllocFloat16(input,  {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(weight, {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(output, {S, H})));

    REQUIRE(IS_OK(alloc->CopyToDevice(input,  x_fp16.data(), x_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(weight, w_fp16.data(), w_fp16.size() * sizeof(uint16_t))));

    auto op = atb_llm::ops::RmsNormOp::Create(1e-6f,
                atb_llm::ops::RmsNormOp::LayerType::NORM);
    REQUIRE(op.get() != nullptr);

    atb::VariantPack vp;
    vp.inTensors  = {input, weight};
    vp.outTensors = {output};

    uint64_t ws_size = 0;
    atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto [ws, ws_st] = runtime->GetWorkspace(ws_size);
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

// ── PRENORM variant runner ────────────────────────────────────────
// Execute one RmsNorm PRENORM case. Returns cosine similarities for
// output and resOut as a pair.
std::pair<float,float> RunPrenormCase(atb_llm::IRuntime* runtime,
                                      const ArrayF32& x_ref,
                                      const ArrayF32& residual_ref,
                                      const ArrayF32& w_ref,
                                      const ArrayF32& expected_out,
                                      const ArrayF32& expected_res_out) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    REQUIRE(x_ref.shape.size() == 2);
    int64_t S = x_ref.shape[0];
    int64_t H = x_ref.shape[1];

    auto x_fp16        = F32ToFp16Buffer(x_ref.data);
    auto residual_fp16 = F32ToFp16Buffer(residual_ref.data);
    auto w_fp16        = F32ToFp16Buffer(w_ref.data);

    atb::Tensor input, residual, weight, output, res_out;
    REQUIRE(IS_OK(alloc->AllocFloat16(input,    {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(residual, {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(weight,   {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(output,   {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(res_out,  {S, H})));

    REQUIRE(IS_OK(alloc->CopyToDevice(input,    x_fp16.data(),        x_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(residual, residual_fp16.data(), residual_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(weight,   w_fp16.data(),        w_fp16.size() * sizeof(uint16_t))));

    // PRENORM: the op reads preNormParam.epsilon (default 1e-5 in ATB),
    // so we pass 1e-5 here for consistency with the reference.
    auto op = atb_llm::ops::RmsNormOp::Create(1e-5f,
                atb_llm::ops::RmsNormOp::LayerType::PRENORM);
    REQUIRE(op.get() != nullptr);

    atb::VariantPack vp;
    // ATB RmsNormOperation PRENORM: inTensors = [x, residual, gamma]
    vp.inTensors  = {input, residual, weight};
    // ATB RmsNormOperation PRENORM: outTensors = [output, resOut]
    vp.outTensors = {output, res_out};

    uint64_t ws_size = 0;
    atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto [ws, ws_st] = runtime->GetWorkspace(ws_size);
        REQUIRE(IS_OK(ws_st));
        ws_ptr = ws;
    }
    atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> host_out_fp16(S * H);
    std::vector<uint16_t> host_res_fp16(S * H);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out_fp16.data(), output,
                                    host_out_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToHost(host_res_fp16.data(), res_out,
                                    host_res_fp16.size() * sizeof(uint16_t))));

    auto host_out_f32 = Fp16ToF32Buffer(host_out_fp16);
    auto host_res_f32 = Fp16ToF32Buffer(host_res_fp16);

    REQUIRE(static_cast<int64_t>(expected_out.data.size()) == S * H);
    REQUIRE(static_cast<int64_t>(expected_res_out.data.size()) == S * H);

    float cs_out = CosineSim(host_out_f32.data(), expected_out.data.data(), S * H);
    float cs_res = CosineSim(host_res_f32.data(), expected_res_out.data.data(), S * H);
    return {cs_out, cs_res};
}

// ── POSTNORM variant runner ───────────────────────────────────────
// Execute one RmsNorm POSTNORM case. Returns cosine similarity for output.
float RunPostnormCase(atb_llm::IRuntime* runtime,
                      const ArrayF32& x_ref,
                      const ArrayF32& residual_ref,
                      const ArrayF32& w_ref,
                      const ArrayF32& expected_out) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    REQUIRE(x_ref.shape.size() == 2);
    int64_t S = x_ref.shape[0];
    int64_t H = x_ref.shape[1];

    auto x_fp16        = F32ToFp16Buffer(x_ref.data);
    auto residual_fp16 = F32ToFp16Buffer(residual_ref.data);
    auto w_fp16        = F32ToFp16Buffer(w_ref.data);

    atb::Tensor input, residual, weight, output;
    REQUIRE(IS_OK(alloc->AllocFloat16(input,    {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(residual, {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(weight,   {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(output,   {S, H})));

    REQUIRE(IS_OK(alloc->CopyToDevice(input,    x_fp16.data(),        x_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(residual, residual_fp16.data(), residual_fp16.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(weight,   w_fp16.data(),        w_fp16.size() * sizeof(uint16_t))));

    // POSTNORM: the op reads postNormParam.epsilon (default 1e-5 in ATB).
    auto op = atb_llm::ops::RmsNormOp::Create(1e-5f,
                atb_llm::ops::RmsNormOp::LayerType::POSTNORM);
    REQUIRE(op.get() != nullptr);

    atb::VariantPack vp;
    // ATB RmsNormOperation POSTNORM: inTensors = [x, residual, gamma]
    vp.inTensors  = {input, residual, weight};
    // ATB RmsNormOperation POSTNORM: outTensors = [output]
    vp.outTensors = {output};

    uint64_t ws_size = 0;
    atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
    REQUIRE(atb_s == atb::NO_ERROR);

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto [ws, ws_st] = runtime->GetWorkspace(ws_size);
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

    REQUIRE(static_cast<int64_t>(expected_out.data.size()) == S * H);
    return CosineSim(host_out_f32.data(), expected_out.data.data(), S * H);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: typical Qwen3VL hidden=2048
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RmsNormOp precision: typical (S=16, H=2048)") {
    LOG_INFO("=== RmsNorm precision: typical ===");

    ArrayF32 x, w, expected;
    REQUIRE(x.Load("/tmp/cpu_op_rms_norm_typical_input.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_rms_norm_typical_weight.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_rms_norm_typical_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunNormCase(runtime.get(), x, w, expected);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: small for debug
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RmsNormOp precision: small (S=4, H=64)") {
    LOG_INFO("=== RmsNorm precision: small ===");

    ArrayF32 x, w, expected;
    REQUIRE(x.Load("/tmp/cpu_op_rms_norm_small_input.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_rms_norm_small_weight.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_rms_norm_small_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunNormCase(runtime.get(), x, w, expected);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: medium — NORM variant
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RmsNormOp precision: medium (S=8, H=256)") {
    LOG_INFO("=== RmsNorm precision: medium ===");

    ArrayF32 x, w, expected;
    REQUIRE(x.Load("/tmp/cpu_op_rms_norm_medium_input.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_rms_norm_medium_weight.bin"));
    REQUIRE(expected.Load("/tmp/cpu_op_rms_norm_medium_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunNormCase(runtime.get(), x, w, expected);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 4: PRENORM typical (S=16, H=2048)
//
// PRENORM: resOut = x + residual;  output = RmsNorm(resOut) * gamma
// ATB inTensors=[x, residual, gamma], outTensors=[output, resOut]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RmsNormOp PRENORM precision: typical (S=16, H=2048)") {
    LOG_INFO("=== RmsNorm PRENORM precision: typical ===");

    ArrayF32 x, residual, w, expected_out, expected_res_out;
    REQUIRE(x.Load("/tmp/cpu_op_rms_norm_prenorm_typical_input.bin"));
    REQUIRE(residual.Load("/tmp/cpu_op_rms_norm_prenorm_typical_residual.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_rms_norm_prenorm_typical_weight.bin"));
    REQUIRE(expected_out.Load("/tmp/cpu_op_rms_norm_prenorm_typical_output.bin"));
    REQUIRE(expected_res_out.Load("/tmp/cpu_op_rms_norm_prenorm_typical_res_out.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto [cs_out, cs_res] = RunPrenormCase(runtime.get(),
                            x, residual, w, expected_out, expected_res_out);
    LOG_INFO("  output cosine = %.6f", cs_out);
    LOG_INFO("  resOut cosine = %.6f", cs_res);
    CHECK(cs_out >= 0.999f);
    CHECK(cs_res >= 0.999f);
}

// ═════════════════════════════════════════════════════════════════
// Case 5: PRENORM small (S=4, H=64)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RmsNormOp PRENORM precision: small (S=4, H=64)") {
    LOG_INFO("=== RmsNorm PRENORM precision: small ===");

    ArrayF32 x, residual, w, expected_out, expected_res_out;
    REQUIRE(x.Load("/tmp/cpu_op_rms_norm_prenorm_small_input.bin"));
    REQUIRE(residual.Load("/tmp/cpu_op_rms_norm_prenorm_small_residual.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_rms_norm_prenorm_small_weight.bin"));
    REQUIRE(expected_out.Load("/tmp/cpu_op_rms_norm_prenorm_small_output.bin"));
    REQUIRE(expected_res_out.Load("/tmp/cpu_op_rms_norm_prenorm_small_res_out.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto [cs_out, cs_res] = RunPrenormCase(runtime.get(),
                            x, residual, w, expected_out, expected_res_out);
    LOG_INFO("  output cosine = %.6f", cs_out);
    LOG_INFO("  resOut cosine = %.6f", cs_res);
    CHECK(cs_out >= 0.999f);
    CHECK(cs_res >= 0.999f);
}

// ═════════════════════════════════════════════════════════════════
// Case 6: POSTNORM typical (S=16, H=2048)
//
// POSTNORM: output = RmsNorm(x + residual) * gamma
// ATB inTensors=[x, residual, gamma], outTensors=[output]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RmsNormOp POSTNORM precision: typical (S=16, H=2048)") {
    LOG_INFO("=== RmsNorm POSTNORM precision: typical ===");

    ArrayF32 x, residual, w, expected_out;
    REQUIRE(x.Load("/tmp/cpu_op_rms_norm_postnorm_typical_input.bin"));
    REQUIRE(residual.Load("/tmp/cpu_op_rms_norm_postnorm_typical_residual.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_rms_norm_postnorm_typical_weight.bin"));
    REQUIRE(expected_out.Load("/tmp/cpu_op_rms_norm_postnorm_typical_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunPostnormCase(runtime.get(), x, residual, w, expected_out);
    LOG_INFO("  output cosine = %.6f", cs);
    CHECK(cs >= 0.999f);
}

// ═════════════════════════════════════════════════════════════════
// Case 7: POSTNORM small (S=4, H=64)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RmsNormOp POSTNORM precision: small (S=4, H=64)") {
    LOG_INFO("=== RmsNorm POSTNORM precision: small ===");

    ArrayF32 x, residual, w, expected_out;
    REQUIRE(x.Load("/tmp/cpu_op_rms_norm_postnorm_small_input.bin"));
    REQUIRE(residual.Load("/tmp/cpu_op_rms_norm_postnorm_small_residual.bin"));
    REQUIRE(w.Load("/tmp/cpu_op_rms_norm_postnorm_small_weight.bin"));
    REQUIRE(expected_out.Load("/tmp/cpu_op_rms_norm_postnorm_small_output.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunPostnormCase(runtime.get(), x, residual, w, expected_out);
    LOG_INFO("  output cosine = %.6f", cs);
    CHECK(cs >= 0.999f);
}
