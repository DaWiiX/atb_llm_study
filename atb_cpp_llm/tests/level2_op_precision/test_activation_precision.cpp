/**
 * Level 2 precision test: ActivationOp (SiLU / GELU / FastGELU) vs PyTorch.
 *
 * Compares each ATB activation operator against the PyTorch reference using
 * cosine similarity. GELU uses the tanh-approximation that ATB implements
 * (PyTorch's `F.gelu(x, approximate="tanh")`). FastGELU uses
 * `x * sigmoid(1.702 * x)`.
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_activation
 *
 * Run: ./test_activation_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/activation_op.h"
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

enum class ActKind { SiLU, GELU, FastGELU };

const char* ActName(ActKind k) {
    switch (k) {
        case ActKind::SiLU:     return "silu";
        case ActKind::GELU:     return "gelu";
        case ActKind::FastGELU: return "fast_gelu";
    }
    return "?";
}

atb_llm::OperationHandle MakeOp(ActKind k) {
    switch (k) {
        case ActKind::SiLU:     return atb_llm::ops::ActivationOp::MakeSiLU();
        case ActKind::GELU:     return atb_llm::ops::ActivationOp::MakeGELU();
        case ActKind::FastGELU: return atb_llm::ops::ActivationOp::MakeFastGELU();
    }
    return atb_llm::OperationHandle(nullptr);
}

// Run one activation case end-to-end. Returns cosine similarity to reference.
float RunCase(atb_llm::IRuntime* runtime, ActKind kind,
              const ArrayF32& x_ref, const ArrayF32& expected) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    REQUIRE(x_ref.shape.size() == 2);
    int64_t S = x_ref.shape[0];
    int64_t H = x_ref.shape[1];

    auto x_fp16 = F32ToFp16Buffer(x_ref.data);

    atb::Tensor input, output;
    REQUIRE(IS_OK(alloc->AllocFloat16(input,  {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(output, {S, H})));
    REQUIRE(IS_OK(alloc->CopyToDevice(input, x_fp16.data(),
                                      x_fp16.size() * sizeof(uint16_t))));

    auto op = MakeOp(kind);
    REQUIRE(op.get() != nullptr);

    atb::VariantPack vp;
    vp.inTensors  = {input};
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

// Load + run a (kind, size) case.
void RunStage(atb_llm::IRuntime* runtime, ActKind kind, const char* size_name) {
    std::string base = std::string("/tmp/cpu_op_activation_") + ActName(kind) + "_" + size_name;
    ArrayF32 x, expected;
    REQUIRE(x.Load(base + "_input.bin"));
    REQUIRE(expected.Load(base + "_output.bin"));
    float cs = RunCase(runtime, kind, x, expected);
    LOG_INFO("  %s/%s  cosine = %.6f", ActName(kind), size_name, cs);
    CHECK(cs >= 0.99f);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: SiLU precision (typical + small)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ActivationOp precision: SiLU") {
    LOG_INFO("=== Activation precision: SiLU ===");
    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    RunStage(runtime.get(), ActKind::SiLU, "typical");
    RunStage(runtime.get(), ActKind::SiLU, "small");
}

// ═════════════════════════════════════════════════════════════════
// Case 2: GELU precision (typical + small)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ActivationOp precision: GELU") {
    LOG_INFO("=== Activation precision: GELU ===");
    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    RunStage(runtime.get(), ActKind::GELU, "typical");
    RunStage(runtime.get(), ActKind::GELU, "small");
}

// ═════════════════════════════════════════════════════════════════
// Case 3: FastGELU precision (typical + small)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ActivationOp precision: FastGELU") {
    LOG_INFO("=== Activation precision: FastGELU ===");
    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    RunStage(runtime.get(), ActKind::FastGELU, "typical");
    RunStage(runtime.get(), ActKind::FastGELU, "small");
}
