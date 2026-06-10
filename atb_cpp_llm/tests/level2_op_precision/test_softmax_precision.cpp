/**
 * Level 2 precision test: SoftmaxOp vs PyTorch F.softmax reference.
 *
 * Cases:
 *   - 1D softmax axes=[-1] on [1, 16]
 *   - 2D softmax axes=[-1] on [4, 8]
 *   - Numerical stability: large/small mix on [1, 8]
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_softmax
 *
 * Run: ./test_softmax_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/softmax_op.h"
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

struct ArrayFp16 {
    std::vector<int64_t> shape;
    std::vector<uint16_t> data;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { LOG_ERROR("Cannot open %s", path.c_str()); return false; }
        int64_t ndim = 0;
        fread(&ndim, sizeof(int64_t), 1, f);
        shape.resize(ndim);
        for (int64_t i = 0; i < ndim; i++) fread(&shape[i], sizeof(int64_t), 1, f);
        data.resize(numel());
        fread(data.data(), sizeof(uint16_t), data.size(), f);
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

std::vector<float> Fp16ToF32Buf(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = atb_llm::Fp16ToF32(src[i]);
    return dst;
}

// Run softmax with axes=[-1] on a 2D input and return decoded fp32 output.
std::vector<float> RunSoftmax(atb_llm::IRuntime* runtime, const ArrayFp16& in) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::SoftmaxOp::Create({-1});
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_out;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in, in.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_out, in.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_in, in.data.data(), in.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors = {t_in};
    vp.outTensors = {t_out};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_w = runtime->GetWorkspace(ws_size); auto& w = __atb_pair_w.first; auto& st = __atb_pair_w.second;
        REQUIRE(IS_OK(st));
        ws = w;
    }
    REQUIRE(op.get()->Execute(vp, ws, ws_size, ctx) == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> host(in.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), t_out, host.size() * sizeof(uint16_t))));
    return Fp16ToF32Buf(host);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: 1D softmax  [1, 16]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SoftmaxOp precision: 1D axes=[-1] [1,16]") {
    LOG_INFO("=== Softmax 1D precision ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_softmax1d_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_softmax1d_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunSoftmax(runtime.get(), in);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);

    // Probabilities must sum to ~1 along last axis
    double sum = 0;
    for (auto v : out_f32) sum += v;
    LOG_INFO("  sum(probs) = %.6f", sum);
    CHECK(std::abs(sum - 1.0) < 1e-2);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: 2D softmax  [4, 8] — axes=[-1]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SoftmaxOp precision: 2D axes=[-1] [4,8]") {
    LOG_INFO("=== Softmax 2D precision ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_softmax2d_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_softmax2d_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunSoftmax(runtime.get(), in);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);

    // Each row must sum to ~1
    int64_t rows = in.shape[0], cols = in.shape[1];
    for (int64_t r = 0; r < rows; r++) {
        double row_sum = 0;
        for (int64_t c = 0; c < cols; c++) row_sum += out_f32[r * cols + c];
        CHECK(std::abs(row_sum - 1.0) < 1e-2);
    }
}

// ═════════════════════════════════════════════════════════════════
// Case 3: numerical stability — input has large values (max=10)
// Softmax must still produce a valid distribution.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SoftmaxOp precision: numerical stability [1,8] large values") {
    LOG_INFO("=== Softmax stability precision ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_softmax_stab_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_softmax_stab_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunSoftmax(runtime.get(), in);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);

    // No NaN/Inf and sums to 1
    double sum = 0;
    bool finite = true;
    for (auto v : out_f32) {
        if (!std::isfinite(v)) finite = false;
        sum += v;
    }
    CHECK(finite);
    LOG_INFO("  sum(probs) = %.6f", sum);
    CHECK(std::abs(sum - 1.0) < 1e-2);
}
