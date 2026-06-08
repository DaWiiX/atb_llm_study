/**
 * Level 2 precision test: ElewiseOp Add/Mul/Muls/Sub/Cast vs NumPy reference.
 *
 * Each case feeds fp16 inputs to the NPU op and compares the fp16 output
 * (decoded back to fp32) against the Python-generated reference (also
 * round-tripped through fp16). Threshold: cosine >= 0.99.
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_elewise
 *
 * Run: ./test_elewise_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/elewise_op.h"
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

struct ArrayF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
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
        fread(data.data(), sizeof(float), data.size(), f);
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

// Run a binary elewise op (Add/Mul/Sub) and return the fp16 output decoded to fp32.
std::vector<float> RunBinaryFp16(atb_llm::IRuntime* runtime,
                                 atb_llm::OperationHandle op,
                                 const ArrayFp16& a,
                                 const ArrayFp16& b) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb::Tensor ta, tb, to;
    REQUIRE(IS_OK(alloc->AllocFloat16(ta, a.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(tb, b.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(to, a.shape)));

    REQUIRE(IS_OK(alloc->CopyToDevice(ta, a.data.data(), a.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(tb, b.data.data(), b.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors = {ta, tb};
    vp.outTensors = {to};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws = nullptr;
    if (ws_size > 0) {
        auto [w, st] = runtime->GetWorkspace(ws_size);
        REQUIRE(IS_OK(st));
        ws = w;
    }
    REQUIRE(op.get()->Execute(vp, ws, ws_size, ctx) == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> out_fp16(a.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(out_fp16.data(), to, out_fp16.size() * sizeof(uint16_t))));
    return Fp16ToF32Buf(out_fp16);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: Add — two fp16 [4, 8] tensors vs numpy
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ElewiseOp precision: Add fp16") {
    LOG_INFO("=== Elewise Add precision ===");
    ArrayFp16 a, b, ref;
    REQUIRE(a.Load("/tmp/cpu_op_elewise_add_a.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_elewise_add_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_elewise_add_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto op = atb_llm::ops::ElewiseOp::MakeAdd();
    REQUIRE(op.get() != nullptr);
    auto out_f32 = RunBinaryFp16(runtime.get(), std::move(op), a, b);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: Mul — element-wise multiply [3, 5]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ElewiseOp precision: Mul fp16") {
    LOG_INFO("=== Elewise Mul precision ===");
    ArrayFp16 a, b, ref;
    REQUIRE(a.Load("/tmp/cpu_op_elewise_mul_a.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_elewise_mul_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_elewise_mul_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto op = atb_llm::ops::ElewiseOp::MakeMul();
    REQUIRE(op.get() != nullptr);
    auto out_f32 = RunBinaryFp16(runtime.get(), std::move(op), a, b);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: Muls — scalar multiply by 2.5
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ElewiseOp precision: Muls(scale=2.5) fp16") {
    LOG_INFO("=== Elewise Muls(2.5) precision ===");
    ArrayFp16 a, ref;
    REQUIRE(a.Load("/tmp/cpu_op_elewise_muls_a.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_elewise_muls_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::ElewiseOp::MakeMuls(2.5f);
    REQUIRE(op.get() != nullptr);

    atb::Tensor ta, to;
    REQUIRE(IS_OK(alloc->AllocFloat16(ta, a.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(to, a.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(ta, a.data.data(), a.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors = {ta};
    vp.outTensors = {to};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws = nullptr;
    if (ws_size > 0) {
        auto [w, st] = runtime->GetWorkspace(ws_size);
        REQUIRE(IS_OK(st));
        ws = w;
    }
    REQUIRE(op.get()->Execute(vp, ws, ws_size, ctx) == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> out_fp16(a.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(out_fp16.data(), to, out_fp16.size() * sizeof(uint16_t))));
    auto out_f32 = Fp16ToF32Buf(out_fp16);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 4: Sub — element-wise subtract [2, 4]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ElewiseOp precision: Sub fp16") {
    LOG_INFO("=== Elewise Sub precision ===");
    ArrayFp16 a, b, ref;
    REQUIRE(a.Load("/tmp/cpu_op_elewise_sub_a.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_elewise_sub_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_elewise_sub_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto op = atb_llm::ops::ElewiseOp::MakeSub();
    REQUIRE(op.get() != nullptr);
    auto out_f32 = RunBinaryFp16(runtime.get(), std::move(op), a, b);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 5: Cast fp16 → fp32 — verify exact upcast
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ElewiseOp precision: Cast fp16→fp32") {
    LOG_INFO("=== Elewise Cast fp16→fp32 precision ===");
    ArrayFp16 a;
    ArrayF32  ref;
    REQUIRE(a.Load("/tmp/cpu_op_elewise_cast_a.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_elewise_cast_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::ElewiseOp::MakeCast(ACL_FLOAT);
    REQUIRE(op.get() != nullptr);

    atb::Tensor ta, to;
    REQUIRE(IS_OK(alloc->AllocFloat16(ta, a.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat32(to, a.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(ta, a.data.data(), a.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors = {ta};
    vp.outTensors = {to};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws = nullptr;
    if (ws_size > 0) {
        auto [w, st] = runtime->GetWorkspace(ws_size);
        REQUIRE(IS_OK(st));
        ws = w;
    }
    REQUIRE(op.get()->Execute(vp, ws, ws_size, ctx) == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<float> out_f32(a.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(out_f32.data(), to, out_f32.size() * sizeof(float))));

    // Cast fp16→fp32 is exact: every element must equal the fp16 reference
    // upcast value bit-for-bit.
    bool exact = true;
    for (size_t i = 0; i < out_f32.size(); i++) {
        if (out_f32[i] != ref.data[i]) {
            LOG_ERROR("  mismatch [%zu]: got %.7g vs ref %.7g", i, out_f32[i], ref.data[i]);
            exact = false;
        }
    }
    CHECK(exact);

    float cs = CosineSim(out_f32.data(), ref.data.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f (expect exact)", cs);
    CHECK(cs >= 0.99f);
}
