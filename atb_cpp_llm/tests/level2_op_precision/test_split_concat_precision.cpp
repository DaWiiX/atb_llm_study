/**
 * Level 2 precision test: SplitOp / ConcatOp vs NumPy reference.
 *
 * Cases:
 *   - Split dim=-1 num=2  on [4, 8]
 *   - Split dim=-1 num=3  on [2, 9]
 *   - Concat dim=0  of two [3, 4]
 *   - Concat dim=-1 of two [4, 3]
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_split_concat
 *
 * Run: ./test_split_concat_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/split_op.h"
#include "ops/concat_op.h"
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

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: Split dim=-1 num=2  [4, 8] → 2 × [4, 4]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SplitOp precision: dim=-1 num=2 [4,8]") {
    LOG_INFO("=== Split num=2 precision ===");
    ArrayFp16 in, p0, p1;
    REQUIRE(in.Load("/tmp/cpu_op_split2_in.bin"));
    REQUIRE(p0.Load("/tmp/cpu_op_split2_p0.bin"));
    REQUIRE(p1.Load("/tmp/cpu_op_split2_p1.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::SplitOp::Create(-1, 2);
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_o0, t_o1;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in, in.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_o0, p0.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_o1, p1.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_in, in.data.data(), in.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors = {t_in};
    vp.outTensors = {t_o0, t_o1};

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

    std::vector<uint16_t> h0(p0.numel()), h1(p1.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(h0.data(), t_o0, h0.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToHost(h1.data(), t_o1, h1.size() * sizeof(uint16_t))));

    auto out0 = Fp16ToF32Buf(h0);
    auto out1 = Fp16ToF32Buf(h1);
    auto ref0 = Fp16ToF32Buf(p0.data);
    auto ref1 = Fp16ToF32Buf(p1.data);

    float cs0 = CosineSim(out0.data(), ref0.data(), out0.size());
    float cs1 = CosineSim(out1.data(), ref1.data(), out1.size());
    LOG_INFO("  cosine part0=%.6f  part1=%.6f", cs0, cs1);
    CHECK(cs0 >= 0.99f);
    CHECK(cs1 >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: Split dim=-1 num=3  [2, 9] → 3 × [2, 3]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SplitOp precision: dim=-1 num=3 [2,9]") {
    LOG_INFO("=== Split num=3 precision ===");
    ArrayFp16 in, p0, p1, p2;
    REQUIRE(in.Load("/tmp/cpu_op_split3_in.bin"));
    REQUIRE(p0.Load("/tmp/cpu_op_split3_p0.bin"));
    REQUIRE(p1.Load("/tmp/cpu_op_split3_p1.bin"));
    REQUIRE(p2.Load("/tmp/cpu_op_split3_p2.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::SplitOp::Create(-1, 3);
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_o0, t_o1, t_o2;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in, in.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_o0, p0.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_o1, p1.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_o2, p2.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_in, in.data.data(), in.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors = {t_in};
    vp.outTensors = {t_o0, t_o1, t_o2};

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

    std::vector<uint16_t> h0(p0.numel()), h1(p1.numel()), h2(p2.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(h0.data(), t_o0, h0.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToHost(h1.data(), t_o1, h1.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToHost(h2.data(), t_o2, h2.size() * sizeof(uint16_t))));

    auto out0 = Fp16ToF32Buf(h0);
    auto out1 = Fp16ToF32Buf(h1);
    auto out2 = Fp16ToF32Buf(h2);
    auto ref0 = Fp16ToF32Buf(p0.data);
    auto ref1 = Fp16ToF32Buf(p1.data);
    auto ref2 = Fp16ToF32Buf(p2.data);

    float cs0 = CosineSim(out0.data(), ref0.data(), out0.size());
    float cs1 = CosineSim(out1.data(), ref1.data(), out1.size());
    float cs2 = CosineSim(out2.data(), ref2.data(), out2.size());
    LOG_INFO("  cosine p0=%.6f  p1=%.6f  p2=%.6f", cs0, cs1, cs2);
    CHECK(cs0 >= 0.99f);
    CHECK(cs1 >= 0.99f);
    CHECK(cs2 >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: Concat dim=0 of two [3, 4]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ConcatOp precision: dim=0 [3,4]+[3,4]") {
    LOG_INFO("=== Concat dim=0 precision ===");
    ArrayFp16 a, b, ref;
    REQUIRE(a.Load("/tmp/cpu_op_concat0_a.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_concat0_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_concat0_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::ConcatOp::Create(0);
    REQUIRE(op.get() != nullptr);

    atb::Tensor ta, tb, to;
    REQUIRE(IS_OK(alloc->AllocFloat16(ta, a.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(tb, b.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(to, ref.shape)));
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

    std::vector<uint16_t> host(ref.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), to, host.size() * sizeof(uint16_t))));
    auto out_f32 = Fp16ToF32Buf(host);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 4: Concat dim=-1 of two [4, 3]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ConcatOp precision: dim=-1 [4,3]+[4,3]") {
    LOG_INFO("=== Concat dim=-1 precision ===");
    ArrayFp16 a, b, ref;
    REQUIRE(a.Load("/tmp/cpu_op_concat1_a.bin"));
    REQUIRE(b.Load("/tmp/cpu_op_concat1_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_concat1_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::ConcatOp::Create(-1);
    REQUIRE(op.get() != nullptr);

    atb::Tensor ta, tb, to;
    REQUIRE(IS_OK(alloc->AllocFloat16(ta, a.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(tb, b.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(to, ref.shape)));
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

    std::vector<uint16_t> host(ref.numel());
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), to, host.size() * sizeof(uint16_t))));
    auto out_f32 = Fp16ToF32Buf(host);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
