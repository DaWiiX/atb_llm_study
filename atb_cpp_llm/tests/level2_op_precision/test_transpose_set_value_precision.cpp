/**
 * Level 2 precision test: TransposeOp + SetValueOp vs NumPy reference.
 *
 * Cases:
 *   - Transpose perm=[1,0] on [3, 5]      (2D)
 *   - Transpose perm=[0,2,1,3] on [2,4,3,5] (4D)
 *   - SetValue: embed a [2, 3] source tile into a [4, 6] canvas at [1:3, 2:5]
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_transpose_set_value
 *
 * Run: ./test_transpose_set_value_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/transpose_op.h"
#include "ops/set_value_op.h"
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

// Run Transpose with a given perm on a fp16 input; output shape derived from ref.
std::vector<float> RunTranspose(atb_llm::IRuntime* runtime,
                                const std::vector<int32_t>& perm,
                                const ArrayFp16& in,
                                const ArrayFp16& ref) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::TransposeOp::Create(perm);
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_out;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in, in.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_out, ref.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_in, in.data.data(), in.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors = {t_in};
    vp.outTensors = {t_out};

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
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), t_out, host.size() * sizeof(uint16_t))));
    return Fp16ToF32Buf(host);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: Transpose perm=[1, 0] on [3, 5] → [5, 3]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("TransposeOp precision: perm=[1,0] [3,5]") {
    LOG_INFO("=== Transpose 2D precision ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_transpose2d_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_transpose2d_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunTranspose(runtime.get(), {1, 0}, in, ref);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f (expect ~1.0, permutation only)", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: Transpose perm=[0, 2, 1, 3] on [2, 4, 3, 5] → [2, 3, 4, 5]
// ═════════════════════════════════════════════════════════════════
TEST_CASE("TransposeOp precision: perm=[0,2,1,3] [2,4,3,5]") {
    LOG_INFO("=== Transpose 4D precision ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_transpose4d_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_transpose4d_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunTranspose(runtime.get(), {0, 2, 1, 3}, in, ref);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f (expect ~1.0, permutation only)", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: SetValue — embed [2, 3] tile into [4, 6] canvas at [1:3, 2:5].
//
// ATB SetValueParam writes a source tensor into a destination region.
// We initialize the dst with zeros, then run SetValue with starts={1,2},
// ends={3,5} and compare to numpy slice assignment.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SetValueOp precision: tile [2,3] into [4,6] at [1:3, 2:5]") {
    LOG_INFO("=== SetValue precision ===");
    ArrayFp16 dst_init, src, ref;
    REQUIRE(dst_init.Load("/tmp/cpu_op_setvalue_dst_init.bin"));
    REQUIRE(src.Load("/tmp/cpu_op_setvalue_src.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_setvalue_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::SetValueOp::Create({1, 2}, {3, 5});
    REQUIRE(op.get() != nullptr);

    // ATB SetValue is an in-place op: dst is both input and output.
    atb::Tensor t_dst, t_src;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_dst, dst_init.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_src, src.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_dst, dst_init.data.data(),
                                      dst_init.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_src, src.data.data(),
                                      src.data.size() * sizeof(uint16_t))));

    // SetValue takes (dst, src) as inputs and writes dst in-place.
    atb::VariantPack vp;
    vp.inTensors = {t_dst, t_src};
    vp.outTensors = {};  // in-place; no output tensors

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
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), t_dst, host.size() * sizeof(uint16_t))));
    auto out_f32 = Fp16ToF32Buf(host);
    auto ref_f32 = Fp16ToF32Buf(ref.data);

    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);

    // Bit-exact check: the values in the embedded region must equal `src`,
    // and the values outside must remain zero.
    int64_t H = dst_init.shape[0], W = dst_init.shape[1];
    bool exact = true;
    for (int64_t i = 0; i < H; i++) {
        for (int64_t j = 0; j < W; j++) {
            float got = out_f32[i * W + j];
            float expected = ref_f32[i * W + j];
            if (got != expected) {
                LOG_ERROR("  mismatch (%lld,%lld): got=%g vs ref=%g",
                          static_cast<long long>(i), static_cast<long long>(j),
                          got, expected);
                exact = false;
            }
        }
    }
    CHECK(exact);
}
