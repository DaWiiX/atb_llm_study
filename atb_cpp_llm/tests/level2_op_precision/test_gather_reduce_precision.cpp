/**
 * Level 2 precision test: GatherOp + ReduceOp vs NumPy reference.
 *
 * Cases:
 *   - GatherOp axis=0 on [8, 4] with indices [2, 5, 0]
 *   - ReduceOp MAX along axis=1 on [3, 5]
 *   - ReduceOp MIN along axis=1 on [3, 5]
 *   - ReduceOp SUM along axis=1 on [3, 5]
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_gather_reduce
 *
 * Run: ./test_gather_reduce_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/gather_op.h"
#include "ops/reduce_op.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
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

struct ArrayI64 {
    std::vector<int64_t> shape;
    std::vector<int64_t> data;
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
        fread(data.data(), sizeof(int64_t), data.size(), f);
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
// Case 1: Gather axis=0 — pick rows 2, 5, 0 from an [8, 4] table
// ═════════════════════════════════════════════════════════════════
TEST_CASE("GatherOp precision: axis=0 [8,4] indices=[2,5,0]") {
    LOG_INFO("=== Gather axis=0 precision ===");
    ArrayFp16 in, ref;
    ArrayI64  idx;
    REQUIRE(in.Load("/tmp/cpu_op_gather_in.bin"));
    REQUIRE(idx.Load("/tmp/cpu_op_gather_idx.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_gather_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::GatherOp::Create(0, 0);
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_idx, t_out;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in, in.shape)));
    REQUIRE(IS_OK(alloc->AllocInt64(t_idx, idx.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_out, ref.shape)));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_in,  in.data.data(),  in.data.size()  * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(t_idx, idx.data.data(), idx.data.size() * sizeof(int64_t))));

    atb::VariantPack vp;
    vp.inTensors = {t_in, t_idx};
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
    auto out_f32 = Fp16ToF32Buf(host);
    auto ref_f32 = Fp16ToF32Buf(ref.data);

    // Gather is a permutation — must be bit-identical fp16 → cosine = 1.0
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f (expect ~1.0)", cs);
    CHECK(cs >= 0.99f);
}

namespace {

// Run a reduce op along axis=1 on a 2D input and return decoded fp32 output.
std::vector<float> RunReduce(atb_llm::IRuntime* runtime,
                             atb_llm::ops::ReduceOp::ReduceType type,
                             const ArrayFp16& in) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::ReduceOp::Create(type, {1});
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_out;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in, in.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_out, {in.shape[0]})));  // reduces axis=1
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

    std::vector<uint16_t> host(in.shape[0]);
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), t_out, host.size() * sizeof(uint16_t))));
    return Fp16ToF32Buf(host);
}

// Decode bf16 (stored as uint16_t bit patterns) to fp32.
inline std::vector<float> Bf16ToF32Buf(const std::vector<uint16_t>& bits) {
    std::vector<float> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        uint32_t u = static_cast<uint32_t>(bits[i]) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(float));
        out[i] = f;
    }
    return out;
}

// Run a reduce op along axis=1 on a 2D bf16 input. Overrides tensor dtype
// to ACL_BF16 after fp16 allocation (since AllocFloat16 is the only public
// 2-byte allocator and ATB ReduceOp MAX/MIN refuses fp16).
std::vector<float> RunReduceBf16(atb_llm::IRuntime* runtime,
                                 atb_llm::ops::ReduceOp::ReduceType type,
                                 const ArrayFp16& in) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    auto op = atb_llm::ops::ReduceOp::Create(type, {1});
    REQUIRE(op.get() != nullptr);

    atb::Tensor t_in, t_out;
    REQUIRE(IS_OK(alloc->AllocFloat16(t_in, in.shape)));
    REQUIRE(IS_OK(alloc->AllocFloat16(t_out, {in.shape[0]})));
    // Reinterpret as bf16 — storage size identical (2 bytes/element).
    t_in.desc.dtype = ACL_BF16;
    t_out.desc.dtype = ACL_BF16;
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

    std::vector<uint16_t> host(in.shape[0]);
    REQUIRE(IS_OK(alloc->CopyToHost(host.data(), t_out, host.size() * sizeof(uint16_t))));
    return Bf16ToF32Buf(host);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 2: Reduce MAX along axis=1 (bf16 — ATB requires bf16 or int32)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ReduceOp precision: MAX axis=1 [3,5] (bf16)") {
    LOG_INFO("=== Reduce MAX precision (bf16) ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_reduce_bf16_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_reduce_bf16_max_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunReduceBf16(runtime.get(), atb_llm::ops::ReduceOp::ReduceType::MAX, in);
    auto ref_f32 = Bf16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: Reduce MIN along axis=1 (bf16 — ATB requires bf16 or int32)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ReduceOp precision: MIN axis=1 [3,5] (bf16)") {
    LOG_INFO("=== Reduce MIN precision (bf16) ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_reduce_bf16_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_reduce_bf16_min_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunReduceBf16(runtime.get(), atb_llm::ops::ReduceOp::ReduceType::MIN, in);
    auto ref_f32 = Bf16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 4: Reduce SUM along axis=1
// ═════════════════════════════════════════════════════════════════
TEST_CASE("ReduceOp precision: SUM axis=1 [3,5]") {
    LOG_INFO("=== Reduce SUM precision ===");
    ArrayFp16 in, ref;
    REQUIRE(in.Load("/tmp/cpu_op_reduce_in.bin"));
    REQUIRE(ref.Load("/tmp/cpu_op_reduce_sum_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto out_f32 = RunReduce(runtime.get(), atb_llm::ops::ReduceOp::ReduceType::SUM, in);
    auto ref_f32 = Fp16ToF32Buf(ref.data);
    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
