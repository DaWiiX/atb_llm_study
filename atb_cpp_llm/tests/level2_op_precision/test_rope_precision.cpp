/**
 * Level 2 precision test: RopeOp vs Python contiguous-half reference.
 *
 * ATB RoPE with rotaryCoeff=2 (Qwen3VL default) applies LLAMA-style
 * contiguous-half rotation:
 *     x1 = x[..., :hd/2]; x2 = x[..., hd/2:]
 *     rotated = concat([-x2, x1], dim=-1)
 *     out = x * cos + rotated * sin
 *
 * Compares NPU fp16 RopeOp output to a Python fp32 reference computed
 * on the same fp16 inputs. Two cases:
 *   * basic — MHA (nh = kvh = 4)
 *   * gqa   — GQA (nh = 12, kvh = 4)
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_rope
 *
 * Run: ./test_rope_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/rope_op.h"
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

// ── Binary file readers (matches gen_cpu_reference.py format) ─────────
struct ArrayFp16 {
    std::vector<int64_t> shape;
    std::vector<uint16_t> data;

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
        fread(data.data(), sizeof(uint16_t), total, f);
        fclose(f);
        return true;
    }
};

struct I32Vec {
    std::vector<int32_t> data;
    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        int64_t count = 0;
        fread(&count, sizeof(int64_t), 1, f);
        data.resize(count);
        fread(data.data(), sizeof(int32_t), count, f);
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

std::vector<float> Fp16Decode(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = atb_llm::Fp16ToF32(src[i]);
    return dst;
}

struct RopeCase {
    int32_t S;
    int32_t nh;
    int32_t kvh;
    int32_t hd;
};

void RunRopeCase(const std::string& name, const RopeCase& cs) {
    LOG_INFO("=== RoPE precision: %s (S=%d nh=%d kvh=%d hd=%d) ===",
             name.c_str(), cs.S, cs.nh, cs.kvh, cs.hd);

    ArrayFp16 q_in, k_in, cos_in, sin_in, rq_ref, rk_ref;
    REQUIRE(q_in.Load(  "/tmp/cpu_op_rope_" + name + "_q.bin"));
    REQUIRE(k_in.Load(  "/tmp/cpu_op_rope_" + name + "_k.bin"));
    REQUIRE(cos_in.Load("/tmp/cpu_op_rope_" + name + "_cos.bin"));
    REQUIRE(sin_in.Load("/tmp/cpu_op_rope_" + name + "_sin.bin"));
    REQUIRE(rq_ref.Load("/tmp/cpu_op_rope_" + name + "_rq_ref.bin"));
    REQUIRE(rk_ref.Load("/tmp/cpu_op_rope_" + name + "_rk_ref.bin"));

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    // ── Allocate device tensors ────────────────────────────────────
    atb::Tensor q_t, k_t, cos_t, sin_t, seqlen_t, rq_t, rk_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(q_t,   {cs.S, cs.nh  * cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(k_t,   {cs.S, cs.kvh * cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(cos_t, {cs.S, cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(sin_t, {cs.S, cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(rq_t,  {cs.S, cs.nh  * cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(rk_t,  {cs.S, cs.kvh * cs.hd})));
    // seqlen tensor: int32, populated via hostData
    REQUIRE(IS_OK(alloc->AllocInt64(seqlen_t, {1})));

    REQUIRE(IS_OK(alloc->CopyToDevice(q_t,   q_in.data.data(),
                                      q_in.data.size()   * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(k_t,   k_in.data.data(),
                                      k_in.data.size()   * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(cos_t, cos_in.data.data(),
                                      cos_in.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(sin_t, sin_in.data.data(),
                                      sin_in.data.size() * sizeof(uint16_t))));

    int32_t seqlen_val = cs.S;  // batch=1
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    // ── Create and execute RopeOp ──────────────────────────────────
    auto op = atb_llm::ops::RopeOp::Create(2);
    REQUIRE(op.get() != nullptr);

    atb::VariantPack vp;
    vp.inTensors  = {q_t, k_t, cos_t, sin_t, seqlen_t};
    vp.outTensors = {rq_t, rk_t};

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

    // ── Copy back and compare ──────────────────────────────────────
    std::vector<uint16_t> rq_host(cs.S * cs.nh  * cs.hd);
    std::vector<uint16_t> rk_host(cs.S * cs.kvh * cs.hd);
    REQUIRE(IS_OK(alloc->CopyToHost(rq_host.data(), rq_t,
                                    rq_host.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToHost(rk_host.data(), rk_t,
                                    rk_host.size() * sizeof(uint16_t))));

    auto rq_f32     = Fp16Decode(rq_host);
    auto rk_f32     = Fp16Decode(rk_host);
    auto rq_ref_f32 = Fp16Decode(rq_ref.data);
    auto rk_ref_f32 = Fp16Decode(rk_ref.data);

    float cs_q = CosineSim(rq_f32.data(), rq_ref_f32.data(), rq_f32.size());
    float cs_k = CosineSim(rk_f32.data(), rk_ref_f32.data(), rk_f32.size());
    LOG_INFO("  cosine(q) = %.6f  cosine(k) = %.6f", cs_q, cs_k);
    CHECK(cs_q >= 0.99f);
    CHECK(cs_k >= 0.99f);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: basic MHA  — nh = kvh = 4, hd = 64
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RopeOp precision: basic (MHA rotaryCoeff=2)") {
    I32Vec meta;
    REQUIRE(meta.Load("/tmp/cpu_op_rope_basic_meta.bin"));
    REQUIRE(meta.data.size() == 4);
    RopeCase cs{meta.data[0], meta.data[1], meta.data[2], meta.data[3]};
    RunRopeCase("basic", cs);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: GQA — nh = 12, kvh = 4, hd = 64
// ═════════════════════════════════════════════════════════════════
TEST_CASE("RopeOp precision: GQA (nh=12, kvh=4)") {
    I32Vec meta;
    REQUIRE(meta.Load("/tmp/cpu_op_rope_gqa_meta.bin"));
    REQUIRE(meta.data.size() == 4);
    RopeCase cs{meta.data[0], meta.data[1], meta.data[2], meta.data[3]};
    RunRopeCase("gqa", cs);
}
