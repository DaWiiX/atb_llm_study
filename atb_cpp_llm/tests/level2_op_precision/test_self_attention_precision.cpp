/**
 * Level 2 precision test: SelfAttentionOp vs PyTorch SDPA reference.
 *
 * ATB SelfAttentionOp uses BSND layout + PA_ENCODER and computes
 *     softmax(q @ k.T / sqrt(hd) + mask) @ v
 * internally (with optional MASK_TYPE_NORM causal/additive mask).
 *
 * Three cases:
 *   * mha_nomask — nh=4,  kvh=4, hd=32, no mask
 *   * gqa_nomask — nh=12, kvh=4, hd=64, no mask (replicated kv heads)
 *   * mha_causal — nh=4,  kvh=4, hd=32, with causal mask (upper-tri = -65504)
 *
 * Compares NPU fp16 output to a PyTorch fp32 SDPA reference computed on
 * the same fp16 inputs (round-tripped). cosine similarity ≥ 0.99.
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage op_self_attention
 *
 * Run: ./test_self_attention_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "ops/self_attention_op.h"
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

// ── Binary file readers (match gen_cpu_reference.py write_fp16 layout) ─
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

struct SaCase {
    int32_t S;
    int32_t nh;
    int32_t kvh;
    int32_t hd;
    bool    use_mask;
};

// Run one SelfAttention case and check cosine similarity vs PyTorch SDPA.
void RunSaCase(const std::string& name, const SaCase& cs) {
    LOG_INFO("=== SA precision: %s (S=%d nh=%d kvh=%d hd=%d mask=%d) ===",
             name.c_str(), cs.S, cs.nh, cs.kvh, cs.hd, cs.use_mask ? 1 : 0);

    ArrayFp16 q_in, k_in, v_in, mask_in, out_ref;
    REQUIRE(q_in.Load("/tmp/cpu_op_sa_" + name + "_q.bin"));
    REQUIRE(k_in.Load("/tmp/cpu_op_sa_" + name + "_k.bin"));
    REQUIRE(v_in.Load("/tmp/cpu_op_sa_" + name + "_v.bin"));
    REQUIRE(out_ref.Load("/tmp/cpu_op_sa_" + name + "_out_ref.bin"));
    if (cs.use_mask) {
        REQUIRE(mask_in.Load("/tmp/cpu_op_sa_" + name + "_mask.bin"));
    }

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    // ── Allocate device tensors (3D BSND layout, B=1 ⇒ B*S = S) ────
    atb::Tensor q_t, k_t, v_t, mask_t, seqlen_t, out_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(q_t,   {cs.S, cs.nh,  cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(k_t,   {cs.S, cs.kvh, cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(v_t,   {cs.S, cs.kvh, cs.hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t, {cs.S, cs.nh,  cs.hd})));
    if (cs.use_mask) {
        REQUIRE(IS_OK(alloc->AllocFloat16(mask_t, {cs.S, cs.S})));
    }
    REQUIRE(IS_OK(alloc->AllocInt64(seqlen_t, {1})));

    REQUIRE(IS_OK(alloc->CopyToDevice(q_t, q_in.data.data(),
                                      q_in.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(k_t, k_in.data.data(),
                                      k_in.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(v_t, v_in.data.data(),
                                      v_in.data.size() * sizeof(uint16_t))));
    if (cs.use_mask) {
        REQUIRE(IS_OK(alloc->CopyToDevice(mask_t, mask_in.data.data(),
                                          mask_in.data.size() * sizeof(uint16_t))));
    }

    int32_t seqlen_val = cs.S;
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    // ── Create and execute SelfAttentionOp ─────────────────────────
    auto op = atb_llm::ops::SelfAttentionOp::Create(cs.nh, cs.kvh, cs.hd, cs.use_mask);
    REQUIRE(op.get() != nullptr);

    atb::VariantPack vp;
    if (cs.use_mask) {
        vp.inTensors = {q_t, k_t, v_t, mask_t, seqlen_t};
    } else {
        vp.inTensors = {q_t, k_t, v_t, seqlen_t};
    }
    vp.outTensors = {out_t};

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

    // ── Copy back and compare ──────────────────────────────────────
    std::vector<uint16_t> out_host(cs.S * cs.nh * cs.hd);
    REQUIRE(IS_OK(alloc->CopyToHost(out_host.data(), out_t,
                                    out_host.size() * sizeof(uint16_t))));

    auto out_f32     = Fp16Decode(out_host);
    auto out_ref_f32 = Fp16Decode(out_ref.data);

    REQUIRE(out_f32.size() == out_ref_f32.size());
    float cs_score = CosineSim(out_f32.data(), out_ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs_score);
    CHECK(cs_score >= 0.99f);
}

SaCase LoadMeta(const std::string& name) {
    I32Vec meta;
    REQUIRE(meta.Load("/tmp/cpu_op_sa_" + name + "_meta.bin"));
    REQUIRE(meta.data.size() == 5);
    return SaCase{
        meta.data[0], meta.data[1], meta.data[2], meta.data[3],
        meta.data[4] != 0
    };
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: MHA without mask  (nh = kvh = 4, hd = 32)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SelfAttentionOp precision: MHA no mask") {
    auto cs = LoadMeta("mha_nomask");
    RunSaCase("mha_nomask", cs);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: GQA without mask  (nh = 12, kvh = 4, hd = 64)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SelfAttentionOp precision: GQA no mask") {
    auto cs = LoadMeta("gqa_nomask");
    RunSaCase("gqa_nomask", cs);
}

// ═════════════════════════════════════════════════════════════════
// Case 3: MHA with causal mask  (upper triangle = -65504 in fp16)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SelfAttentionOp precision: MHA causal mask") {
    auto cs = LoadMeta("mha_causal");
    RunSaCase("mha_causal", cs);
}
