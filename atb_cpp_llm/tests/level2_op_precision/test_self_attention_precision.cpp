/**
 * Level 2 precision test: SelfAttentionOp vs PyTorch SDPA reference.
 *
 * ATB SelfAttentionOp uses BSND layout + PA_ENCODER and computes
 *     softmax(q @ k.T / sqrt(hd) + mask) @ v
 * internally (with optional MASK_TYPE_NORM causal/additive mask).
 *
 * On 310P: mask must be in NZ (FRACTAL_NZ) format.
 * On 910B: mask must be in ND format.
 *
 * Compares NPU fp16 output to a PyTorch fp32 SDPA reference computed on
 * the same fp16 inputs (round-tripped). cosine similarity >= 0.99.
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
#include "utils/cpp11_compat.h"
#include "test_mask_helper.h"

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

// ── Run one SelfAttention case ──────────────────────────────────────
// name:     test case name (matches .bin file prefix)
// S, nh, kvh, hd, use_mask: decoded from meta file
// expected_result: true if the test should pass, false if it should fail
//                  (e.g. non-16-aligned S on 310P may fail)
struct SaResult {
    bool    ok;
    float   cosine;
    int32_t atb_error;
    std::string error_msg;
};

SaResult RunOneCase(const std::string& name, int32_t S, int32_t nh,
                    int32_t kvh, int32_t hd, bool use_mask) {
    SaResult res = {false, 0.0f, 0, ""};

    ArrayFp16 q_in, k_in, v_in, mask_in, out_ref;
    if (!q_in.Load("/tmp/cpu_op_sa_" + name + "_q.bin")) {
        res.error_msg = "Cannot load q reference data";
        return res;
    }
    if (!k_in.Load("/tmp/cpu_op_sa_" + name + "_k.bin")) {
        res.error_msg = "Cannot load k reference data";
        return res;
    }
    if (!v_in.Load("/tmp/cpu_op_sa_" + name + "_v.bin")) {
        res.error_msg = "Cannot load v reference data";
        return res;
    }
    if (!out_ref.Load("/tmp/cpu_op_sa_" + name + "_out_ref.bin")) {
        res.error_msg = "Cannot load out_ref reference data";
        return res;
    }
    if (use_mask) {
        if (!mask_in.Load("/tmp/cpu_op_sa_" + name + "_mask.bin")) {
            res.error_msg = "Cannot load mask reference data";
            return res;
        }
    }

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    if (!runtime) {
        res.error_msg = "Cannot create runtime";
        return res;
    }
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    // ── Allocate and upload Q/K/V ──────────────────────────────────
    atb::Tensor q_t, k_t, v_t, mask_t, out_t, seqlen_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(q_t,   {S, nh,  hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(k_t,   {S, kvh, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(v_t,   {S, kvh, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t, {S, nh,  hd})));

    REQUIRE(IS_OK(alloc->CopyToDevice(q_t, q_in.data.data(),
                                      q_in.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(k_t, k_in.data.data(),
                                      k_in.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(v_t, v_in.data.data(),
                                      v_in.data.size() * sizeof(uint16_t))));

    // ── Prepare mask ──────────────────────────────────────────────
    // Platform format handled automatically by test::UploadMask
    if (use_mask) {
        atb_llm::test::UploadMask(alloc, mask_in.data.data(), S, mask_t);
    }

    // ── Seqlen tensor ─────────────────────────────────────────────
    int32_t seqlen_val = S;
    seqlen_t.desc.dtype = ACL_INT32;
    seqlen_t.desc.format = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    // ── Create and execute SelfAttentionOp ────────────────────────
    auto op = atb_llm::ops::SelfAttentionOp::Create(nh, kvh, hd, use_mask);
    if (!op.get()) {
        res.error_msg = "CreateOp failed";
        return res;
    }

    atb::VariantPack vp;
    if (use_mask) {
        vp.inTensors = {q_t, k_t, v_t, mask_t, seqlen_t};
    } else {
        vp.inTensors = {q_t, k_t, v_t, seqlen_t};
    }
    vp.outTensors = {out_t};

    uint64_t ws_size = 0;
    atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        res.atb_error = static_cast<int32_t>(atb_s);
        res.error_msg = "Setup failed (error " + std::to_string(res.atb_error) + ")";
        return res;
    }

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime->GetWorkspace(ws_size);
        auto& ws = __atb_pair_ws.first;
        auto& ws_st = __atb_pair_ws.second;
        if (!IS_OK(ws_st)) {
            res.error_msg = "GetWorkspace failed";
            return res;
        }
        ws_ptr = ws;
    }

    atb_s = op.get()->Execute(vp, ws_ptr, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        res.atb_error = static_cast<int32_t>(atb_s);
        res.error_msg = "Execute failed (error " + std::to_string(res.atb_error) + ")";
        return res;
    }
    runtime->Synchronize();

    // ── Copy back and compare ─────────────────────────────────────
    std::vector<uint16_t> out_host(static_cast<size_t>(S) * nh * hd);
    REQUIRE(IS_OK(alloc->CopyToHost(out_host.data(), out_t,
                                    out_host.size() * sizeof(uint16_t))));

    auto out_f32     = Fp16Decode(out_host);
    auto out_ref_f32 = Fp16Decode(out_ref.data);

    REQUIRE(out_f32.size() == out_ref_f32.size());
    res.cosine = CosineSim(out_f32.data(), out_ref_f32.data(), out_f32.size());
    res.ok = true;
    return res;
}

// ── Test case descriptor ───────────────────────────────────────────
struct TestCase {
    std::string name;
    int32_t S, nh, kvh, hd;
    bool use_mask;
    bool skip_on_910b;  // if true, only run on 310P
};

void RunTestCase(const TestCase& tc) {
    std::string skip_reason;
    if (tc.skip_on_910b && !atb_llm::Is310P()) {
        MESSAGE("Skipping ", tc.name, " (310P-only test)");
        return;
    }
    INFO("=== SA precision: ", tc.name,
         " (S=", tc.S, " nh=", tc.nh, " kvh=", tc.kvh,
         " hd=", tc.hd, " mask=", tc.use_mask ? "causal" : "none", ") ===");
    CAPTURE(tc.name);
    CAPTURE(tc.S);
    CAPTURE(tc.nh);
    CAPTURE(tc.kvh);
    CAPTURE(tc.hd);

    auto res = RunOneCase(tc.name, tc.S, tc.nh, tc.kvh, tc.hd, tc.use_mask);

    if (!res.ok) {
        if (res.atb_error != 0) {
            FAIL_CHECK(tc.name << " ATB error " << res.atb_error
                       << ": " << res.error_msg);
        } else {
            FAIL_CHECK(tc.name << ": " << res.error_msg);
        }
        return;
    }

    INFO(tc.name, " cosine = ", res.cosine);
    CHECK(res.cosine >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════════
// Helper to load meta and create test case
// ═════════════════════════════════════════════════════════════════════
TestCase LoadMeta(const std::string& name, bool skip_on_910b = false) {
    I32Vec meta;
    REQUIRE(meta.Load("/tmp/cpu_op_sa_" + name + "_meta.bin"));
    REQUIRE(meta.data.size() == 5);
    return TestCase{
        name,
        meta.data[0], meta.data[1], meta.data[2], meta.data[3],
        meta.data[4] != 0,
        skip_on_910b
    };
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════
// Group 1: Basic (works on both platforms)
// ═════════════════════════════════════════════════════════════════════

TEST_CASE("SelfAttentionOp: MHA no mask (S=8, nh=4, hd=32)") {
    RunTestCase(LoadMeta("mha_nomask"));
}

TEST_CASE("SelfAttentionOp: GQA no mask (S=8, nh=12, kvh=4, hd=64)") {
    RunTestCase(LoadMeta("gqa_nomask"));
}

TEST_CASE("SelfAttentionOp: MHA causal mask (S=8, nh=4, hd=32)") {
    RunTestCase(LoadMeta("mha_causal"));
}

// ═════════════════════════════════════════════════════════════════════
// Group 2: NZ mask — different S values (critical for 310P)
// ═════════════════════════════════════════════════════════════════════

TEST_CASE("SelfAttentionOp: MHA causal S=4 (not 16-aligned)") {
    RunTestCase(LoadMeta("mha_causal_s4"));
}

TEST_CASE("SelfAttentionOp: MHA causal S=16 (16-aligned)") {
    RunTestCase(LoadMeta("mha_causal_s16"));
}

TEST_CASE("SelfAttentionOp: MHA causal S=32 (16-aligned)") {
    RunTestCase(LoadMeta("mha_causal_s32"));
}

// ═════════════════════════════════════════════════════════════════════
// Group 3: NZ mask — real model params (nh=16, hd=128)
// ═════════════════════════════════════════════════════════════════════

TEST_CASE("SelfAttentionOp: MHA causal hd=128 S=4 (real model)") {
    RunTestCase(LoadMeta("mha_causal_hd128_s4"));
}

TEST_CASE("SelfAttentionOp: MHA causal hd=128 S=16 (real model)") {
    RunTestCase(LoadMeta("mha_causal_hd128_s16"));
}

TEST_CASE("SelfAttentionOp: MHA no mask hd=128 S=16 (sanity)") {
    RunTestCase(LoadMeta("mha_nomask_hd128_s16"));
}

// ═════════════════════════════════════════════════════════════════════
// Group 4: GQA with causal mask (verified on 310P, cos=1.0)
// ═════════════════════════════════════════════════════════════════════

TEST_CASE("SelfAttentionOp: GQA causal mask (S=8, nh=12, kvh=4, hd=64)") {
    RunTestCase(LoadMeta("gqa_causal"));
}
