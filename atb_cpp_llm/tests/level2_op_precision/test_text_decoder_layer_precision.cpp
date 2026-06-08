/**
 * Level 2 precision test: TextDecoderLayerGraph vs PyTorch CPU reference.
 *
 * Compares the NPU fp16 full text decoder layer output to a pure PyTorch
 * reference (no torch_atb dependency) via cosine similarity.
 *
 * Layer composition (matches text_decoder_layer.py):
 *   h = x
 *   normed = RMSNorm(h, iln_w)
 *   attn_out = SelfAttention(normed, q_w, k_w, v_w, o_w, qn_w, kn_w, cos, sin, [mask])
 *   h = h + attn_out
 *   normed2 = RMSNorm(h, pln_w)
 *   mlp_out = SwiGluMLP(normed2, gate_w, up_w, down_w)
 *   out = h + mlp_out
 *
 * VariantPack input order (no mask, use_qk_norm=true, 15 inputs):
 *   hidden_states, q_w, k_w, v_w, o_w, qn_w, kn_w,
 *   gate_w, up_w, down_w, iln_w, pln_w,
 *   cos, sin, seqlen
 * With mask: insert mask before seqlen (16 inputs).
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage text_decoder_layer
 *
 * Run: ./test_text_decoder_layer_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "components/text/decoder_layer_graph.h"
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

struct ArrayFP16 {
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

struct ArrayI32 {
    std::vector<int32_t> data;
    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { LOG_ERROR("Cannot open %s", path.c_str()); return false; }
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

std::vector<float> Fp16ToF32(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = atb_llm::Fp16ToF32(src[i]);
    return dst;
}

struct Case {
    std::string name;
    int32_t S, nh, kvh, hd, I;
    bool use_mask;
};

float RunDecoderLayer(atb_llm::IRuntime* runtime, const Case& c) {
    int32_t S = c.S, nh = c.nh, kvh = c.kvh, hd = c.hd, I = c.I;
    int32_t H = nh * hd;
    int32_t Hkv = kvh * hd;
    bool use_mask = c.use_mask;
    const std::string pfx = "/tmp/cpu_dec_" + c.name + "_";

    // Load every fp16 tensor from disk
    ArrayFP16 x, qw, kw, vw, ow, qnw, knw;
    ArrayFP16 gw, uw, dw, ilnw, plnw, cos_a, sin_a, mask_a, ref;
    REQUIRE(x.Load   (pfx + "x.bin"));
    REQUIRE(qw.Load  (pfx + "q_w.bin"));
    REQUIRE(kw.Load  (pfx + "k_w.bin"));
    REQUIRE(vw.Load  (pfx + "v_w.bin"));
    REQUIRE(ow.Load  (pfx + "o_w.bin"));
    REQUIRE(qnw.Load (pfx + "qn_w.bin"));
    REQUIRE(knw.Load (pfx + "kn_w.bin"));
    REQUIRE(gw.Load  (pfx + "gate_w.bin"));
    REQUIRE(uw.Load  (pfx + "up_w.bin"));
    REQUIRE(dw.Load  (pfx + "down_w.bin"));
    REQUIRE(ilnw.Load(pfx + "iln_w.bin"));
    REQUIRE(plnw.Load(pfx + "pln_w.bin"));
    REQUIRE(cos_a.Load(pfx + "cos.bin"));
    REQUIRE(sin_a.Load(pfx + "sin.bin"));
    if (use_mask) REQUIRE(mask_a.Load(pfx + "mask.bin"));
    REQUIRE(ref.Load (pfx + "out_ref.bin"));

    // Build the decoder layer
    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::text::TextDecoderLayerGraph::Build(
        "PrecDecoderLayer_" + c.name,
        nh, kvh, hd, S, 1e-6f, use_mask, op);
    REQUIRE(IS_OK(s));
    REQUIRE(op.get() != nullptr);
    REQUIRE(op.get()->GetInputNum() == static_cast<uint32_t>(use_mask ? 16 : 15));

    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    // Allocate device tensors
    atb::Tensor in_x, in_qw, in_kw, in_vw, in_ow, in_qnw, in_knw;
    atb::Tensor in_gw, in_uw, in_dw, in_iln, in_pln;
    atb::Tensor in_cos, in_sin, in_mask, in_seqlen, out_t;

    REQUIRE(IS_OK(alloc->AllocFloat16(in_x,   {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_qw,  {H,   H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_kw,  {Hkv, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_vw,  {Hkv, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_ow,  {H,   H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_qnw, {hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_knw, {hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_gw,  {I, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_uw,  {I, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_dw,  {H, I})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_iln, {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_pln, {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_cos, {S, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_sin, {S, hd})));
    if (use_mask) {
        REQUIRE(IS_OK(alloc->AllocFloat16(in_mask, {S, S})));
    }
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t, {S, H})));

    auto upload = [&](atb::Tensor& dst, const ArrayFP16& src) {
        REQUIRE(IS_OK(alloc->CopyToDevice(dst, src.data.data(),
                                          src.data.size() * sizeof(uint16_t))));
    };
    upload(in_x,   x);
    upload(in_qw,  qw);  upload(in_kw,  kw);  upload(in_vw, vw);  upload(in_ow, ow);
    upload(in_qnw, qnw); upload(in_knw, knw);
    upload(in_gw,  gw);  upload(in_uw,  uw);  upload(in_dw, dw);
    upload(in_iln, ilnw); upload(in_pln, plnw);
    upload(in_cos, cos_a); upload(in_sin, sin_a);
    if (use_mask) upload(in_mask, mask_a);

    // seqlen: single int32 = S (batch=1)
    int32_t seqlen_val = S;
    in_seqlen.desc.dtype = ACL_INT32;
    in_seqlen.desc.format = ACL_FORMAT_ND;
    in_seqlen.desc.shape.dimNum = 1;
    in_seqlen.desc.shape.dims[0] = 1;
    in_seqlen.dataSize = sizeof(int32_t);
    in_seqlen.hostData = &seqlen_val;

    atb::VariantPack vp;
    if (use_mask) {
        vp.inTensors = {
            in_x, in_qw, in_kw, in_vw, in_ow, in_qnw, in_knw,
            in_gw, in_uw, in_dw, in_iln, in_pln,
            in_cos, in_sin, in_mask, in_seqlen};
    } else {
        vp.inTensors = {
            in_x, in_qw, in_kw, in_vw, in_ow, in_qnw, in_knw,
            in_gw, in_uw, in_dw, in_iln, in_pln,
            in_cos, in_sin, in_seqlen};
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

    std::vector<uint16_t> host_out(S * H);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out.data(), out_t,
                                    host_out.size() * sizeof(uint16_t))));
    auto host_f32 = Fp16ToF32(host_out);
    auto ref_f32  = Fp16ToF32(ref.data);
    REQUIRE(static_cast<int64_t>(ref_f32.size()) == S * H);
    return CosineSim(host_f32.data(), ref_f32.data(), S * H);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: small, no mask (nh=4, kvh=4, hd=32, I=64, S=4)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("TextDecoderLayerGraph precision: small no-mask") {
    LOG_INFO("=== TextDecoderLayerGraph precision: small no-mask ===");

    ArrayI32 meta;
    REQUIRE(meta.Load("/tmp/cpu_dec_small_nomask_meta.bin"));
    REQUIRE(meta.data.size() == 6);
    Case c;
    c.name = "small_nomask";
    c.S    = meta.data[0];
    c.nh   = meta.data[1];
    c.kvh  = meta.data[2];
    c.hd   = meta.data[3];
    c.I    = meta.data[4];
    c.use_mask = (meta.data[5] != 0);
    REQUIRE_FALSE(c.use_mask);

    auto runtime = atb_llm::CreateRuntime(0, 3LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunDecoderLayer(runtime.get(), c);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: GQA with causal mask (nh=12, kvh=4, hd=64, I=256, S=8)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("TextDecoderLayerGraph precision: GQA with mask") {
    LOG_INFO("=== TextDecoderLayerGraph precision: GQA + mask ===");

    ArrayI32 meta;
    REQUIRE(meta.Load("/tmp/cpu_dec_gqa_mask_meta.bin"));
    REQUIRE(meta.data.size() == 6);
    Case c;
    c.name = "gqa_mask";
    c.S    = meta.data[0];
    c.nh   = meta.data[1];
    c.kvh  = meta.data[2];
    c.hd   = meta.data[3];
    c.I    = meta.data[4];
    c.use_mask = (meta.data[5] != 0);
    REQUIRE(c.use_mask);

    auto runtime = atb_llm::CreateRuntime(0, 3LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunDecoderLayer(runtime.get(), c);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
