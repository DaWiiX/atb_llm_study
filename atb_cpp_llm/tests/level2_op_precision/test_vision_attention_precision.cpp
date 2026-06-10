/**
 * Level 2 precision test: VisionAttentionGraph vs PyTorch reference.
 *
 * Pipelines tested:
 *   1) cos=1, sin=0 identity RoPE  (historical case, validates the
 *      QKV / SDPA / projection arithmetic without exercising rotation).
 *   2) Real vision RoPE cos/sin built from VisionRotaryEmbedding (matches
 *      what the Python engine feeds into ATB RopeOp). This actually
 *      exercises the half-rotation branch end-to-end.
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage vision_attention
 *
 * Run: ./test_vision_attention_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "components/vision/vision_attention_graph.h"
#include "engine/runtime_impl.h"
#include "utils/float_utils.h"
#include "log/logger.h"

#include "acl/acl.h"

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

struct MetaI32 {
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

std::vector<float> Fp16ToF32(const std::vector<uint16_t>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) dst[i] = atb_llm::Fp16ToF32(src[i]);
    return dst;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case: VisionAttention end-to-end precision
// ═════════════════════════════════════════════════════════════════
TEST_CASE("VisionAttentionGraph precision: N=16 nh=4 hd=32") {
    LOG_INFO("=== VisionAttention precision ===");

    ArrayFp16 x, qkv_w, qkv_b, proj_w, proj_b, cos, sin, ref;
    MetaI32 meta;
    REQUIRE(x.Load("/tmp/cpu_vision_attn_x.bin"));
    REQUIRE(qkv_w.Load("/tmp/cpu_vision_attn_qkv_w.bin"));
    REQUIRE(qkv_b.Load("/tmp/cpu_vision_attn_qkv_b.bin"));
    REQUIRE(proj_w.Load("/tmp/cpu_vision_attn_proj_w.bin"));
    REQUIRE(proj_b.Load("/tmp/cpu_vision_attn_proj_b.bin"));
    REQUIRE(cos.Load("/tmp/cpu_vision_attn_cos.bin"));
    REQUIRE(sin.Load("/tmp/cpu_vision_attn_sin.bin"));
    REQUIRE(ref.Load("/tmp/cpu_vision_attn_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_vision_attn_meta.bin"));
    REQUIRE(meta.data.size() == 3);
    int64_t N  = meta.data[0];
    int64_t nh = meta.data[1];
    int64_t hd = meta.data[2];
    int64_t hidden = nh * hd;

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb_llm::OperationHandle op;
    REQUIRE(IS_OK(atb_llm::components::VisionAttentionGraph::Build(
        "VAttnPrec", static_cast<int32_t>(nh), static_cast<int32_t>(hd), op)));
    REQUIRE(op.get() != nullptr);

    atb::Tensor x_t, qkvw_t, qkvb_t, projw_t, projb_t, cos_t, sin_t, seqlen_t, out_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(x_t,     {N, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(qkvw_t,  {3 * hidden, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(qkvb_t,  {3 * hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(projw_t, {hidden, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(projb_t, {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(cos_t,   {N, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(sin_t,   {N, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t,   {N, hidden})));

    REQUIRE(IS_OK(alloc->CopyToDevice(x_t,     x.data.data(),      x.data.size()      * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(qkvw_t,  qkv_w.data.data(),  qkv_w.data.size()  * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(qkvb_t,  qkv_b.data.data(),  qkv_b.data.size()  * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(projw_t, proj_w.data.data(), proj_w.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(projb_t, proj_b.data.data(), proj_b.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(cos_t,   cos.data.data(),    cos.data.size()    * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(sin_t,   sin.data.data(),    sin.data.size()    * 2)));

    // seqlen: host int32, [1] with value N
    int32_t seqlen_val = static_cast<int32_t>(N);
    seqlen_t.desc.dtype     = ACL_INT32;
    seqlen_t.desc.format    = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    atb::VariantPack vp;
    vp.inTensors  = {x_t, qkvw_t, qkvb_t, projw_t, projb_t, cos_t, sin_t, seqlen_t};
    vp.outTensors = {out_t};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_st = __atb_pair_ws.second;
        REQUIRE(IS_OK(ws_st));
        ws_ptr = ws;
    }
    REQUIRE(op.get()->Execute(vp, ws_ptr, ws_size, ctx) == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> host_out(N * hidden);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out.data(), out_t, host_out.size() * 2)));
    auto out_f32 = Fp16ToF32(host_out);
    auto ref_f32 = Fp16ToF32(ref.data);
    REQUIRE(out_f32.size() == ref_f32.size());

    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case: VisionAttention end-to-end precision with REAL vision RoPE
//   Loads cpu_vision_attn_{cos,sin,ref}_real.bin which were built from
//   VisionRotaryEmbedding (grid [[1,4,4]] → 16 tokens, hd=32). Same
//   weights/input as the identity case; only cos/sin and ref change.
// ═════════════════════════════════════════════════════════════════
TEST_CASE("VisionAttentionGraph precision: real RoPE N=16 nh=4 hd=32") {
    LOG_INFO("=== VisionAttention precision (real RoPE) ===");

    ArrayFp16 x, qkv_w, qkv_b, proj_w, proj_b, cos, sin, ref;
    MetaI32 meta;
    REQUIRE(x.Load("/tmp/cpu_vision_attn_x.bin"));
    REQUIRE(qkv_w.Load("/tmp/cpu_vision_attn_qkv_w.bin"));
    REQUIRE(qkv_b.Load("/tmp/cpu_vision_attn_qkv_b.bin"));
    REQUIRE(proj_w.Load("/tmp/cpu_vision_attn_proj_w.bin"));
    REQUIRE(proj_b.Load("/tmp/cpu_vision_attn_proj_b.bin"));
    REQUIRE(cos.Load("/tmp/cpu_vision_attn_cos_real.bin"));
    REQUIRE(sin.Load("/tmp/cpu_vision_attn_sin_real.bin"));
    REQUIRE(ref.Load("/tmp/cpu_vision_attn_ref_real.bin"));
    REQUIRE(meta.Load("/tmp/cpu_vision_attn_meta.bin"));
    REQUIRE(meta.data.size() == 3);
    int64_t N  = meta.data[0];
    int64_t nh = meta.data[1];
    int64_t hd = meta.data[2];
    int64_t hidden = nh * hd;

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb_llm::OperationHandle op;
    REQUIRE(IS_OK(atb_llm::components::VisionAttentionGraph::Build(
        "VAttnPrecReal", static_cast<int32_t>(nh), static_cast<int32_t>(hd), op)));
    REQUIRE(op.get() != nullptr);

    atb::Tensor x_t, qkvw_t, qkvb_t, projw_t, projb_t, cos_t, sin_t, seqlen_t, out_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(x_t,     {N, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(qkvw_t,  {3 * hidden, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(qkvb_t,  {3 * hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(projw_t, {hidden, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(projb_t, {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(cos_t,   {N, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(sin_t,   {N, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t,   {N, hidden})));

    REQUIRE(IS_OK(alloc->CopyToDevice(x_t,     x.data.data(),      x.data.size()      * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(qkvw_t,  qkv_w.data.data(),  qkv_w.data.size()  * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(qkvb_t,  qkv_b.data.data(),  qkv_b.data.size()  * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(projw_t, proj_w.data.data(), proj_w.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(projb_t, proj_b.data.data(), proj_b.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(cos_t,   cos.data.data(),    cos.data.size()    * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(sin_t,   sin.data.data(),    sin.data.size()    * 2)));

    int32_t seqlen_val = static_cast<int32_t>(N);
    seqlen_t.desc.dtype     = ACL_INT32;
    seqlen_t.desc.format    = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    atb::VariantPack vp;
    vp.inTensors  = {x_t, qkvw_t, qkvb_t, projw_t, projb_t, cos_t, sin_t, seqlen_t};
    vp.outTensors = {out_t};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime->GetWorkspace(ws_size); auto& ws = __atb_pair_ws.first; auto& ws_st = __atb_pair_ws.second;
        REQUIRE(IS_OK(ws_st));
        ws_ptr = ws;
    }
    REQUIRE(op.get()->Execute(vp, ws_ptr, ws_size, ctx) == atb::NO_ERROR);
    runtime->Synchronize();

    std::vector<uint16_t> host_out(N * hidden);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out.data(), out_t, host_out.size() * 2)));
    auto out_f32 = Fp16ToF32(host_out);
    auto ref_f32 = Fp16ToF32(ref.data);
    REQUIRE(out_f32.size() == ref_f32.size());

    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine (real RoPE) = %.6f", cs);
    CHECK(cs >= 0.999f);
}
