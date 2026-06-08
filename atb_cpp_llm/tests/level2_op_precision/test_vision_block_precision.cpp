/**
 * Level 2 precision test: VisionBlockGraph vs PyTorch reference.
 *
 * Pipeline (cos=1, sin=0 identity RoPE):
 *   hidden -> LayerNorm -> VisionAttention(+residual)
 *          -> LayerNorm -> VisionMLP(+residual) -> output
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage vision_block
 *
 * Run: ./test_vision_block_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "components/vision/vision_block_graph.h"
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
// Case: VisionBlock end-to-end precision (LN → Attn → LN → MLP)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("VisionBlockGraph precision: N=16 nh=4 hd=32") {
    LOG_INFO("=== VisionBlock precision ===");

    ArrayFp16 x, qkv_w, qkv_b, proj_w, proj_b;
    ArrayFp16 fc1_w, fc1_b, fc2_w, fc2_b;
    ArrayFp16 n1_w, n1_b, n2_w, n2_b;
    ArrayFp16 cos, sin, ref;
    MetaI32 meta;
    REQUIRE(x.Load("/tmp/cpu_vision_block_x.bin"));
    REQUIRE(qkv_w.Load("/tmp/cpu_vision_block_qkv_w.bin"));
    REQUIRE(qkv_b.Load("/tmp/cpu_vision_block_qkv_b.bin"));
    REQUIRE(proj_w.Load("/tmp/cpu_vision_block_proj_w.bin"));
    REQUIRE(proj_b.Load("/tmp/cpu_vision_block_proj_b.bin"));
    REQUIRE(fc1_w.Load("/tmp/cpu_vision_block_fc1_w.bin"));
    REQUIRE(fc1_b.Load("/tmp/cpu_vision_block_fc1_b.bin"));
    REQUIRE(fc2_w.Load("/tmp/cpu_vision_block_fc2_w.bin"));
    REQUIRE(fc2_b.Load("/tmp/cpu_vision_block_fc2_b.bin"));
    REQUIRE(n1_w.Load("/tmp/cpu_vision_block_n1_w.bin"));
    REQUIRE(n1_b.Load("/tmp/cpu_vision_block_n1_b.bin"));
    REQUIRE(n2_w.Load("/tmp/cpu_vision_block_n2_w.bin"));
    REQUIRE(n2_b.Load("/tmp/cpu_vision_block_n2_b.bin"));
    REQUIRE(cos.Load("/tmp/cpu_vision_block_cos.bin"));
    REQUIRE(sin.Load("/tmp/cpu_vision_block_sin.bin"));
    REQUIRE(ref.Load("/tmp/cpu_vision_block_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_vision_block_meta.bin"));
    REQUIRE(meta.data.size() == 4);
    int64_t N  = meta.data[0];
    int64_t nh = meta.data[1];
    int64_t hd = meta.data[2];
    int64_t I  = meta.data[3];
    int64_t hidden = nh * hd;

    auto runtime = atb_llm::CreateRuntime(0, 3LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb_llm::OperationHandle op;
    REQUIRE(IS_OK(atb_llm::components::VisionBlockGraph::Build(
        "VBlockPrec",
        static_cast<int32_t>(nh), static_cast<int32_t>(hd), 1e-6f, op)));
    REQUIRE(op.get() != nullptr);

    atb::Tensor x_t, qkvw_t, qkvb_t, projw_t, projb_t;
    atb::Tensor fc1w_t, fc1b_t, fc2w_t, fc2b_t;
    atb::Tensor n1w_t, n1b_t, n2w_t, n2b_t;
    atb::Tensor cos_t, sin_t, seqlen_t, out_t;

    REQUIRE(IS_OK(alloc->AllocFloat16(x_t,     {N, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(qkvw_t,  {3 * hidden, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(qkvb_t,  {3 * hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(projw_t, {hidden, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(projb_t, {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc1w_t,  {I, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc1b_t,  {I})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc2w_t,  {hidden, I})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc2b_t,  {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(n1w_t,   {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(n1b_t,   {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(n2w_t,   {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(n2b_t,   {hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(cos_t,   {N, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(sin_t,   {N, hd})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t,   {N, hidden})));

    auto put = [&](atb::Tensor& t, const ArrayFp16& a) {
        REQUIRE(IS_OK(alloc->CopyToDevice(t, a.data.data(), a.data.size() * 2)));
    };
    put(x_t, x);
    put(qkvw_t, qkv_w); put(qkvb_t, qkv_b);
    put(projw_t, proj_w); put(projb_t, proj_b);
    put(fc1w_t, fc1_w); put(fc1b_t, fc1_b);
    put(fc2w_t, fc2_w); put(fc2b_t, fc2_b);
    put(n1w_t, n1_w); put(n1b_t, n1_b);
    put(n2w_t, n2_w); put(n2b_t, n2_b);
    put(cos_t, cos); put(sin_t, sin);

    int32_t seqlen_val = static_cast<int32_t>(N);
    seqlen_t.desc.dtype     = ACL_INT32;
    seqlen_t.desc.format    = ACL_FORMAT_ND;
    seqlen_t.desc.shape.dimNum = 1;
    seqlen_t.desc.shape.dims[0] = 1;
    seqlen_t.dataSize = sizeof(int32_t);
    seqlen_t.hostData = &seqlen_val;

    atb::VariantPack vp;
    vp.inTensors = {
        x_t,
        qkvw_t, qkvb_t, projw_t, projb_t,
        fc1w_t, fc1b_t, fc2w_t, fc2b_t,
        n1w_t, n1b_t, n2w_t, n2b_t,
        cos_t, sin_t, seqlen_t,
    };
    vp.outTensors = {out_t};

    uint64_t ws_size = 0;
    REQUIRE(op.get()->Setup(vp, ws_size, ctx) == atb::NO_ERROR);
    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto [ws, ws_st] = runtime->GetWorkspace(ws_size);
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
