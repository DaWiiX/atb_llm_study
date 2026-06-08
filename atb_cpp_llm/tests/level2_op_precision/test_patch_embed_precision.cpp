/**
 * Level 2 precision test: PatchEmbedGraph vs PyTorch reference.
 *
 * Pipeline:
 *   pixels (N*K,) -> reshape (N, K) -> Linear(+bias) -> (N, embed_dim)
 *   where K = in_channels * temporal_patch_size * patch_size * patch_size
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage patch_embed
 *
 * Run: ./test_patch_embed_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "components/vision/patch_embed_graph.h"
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
// Case: PatchEmbed end-to-end precision (flatten + Linear)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("PatchEmbedGraph precision: N=4 in_ch=3 tp=2 p=14 embed=64") {
    LOG_INFO("=== PatchEmbed precision ===");

    ArrayFp16 pixels, w, b, ref;
    MetaI32 meta;
    REQUIRE(pixels.Load("/tmp/cpu_patch_embed_pixels.bin"));
    REQUIRE(w.Load("/tmp/cpu_patch_embed_w.bin"));
    REQUIRE(b.Load("/tmp/cpu_patch_embed_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_patch_embed_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_patch_embed_meta.bin"));
    REQUIRE(meta.data.size() == 5);
    int64_t N           = meta.data[0];
    int64_t in_channels = meta.data[1];
    int64_t tp          = meta.data[2];
    int64_t p           = meta.data[3];
    int64_t embed_dim   = meta.data[4];
    int64_t K = in_channels * tp * p * p;

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb_llm::OperationHandle op;
    REQUIRE(IS_OK(atb_llm::components::PatchEmbedGraph::Build(
        "PEPrec",
        static_cast<int32_t>(in_channels),
        static_cast<int32_t>(tp),
        static_cast<int32_t>(p),
        static_cast<int32_t>(embed_dim), op)));
    REQUIRE(op.get() != nullptr);

    atb::Tensor pixels_t, w_t, b_t, out_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(pixels_t, {N * K})));
    REQUIRE(IS_OK(alloc->AllocFloat16(w_t,      {embed_dim, K})));
    REQUIRE(IS_OK(alloc->AllocFloat16(b_t,      {embed_dim})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t,    {N, embed_dim})));

    REQUIRE(IS_OK(alloc->CopyToDevice(pixels_t, pixels.data.data(), pixels.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(w_t,      w.data.data(),      w.data.size()      * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(b_t,      b.data.data(),      b.data.size()      * 2)));

    atb::VariantPack vp;
    vp.inTensors  = {pixels_t, w_t, b_t};
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

    std::vector<uint16_t> host_out(N * embed_dim);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out.data(), out_t, host_out.size() * 2)));
    auto out_f32 = Fp16ToF32(host_out);
    auto ref_f32 = Fp16ToF32(ref.data);
    REQUIRE(out_f32.size() == ref_f32.size());

    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
