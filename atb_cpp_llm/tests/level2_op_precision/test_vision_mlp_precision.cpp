/**
 * Level 2 precision test: VisionMlpGraph vs PyTorch reference.
 *
 * Pipeline: fc1 Linear(+bias) -> GELU(tanh) -> fc2 Linear(+bias)
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage vision_mlp
 *
 * Run: ./test_vision_mlp_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "components/vision/vision_mlp_graph.h"
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
// Case: VisionMLP end-to-end precision
// ═════════════════════════════════════════════════════════════════
TEST_CASE("VisionMlpGraph precision: N=8 H=64 I=128") {
    LOG_INFO("=== VisionMLP precision ===");

    ArrayFp16 x, fc1_w, fc1_b, fc2_w, fc2_b, ref;
    MetaI32 meta;
    REQUIRE(x.Load("/tmp/cpu_vision_mlp_x.bin"));
    REQUIRE(fc1_w.Load("/tmp/cpu_vision_mlp_fc1_w.bin"));
    REQUIRE(fc1_b.Load("/tmp/cpu_vision_mlp_fc1_b.bin"));
    REQUIRE(fc2_w.Load("/tmp/cpu_vision_mlp_fc2_w.bin"));
    REQUIRE(fc2_b.Load("/tmp/cpu_vision_mlp_fc2_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_vision_mlp_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_vision_mlp_meta.bin"));
    REQUIRE(meta.data.size() == 3);
    int64_t N = meta.data[0], H = meta.data[1], I = meta.data[2];

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb_llm::OperationHandle op;
    REQUIRE(IS_OK(atb_llm::components::VisionMlpGraph::Build("VMLPPrec", op)));
    REQUIRE(op.get() != nullptr);

    atb::Tensor x_t, fc1w_t, fc1b_t, fc2w_t, fc2b_t, out_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(x_t,    {N, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc1w_t, {I, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc1b_t, {I})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc2w_t, {H, I})));
    REQUIRE(IS_OK(alloc->AllocFloat16(fc2b_t, {H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t,  {N, H})));

    REQUIRE(IS_OK(alloc->CopyToDevice(x_t,    x.data.data(),     x.data.size()     * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(fc1w_t, fc1_w.data.data(), fc1_w.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(fc1b_t, fc1_b.data.data(), fc1_b.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(fc2w_t, fc2_w.data.data(), fc2_w.data.size() * 2)));
    REQUIRE(IS_OK(alloc->CopyToDevice(fc2b_t, fc2_b.data.data(), fc2_b.data.size() * 2)));

    atb::VariantPack vp;
    vp.inTensors  = {x_t, fc1w_t, fc1b_t, fc2w_t, fc2b_t};
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

    std::vector<uint16_t> host_out(N * H);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out.data(), out_t, host_out.size() * 2)));
    auto out_f32 = Fp16ToF32(host_out);
    auto ref_f32 = Fp16ToF32(ref.data);
    REQUIRE(out_f32.size() == ref_f32.size());

    float cs = CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
