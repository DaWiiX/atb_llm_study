/**
 * Level 2 precision test: SwiGluMlpGraph vs PyTorch CPU reference.
 *
 * Compares the NPU fp16 SwiGLU MLP output to a pure PyTorch reference
 * (no torch_atb dependency) via cosine similarity.
 *
 * SwiGLU math (matches text_mlp.py):
 *   gate = SiLU(x @ gate_w.T)
 *   up   = x @ up_w.T
 *   out  = (gate * up) @ down_w.T
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage swiglu_mlp
 *
 * Run: ./test_swiglu_mlp_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "components/common/swiglu_mlp_graph.h"
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
    std::vector<uint16_t> data;  // raw fp16 bits

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

// Run one SwiGLU case. Returns cosine similarity vs the reference output.
float RunCase(atb_llm::IRuntime* runtime,
              const ArrayFP16& x, const ArrayFP16& gw,
              const ArrayFP16& uw, const ArrayFP16& dw,
              const ArrayFP16& expected,
              int32_t S, int32_t H, int32_t I) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    atb_llm::OperationHandle op;
    atb_llm::Status s = atb_llm::components::SwiGluMlpGraph::Build("PrecSwiGLU", op);
    REQUIRE(IS_OK(s));
    REQUIRE(op.get() != nullptr);

    atb::Tensor in_x, in_gw, in_uw, in_dw, out_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(in_x,  {S, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_gw, {I, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_uw, {I, H})));
    REQUIRE(IS_OK(alloc->AllocFloat16(in_dw, {H, I})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t, {S, H})));

    REQUIRE(IS_OK(alloc->CopyToDevice(in_x,  x.data.data(),  x.data.size()  * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(in_gw, gw.data.data(), gw.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(in_uw, uw.data.data(), uw.data.size() * sizeof(uint16_t))));
    REQUIRE(IS_OK(alloc->CopyToDevice(in_dw, dw.data.data(), dw.data.size() * sizeof(uint16_t))));

    atb::VariantPack vp;
    vp.inTensors  = {in_x, in_gw, in_uw, in_dw};
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
    auto ref_f32  = Fp16ToF32(expected.data);
    REQUIRE(static_cast<int64_t>(ref_f32.size()) == S * H);
    return CosineSim(host_f32.data(), ref_f32.data(), S * H);
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: small (S=4, H=64, I=128)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SwiGluMlpGraph precision: small (S=4, H=64, I=128)") {
    LOG_INFO("=== SwiGluMlpGraph precision: small ===");

    ArrayFP16 x, gw, uw, dw, ref;
    ArrayI32  meta;
    REQUIRE(x.Load   ("/tmp/cpu_swiglu_mlp_small_x.bin"));
    REQUIRE(gw.Load  ("/tmp/cpu_swiglu_mlp_small_gate_w.bin"));
    REQUIRE(uw.Load  ("/tmp/cpu_swiglu_mlp_small_up_w.bin"));
    REQUIRE(dw.Load  ("/tmp/cpu_swiglu_mlp_small_down_w.bin"));
    REQUIRE(ref.Load ("/tmp/cpu_swiglu_mlp_small_out_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_swiglu_mlp_small_meta.bin"));
    REQUIRE(meta.data.size() == 3);
    int32_t S = meta.data[0], H = meta.data[1], I = meta.data[2];

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunCase(runtime.get(), x, gw, uw, dw, ref, S, H, I);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: typical Qwen3VL (S=16, H=2048, I=6144)
// ═════════════════════════════════════════════════════════════════
TEST_CASE("SwiGluMlpGraph precision: typical (S=16, H=2048, I=6144)") {
    LOG_INFO("=== SwiGluMlpGraph precision: typical Qwen3VL ===");

    ArrayFP16 x, gw, uw, dw, ref;
    ArrayI32  meta;
    REQUIRE(x.Load   ("/tmp/cpu_swiglu_mlp_typical_x.bin"));
    REQUIRE(gw.Load  ("/tmp/cpu_swiglu_mlp_typical_gate_w.bin"));
    REQUIRE(uw.Load  ("/tmp/cpu_swiglu_mlp_typical_up_w.bin"));
    REQUIRE(dw.Load  ("/tmp/cpu_swiglu_mlp_typical_down_w.bin"));
    REQUIRE(ref.Load ("/tmp/cpu_swiglu_mlp_typical_out_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_swiglu_mlp_typical_meta.bin"));
    REQUIRE(meta.data.size() == 3);
    int32_t S = meta.data[0], H = meta.data[1], I = meta.data[2];

    auto runtime = atb_llm::CreateRuntime(0, 4LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    float cs = RunCase(runtime.get(), x, gw, uw, dw, ref, S, H, I);
    LOG_INFO("  cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
