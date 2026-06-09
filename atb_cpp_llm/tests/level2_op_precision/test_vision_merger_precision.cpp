/**
 * Level 2 precision test: VisionMergerGraph + DeepstackGraph vs PyTorch ref.
 *
 * Two cases (same arithmetic, different op ordering):
 *   main      — LayerNorm(hidden)  -> reshape(group_4) -> fc1 -> GELU -> fc2
 *   deepstack — reshape(group_4)   -> LayerNorm(grouped) -> fc1 -> GELU -> fc2
 *
 * Prerequisites:
 *     python tests/gen_cpu_reference.py --stage vision_merger
 *
 * Run: ./test_vision_merger_precision
 * Requires: NPU device + ATB/ACL runtime.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/tensor_allocator.h"
#include "components/vision/vision_merger_graph.h"
#include "components/vision/deepstack_graph.h"
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

// Helper: run a single merger (main or deepstack) and return cosine vs ref.
float RunMergerCase(atb_llm::IRuntime* runtime,
                    atb_llm::OperationHandle& op,
                    const ArrayFp16& x, const ArrayFp16& n_w, const ArrayFp16& n_b,
                    const ArrayFp16& f1_w, const ArrayFp16& f1_b,
                    const ArrayFp16& f2_w, const ArrayFp16& f2_b,
                    const ArrayFp16& ref,
                    int64_t N, int64_t hidden, int64_t merge, int64_t out_hidden,
                    bool is_deepstack) {
    auto* alloc = runtime->GetAllocator();
    auto* ctx   = runtime->GetContext();

    int64_t mer_hidden = hidden * merge * merge;
    int64_t groups = (N * hidden) / mer_hidden;
    int64_t norm_dim = is_deepstack ? mer_hidden : hidden;

    atb::Tensor x_t, nw_t, nb_t, f1w_t, f1b_t, f2w_t, f2b_t, out_t;
    REQUIRE(IS_OK(alloc->AllocFloat16(x_t,    {N, hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(nw_t,   {norm_dim})));
    REQUIRE(IS_OK(alloc->AllocFloat16(nb_t,   {norm_dim})));
    REQUIRE(IS_OK(alloc->AllocFloat16(f1w_t,  {mer_hidden, mer_hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(f1b_t,  {mer_hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(f2w_t,  {out_hidden, mer_hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(f2b_t,  {out_hidden})));
    REQUIRE(IS_OK(alloc->AllocFloat16(out_t,  {groups, out_hidden})));

    auto put = [&](atb::Tensor& t, const ArrayFp16& a) {
        REQUIRE(IS_OK(alloc->CopyToDevice(t, a.data.data(), a.data.size() * 2)));
    };
    put(x_t,   x);
    put(nw_t,  n_w);  put(nb_t,  n_b);
    put(f1w_t, f1_w); put(f1b_t, f1_b);
    put(f2w_t, f2_w); put(f2b_t, f2_b);

    atb::VariantPack vp;
    vp.inTensors  = {x_t, nw_t, nb_t, f1w_t, f1b_t, f2w_t, f2b_t};
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

    std::vector<uint16_t> host_out(groups * out_hidden);
    REQUIRE(IS_OK(alloc->CopyToHost(host_out.data(), out_t, host_out.size() * 2)));
    auto out_f32 = Fp16ToF32(host_out);
    auto ref_f32 = Fp16ToF32(ref.data);
    REQUIRE(out_f32.size() == ref_f32.size());
    return CosineSim(out_f32.data(), ref_f32.data(), out_f32.size());
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
// Case 1: VisionMerger (main) — LayerNorm → reshape → fc1 → GELU → fc2
// ═════════════════════════════════════════════════════════════════
TEST_CASE("VisionMergerGraph precision: main (N=8 hidden=64 merge=2)") {
    LOG_INFO("=== VisionMerger (main) precision ===");

    ArrayFp16 x, n_w, n_b, f1_w, f1_b, f2_w, f2_b, ref;
    MetaI32 meta;
    REQUIRE(x.Load("/tmp/cpu_vision_merger_main_x.bin"));
    REQUIRE(n_w.Load("/tmp/cpu_vision_merger_main_n_w.bin"));
    REQUIRE(n_b.Load("/tmp/cpu_vision_merger_main_n_b.bin"));
    REQUIRE(f1_w.Load("/tmp/cpu_vision_merger_main_f1_w.bin"));
    REQUIRE(f1_b.Load("/tmp/cpu_vision_merger_main_f1_b.bin"));
    REQUIRE(f2_w.Load("/tmp/cpu_vision_merger_main_f2_w.bin"));
    REQUIRE(f2_b.Load("/tmp/cpu_vision_merger_main_f2_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_vision_merger_main_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_vision_merger_main_meta.bin"));
    REQUIRE(meta.data.size() == 5);
    int64_t N          = meta.data[0];
    int64_t hidden     = meta.data[1];
    int64_t merge      = meta.data[2];
    int64_t out_hidden = meta.data[3];
    bool    is_ds      = (meta.data[4] != 0);
    REQUIRE(!is_ds);

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    atb_llm::OperationHandle op;
    REQUIRE(IS_OK(atb_llm::components::VisionMergerGraph::Build(
        "VMergerPrecMain",
        static_cast<int32_t>(hidden),
        static_cast<int32_t>(merge),
        /*is_deepstack=*/false, 1e-6f, op)));
    REQUIRE(op.get() != nullptr);

    float cs = RunMergerCase(runtime.get(), op,
                             x, n_w, n_b, f1_w, f1_b, f2_w, f2_b, ref,
                             N, hidden, merge, out_hidden,
                             /*is_deepstack=*/false);
    LOG_INFO("  main cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}

// ═════════════════════════════════════════════════════════════════
// Case 2: DeepstackGraph — reshape → LayerNorm → fc1 → GELU → fc2
// ═════════════════════════════════════════════════════════════════
TEST_CASE("DeepstackGraph precision: (N=8 hidden=64 merge=2)") {
    LOG_INFO("=== Deepstack precision ===");

    ArrayFp16 x, n_w, n_b, f1_w, f1_b, f2_w, f2_b, ref;
    MetaI32 meta;
    REQUIRE(x.Load("/tmp/cpu_vision_merger_deepstack_x.bin"));
    REQUIRE(n_w.Load("/tmp/cpu_vision_merger_deepstack_n_w.bin"));
    REQUIRE(n_b.Load("/tmp/cpu_vision_merger_deepstack_n_b.bin"));
    REQUIRE(f1_w.Load("/tmp/cpu_vision_merger_deepstack_f1_w.bin"));
    REQUIRE(f1_b.Load("/tmp/cpu_vision_merger_deepstack_f1_b.bin"));
    REQUIRE(f2_w.Load("/tmp/cpu_vision_merger_deepstack_f2_w.bin"));
    REQUIRE(f2_b.Load("/tmp/cpu_vision_merger_deepstack_f2_b.bin"));
    REQUIRE(ref.Load("/tmp/cpu_vision_merger_deepstack_ref.bin"));
    REQUIRE(meta.Load("/tmp/cpu_vision_merger_deepstack_meta.bin"));
    REQUIRE(meta.data.size() == 5);
    int64_t N          = meta.data[0];
    int64_t hidden     = meta.data[1];
    int64_t merge      = meta.data[2];
    int64_t out_hidden = meta.data[3];
    bool    is_ds      = (meta.data[4] != 0);
    REQUIRE(is_ds);

    auto runtime = atb_llm::CreateRuntime(0, 2LL * 1024 * 1024 * 1024);
    REQUIRE(runtime != nullptr);

    atb_llm::OperationHandle op;
    REQUIRE(IS_OK(atb_llm::components::DeepstackGraph::Build(
        "DSPrec",
        static_cast<int32_t>(hidden),
        static_cast<int32_t>(merge), 1e-6f, op)));
    REQUIRE(op.get() != nullptr);

    float cs = RunMergerCase(runtime.get(), op,
                             x, n_w, n_b, f1_w, f1_b, f2_w, f2_b, ref,
                             N, hidden, merge, out_hidden,
                             /*is_deepstack=*/true);
    LOG_INFO("  deepstack cosine = %.6f", cs);
    CHECK(cs >= 0.99f);
}
