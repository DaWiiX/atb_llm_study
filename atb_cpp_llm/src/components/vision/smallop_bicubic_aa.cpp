#include "components/vision/smallop_bicubic_aa.h"

#include "acl/acl.h"
#include "atb/atb_infer.h"
#include "atb_llm/runtime.h"
#include "atb_llm/operation_handle.h"
#include "core/tensor_allocator.h"
#include "families/base_model.h"   // ExecuteOperation (shared single-op runner)
#include "log/logger.h"
#include "ops/linear_op.h"
#include "ops/transpose_op.h"
#include "utils/float_utils.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace {

// PIL bicubic filter (a = -0.5), grouped Horner form (Resample.c:46-62).
// Computed in double — matches the CPU PIL 8bpc path's coefficient regime.
inline double BicubicFilter(double x) {
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

// Precompute PIL bicubic AA coefficients (Resample.c precompute_coeffs:181-267
// with bicubic support=2.0). Output is FP32 sparse weights with the bounds
// (xmin / k_count) per output index — the densification happens in
// BuildDenseWeightFp16 below. Does NOT do the 8bpc fixed-point conversion
// (normalize_coeffs_8bpc:269-284): keeps fp32 floats so the NPU MatMul can
// stay numerically agile without a clip8 chain.
//
// Layout:
//   bounds[2*j + 0] = xmin_j         (first source index touched by output j)
//   bounds[2*j + 1] = k_count_j      (= xmax - xmin, number of valid taps)
//   weights[j * ksize + k]           (k in [0, k_count_j); [k_count_j, ksize)
//                                     stay 0 — kept for caller's convenience
//                                     even though the dense build ignores them)
//
// Returns ksize (worst-case taps per output). Matches Resample.c byte-for-byte
// except for the int32 quantization step.
int PrecomputeCoeffsFloat(int in_size, int out_size,
                           std::vector<float>& weights,
                           std::vector<int>& bounds) {
    const double scale       = static_cast<double>(in_size) / out_size;
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support     = 2.0 * filterscale;          // BICUBIC.support = 2.0
    const int ksize          = static_cast<int>(std::ceil(support)) * 2 + 1;

    weights.assign(static_cast<size_t>(out_size) * ksize, 0.0f);
    bounds.assign(static_cast<size_t>(out_size) * 2, 0);

    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double rss    = 1.0 / filterscale;
        // (int) is truncation toward 0 — NOT floor (Resample.c:236,241).
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;  // xmax now = k_count (# of valid taps)

        double ww = 0.0;
        std::vector<double> k_double(static_cast<size_t>(xmax), 0.0);
        for (int x = 0; x < xmax; ++x) {
            const double w = BicubicFilter((x + xmin - center + 0.5) * rss);
            k_double[x] = w;
            ww += w;
        }
        if (ww != 0.0) {
            for (int x = 0; x < xmax; ++x) k_double[x] /= ww;
        }

        float* k_out = &weights[static_cast<size_t>(xx) * ksize];
        for (int x = 0; x < xmax; ++x) {
            k_out[x] = static_cast<float>(k_double[x]);
        }
        // [xmax, ksize) already zero from weights.assign().
        bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
        bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    return ksize;
}

// Spread the sparse PIL coefficients into a dense fp16 weight matrix
// W_dense[out_size, in_size] in row-major layout. Most entries are 0; the
// non-zeros live in a band of width ksize starting at bounds[j*2].
// fp32 -> fp16 uses Fp32ToFp16 (CANN aclFloatToFloat16, RNE rounding).
void BuildDenseWeightFp16(int in_size, int out_size,
                           std::vector<uint16_t>& dense_fp16) {
    std::vector<float> weights;
    std::vector<int>   bounds;
    const int ksize = PrecomputeCoeffsFloat(in_size, out_size, weights, bounds);

    const size_t total = static_cast<size_t>(out_size) * in_size;
    dense_fp16.assign(total, Fp32ToFp16(0.0f));  // 0x0000 — zero in fp16 too
    for (int j = 0; j < out_size; ++j) {
        const int xmin    = bounds[static_cast<size_t>(j) * 2 + 0];
        const int k_count = bounds[static_cast<size_t>(j) * 2 + 1];
        const float* k    = &weights[static_cast<size_t>(j) * ksize];
        uint16_t* row     = &dense_fp16[static_cast<size_t>(j) * in_size];
        for (int t = 0; t < k_count; ++t) {
            row[xmin + t] = Fp32ToFp16(k[t]);
        }
    }
}

// Build an atb::Tensor that ALIASES an external device pointer (no allocation,
// no tracking, no free on destruction). Used to wrap the caller-owned
// input/output buffers so they can be passed into ATB ops.
atb::Tensor MakeViewFp16(void* device_ptr, std::vector<int64_t> shape) {
    atb::Tensor t{};
    t.desc.dtype  = ACL_FLOAT16;
    t.desc.format = ACL_FORMAT_ND;
    t.desc.shape.dimNum = static_cast<uint64_t>(shape.size());
    int64_t elems = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        t.desc.shape.dims[i] = shape[i];
        elems *= shape[i];
    }
    t.deviceData = device_ptr;
    t.dataSize   = static_cast<size_t>(elems) * sizeof(uint16_t);
    t.hostData   = nullptr;
    return t;
}

// View-style reshape: change dimNum/dims on an existing tensor without
// touching deviceData / dataSize. The new shape must have the same element
// count as the old (caller's responsibility — total bytes are unchanged).
void ReshapeView(atb::Tensor& t, std::vector<int64_t> new_shape) {
    t.desc.shape.dimNum = static_cast<uint64_t>(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
        t.desc.shape.dims[i] = new_shape[i];
    }
}

}  // anonymous namespace

aclError NpuBicubicResizeAASmallOp(const void* input_device,
                                    int in_h, int in_w, int channels,
                                    int out_h, int out_w,
                                    IRuntime* runtime,
                                    void* output_device) {
    if (runtime == nullptr || input_device == nullptr || output_device == nullptr) {
        LOG_ERROR("NpuBicubicResizeAASmallOp: null parameter "
                  "(runtime=%p input=%p output=%p)",
                  static_cast<const void*>(runtime), input_device, output_device);
        return ACL_ERROR_INVALID_PARAM;
    }
    if (channels <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        LOG_ERROR("NpuBicubicResizeAASmallOp: invalid dims C=%d Hin=%d Win=%d Hout=%d Wout=%d",
                  channels, in_h, in_w, out_h, out_w);
        return ACL_ERROR_INVALID_PARAM;
    }

    const bool need_h = (out_w != in_w);
    const bool need_v = (out_h != in_h);

    // Identity (both axes unchanged) -> async device-to-device memcpy.
    // Skips the entire op chain; saves the only work to a single copy.
    if (!need_h && !need_v) {
        const size_t bytes = static_cast<size_t>(channels) * in_h * in_w * sizeof(uint16_t);
        aclError e = aclrtMemcpyAsync(output_device, bytes,
                                       input_device, bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE,
                                       runtime->GetStream());
        if (e != ACL_SUCCESS) {
            LOG_ERROR("NpuBicubicResizeAASmallOp: identity memcpy failed aclError=%d",
                      static_cast<int>(e));
        }
        return e;
    }

    auto* alloc = runtime->GetAllocator();

    // Intermediate device tensors (all 0-initialized; cleanup checks deviceData).
    atb::Tensor w_h{}, w_v{}, h_out{}, h_tr{}, v_out{};
    Status ret = STATUS_OK;
    aclError ret_acl = ACL_SUCCESS;

    // Width after the H pass: out_w if H is active, else the original in_w.
    const int64_t mid_w = need_h ? out_w : in_w;

    if (need_h) {
        std::vector<uint16_t> w_h_host;
        BuildDenseWeightFp16(in_w, out_w, w_h_host);  // [out_w, in_w]

        ret = alloc->AllocFloat16(w_h, {out_w, in_w});
        if (ret != STATUS_OK) { LOG_ERROR("NpuBicubicResizeAASmallOp: AllocFloat16 w_h failed"); ret_acl = ACL_ERROR_BAD_ALLOC; goto cleanup; }
        ret = alloc->CopyToDevice(w_h, w_h_host.data(), w_h_host.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) { LOG_ERROR("NpuBicubicResizeAASmallOp: CopyToDevice w_h failed"); ret_acl = ACL_ERROR_INTERNAL_ERROR; goto cleanup; }

        // H pass: input [C*H, in_w] (view) @ W_h^T [out_w, in_w] -> h_out [C*H, out_w].
        const int64_t rows = static_cast<int64_t>(channels) * in_h;
        atb::Tensor input_view = MakeViewFp16(const_cast<void*>(input_device),
                                              {rows, in_w});

        if (!need_v) {
            // No V pass: write Linear output directly into the caller's buffer.
            atb::Tensor output_view = MakeViewFp16(output_device, {rows, out_w});
            atb::VariantPack vp;
            vp.inTensors  = {input_view, w_h};
            vp.outTensors = {output_view};
            OperationHandle op = ops::LinearOp::Create(/*has_bias=*/false,
                                                       /*transpose_a=*/false,
                                                       /*transpose_b=*/true);
            uint64_t ws_size = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws_size);
            if (ret != STATUS_OK) { ret_acl = ACL_ERROR_INTERNAL_ERROR; goto cleanup; }
            // Done — w_h freed in cleanup.
            goto cleanup;
        }

        // V is also active: H output goes into a tracked intermediate buffer.
        ret = alloc->AllocFloat16(h_out, {rows, out_w});
        if (ret != STATUS_OK) { LOG_ERROR("NpuBicubicResizeAASmallOp: AllocFloat16 h_out failed"); ret_acl = ACL_ERROR_BAD_ALLOC; goto cleanup; }

        {
            atb::VariantPack vp;
            vp.inTensors  = {input_view, w_h};
            vp.outTensors = {h_out};
            OperationHandle op = ops::LinearOp::Create(false, false, true);
            uint64_t ws_size = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws_size);
            if (ret != STATUS_OK) { ret_acl = ACL_ERROR_INTERNAL_ERROR; goto cleanup; }
        }
        // H output is no longer needed in 2-D form; reshape view to 4-D
        // [1, C, H, out_w] so the subsequent Transpose sees it as 4-D NCHW.
        ReshapeView(h_out, {1, channels, in_h, out_w});

        // W_h is no longer read after the H pass (Tpre/V use a different weight).
        // Freed in cleanup after the end-of-pipeline sync — NOT eagerly: the async
        // H Linear may still be reading w_h, and aclrtFree is not stream-ordered.
    }
    // else: H untouched. input_device is the start of the V pipeline.

    if (need_v) {
        // h_tr is the H result transposed [1, C, H, mid_w] -> [1, C, mid_w, H].
        ret = alloc->AllocFloat16(h_tr, {1, channels, mid_w, in_h});
        if (ret != STATUS_OK) { LOG_ERROR("NpuBicubicResizeAASmallOp: AllocFloat16 h_tr failed"); ret_acl = ACL_ERROR_BAD_ALLOC; goto cleanup; }

        // 4-D source for transpose: either the H output (h_out, already 4-D
        // viewed [1,C,H,out_w]) or the caller's input (viewed [1,C,H,in_w]).
        atb::Tensor pre_transpose_src;
        if (need_h) {
            pre_transpose_src = h_out;       // shape [1,C,H,out_w]
        } else {
            pre_transpose_src = MakeViewFp16(const_cast<void*>(input_device),
                                              {1, channels, in_h, in_w});
        }

        {
            atb::VariantPack vp;
            vp.inTensors  = {pre_transpose_src};
            vp.outTensors = {h_tr};
            OperationHandle op = ops::TransposeOp::Create({0, 1, 3, 2});
            uint64_t ws_size = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws_size);
            if (ret != STATUS_OK) { ret_acl = ACL_ERROR_INTERNAL_ERROR; goto cleanup; }
        }
        // h_out is no longer read after Tpre (V reads h_tr). Deferred to cleanup
        // (post-sync) — NOT freed eagerly: the async Tpre may still be reading it.

        // Build the V weight [out_h, H] and upload.
        std::vector<uint16_t> w_v_host;
        BuildDenseWeightFp16(in_h, out_h, w_v_host);

        ret = alloc->AllocFloat16(w_v, {out_h, in_h});
        if (ret != STATUS_OK) { LOG_ERROR("NpuBicubicResizeAASmallOp: AllocFloat16 w_v failed"); ret_acl = ACL_ERROR_BAD_ALLOC; goto cleanup; }
        ret = alloc->CopyToDevice(w_v, w_v_host.data(), w_v_host.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) { LOG_ERROR("NpuBicubicResizeAASmallOp: CopyToDevice w_v failed"); ret_acl = ACL_ERROR_INTERNAL_ERROR; goto cleanup; }

        // V pass: h_tr viewed as [C*mid_w, H] @ W_v^T [out_h, H] -> [C*mid_w, out_h].
        ReshapeView(h_tr, {static_cast<int64_t>(channels) * mid_w, in_h});

        ret = alloc->AllocFloat16(v_out, {static_cast<int64_t>(channels) * mid_w, out_h});
        if (ret != STATUS_OK) { LOG_ERROR("NpuBicubicResizeAASmallOp: AllocFloat16 v_out failed"); ret_acl = ACL_ERROR_BAD_ALLOC; goto cleanup; }

        {
            atb::VariantPack vp;
            vp.inTensors  = {h_tr, w_v};
            vp.outTensors = {v_out};
            OperationHandle op = ops::LinearOp::Create(false, false, true);
            uint64_t ws_size = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws_size);
            if (ret != STATUS_OK) { ret_acl = ACL_ERROR_INTERNAL_ERROR; goto cleanup; }
        }
        // h_tr / w_v no longer read after the V pass; freed in cleanup (post-sync).

        // Final transpose v_out [1,C,mid_w,out_h] -> output_view [1,C,out_h,mid_w].
        ReshapeView(v_out, {1, channels, mid_w, out_h});
        atb::Tensor output_view = MakeViewFp16(output_device,
                                                {1, channels, out_h, mid_w});

        {
            atb::VariantPack vp;
            vp.inTensors  = {v_out};
            vp.outTensors = {output_view};
            OperationHandle op = ops::TransposeOp::Create({0, 1, 3, 2});
            uint64_t ws_size = 0;
            ret = ExecuteOperation(op.get(), vp, runtime, ws_size);
            if (ret != STATUS_OK) { ret_acl = ACL_ERROR_INTERNAL_ERROR; goto cleanup; }
        }
        // v_out no longer read after Tpost; freed in cleanup (post-sync).
    }

cleanup:
    // Synchronize before freeing intermediates: the upstream ops (H Linear,
    // Tpre Transpose, V Linear, Tpost Transpose) were enqueued asynchronously
    // and still read w_h/h_out/h_tr/w_v/v_out. aclrtFree (used by alloc->Free)
    // reclaims memory IMMEDIATELY (not stream-ordered — verified empirically),
    // so we must drain the queue before any cleanup Free. Mirrors
    // qwen3vl_preprocess.cpp:580 and :703 sync-before-free discipline.
    // Reached by every intermediate-allocating path (H-only goto's here too);
    // the identity path returns earlier without allocating, so it skips this.
    if (ret == STATUS_OK) {
        Status sync_ret = runtime->Synchronize();
        if (sync_ret != STATUS_OK) {
            LOG_ERROR("NpuBicubicResizeAASmallOp: Synchronize before cleanup failed");
            ret = sync_ret;                 // error return; fall through to free
            ret_acl = ACL_ERROR_INTERNAL_ERROR;
        }
    }
    // Reverse allocation order; Free() is idempotent (skips nullptr).
    if (v_out.deviceData != nullptr) alloc->Free(v_out);
    if (h_tr.deviceData  != nullptr) alloc->Free(h_tr);
    if (h_out.deviceData != nullptr) alloc->Free(h_out);
    if (w_v.deviceData   != nullptr) alloc->Free(w_v);
    if (w_h.deviceData   != nullptr) alloc->Free(w_h);

    return (ret == STATUS_OK) ? ACL_SUCCESS : ret_acl;
}

}  // namespace atb_llm
