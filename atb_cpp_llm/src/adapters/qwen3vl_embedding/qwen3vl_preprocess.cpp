#include "adapters/qwen3vl_embedding/qwen3vl_preprocess.h"
#include "adapters/qwen3vl_embedding/qwen3vl_weights.h"
#include "utils/float_utils.h"
#include "log/logger.h"
#include "utils/cpp11_compat.h"
#include "atb_llm/runtime.h"
#include "ops/elewise_op.h"
#include "ops/transpose_op.h"
#include "components/vision/aclnn_bicubic_resize.h"
#include "components/vision/smallop_bicubic_aa.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

// ── PIL 8bpc fixed-point bicubic (AA) — bit-exact port of Resample.c ──────
// File-local helpers (not exported in the .h). These reproduce Pillow's
// Image.resize(BICUBIC) integer pipeline exactly: 8-bit integer accumulation
// with PRECISION_BITS fractional bits, antialias filter support that grows
// with the downsample factor, and (int)-truncated coefficient quantization.
namespace {

// 32-bit accumulator budget: 8 result bits + 2 overflow bits ⇒ 22 fractional
// bits (Resample.c:92).
constexpr int kPrecisionBits = 32 - 8 - 2;  // 22

// Fixed-point clip to uint8 (Resample.c clip8, :176-179). Bit-exact with the
// 1280-entry lookup table: the table is indexed by (in >> PRECISION_BITS) and
// saturates to [0,255], which is exactly what this arithmetic shift + clamp
// computes — so the table itself is not ported.
static inline uint8_t Clip8(int v) {
    int s = v >> kPrecisionBits;  // arithmetic (sign-preserving) right shift
    if (s < 0) return 0;
    if (s > 255) return 255;
    return static_cast<uint8_t>(s);
}

// Bicubic convolution kernel, a=-0.5, grouped (Horner) form (Resample.c:46-62).
static double BicubicFilter(double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

// Precompute fixed-point resampling coefficients for one axis
// (Resample.c precompute_coeffs:181-267 with in0=0, in1=inSize, BICUBIC
// support=2.0; followed by normalize_coeffs_8bpc:269-284).
//   bounds[xx*2+0] = xmin (first source index), bounds[xx*2+1] = xmax (count).
//   ki = outSize*ksize fixed-point int32 weights ([xmax,ksize) left as 0).
// Returns ksize (taps per output sample).
static int PrecomputeCoeffs8bpc(int32_t inSize, int32_t outSize,
                                std::vector<int>& bounds,
                                std::vector<int32_t>& ki) {
    double scale = static_cast<double>(inSize) / outSize;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = 2.0 * filterscale;
    int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

    bounds.resize(static_cast<size_t>(outSize) * 2);
    std::vector<double> kk(static_cast<size_t>(outSize) * ksize, 0.0);

    for (int xx = 0; xx < outSize; xx++) {
        double center = (xx + 0.5) * scale;
        double rss = 1.0 / filterscale;
        // (int) truncates toward zero — NOT floor (Resample.c:236,241).
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        xmax -= xmin;

        double* k = &kk[static_cast<size_t>(xx) * ksize];
        double ww = 0.0;
        for (int x = 0; x < xmax; x++) {
            double w = BicubicFilter((x + xmin - center + 0.5) * rss);
            k[x] = w;
            ww += w;
        }
        if (ww != 0.0) {
            for (int x = 0; x < xmax; x++) k[x] /= ww;
        }
        // [xmax, ksize) stay 0 (kk was zero-initialised).
        bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
        bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }

    // normalize_coeffs_8bpc: (int) truncation (NOT round) of w * 2^PRECISION_BITS,
    // biased by ±0.5 toward the sign of w (Resample.c:277-283).
    ki.resize(static_cast<size_t>(outSize) * ksize);
    for (size_t i = 0; i < kk.size(); i++) {
        double w = kk[i];
        ki[i] = (w < 0)
                    ? static_cast<int>(-0.5 + w * (1 << kPrecisionBits))
                    : static_cast<int>(0.5 + w * (1 << kPrecisionBits));
    }
    return ksize;
}

// Horizontal pass: in [C,in_h,in_w] uint8 → out [C,in_h,out_w] uint8
// (Resample.c ImagingResampleHorizontal_8bpc, single-band path).
static void HorizontalPass8bpc(const uint8_t* in, int32_t in_h, int32_t in_w,
                               int32_t channels, int32_t out_w,
                               int ksize, const int* bounds, const int32_t* ki,
                               uint8_t* out) {
    for (int32_t c = 0; c < channels; c++) {
        for (int32_t ih = 0; ih < in_h; ih++) {
            for (int32_t ow = 0; ow < out_w; ow++) {
                int xmin = bounds[ow * 2 + 0];
                int xmax = bounds[ow * 2 + 1];
                const int32_t* k = &ki[static_cast<size_t>(ow) * ksize];
                int ss = 1 << (kPrecisionBits - 1);  // rounding bias
                for (int x = 0; x < xmax; x++) {
                    ss += static_cast<int>(
                              in[(static_cast<size_t>(c) * in_h + ih) * in_w + (xmin + x)]) *
                          k[x];
                }
                out[(static_cast<size_t>(c) * in_h + ih) * out_w + ow] = Clip8(ss);
            }
        }
    }
}

// Vertical pass: in [C,in_h,out_w] uint8 → out [C,out_h,out_w] uint8
// (Resample.c ImagingResampleVertical_8bpc, single-band path).
static void VerticalPass8bpc(const uint8_t* in, int32_t in_h, int32_t out_w,
                             int32_t channels, int32_t out_h,
                             int ksize, const int* bounds, const int32_t* ki,
                             uint8_t* out) {
    for (int32_t c = 0; c < channels; c++) {
        for (int32_t oh = 0; oh < out_h; oh++) {
            int ymin = bounds[oh * 2 + 0];
            int ymax = bounds[oh * 2 + 1];
            const int32_t* k = &ki[static_cast<size_t>(oh) * ksize];
            for (int32_t ow = 0; ow < out_w; ow++) {
                int ss = 1 << (kPrecisionBits - 1);  // rounding bias
                for (int y = 0; y < ymax; y++) {
                    ss += static_cast<int>(
                              in[(static_cast<size_t>(c) * in_h + (ymin + y)) * out_w + ow]) *
                          k[y];
                }
                out[(static_cast<size_t>(c) * out_h + oh) * out_w + ow] = Clip8(ss);
            }
        }
    }
}

// Two-pass separable resample (Resample.c ImagingResampleInner): horizontal
// then vertical, each skipped when its axis is unchanged. Output is uint8
// [C,out_h,out_w]. Identity (no axis changes) copies the input through —
// matching Pillow's ImagingCopy fast path (e.g. 416×672 → 416×672).
static void ResampleCore8bpc(const uint8_t* in, int32_t in_h, int32_t in_w,
                             int32_t channels, int32_t out_h, int32_t out_w,
                             std::vector<uint8_t>& out) {
    bool need_h = (out_w != in_w);
    bool need_v = (out_h != in_h);

    if (!need_h && !need_v) {
        out.assign(in, in + static_cast<size_t>(channels) * in_h * in_w);
        return;
    }

    // --- Horizontal pass (or identity) → intermediate [C, in_h, out_w] ---
    std::vector<uint8_t> tmp;
    const uint8_t* hsrc;
    if (need_h) {
        std::vector<int> bh;
        std::vector<int32_t> kih;
        int ksize_h = PrecomputeCoeffs8bpc(in_w, out_w, bh, kih);
        tmp.resize(static_cast<size_t>(channels) * in_h * out_w);
        HorizontalPass8bpc(in, in_h, in_w, channels, out_w,
                           ksize_h, bh.data(), kih.data(), tmp.data());
        hsrc = tmp.data();
    } else {
        hsrc = in;  // intermediate width == in_w == out_w
    }

    // --- Vertical pass (or identity) → out [C, out_h, out_w] ---
    if (need_v) {
        std::vector<int> bv;
        std::vector<int32_t> kiv;
        int ksize_v = PrecomputeCoeffs8bpc(in_h, out_h, bv, kiv);
        out.resize(static_cast<size_t>(channels) * out_h * out_w);
        VerticalPass8bpc(hsrc, in_h, out_w, channels, out_h,
                         ksize_v, bv.data(), kiv.data(), out.data());
    } else {
        out.assign(hsrc, hsrc + static_cast<size_t>(channels) * in_h * out_w);
    }
}

}  // namespace

namespace atb_llm {
namespace adapters {

void BicubicResize(const uint8_t* input, int32_t in_h, int32_t in_w,
                   int32_t channels, int32_t out_h, int32_t out_w,
                   float* output) {
    // PIL 8bpc fixed-point AA resample → uint8, then widen to float.
    std::vector<uint8_t> u8;
    ResampleCore8bpc(input, in_h, in_w, channels, out_h, out_w, u8);
    size_t n = static_cast<size_t>(channels) * out_h * out_w;
    for (size_t i = 0; i < n; i++) {
        output[i] = static_cast<float>(u8[i]);
    }
}

// Python-style banker's rounding: round half to even (matches Python 3 round())
static int32_t BankersRound(float x) {
    float floor_val = std::floor(x);
    float frac = x - floor_val;
    if (frac < 0.5f) return static_cast<int32_t>(floor_val);
    if (frac > 0.5f) return static_cast<int32_t>(floor_val + 1.0f);
    // Exactly 0.5 — round to nearest even
    int32_t lower = static_cast<int32_t>(floor_val);
    return (lower % 2 == 0) ? lower : lower + 1;
}

void SmartResize(int32_t height, int32_t width,
                 int32_t factor, int32_t min_pixels, int32_t max_pixels,
                 int32_t& new_height, int32_t& new_width) {
    // Round to nearest multiple of factor (banker's rounding, matches Python)
    int32_t h_bar = BankersRound(static_cast<float>(height) / factor) * factor;
    int32_t w_bar = BankersRound(static_cast<float>(width) / factor) * factor;

    int64_t area = static_cast<int64_t>(h_bar) * w_bar;
    if (area > max_pixels) {
        double beta = std::sqrt(static_cast<double>(height * width) / max_pixels);
        h_bar = std::max(factor, static_cast<int32_t>(
                                     std::floor(height / beta / factor)) *
                                     factor);
        w_bar = std::max(factor, static_cast<int32_t>(
                                     std::floor(width / beta / factor)) *
                                     factor);
    } else if (area < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
        h_bar = static_cast<int32_t>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int32_t>(std::ceil(width * beta / factor)) * factor;
    }

    new_height = h_bar;
    new_width = w_bar;
}

Status PreprocessImage(const uint8_t* image,
                       int32_t channels, int32_t height, int32_t width,
                       const Qwen3VLConfig& config,
                       uint16_t* pixel_values,
                       int64_t& num_patches,
                       int64_t* grid_thw) {
    int32_t patch_size = config.pp_patch_size;
    int32_t tp = config.pp_temporal_patch_size;
    int32_t merge_size = config.pp_merge_size;
    int32_t factor = patch_size * merge_size;

    // 1. Smart resize
    int32_t new_h, new_w;
    SmartResize(height, width, factor,
                config.pp_min_pixels, config.pp_max_pixels,
                new_h, new_w);

    // 2. Bicubic resize (bit-exact PIL 8bpc fixed-point AA) → uint8 [C,new_h,new_w].
    std::vector<uint8_t> resized_u8;
    ResampleCore8bpc(image, height, width, channels, new_h, new_w, resized_u8);

    // 3. Rescale + per-channel normalize: (x/255 - mean) / std.
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std_val[3] = {0.5f, 0.5f, 0.5f};
    if (config.pp_image_mean.size() >= 3) {
        mean[0] = config.pp_image_mean[0];
        mean[1] = config.pp_image_mean[1];
        mean[2] = config.pp_image_mean[2];
    }
    if (config.pp_image_std.size() >= 3) {
        std_val[0] = config.pp_image_std[0];
        std_val[1] = config.pp_image_std[1];
        std_val[2] = config.pp_image_std[2];
    }

    size_t resized_size = static_cast<size_t>(channels) * new_h * new_w;
    std::vector<float> resized(resized_size);
    size_t plane = static_cast<size_t>(new_h) * new_w;
    for (int32_t c = 0; c < channels; c++) {
        float mc = mean[c];
        float sc = std_val[c];
        size_t base = static_cast<size_t>(c) * plane;
        for (size_t p = 0; p < plane; p++) {
            size_t idx = base + p;
            resized[idx] = (resized_u8[idx] / 255.0f - mc) / sc;
        }
    }

    // 4. Pad to temporal_patch_size (single image -> 2 frames by duplicating)
    int32_t total_frames = tp;           // For single image, we pad to tp=2
    int32_t grid_t = total_frames / tp;  // = 1
    int32_t grid_h = new_h / patch_size;
    int32_t grid_w = new_w / patch_size;

    grid_thw[0] = grid_t;
    grid_thw[1] = grid_h;
    grid_thw[2] = grid_w;

    // 5. Patch extraction with spatial merge ordering
    // Python: frames.reshape(grid_t, tp, C, grid_h//ms, ms, ps, grid_w//ms, ms, ps)
    //         .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    //         .reshape(grid_t*grid_h*grid_w, C*tp*ps*ps)
    //
    // After permute, the last dim order is: C, tp, ps_h, ps_w
    // So each patch element is indexed by (c, f, ph, pw) with strides:
    //   offset = c * tp*ps*ps + f * ps*ps + ph * ps + pw

    int64_t patch_dim = static_cast<int64_t>(channels) * tp * patch_size * patch_size;
    num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;

    int64_t merged_h = grid_h / merge_size;
    int64_t merged_w = grid_w / merge_size;

    // Single-image input: all tp frames read the same `resized` data, so the
    // f=0 block is computed once and memcpy'd to f=1..tp-1 (halves read+convert
    // work in the temporal dimension).
    size_t ps2 = static_cast<size_t>(patch_size) * patch_size;
    int64_t patch_idx = 0;
    for (int64_t br = 0; br < merged_h; br++) {
        for (int64_t bc = 0; bc < merged_w; bc++) {
            for (int64_t ir = 0; ir < merge_size; ir++) {
                for (int64_t ic = 0; ic < merge_size; ic++) {
                    uint16_t* pixel_base = pixel_values + patch_idx * patch_dim;
                    int64_t row_base = br * merge_size * patch_size + ir * patch_size;
                    int64_t col_base = bc * merge_size * patch_size + ic * patch_size;
                    for (int64_t c = 0; c < channels; c++) {
                        const float* resized_c =
                            resized.data() + c * new_h * new_w;
                        // f=0 block: read each pixel once, convert once.
                        uint16_t* f0 = pixel_base + c * tp * ps2;
                        for (int64_t ph = 0; ph < patch_size; ph++) {
                            const float* src_row =
                                resized_c + (row_base + ph) * new_w + col_base;
                            uint16_t* dst_row = f0 + ph * patch_size;
                            for (int64_t pw = 0; pw < patch_size; pw++) {
                                dst_row[pw] = atb_llm::Fp32ToFp16(src_row[pw]);
                            }
                        }
                        // Duplicate f=0 to f=1..tp-1 (single image → identical frames).
                        for (int32_t f = 1; f < tp; f++) {
                            std::memcpy(f0 + f * ps2, f0, ps2 * sizeof(uint16_t));
                        }
                    }
                    patch_idx++;
                }
            }
        }
    }

    LOG_INFO("Preprocessed image: %dx%d -> %dx%d, grid=(%ld,%ld,%ld), patches=%ld, dim=%ld",
             height, width, new_h, new_w,
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(num_patches), static_cast<long>(patch_dim));

    return STATUS_OK;
}

// ── Internal: NPU preprocess steps 1-5 (SmartResize → H2D → bicubic →
// normalize → AsStrided+Transpose). Shared by both PreprocessImageNpu
// overloads (DRY). Leaves the device-resident patch output in @p transpose_out
// (allocator-tracked; caller owns — must Free or Detach). All intermediates
// are freed here; on error transpose_out is freed too. No D2H: the device
// tensor stays on NPU so path C can feed the vision pipeline directly.
static Status PreprocessImageNpuInternal(IRuntime* runtime,
                                         const uint8_t* image,
                                         int32_t channels, int32_t height, int32_t width,
                                         const Qwen3VLConfig& config,
                                         atb::Tensor& transpose_out,
                                         int64_t& num_patches,
                                         int64_t* grid_thw) {
    if (runtime == nullptr || image == nullptr || grid_thw == nullptr) {
        LOG_ERROR("PreprocessImageNpuInternal: null parameter (runtime=%p, image=%p, grid_thw=%p)",
                  static_cast<const void*>(runtime), static_cast<const void*>(image),
                  static_cast<const void*>(grid_thw));
        return ERROR_INVALID_PARAM;
    }

    int32_t patch_size = config.pp_patch_size;
    int32_t tp = config.pp_temporal_patch_size;
    int32_t merge_size = config.pp_merge_size;
    int32_t factor = patch_size * merge_size;

    // --- Step 1: SmartResize (CPU) ------------------------------------
    int32_t new_h, new_w;
    SmartResize(height, width, factor,
                config.pp_min_pixels, config.pp_max_pixels,
                new_h, new_w);

    // --- Resources (all declared upfront for goto-cleanup) ------------
    auto* alloc = runtime->GetAllocator();
    auto* ctx = runtime->GetContext();
    Status ret = STATUS_OK;

    // Value-initialize all tensor handles: deviceData starts as nullptr so
    // the cleanup block can distinguish allocated vs unallocated tensors.
    // transpose_out is the caller-owned out-param (value-initialised by caller).
    atb::Tensor input_tensor{};
    atb::Tensor resized_tensor{};
    atb::Tensor normalize_tmp{};
    atb::Tensor mean_bc{};
    atb::Tensor inv_std_bc{};
    atb::Tensor asstrided_out{};

    // --- Step 2: H2D -- uint8->fp16 conversion + copy to device ---------
    int64_t in_elems = static_cast<int64_t>(channels) * height * width;
    std::vector<uint16_t> in_fp16(in_elems);
    for (int64_t i = 0; i < in_elems; i++) {
        in_fp16[i] = Fp32ToFp16(static_cast<float>(image[i]));
    }

    ret = alloc->AllocFloat16(input_tensor, {1, channels, height, width});
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: AllocFloat16 input failed");
        goto cleanup;
    }
    ret = alloc->CopyToDevice(input_tensor, in_fp16.data(), in_elems * sizeof(uint16_t));
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: CopyToDevice input failed");
        goto cleanup;
    }

    // --- Step 3: Bicubic Resize on NPU ---------------------------------
    ret = alloc->AllocFloat16(resized_tensor, {1, channels, new_h, new_w});
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: AllocFloat16 resized failed");
        goto cleanup;
    }

    // AA pre-filtering is only beneficial when genuinely downsampling.
    // At identity / near-identity scales AA smooths the image and
    // degrades precision vs CPU (cos drops to ~0.950), so non-downsample
    // paths always use the non-AA aclnn op. When downsampling, 910B uses the
    // hardware AA op; 310P (where the AA op is unsupported) uses a small-op
    // AA assembly that is numerically equivalent (see dispatch below).
    {
        bool downsample = (new_h < height || new_w < width);
        if (downsample) {
            if (Is910B()) {
                // 910B: hardware AA op (aclnnUpsampleBicubic2dAA), fastest path.
                ret = NpuBicubicResizeAA(runtime,
                                         input_tensor.deviceData,
                                         channels, height, width, new_h, new_w,
                                         resized_tensor.deviceData);
            } else {
                // 310P: aclnnUpsampleBicubic2dAA unsupported (aclnnStatus=561103).
                // Stack ATB small ops (Linear x2 + Transpose x2) into a
                // PIL-equivalent separable AA bicubic. Numerically equivalent to
                // 910B's hardware AA (verified Batch B: end-to-end cos matches
                // 910B aclnn AA to ~6 decimals).
                aclError ae = NpuBicubicResizeAASmallOp(
                    input_tensor.deviceData, height, width, channels,
                    new_h, new_w, runtime, resized_tensor.deviceData);
                ret = (ae == ACL_SUCCESS) ? STATUS_OK : ERROR_INFERENCE;
            }
        } else {
            // Non-downsample (identity / upsample): non-AA aclnn, cross-platform.
            ret = NpuBicubicResize(runtime,
                                   input_tensor.deviceData,
                                   channels, height, width, new_h, new_w,
                                   resized_tensor.deviceData);
        }
    }
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: NpuBicubicResize failed");
        goto cleanup;
    }

    // Free input tensor early (no longer needed); Free() sets deviceData to
    // nullptr so the cleanup block will skip it — no double-free.
    alloc->Free(input_tensor);

    // --- Step 4–6: Normalize + D2H + patch extraction -----------------
    // Nested scope isolates all normalization local variables (mean, std_val,
    // mean_neg, inv_std, RunAtbOp lambda) so that gotos from steps 2-3
    // (above) don't cross their initialization — C++ forbids goto over
    // variables with non-trivial destructors.
    {
        // ── Step 4: Normalize on NPU via 3 ATB ElewiseOps ──
        // Formula: (x/255 - mean) / std
        //
        // ATB has no scalar add/subtract (no ELEWISE_ADDS).  We use broadcast
        // tensors instead: ELEWISE_ADD/ELEWISE_MUL broadcast when corresponding
        // dims are equal or at least one is 1.  A (1,C,1,1) tensor broadcasts
        // to (1,C,H,W) -- one scalar per channel, perfect for per-channel mean/std.
        //
        // Ops: 4a. MULS(1/255)
        //      4b. ADD(-mean broadcast)  --  x/255 + (-mean) = x/255 - mean
        //      4c. MUL(1/std broadcast)  --  (x/255 - mean) * (1/std)

        float mean[3] = {0.5f, 0.5f, 0.5f};
        float std_val[3] = {0.5f, 0.5f, 0.5f};
        if (config.pp_image_mean.size() >= 3) {
            mean[0] = config.pp_image_mean[0];
            mean[1] = config.pp_image_mean[1];
            mean[2] = config.pp_image_mean[2];
        }
        if (config.pp_image_std.size() >= 3) {
            std_val[0] = config.pp_image_std[0];
            std_val[1] = config.pp_image_std[1];
            std_val[2] = config.pp_image_std[2];
        }

        // Intermediate tensor (same shape as resized, used as ping-pong buffer)
        ret = alloc->AllocFloat16(normalize_tmp, {1, channels, new_h, new_w});
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: AllocFloat16 normalize_tmp failed");
            goto cleanup;
        }

        // Per-channel broadcast tensors (1, C, 1, 1) -- just 3 elements each
        std::vector<uint16_t> mean_neg(static_cast<size_t>(channels));
        std::vector<uint16_t> inv_std(static_cast<size_t>(channels));
        for (int32_t c = 0; c < channels; c++) {
            mean_neg[c] = Fp32ToFp16(-mean[c]);
            inv_std[c] = Fp32ToFp16(1.0f / std_val[c]);
        }

        ret = alloc->AllocFloat16(mean_bc, {1, channels, 1, 1});
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: AllocFloat16 mean_bc failed");
            goto cleanup;
        }
        ret = alloc->AllocFloat16(inv_std_bc, {1, channels, 1, 1});
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: AllocFloat16 inv_std_bc failed");
            goto cleanup;
        }
        ret = alloc->CopyToDevice(mean_bc, mean_neg.data(), mean_neg.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) goto cleanup;
        ret = alloc->CopyToDevice(inv_std_bc, inv_std.data(), inv_std.size() * sizeof(uint16_t));
        if (ret != STATUS_OK) goto cleanup;

        // Run a single ATB operation (Setup -> GetWorkspace -> Execute).  Works
        // for any atb::Operation wrapped in OperationHandle (Elewise, AsStrided,
        // Transpose, ...).  op is moved in and destroyed when the call returns.
        auto RunAtbOp = [&](OperationHandle op, atb::VariantPack& vp) -> Status {
            uint64_t ws_size = 0;
            atb::Status atb_s = op.get()->Setup(vp, ws_size, ctx);
            if (atb_s != atb::NO_ERROR) {
                LOG_ERROR("PreprocessImageNpu: ATB op Setup failed, atbStatus=%d",
                          static_cast<int>(atb_s));
                return ERROR_INFERENCE;
            }
            uint8_t* ws = nullptr;
            if (ws_size > 0) {
                auto wp = runtime->GetWorkspace(ws_size);
                ws = wp.first;
                if (wp.second != STATUS_OK) {
                    LOG_ERROR("PreprocessImageNpu: GetWorkspace failed, size=%lu",
                              static_cast<unsigned long>(ws_size));
                    return wp.second;
                }
            }
            atb_s = op.get()->Execute(vp, ws, ws_size, ctx);
            if (atb_s != atb::NO_ERROR) {
                LOG_ERROR("PreprocessImageNpu: ATB op Execute failed, atbStatus=%d",
                          static_cast<int>(atb_s));
                return ERROR_INFERENCE;
            }
            return STATUS_OK;
        };

        // 4a. MULS(1/255): resized -> normalize_tmp
        {
            atb::VariantPack vp;
            vp.inTensors = {resized_tensor};
            vp.outTensors = {normalize_tmp};
            ret = RunAtbOp(ops::ElewiseOp::MakeMuls(1.0f / 255.0f), vp);
            if (ret != STATUS_OK) goto cleanup;
        }

        // 4b. ADD(-mean broadcast): normalize_tmp + mean_bc -> resized_tensor (overwrite)
        {
            atb::VariantPack vp;
            vp.inTensors = {normalize_tmp, mean_bc};
            vp.outTensors = {resized_tensor};
            ret = RunAtbOp(ops::ElewiseOp::MakeAdd(), vp);
            if (ret != STATUS_OK) goto cleanup;
        }

        // 4c. MUL(1/std broadcast): resized_tensor * inv_std_bc -> normalize_tmp (final)
        {
            atb::VariantPack vp;
            vp.inTensors = {resized_tensor, inv_std_bc};
            vp.outTensors = {normalize_tmp};
            ret = RunAtbOp(ops::ElewiseOp::MakeMul(), vp);
            if (ret != STATUS_OK) goto cleanup;
        }

        // Ensure all NPU ops complete before D2H
        ret = runtime->Synchronize();
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpu: Synchronize after normalize failed");
            goto cleanup;
        }

        // Free broadcast tensors and bicubic output (no longer needed);
        // Free() sets deviceData to nullptr so cleanup skips them.
        alloc->Free(mean_bc);
        alloc->Free(inv_std_bc);
        alloc->Free(resized_tensor);

        // ── Step 5: Patch extraction on NPU (AsStrided + 8-D Transpose) ──
        // Replaces the old D2H + CPU patch loop.  normalize_tmp [1,C,new_h,new_w]
        // is logically [C, new_h, new_w] = [C, merged_h*ms*ps, merged_w*ms*ps].
        // AsStrided reinterprets it as the 8-D patch grid [tp, C, merged_h, ms,
        // ps, merged_w, ms, ps] with a 0 element-stride on tp (broadcasts the
        // single frame to all tp frames — matches the CPU loop's f=0 -> f=1..tp-1
        // memcpy).  Transpose perm [2,5,3,6,1,0,4,7] reorders to [merged_h,
        // merged_w, ms, ms, C, tp, ps, ps], whose row-major layout is exactly
        // [num_patches, patch_dim].  Bit-exact vs CPU (test_patch_transpose_spike).
        int32_t grid_t = 1;
        int32_t grid_h = new_h / patch_size;
        int32_t grid_w = new_w / patch_size;
        int64_t merged_h = grid_h / merge_size;
        int64_t merged_w = grid_w / merge_size;

        grid_thw[0] = grid_t;
        grid_thw[1] = grid_h;
        grid_thw[2] = grid_w;
        num_patches = static_cast<int64_t>(grid_t) * grid_h * grid_w;

        // 8-D shapes (grid_t=1 squeezed; tp handled by AsStrided broadcast).
        // shape7 = [C, merged_h, ms, ps, merged_w, ms, ps] is the contiguous
        // row-major view of [C, new_h, new_w]; shape8 prepends tp.
        const int64_t ms = merge_size;
        const int64_t ps = patch_size;
        std::vector<int64_t> shape7 = {channels, merged_h, ms, ps, merged_w, ms, ps};
        std::vector<int64_t> shape8 = {tp, channels, merged_h, ms, ps, merged_w, ms, ps};

        // Row-major element strides for shape7 (verified bit-exact in the spike);
        // tp dim gets stride 0 (broadcast).
        std::vector<int64_t> stride7(7);
        {
            int64_t s = 1;
            for (int i = 6; i >= 0; --i) { stride7[i] = s; s *= shape7[i]; }
        }
        std::vector<int64_t> stride8;
        stride8.push_back(0);
        for (int64_t s7 : stride7) stride8.push_back(s7);

        // 5a. AsStrided: normalize_tmp -> asstrided_out [shape8, stride8, offset 0]
        {
            atb::infer::AsStridedParam param;
            for (int64_t s : shape8)  param.size.push_back(s);
            for (int64_t s : stride8) param.stride.push_back(s);
            param.offset.push_back(0);

            atb::Operation* raw = nullptr;
            atb::Status atb_s = atb::CreateOperation(param, &raw);
            if (atb_s != atb::NO_ERROR || raw == nullptr) {
                LOG_ERROR("PreprocessImageNpu: AsStrided CreateOperation failed, atbStatus=%d",
                          static_cast<int>(atb_s));
                ret = ERROR_INFERENCE;
                goto cleanup;
            }
            OperationHandle asstrided_op(raw);

            ret = alloc->AllocFloat16(asstrided_out, shape8);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: AllocFloat16 asstrided_out failed");
                goto cleanup;
            }

            atb::VariantPack vp;
            vp.inTensors  = {normalize_tmp};
            vp.outTensors = {asstrided_out};
            ret = RunAtbOp(std::move(asstrided_op), vp);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: AsStrided execute failed");
                goto cleanup;
            }
        }

        // 5b. Transpose perm [2,5,3,6,1,0,4,7]:
        //   [tp,C,mh,ms,ps,mw,ms,ps] -> [mh,mw,ms,ms,C,tp,ps,ps]
        // whose row-major layout == [num_patches, patch_dim].
        {
            const std::vector<int32_t> perm = {2, 5, 3, 6, 1, 0, 4, 7};
            std::vector<int64_t> perm_out_shape(8);
            for (int i = 0; i < 8; i++) {
                perm_out_shape[i] = shape8[perm[i]];
            }

            OperationHandle transpose_op = ops::TransposeOp::Create(perm);
            if (!transpose_op) {
                LOG_ERROR("PreprocessImageNpu: TransposeOp::Create failed");
                ret = ERROR_INFERENCE;
                goto cleanup;
            }

            ret = alloc->AllocFloat16(transpose_out, perm_out_shape);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: AllocFloat16 transpose_out failed");
                goto cleanup;
            }

            atb::VariantPack vp;
            vp.inTensors  = {asstrided_out};
            vp.outTensors = {transpose_out};
            ret = RunAtbOp(std::move(transpose_op), vp);
            if (ret != STATUS_OK) {
                LOG_ERROR("PreprocessImageNpu: Transpose execute failed");
                goto cleanup;
            }
        }

        // ── Step 6: free intermediates (no D2H — path C keeps the device tensor) ──
        // transpose_out is already laid out as [num_patches, patch_dim] fp16.
        // The sync ensures the async AsStrided+Transpose complete before we free
        // their inputs (normalize_tmp, asstrided_out); it also makes transpose_out
        // valid for the caller (D2H in the CPU overload, direct feed in path C).
        // transpose_out itself is NOT freed here — ownership transfers to the caller.
        ret = runtime->Synchronize();
        if (ret != STATUS_OK) {
            LOG_ERROR("PreprocessImageNpuInternal: Synchronize before freeing intermediates failed");
            goto cleanup;
        }

        // Free patch-pipeline intermediates (normalize_tmp was the AsStrided
        // input; asstrided_out the Transpose input). transpose_out is returned.
        alloc->Free(normalize_tmp);
        alloc->Free(asstrided_out);
    }

    return STATUS_OK;  // transpose_out valid, caller owns (Free or Detach)

cleanup:
    // Free in reverse allocation order; Free() is idempotent (checks deviceData
    // != nullptr and nulls it afterwards), so already-freed tensors are safe.
    if (transpose_out.deviceData != nullptr)   { alloc->Free(transpose_out); }
    if (asstrided_out.deviceData != nullptr)   { alloc->Free(asstrided_out); }
    if (inv_std_bc.deviceData != nullptr)      { alloc->Free(inv_std_bc); }
    if (mean_bc.deviceData != nullptr)         { alloc->Free(mean_bc); }
    if (normalize_tmp.deviceData != nullptr)   { alloc->Free(normalize_tmp); }
    if (resized_tensor.deviceData != nullptr)  { alloc->Free(resized_tensor); }
    if (input_tensor.deviceData != nullptr)    { alloc->Free(input_tensor); }

    return ret;
}

// ── CPU-pointer overload: NPU preprocess + D2H into a host buffer. ──
// Existing public contract (preprocessed pixel_values on host). The device
// tensor from the internal helper is D2H'd here then freed.
Status PreprocessImageNpu(IRuntime* runtime,
                          const uint8_t* image,
                          int32_t channels, int32_t height, int32_t width,
                          const Qwen3VLConfig& config,
                          uint16_t* pixel_values,
                          int64_t& num_patches,
                          int64_t* grid_thw) {
    if (runtime == nullptr || image == nullptr || pixel_values == nullptr || grid_thw == nullptr) {
        LOG_ERROR("PreprocessImageNpu: null parameter (runtime=%p, image=%p, pixel_values=%p, grid_thw=%p)",
                  static_cast<const void*>(runtime), static_cast<const void*>(image),
                  static_cast<const void*>(pixel_values), static_cast<const void*>(grid_thw));
        return ERROR_INVALID_PARAM;
    }

    atb::Tensor transpose_out{};
    Status ret = PreprocessImageNpuInternal(runtime, image, channels, height, width,
                                            config, transpose_out, num_patches, grid_thw);
    if (ret != STATUS_OK) return ret;

    // transpose_out is allocator-tracked and already synchronised by the helper
    // (its free-safety sync also makes the Transpose output valid for D2H), so a
    // single D2H delivers pixel_values directly — no CPU rearrange, no extra sync.
    auto* alloc = runtime->GetAllocator();
    int64_t patch_dim = static_cast<int64_t>(channels) * config.pp_temporal_patch_size *
                        config.pp_patch_size * config.pp_patch_size;
    ret = alloc->CopyToHost(pixel_values, transpose_out,
                            static_cast<size_t>(num_patches) * patch_dim * sizeof(uint16_t));
    alloc->Free(transpose_out);  // tracked; free regardless of D2H result
    if (ret != STATUS_OK) {
        LOG_ERROR("PreprocessImageNpu: D2H transpose_out failed");
        return ret;
    }

    int32_t patch_size = config.pp_patch_size;
    LOG_INFO("PreprocessImageNpu: %dx%d -> %dx%d, grid=(%ld,%ld,%ld), patches=%ld, dim=%ld",
             height, width,
             static_cast<long>(grid_thw[1]) * patch_size, static_cast<long>(grid_thw[2]) * patch_size,
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(num_patches), static_cast<long>(patch_dim));
    return STATUS_OK;
}

// ── Device-tensor overload (path C): no D2H — the device tensor stays on NPU
// and is handed to the caller (ownership transferred via Detach). The caller
// wraps it with NpuTensor::Adopt (or aclrtFree's it) and feeds it directly to
// the vision pipeline, eliminating the D2H (here) → H2D (ForwardWithTiming)
// round trip of the CPU-pointer overload above. ──
Status PreprocessImageNpu(IRuntime* runtime,
                          const uint8_t* image,
                          int32_t channels, int32_t height, int32_t width,
                          const Qwen3VLConfig& config,
                          atb::Tensor& pixel_values_npu,
                          int64_t& num_patches,
                          int64_t* grid_thw) {
    if (runtime == nullptr || image == nullptr || grid_thw == nullptr) {
        LOG_ERROR("PreprocessImageNpu(device): null parameter (runtime=%p, image=%p, grid_thw=%p)",
                  static_cast<const void*>(runtime), static_cast<const void*>(image),
                  static_cast<const void*>(grid_thw));
        return ERROR_INVALID_PARAM;
    }

    Status ret = PreprocessImageNpuInternal(runtime, image, channels, height, width,
                                            config, pixel_values_npu, num_patches, grid_thw);
    if (ret != STATUS_OK) return ret;

    // Transfer device ownership to the caller: untrack from the allocator so
    // NpuTensor::Adopt (aclrtFree) won't clash with ~TensorAllocator/FreeAll.
    runtime->GetAllocator()->Detach(pixel_values_npu);

    // PatchEmbedGraph reads the input's desc.dims[0] to recover N (it assumes a
    // 1-D flat input of N*patch_dim). The Transpose output is 8-D; reinterpret
    // it as 1-D {num_patches*patch_dim} (same deviceData/dataSize, row-major).
    int64_t patch_dim = static_cast<int64_t>(channels) * config.pp_temporal_patch_size *
                        config.pp_patch_size * config.pp_patch_size;
    pixel_values_npu.desc.shape.dimNum = 1;
    pixel_values_npu.desc.shape.dims[0] = num_patches * patch_dim;
    // dataSize already == num_patches * patch_dim * sizeof(fp16); unchanged.

    int32_t patch_size = config.pp_patch_size;
    LOG_INFO("PreprocessImageNpu(device): %dx%d -> %dx%d, grid=(%ld,%ld,%ld), patches=%ld, dim=%ld [path C, no D2H]",
             height, width,
             static_cast<long>(grid_thw[1]) * patch_size, static_cast<long>(grid_thw[2]) * patch_size,
             static_cast<long>(grid_thw[0]), static_cast<long>(grid_thw[1]),
             static_cast<long>(grid_thw[2]),
             static_cast<long>(num_patches), static_cast<long>(patch_dim));
    return STATUS_OK;
}

}  // namespace adapters
}  // namespace atb_llm
