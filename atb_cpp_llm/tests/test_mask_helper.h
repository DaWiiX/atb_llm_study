#pragma once
/// Shared helper: upload mask tensors in platform-correct format.
///
/// 910B: ND  [S, S]
/// 310P: NZ  [1, ceil(S/16), ceil(S/16)*16, 16]  (FRACTAL_NZ)
///
/// The production equivalent lives in qwen3vl_model.cpp:570-581.

#include "core/tensor_allocator.h"
#include "utils/float_utils.h"
#include "util/cpp11_compat.h"
#include "runners/text_runner.h"   // MakeCausalMaskFp16
#include <vector>
#include <cstdint>

namespace atb_llm {
namespace test {

/// Upload mask data in the correct format for the current platform.
///
/// @param alloc        TensorAllocator from the runtime
/// @param mask_nd_data fp16 mask in ND (row-major) layout, S*S elements
/// @param S            sequence length
/// @param mask_t       [out] allocated device tensor, correct format & shape
inline void UploadMask(TensorAllocator* alloc,
                       const uint16_t* mask_nd_data,
                       int64_t S,
                       atb::Tensor& mask_t) {
    if (Is310P()) {
        int64_t s_pad = ((S + 15) / 16) * 16;
        int64_t n1    = (S + 15) / 16;
        int64_t nz_elems = n1 * s_pad * 16;
        alloc->AllocFloat16(mask_t, {1, n1, s_pad, 16});
        mask_t.desc.format = ACL_FORMAT_FRACTAL_NZ;
        std::vector<uint16_t> nz(static_cast<size_t>(nz_elems));
        ConvertNdToNzFp16(mask_nd_data, S, S, nz.data());
        alloc->CopyToDevice(mask_t, nz.data(),
                            static_cast<size_t>(nz_elems) * sizeof(uint16_t));
    } else {
        alloc->AllocFloat16(mask_t, {S, S});
        alloc->CopyToDevice(mask_t, mask_nd_data,
                            static_cast<size_t>(S) * S * sizeof(uint16_t));
    }
}

/// Generate a causal (upper-triangular) mask and upload in platform format.
///
/// @param alloc  TensorAllocator from the runtime
/// @param S      sequence length
/// @param mask_t [out] allocated device tensor, correct format & shape
inline void UploadCausalMask(TensorAllocator* alloc,
                              int64_t S,
                              atb::Tensor& mask_t) {
    if (Is310P()) {
        // 310P: generate causal mask directly in NZ layout (no intermediate ND)
        int64_t s_pad = ((S + 15) / 16) * 16;
        int64_t n1    = (S + 15) / 16;
        int64_t nz_elems = n1 * s_pad * 16;
        alloc->AllocFloat16(mask_t, {1, n1, s_pad, 16});
        mask_t.desc.format = ACL_FORMAT_FRACTAL_NZ;
        std::vector<uint16_t> nz(static_cast<size_t>(nz_elems));
        runners::MakeCausalMaskNzFp16(static_cast<int32_t>(S), nz.data(), s_pad, n1);
        alloc->CopyToDevice(mask_t, nz.data(),
                            static_cast<size_t>(nz_elems) * sizeof(uint16_t));
    } else {
        // 910B: standard ND causal mask
        std::vector<uint16_t> mask_nd(static_cast<size_t>(S) * S);
        runners::MakeCausalMaskFp16(static_cast<int32_t>(S), mask_nd.data());
        alloc->AllocFloat16(mask_t, {S, S});
        alloc->CopyToDevice(mask_t, mask_nd.data(),
                            static_cast<size_t>(S) * S * sizeof(uint16_t));
    }
}

}  // namespace test
}  // namespace atb_llm
