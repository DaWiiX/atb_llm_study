#pragma once

#include <cstdint>
#include <cstring>
#include <acl/acl.h>

namespace atb_llm {

/// Type-pun a float to its uint32_t bit representation via std::memcpy.
inline uint32_t FloatToUint32(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/// fp16 -> fp32 using official CANN API
inline float Fp16ToF32(uint16_t fp16_bits) {
    return aclFloat16ToFloat(static_cast<aclFloat16>(fp16_bits));
}

/// fp32 -> fp16 using official CANN API
inline uint16_t Fp32ToFp16(float val) {
    return static_cast<uint16_t>(aclFloatToFloat16(val));
}

/// Convert bf16 (uint16 raw bits) to float16 (uint16 raw bits).
/// Uses bf16 → fp32 → fp16 path via CANN APIs, which applies
/// round-to-nearest-even matching Python's bf16→f32→f16 behavior.
/// The old bit-manipulation path truncated mantissa bits (zeroing
/// the low 3 fp16 mantissa bits), causing systematic rounding bias.
inline uint16_t Bf16ToFp16(uint16_t bf16_bits) {
    // Reconstruct fp32 from bf16 (bf16 is upper 16 bits of fp32)
    uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16;
    float f32_value;
    std::memcpy(&f32_value, &f32_bits, sizeof(float));
    // Convert fp32 → fp16 using CANN round-to-nearest-even
    return Fp32ToFp16(f32_value);
}

/// Convert bf16 buffer to fp16 buffer (element-wise).
inline void Bf16ToFp16Buffer(const uint16_t* src, uint16_t* dst, size_t num_elements) {
    for (size_t i = 0; i < num_elements; i++) {
        dst[i] = Bf16ToFp16(src[i]);
    }
}

}  // namespace atb_llm
