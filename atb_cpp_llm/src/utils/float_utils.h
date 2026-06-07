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

/// Convert bf16 (uint16 raw bits) to float16 (uint16 raw bits) via bit manipulation.
/// bf16: 1 sign + 8 exp + 7 mantissa (stored in upper 16 bits of float32)
/// fp16: 1 sign + 5 exp + 10 mantissa
inline uint16_t Bf16ToFp16(uint16_t bf16_bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16;
    uint32_t sign = (f32_bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((f32_bits >> 23) & 0xFF) - 127;

    if (exp == 128) {
        // Inf or NaN -> clamp to fp16 inf
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    if (exp < -24) {
        // Too small for fp16 denormals -> zero
        return static_cast<uint16_t>(sign);
    }
    if (exp < -14) {
        // fp16 denormal range
        uint32_t mantissa = (f32_bits & 0x7FFFFF) | 0x800000;
        int32_t shift = -14 - exp;
        mantissa >>= (shift + 13);
        return static_cast<uint16_t>(sign | mantissa);
    }
    if (exp > 15) {
        // Overflow -> fp16 max
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    // Normal fp16
    uint16_t fp16_exp = static_cast<uint16_t>(exp + 15);
    uint16_t fp16_mantissa = static_cast<uint16_t>((f32_bits >> 13) & 0x3FF);
    return static_cast<uint16_t>(sign | (fp16_exp << 10) | fp16_mantissa);
}

/// Convert bf16 buffer to fp16 buffer (element-wise).
inline void Bf16ToFp16Buffer(const uint16_t* src, uint16_t* dst, size_t num_elements) {
    for (size_t i = 0; i < num_elements; i++) {
        dst[i] = Bf16ToFp16(src[i]);
    }
}

}  // namespace atb_llm
