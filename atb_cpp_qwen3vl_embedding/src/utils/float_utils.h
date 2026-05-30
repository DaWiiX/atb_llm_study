#pragma once

#include <cstdint>
#include <cstring>

namespace atb_llm {

/// Type-pun a float to its uint32_t bit representation via std::memcpy.
/// This avoids undefined behavior from reinterpret_cast<uint32_t&>(float_val).
inline uint32_t FloatToUint32(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/// Correct fp16 -> fp32 conversion via IEEE 754 bit manipulation.
inline float Fp16ToF32(uint16_t fp16_bits) {
    uint32_t sign = (fp16_bits >> 15) & 1;
    uint32_t exp = (fp16_bits >> 10) & 0x1F;
    uint32_t mantissa = fp16_bits & 0x3FF;
    uint32_t f32_bits;
    if (exp == 0) {
        if (mantissa == 0) {
            f32_bits = sign << 31;
        } else {
            // Denormalized fp16 -> normalize for fp32
            exp = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF;
            f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f32_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        // Normalized
        f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    float val;
    std::memcpy(&val, &f32_bits, sizeof(float));
    return val;
}

/// Correct fp32 -> fp16 conversion via IEEE 754 bit manipulation.
inline uint16_t Fp32ToFp16(float val) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &val, sizeof(float));
    uint32_t sign = (f32_bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((f32_bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32_bits & 0x7FFFFF;

    if (exp == 128) {
        // Inf or NaN
        return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
    }
    if (exp > 15) {
        // Overflow -> fp16 inf
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    if (exp < -24) {
        // Too small -> zero
        return static_cast<uint16_t>(sign);
    }
    if (exp < -14) {
        // Denormalized fp16
        mantissa = (mantissa | 0x800000) >> (-14 - exp);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    // Normal fp16
    return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mantissa >> 13));
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
