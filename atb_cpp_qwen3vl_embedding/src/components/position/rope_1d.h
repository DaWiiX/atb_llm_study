#pragma once
#include <cstdint>
#include <vector>
#include <cmath>

namespace atb_llm {
namespace components {

/// Standard 1D Rotary Position Embedding (RoPE).
///
/// This is CPU-side computation (no ATB graph).
/// Takes 1D position_ids and produces cos/sin values for the ATB RopeOperation.
///
/// Unlike MRoPE (which handles 3D T/H/W positions with interleaved sections),
/// this class implements the standard 1D RoPE used by most transformer models
/// (LLaMA, GPT-NeoX, etc.).
///
/// Usage:
///   Rope1D rope(head_dim=128, rope_theta=10000.0f, rotary_dim=128);
///   rope.Compute(position_ids, seq_len, cos_out, sin_out);
///
/// Output shape: (seq_len, head_dim), float32, compatible with RopeOp inputs.
class Rope1D {
public:
    /// Constructor.
    /// @param head_dim      Dimension per attention head (e.g. 128)
    /// @param rope_theta    Base frequency (e.g. 10000.0f, or 5000000.0f for Qwen3)
    /// @param rotary_dim    Number of dimensions to apply rotary embedding to.
    ///                      Must be <= head_dim and even. Defaults to head_dim.
    ///                      When rotary_dim < head_dim, the trailing dimensions
    ///                      are left unchanged (pass-through).
    Rope1D(int32_t head_dim, float rope_theta, int32_t rotary_dim = -1);

    /// Compute 1D RoPE cos/sin from position IDs.
    ///
    /// @param position_ids  1D position IDs, shape (S,) as int64. Each element
    ///                       is the position index for that sequence token.
    /// @param seq_len       Sequence length S
    /// @param cos_out       Output: cos values, shape (S, head_dim), float32, pre-allocated
    /// @param sin_out       Output: sin values, shape (S, head_dim), float32, pre-allocated
    void Compute(const int64_t* position_ids, int64_t seq_len,
                 float* cos_out, float* sin_out) const;

    /// Generate cos/sin for a contiguous range of positions [0, seq_len).
    ///
    /// Convenience method for prefill / decode phases where positions are
    /// simply 0, 1, 2, ..., seq_len-1.
    ///
    /// @param seq_len       Sequence length S
    /// @param cos_out       Output: cos values, shape (S, head_dim), float32, pre-allocated
    /// @param sin_out       Output: sin values, shape (S, head_dim), float32, pre-allocated
    void ComputeContiguous(int64_t seq_len, float* cos_out, float* sin_out) const;

    int32_t HeadDim() const {
        return head_dim_;
    }
    int32_t RotaryDim() const {
        return rotary_dim_;
    }
    float RopeTheta() const {
        return rope_theta_;
    }

private:
    int32_t head_dim_;
    float rope_theta_;
    int32_t rotary_dim_;           // number of dims with rotary embedding (even, <= head_dim)
    int32_t half_dim_;             // rotary_dim / 2
    std::vector<float> inv_freq_;  // rotary_dim/2 elements

    /// Compute inv_freq table: inv_freq[i] = 1.0 / (theta^(2i/rotary_dim))
    void BuildInvFreq();
};

// ═════════════════════════════════════════════════════════════════════
// Inline implementation
// ═════════════════════════════════════════════════════════════════════

inline Rope1D::Rope1D(int32_t head_dim, float rope_theta, int32_t rotary_dim) : head_dim_(head_dim), rope_theta_(rope_theta) {
    // Default rotary_dim = head_dim (full rotation)
    if (rotary_dim <= 0) {
        rotary_dim_ = head_dim;
    } else {
        rotary_dim_ = rotary_dim;
    }

    // Validate: rotary_dim must be even and <= head_dim
    if (rotary_dim_ > head_dim_) {
        rotary_dim_ = head_dim_;
    }
    if (rotary_dim_ % 2 != 0) {
        rotary_dim_--;  // round down to even
    }

    half_dim_ = rotary_dim_ / 2;
    BuildInvFreq();
}

inline void Rope1D::BuildInvFreq() {
    inv_freq_.resize(half_dim_);
    for (int32_t i = 0; i < half_dim_; i++) {
        inv_freq_[i] = 1.0f / std::pow(rope_theta_,
                                       static_cast<float>(2 * i) / rotary_dim_);
    }
}

inline void Rope1D::Compute(const int64_t* position_ids, int64_t seq_len,
                            float* cos_out, float* sin_out) const {
    // For each position s and frequency index i:
    //   freq = inv_freq[i] * position_ids[s]
    //   cos_out[s, i] = cos(freq)
    //   sin_out[s, i] = sin(freq)
    //
    // The first rotary_dim columns get cos/sin values.
    // If rotary_dim < head_dim, remaining columns are zeroed
    // (the ATB RopeOp handles the rotation in the first rotary_dim dims).

    for (int64_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(position_ids[s]);
        float* c_row = cos_out + s * head_dim_;
        float* s_row = sin_out + s * head_dim_;

        // Rotary dimensions: cos/sin for first half_dim_ frequencies,
        // duplicated to fill rotary_dim_ columns (LLAMA-style contiguous-half).
        for (int32_t i = 0; i < half_dim_; i++) {
            float val = inv_freq_[i] * pos;
            float c = std::cos(val);
            float sn = std::sin(val);
            // Contiguous-half layout: [i] and [i + half_dim_] get the same value
            c_row[i] = c;
            c_row[i + half_dim_] = c;
            s_row[i] = sn;
            s_row[i + half_dim_] = sn;
        }

        // Zero out remaining dimensions (if rotary_dim < head_dim)
        for (int32_t i = rotary_dim_; i < head_dim_; i++) {
            c_row[i] = 0.0f;
            s_row[i] = 0.0f;
        }
    }
}

inline void Rope1D::ComputeContiguous(int64_t seq_len, float* cos_out,
                                      float* sin_out) const {
    // Generate positions [0, 1, 2, ..., seq_len-1] and compute
    for (int64_t s = 0; s < seq_len; s++) {
        float pos = static_cast<float>(s);
        float* c_row = cos_out + s * head_dim_;
        float* s_row = sin_out + s * head_dim_;

        for (int32_t i = 0; i < half_dim_; i++) {
            float val = inv_freq_[i] * pos;
            float c = std::cos(val);
            float sn = std::sin(val);
            c_row[i] = c;
            c_row[i + half_dim_] = c;
            s_row[i] = sn;
            s_row[i + half_dim_] = sn;
        }

        for (int32_t i = rotary_dim_; i < head_dim_; i++) {
            c_row[i] = 0.0f;
            s_row[i] = 0.0f;
        }
    }
}

}  // namespace components
}  // namespace atb_llm
