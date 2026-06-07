#pragma once
#include <cstdint>

namespace atb_llm {

// ── 注意力类型 ─────────────────────────────────────────────
enum class AttnType {
    GQA,    // Grouped-Query Attention (Qwen3, LLaMA, etc.)
    MHA,    // Multi-Head Attention (standard, all heads equal)
    MLA     // Multi-head Latent Attention (DeepSeek-V2/V3)
};

// ── MLP 类型 ───────────────────────────────────────────────
enum class MlpType {
    SwiGLU, // gate + SiLU, up, mul, down (Qwen3, LLaMA)
    GeGLU,  // gate + GELU, up, mul, down
    GELU,   // fc1 + GELU + fc2 (ViT-style)
    MoE     // Mixture of Experts
};

// ── 归一化类型 ─────────────────────────────────────────────
enum class NormType {
    RMSNorm,    // Root Mean Square Layer Normalization
    LayerNorm   // Standard Layer Normalization
};

// ── RoPE 类型 ──────────────────────────────────────────────
enum class RopeType {
    MRoPE_3D,   // Multi-dimensional RoPE (Qwen3VL)
    Rope1D,     // Standard 1D RoPE
    YaRN        // YaRN extension
};

// ── 注意力配置 ─────────────────────────────────────────────
struct AttnConfig {
    AttnType  type         = AttnType::GQA;
    int32_t   num_heads    = 16;
    int32_t   num_kv_heads = 16;
    int32_t   head_dim     = 128;
    int32_t   seq_len      = 1;
    int32_t   batch_size   = 1;      // batch dimension (future use)
    float     epsilon      = 1e-6f;
    bool      use_qk_norm  = true;
    bool      use_bias     = false;
    bool      use_mask     = true;
    int32_t   rotary_dim   = 2;
    // MLA specific
    int32_t   kv_lora_rank = 0;
    int32_t   q_lora_rank  = 0;
};

// ── MLP 配置 ───────────────────────────────────────────────
struct MlpConfig {
    MlpType   type              = MlpType::SwiGLU;
    int32_t   intermediate_size = 4096;
    int32_t   batch_size        = 1;   // batch dimension (future use)
    bool      use_bias          = false;
    // MoE specific
    int32_t   num_experts       = 0;
    int32_t   top_k             = 0;
};

// ── 归一化配置 ─────────────────────────────────────────────
struct NormConfig {
    NormType  type        = NormType::RMSNorm;
    float     epsilon     = 1e-6f;
    bool      use_bias    = false;
};

// ── 层描述符（单层解码器完整配置）──────────────────────────
struct LayerDescriptor {
    AttnConfig attn;
    MlpConfig  mlp;
    NormConfig input_norm;
    NormConfig post_norm;
};

} // namespace atb_llm
