#pragma once

namespace atb_llm {

// ── Per-stage timing breakdown (milliseconds) ─────────────
struct StageTimings {
    double preprocess_ms = 0;    // Image preprocessing (NPU path C, or 0 if external PREPROCESSED)
    double vision_pos_ms = 0;    // Vision pos_embed + RoPE (CPU + NPU)
    double vision_model_ms = 0;  // Vision model: first_layer + blocks + merger (NPU)
    double text_embed_ms = 0;    // Text embedding lookup + vision injection (CPU)
    double position_ids_ms = 0;  // MRoPE position IDs (CPU)
    double text_model_ms = 0;    // Text decoder layers + FinalNorm (NPU)
    double pooling_ms = 0;       // Pooling + normalization (CPU)
    double e2e_ms = 0;           // Total end-to-end
};

} // namespace atb_llm
