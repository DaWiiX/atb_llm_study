#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <acl/acl.h>

namespace atb_llm {

// ── 输入模式 ─────────────────────────────────────────────
enum class InputMode {
    TEXT_ONLY,          // 纯文本推理
    IMAGE_ONLY,         // 纯图片 -> 视觉特征
    IMAGE_AND_TEXT,     // 图文混合（原始图片 + 文本）
    PREPROCESSED        // 已预处理（pixel_values + grid_thw + 文本）
};

// ── 原始图片输入 ─────────────────────────────────────────
struct RawImage {
    const uint8_t* data = nullptr;   // NCHW uint8，像素值 [0, 255]
    int64_t channels = 0;            // 通常 3
    int64_t height = 0;
    int64_t width = 0;
};

// ── 预处理后的图片输入 ───────────────────────────────────
struct PreprocessedImage {
    const void* pixel_values = nullptr;    // (N, patch_dim) float16/float32
    int64_t num_patches = 0;               // N
    int64_t patch_dim = 0;                 // 每个 patch 的维度
    const int64_t* grid_thw = nullptr;     // (3,) [grid_t, grid_h, grid_w]
    aclDataType dtype = ACL_DT_UNDEFINED;  // ACL_FLOAT16 或 ACL_FLOAT32
};

// ── 文本输入 ─────────────────────────────────────────────
struct TextInput {
    const int64_t* input_ids = nullptr;    // (B, S) token IDs
    int64_t batch_size = 0;                // B
    int64_t seq_length = 0;                // S
};

// ── 推理请求 ─────────────────────────────────────────────
struct InferRequest {
    InputMode mode = InputMode::TEXT_ONLY;
    TextInput text;
    RawImage raw_image;              // mode in {IMAGE_ONLY, IMAGE_AND_TEXT}
    PreprocessedImage preprocessed;  // mode == PREPROCESSED
};

// ── 推理结果 ─────────────────────────────────────────────
// 所有权：Engine 内部管理内存，用户在下一次 Forward/Encode 调用前数据有效
struct InferResult {
    std::vector<uint8_t> data;   // 输出数据（host 内存，RAII 管理）
    std::vector<int64_t> shape;
    aclDataType dtype = ACL_DT_UNDEFINED;

    /// 获取 typed 指针（方便使用）
    template <typename T>
    const T* As() const {
        return reinterpret_cast<const T*>(data.data());
    }
};

// ── 引擎配置 ─────────────────────────────────────────────
struct EngineConfig {
    std::string model_dir;   // 包含 config.json + model.safetensors
    int64_t buffer_size = 0;     // ATB buffer size (bytes)，0 = 自动
    int device_id = 0;           // NPU 设备 ID
    bool normalize = false;      // 输出是否 L2 归一化
};

// ── 状态码 ───────────────────────────────────────────────
enum Status : int32_t {
    STATUS_OK = 0,
    ERROR_INVALID_PARAM = -1,
    ERROR_FILE_NOT_FOUND = -2,
    ERROR_WEIGHT_LOAD = -3,
    ERROR_GRAPH_BUILD = -4,
    ERROR_NPU_MEMORY = -5,
    ERROR_INFERENCE = -6,
    ERROR_UNSUPPORTED = -7,
    ERROR_ATB_BASE = -1000,
};

} // namespace atb_llm
