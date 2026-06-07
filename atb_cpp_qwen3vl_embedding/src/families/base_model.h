#pragma once
#include "atb_llm/model.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace families {

/// 模型基类 -- 提供多模型共享的工具方法
///
/// 继承 IModel，添加 ExecuteGraph、EmbeddingLookup、FindImageTokenPositions、
/// RunPooling、InjectDeepstack 等通用工具。子类（如 Qwen3VLModel）继承 BaseModel，
/// 不再重复实现这些方法。
class BaseModel : public IModel {
public:
    BaseModel() = default;
    ~BaseModel() override = default;

protected:
    /// Runtime pointer -- set by subclass in Load()
    IRuntime* runtime_ = nullptr;

    // ── NPU 图执行 ────────────────────────────────────────
    /// 执行 ATB 图：Setup → GetWorkspace → Execute → Synchronize
    Status ExecuteGraph(OperationHandle& graph, atb::VariantPack& vp);

    // ── CPU 嵌入查找 ──────────────────────────────────────
    /// fp16 嵌入表查找：token_ids → fp16 向量
    void EmbeddingLookup(const int64_t* input_ids, int64_t seq_len,
                         const uint16_t* embed_table,
                         int64_t hidden_size, int64_t vocab_size,
                         uint16_t* output);

    // ── 图像 token 定位 ───────────────────────────────────
    /// 在 input_ids 中查找等于 image_token_id 的位置
    static std::vector<int64_t> FindImageTokenPositions(
        const int64_t* input_ids, int64_t seq_len,
        int64_t image_token_id);

    // ── 池化策略 ──────────────────────────────────────────
    enum class PoolingStrategy {
        LAST_TOKEN,   // 取最后一个 token（Qwen3VL embedding 模式）
        MEAN,         // 均值池化
        CLS           // 取 CLS token
    };

    /// 从 hidden_states 中提取池化结果
    Status RunPooling(const uint16_t* hidden_states, int64_t seq_len,
                      int64_t hidden_size, bool normalize,
                      PoolingStrategy strategy,
                      InferResult& result);

    // ── Deepstack 注入 ────────────────────────────────────
    /// 将 deepstack 特征加法注入到 hidden states 的特定位置
    void InjectDeepstack(NpuTensor& hidden_npu,
                         const std::vector<uint16_t>& ds_feat,
                         const std::vector<int64_t>& positions,
                         int64_t seq_len, int64_t hidden_size,
                         int64_t feat_dim,
                         IRuntime* runtime);
};

}  // namespace families
}  // namespace atb_llm
