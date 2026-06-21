#pragma once
#include "atb_llm/model.h"
#include "atb_llm/runtime.h"
#include "core/raii.h"
#include "core/npu_tensor.h"
#include "log/logger.h"
#include <cstdint>
#include <vector>

namespace atb_llm {
namespace families {

/// 模型基类 -- 提供多模型共享的工具方法
///
/// 继承 IModel，添加 ExecuteGraph、EmbeddingLookup、FindImageTokenPositions、
/// RunPooling 等通用工具。子类（如 Qwen3VLModel）继承 BaseModel，
/// 不再重复实现这些方法。
/// Deepstack 注入逻辑已移至 components::DeepstackFusion。
class BaseModel : public IModel {
public:
    BaseModel() = default;
    ~BaseModel() override = default;

protected:
    /// Runtime pointer -- set by subclass in Load()
    IRuntime* runtime_ = nullptr;

    // ── NPU 图执行 ────────────────────────────────────────
    /// 执行 ATB 图：Setup → GetWorkspace → Execute（默认异步；ATB_ENABLE_PER_OP_SYNC=1 时加 Synchronize）
    Status ExecuteGraph(OperationHandle& graph, atb::VariantPack& vp);

#ifdef DEBUG
    /// Debug-only: execute graph with VariantPack size validation.
    /// Checks that vp.inTensors.size() == expected_in_count and
    /// vp.outTensors.size() == expected_out_count before execution.
    /// Returns ERROR_INVALID_PARAM on mismatch; otherwise delegates
    /// to ExecuteGraph().
    Status ExecuteGraphChecked(OperationHandle& graph, atb::VariantPack& vp,
                               size_t expected_in_count, size_t expected_out_count,
                               const char* context = "");
#endif

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

public:
    // ── 池化策略 ──────────────────────────────────────────
    enum class PoolingStrategy {
        LAST_TOKEN,         // 取最后一个 token (seq_len-1)
        LAST_TOKEN_BY_MASK, // 用 attention_mask 找最后一个非 padding token
        MEAN,               // 均值池化
        CLS                 // 取 CLS token
    };

    /// 从 hidden_states 中提取池化结果
    Status RunPooling(const uint16_t* hidden_states, int64_t seq_len,
                      int64_t hidden_size, bool normalize,
                      PoolingStrategy strategy,
                      InferResult& result,
                      const int64_t* attention_mask = nullptr);
};

}  // namespace families

/**
 * @brief Execute an ATB operation with Setup→GetWorkspace→Execute lifecycle.
 *
 * Encapsulates the repeated pattern of calling Setup to query workspace size,
 * allocating workspace via the runtime, then calling Execute. Optional
 * Synchronize after Execute for ordering guarantees across streams.
 *
 * @param op       The ATB operation (graph or single op) to execute
 * @param vp       Input/output tensor VariantPack
 * @param runtime  Runtime for workspace allocation, context, and sync
 * @param ws_size  (in/out) Workspace size in bytes. Pass 0 on first call;
 *                 updated to the actual size allocated. Reuse across calls
 *                 to the same op to avoid redundant Setup queries.
 * @param sync     If true, call runtime->Synchronize() after Execute
 * @return Status  STATUS_OK on success, or an error code on failure
 */
inline Status ExecuteOperation(atb::Operation* op,
                               atb::VariantPack& vp,
                               IRuntime* runtime,
                               uint64_t& ws_size,
                               bool sync = false) {
    auto* ctx = runtime->GetContext();

    atb::Status atb_s = op->Setup(vp, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("ExecuteOperation: Setup failed with status %d",
                  static_cast<int>(atb_s));
        return ERROR_GRAPH_BUILD;
    }

    uint8_t* ws_ptr = nullptr;
    if (ws_size > 0) {
        auto __atb_pair_ws = runtime->GetWorkspace(ws_size);
        auto& ws = __atb_pair_ws.first;
        auto& ws_s = __atb_pair_ws.second;
        ws_ptr = ws;
        if (ws_s != STATUS_OK) {
            LOG_ERROR("ExecuteOperation: GetWorkspace(%lu) failed", ws_size);
            return ws_s;
        }
        if (ws_ptr == nullptr) {
            LOG_ERROR("ExecuteOperation: workspace pointer is null despite size=%lu",
                      ws_size);
            return ERROR_NPU_MEMORY;
        }
    } else {
        // GRAPH_LAUNCH_MODE requires a non-null workspace device pointer.
        auto __atb_pair_ws = runtime->GetWorkspace(1);
        auto& ws = __atb_pair_ws.first;
        auto& ws_s = __atb_pair_ws.second;
        if (ws_s == STATUS_OK && ws != nullptr) {
            ws_ptr = ws;
            ws_size = 1;
        }
    }

    atb_s = op->Execute(vp, ws_ptr, ws_size, ctx);
    if (atb_s != atb::NO_ERROR) {
        LOG_ERROR("ExecuteOperation: Execute failed with status %d",
                  static_cast<int>(atb_s));
        return ERROR_INFERENCE;
    }

    if (sync && runtime) {
        Status sync_s = runtime->Synchronize();
        if (sync_s != STATUS_OK) {
            LOG_ERROR("ExecuteOperation: Synchronize failed with status %d",
                      static_cast<int>(sync_s));
            return sync_s;
        }
    }

    return STATUS_OK;
}

}  // namespace atb_llm
