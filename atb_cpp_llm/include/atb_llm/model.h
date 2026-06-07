#pragma once
#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace atb_llm {

// Forward declaration for JsonConfig (used in RegistryEntry::CompatibilityCheck)
class JsonConfig;

/// 模型抽象接口 -- 每个模型实现此接口
///
/// 设计原则：
///   1. Engine 不知道模型内部结构，只调用 Load + Forward/Encode
///   2. 模型拥有完整的推理流程控制权（图构建、权重管理、执行顺序）
///   3. 模型通过 IRuntime 访问 NPU 资源，无需自己管理 Context/Stream
///   4. 组件层（TextAttention、SwiGluMLP 等）是可选的工具，非强制
class IModel {
public:
    virtual ~IModel() = default;

    // ── 生命周期 ─────────────────────────────────────────
    /// 加载模型：解析配置、加载权重、构建 ATB 图
    virtual Status Load(const std::string& model_dir, IRuntime* runtime) = 0;

    // ── 推理 ─────────────────────────────────────────────
    /// 推理入口：接受 InferRequest，输出 InferResult
    /// 模型自行决定内部流程
    virtual Status Forward(const InferRequest& request, InferResult& result) = 0;

    /// Forward with per-stage timing breakdown.
    /// Default implementation: wraps Forward() with e2e timer only.
    virtual Status ForwardWithTiming(const InferRequest& request,
                                      InferResult& result,
                                      StageTimings& timings);

    // ── 元信息（可选，用于引擎层的优化决策） ─────────────
    virtual const char* GetName() const { return "unknown"; }
    virtual bool HasVision() const { return false; }
    virtual bool IsMoE() const { return false; }
    /// 返回 true 表示模型需要 KV Cache 进行增量解码（生成式模型）。
    /// 默认 false（embedding/多模态模型不需要 KV Cache）。
    virtual bool IsGenerative() const { return false; }
};

/// 模型工厂函数类型
using ModelFactory = std::function<std::unique_ptr<IModel>()>;

/// 注册表条目：包含工厂函数、优先级和兼容性检查
struct RegistryEntry {
    std::string model_type;        // 注册名称
    ModelFactory factory;          // 工厂函数
    int32_t priority = 0;          // 优先级（越大越优先）
    using CompatibilityCheck = std::function<bool(const std::string& model_type, const JsonConfig& cfg)>;
    CompatibilityCheck compat_check = nullptr;  // 兼容性检查函数
};

/// 注册模型工厂（旧接口，保持向后兼容）
void RegisterModelFactory(const std::string& model_type, ModelFactory factory);

/// 注册 RegistryEntry
void RegisterModelEntry(RegistryEntry entry);

/// 自动检测模型类型并创建实例
std::unique_ptr<IModel> CreateModel(const std::string& model_dir);

/// 注册模型工厂宏（旧接口，保持向后兼容）
#define REGISTER_MODEL(type_name, factory_fn)                          \
    static bool _reg_##type_name = []() {                              \
        atb_llm::RegisterModelFactory(#type_name, factory_fn);          \
        return true;                                                   \
    }();

/// 注册模型工厂宏（带兼容性检查）
#define REGISTER_MODEL_WITH_CHECK(type_name, factory_fn, check_fn, prio)  \
    static bool _reg_##type_name = []() {                                 \
        atb_llm::RegistryEntry entry;                                      \
        entry.model_type = #type_name;                                     \
        entry.factory = factory_fn;                                        \
        entry.compat_check = check_fn;                                     \
        entry.priority = prio;                                             \
        atb_llm::RegisterModelEntry(std::move(entry));                     \
        return true;                                                       \
    }();

} // namespace atb_llm
