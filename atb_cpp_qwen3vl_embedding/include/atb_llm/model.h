#pragma once
#include "atb_llm/types.h"
#include "atb_llm/runtime.h"
#include <functional>
#include <memory>
#include <string>

namespace atb_llm {

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

    // ── 元信息（可选，用于引擎层的优化决策） ─────────────
    virtual const char* GetName() const { return "unknown"; }
    virtual bool HasVision() const { return false; }
    virtual bool IsMoE() const { return false; }
};

/// 模型工厂函数类型
using ModelFactory = std::function<std::unique_ptr<IModel>()>;

/// 注册模型工厂
void RegisterModelFactory(const std::string& model_type, ModelFactory factory);

/// 自动检测模型类型并创建实例
std::unique_ptr<IModel> CreateModel(const std::string& model_dir);

/// 注册模型工厂宏
#define REGISTER_MODEL(type_name, factory_fn)                          \
    static bool _reg_##type_name = []() {                              \
        RegisterModelFactory(#type_name, factory_fn);                  \
        return true;                                                   \
    }();

} // namespace atb_llm
