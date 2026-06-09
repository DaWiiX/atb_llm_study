# ATB C++ LLM Engine — 设计文档 v2

## 1. 概述

### 1.1 目标

构建一个围绕 Ascend Transformer Boost (ATB) 的 C++ 多模态大模型推理引擎库。

**已验证可适配的模型族**：
- Qwen 系列：Qwen3、Qwen3VL、Qwen2.5VL、Qwen3.5、Qwen3.6
- MoE 模型：DeepSeek-V2/V3、Mixtral、Qwen-MoE
- 未来模型：任何基于 Transformer decoder/encoder 架构的模型

### 1.2 核心设计原则

```
Engine 只管「怎么跑」—— 内存、Stream、Context、Workspace
Model 决定「跑什么」—— 图结构、权重映射、推理流程、后处理
```

### 1.3 命名

原 `atb_cpp_qwen3vl_embedding` → `atb_cpp_llm_engine`

---

## 2. v1 接口诊断：为什么撑不住

v1 的 `ModelAdapter` 接口有 **4 个致命问题**：

| 问题 | v1 接口 | 影响范围 |
|---|---|---|
| 图构建方法写死架构 | `BuildVisionFirstLayerGraph`、`BuildDeepstackMergerGraph` | MoE、非 ViT 视觉编码器全部不适配 |
| Engine 硬编码流水线 | Engine 内部假设 Vision→注入→Text→Norm 流程 | 纯文本模型、cross-attention 融合模型不兼容 |
| 权重访问太具体 | `GetTextLayerWeights(layer_idx)` 假设每层结构相同 | MoE 模型每层权重数量不同 |
| 缺少关键算子 | 无 MoE routing、MLA、cross-attention | DeepSeek 系列完全无法适配 |

**根本原因**：v1 把 ModelAdapter 当成了"配置适配器"，但实际上应该是"推理程序"。

---

## 3. 修订后的分层架构

```
┌────────────────────────────────────────────────────────────┐
│                      Public API                             │
│  engine.h: Create / Forward / Encode / Destroy              │
│  types.h: InferRequest / InferResult / EngineConfig         │
│  model.h: IModel 抽象接口（唯一的模型扩展点）               │
├────────────────────────────────────────────────────────────┤
│                     Engine Layer                            │
│  职责：NPU 设备管理、Context/Stream 生命周期、              │
│  Workspace 池、权重加载工具、Tensor 分配工具                │
│  不涉及任何模型结构知识                                      │
├────────────────────────────────────────────────────────────┤
│                    Model Layer                              │
│  每个模型实现 IModel 接口，拥有完整的推理流程控制权          │
│  Qwen3VLModel / Qwen3Model / DeepSeekV2Model / ...         │
├────────────────────────────────────────────────────────────┤
│                   Component Layer                           │
│  可复用的 ATB 图组件（非强制，模型可选择性使用）             │
│  TextAttention / SwiGluMLP / MoERouter / MlaAttention /     │
│  VisionBlock / RmsNormGraph / LinearGraph / ...             │
├────────────────────────────────────────────────────────────┤
│                      Ops Layer                              │
│  ATB 算子的 C++ 封装（1:1 映射 ATB 原生参数结构体）         │
│  LinearOp / RmsNormOp / SelfAttentionOp / RopeOp /          │
│  ElewiseOp / ActivationOp / SplitOp / ...                   │
├────────────────────────────────────────────────────────────┤
│                      Core Layer                             │
│  ContextManager / TensorAllocator / GraphBuilder            │
│  BufferPool / StreamGuard / WorkspaceGuard                  │
├────────────────────────────────────────────────────────────┤
│                       IO Layer                              │
│  SafetensorsReader / JsonConfig / WeightLoader              │
├────────────────────────────────────────────────────────────┤
│                    ATB / ACL Runtime                        │
└────────────────────────────────────────────────────────────┘
```

**与 v1 的根本区别**：Engine 不再了解模型内部结构。模型拥有完整的推理流程控制权。

---

## 4. 核心接口定义

### 4.1 公共类型 (`include/atb_llm/types.h`)

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <acl/acl.h>

namespace atb_llm {

// ── 输入模式 ─────────────────────────────────────────────
enum class InputMode {
    TEXT_ONLY,          // 纯文本推理
    IMAGE_ONLY,         // 纯图片 → 视觉特征
    IMAGE_AND_TEXT,     // 图文混合（原始图片 + 文本）
    PREPROCESSED        // 已预处理（pixel_values + grid_thw + 文本）
};

// ── 原始图片输入 ─────────────────────────────────────────
struct RawImage {
    const uint8_t* data;   // NCHW uint8，像素值 [0, 255]
    int64_t channels;      // 通常 3
    int64_t height;
    int64_t width;
};

// ── 预处理后的图片输入 ───────────────────────────────────
struct PreprocessedImage {
    const void* pixel_values;    // (N, patch_dim) float16/float32
    int64_t num_patches;         // N
    int64_t patch_dim;           // 每个 patch 的维度
    const int64_t* grid_thw;     // (3,) [grid_t, grid_h, grid_w]
    aclDataType dtype;           // ACL_FLOAT16 或 ACL_FLOAT32
};

// ── 文本输入 ─────────────────────────────────────────────
struct TextInput {
    const int64_t* input_ids;    // (B, S) token IDs
    int64_t batch_size;          // B
    int64_t seq_length;          // S
};

// ── 推理请求 ─────────────────────────────────────────────
struct InferRequest {
    InputMode mode;
    TextInput text;
    RawImage raw_image;              // mode ∈ {IMAGE_ONLY, IMAGE_AND_TEXT}
    PreprocessedImage preprocessed;  // mode == PREPROCESSED
};

// ── 推理结果 ─────────────────────────────────────────────
// 所有权：Engine 内部管理内存，用户在下一次 Forward/Encode 调用前数据有效
struct InferResult {
    std::vector<uint8_t> data;   // 输出数据（host 内存，RAII 管理）
    std::vector<int64_t> shape;
    aclDataType dtype;
    /// 获取 typed 指针（方便使用）
    template <typename T> const T* As() const {
        return reinterpret_cast<const T*>(data.data());
    }
};

// ── 引擎配置 ─────────────────────────────────────────────
struct EngineConfig {
    std::string model_dir;   // 包含 config.json + model.safetensors
    int64_t buffer_size;     // ATB buffer size (bytes)，0 = 自动
    int device_id;           // NPU 设备 ID
    bool normalize;          // 输出是否 L2 归一化
};

// ── 状态码 ───────────────────────────────────────────────
enum Status : int32_t {
    OK = 0,
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
```

### 4.2 运行时服务接口 (`include/atb_llm/runtime.h`)

Engine 提供给 Model 的"服务"——Model 通过此接口访问 NPU 资源，无需自己管理：

```cpp
#pragma once
#include "atb_llm/types.h"
#include "atb/atb_infer.h"

namespace atb_llm {

class TensorAllocator;
class WeightLoader;

/// 运行时服务 —— Engine 提供给 Model 的资源访问接口
class IRuntime {
public:
    virtual ~IRuntime() = default;

    // ── ATB 资源 ─────────────────────────────────────────
    virtual atb::Context* GetContext() = 0;
    virtual aclrtStream GetStream() = 0;
    virtual Status Synchronize() = 0;

    // ── 内存管理 ─────────────────────────────────────────
    virtual TensorAllocator* GetAllocator() = 0;
    virtual uint8_t* GetWorkspace(uint64_t required_size) = 0;

    // ── 权重加载 ─────────────────────────────────────────
    virtual WeightLoader* GetWeightLoader() = 0;

    // ── 图构建辅助 ───────────────────────────────────────
    /// 创建 GraphOpBuilder（RAII，析构时自动调用 DestroyGraphOpBuilder）
    virtual GraphOpBuilderPtr CreateGraphBuilder(const std::string& name) = 0;
};

} // namespace atb_llm
```

### 4.3 模型接口 (`include/atb_llm/model.h`)

**唯一的模型扩展点。** 模型拥有完整的推理流程控制权。

```cpp
#pragma once
#include "atb_llm/types.h"
#include "atb_llm/runtime.h"

namespace atb_llm {

/// 模型抽象接口 —— 每个模型实现此接口
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

} // namespace atb_llm
```

**v1 → v2 关键变化**：
- 没有 `Build*Graph` 方法 —— 模型在 `Load()` 内部自行构建
- 没有 `Get*Weights` 方法 —— 模型自己管理权重
- 没有 `ComputePositionIds` —— 模型在 `Forward()` 内部自行处理
- Engine 通过 `IRuntime` 提供资源，不干预模型逻辑

### 4.4 引擎接口 (`include/atb_llm/engine.h`)

Engine 变成了一个极薄的壳：

```cpp
#pragma once
#include "atb_llm/types.h"
#include "atb_llm/model.h"
#include <memory>

namespace atb_llm {

class LLMEngine {
public:
    static Status Create(const EngineConfig& config,
                         std::unique_ptr<LLMEngine>& engine);
    Status Forward(const InferRequest& request, InferResult& result);
    Status Encode(const InferRequest& request, InferResult& result);
    ~LLMEngine();

private:
    LLMEngine();
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace atb_llm
```

Engine::Impl 的实现极简：

```cpp
class LLMEngine::Impl {
    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<IModel> model_;

    Status Init(const EngineConfig& config) {
        // 1. 初始化 ACL + ATB Context + Stream
        runtime_ = CreateRuntime(config.device_id, config.buffer_size);
        // 2. 打开权重文件
        runtime_->GetWeightLoader()->LoadFromFile(
            config.model_dir + "/model.safetensors");
        // 3. 自动检测模型类型并创建
        model_ = CreateModel(config.model_dir);
        // 4. 加载模型（模型自行完成图构建和权重加载）
        return model_->Load(config.model_dir, runtime_.get());
    }

    Status Forward(const InferRequest& req, InferResult& res) {
        return model_->Forward(req, res);  // 完全委托给模型
    }
};
```

### 4.5 Core 层设计

#### 4.5.1 RAII 资源包装器

ATB 资源必须显式释放，否则内存泄漏。所有 ATB 资源在 Core 层通过 RAII 包装器管理：

```cpp
// ── Operation* 包装器 ─────────────────────────────────────
class OperationHandle {
    atb::Operation* op_ = nullptr;
public:
    explicit OperationHandle(atb::Operation* op = nullptr) : op_(op) {}
    ~OperationHandle() { if (op_) atb::DestroyOperation(op_); }
    OperationHandle(OperationHandle&& o) noexcept : op_(std::exchange(o.op_, nullptr)) {}
    OperationHandle& operator=(OperationHandle&& o) noexcept {
        if (this != &o) { if (op_) atb::DestroyOperation(op_); op_ = std::exchange(o.op_, nullptr); }
        return *this;
    }
    atb::Operation* get() const { return op_; }
    atb::Operation* release() { return std::exchange(op_, nullptr); }
    explicit operator bool() const { return op_ != nullptr; }
};

// ── GraphOpBuilder* 包装器 ────────────────────────────────
struct GraphOpBuilderDeleter {
    void operator()(atb::GraphOpBuilder* p) { if (p) atb::DestroyGraphOpBuilder(p); }
};
using GraphOpBuilderPtr = std::unique_ptr<atb::GraphOpBuilder, GraphOpBuilderDeleter>;

// ── Context* 包装器 ──────────────────────────────────────
class ContextHandle {
    atb::Context* ctx_ = nullptr;
public:
    explicit ContextHandle(atb::Context* ctx = nullptr) : ctx_(ctx) {}
    ~ContextHandle() { if (ctx_) atb::DestroyContext(ctx_); }
    atb::Context* get() const { return ctx_; }
};
```

#### 4.5.2 ContextManager

```cpp
class ContextManager {
public:
    ContextManager(int device_id);
    ~ContextManager();
    atb::Context* GetContext();
    aclrtStream GetStream();
    Status Synchronize();
    // RAII: 析构时自动 DestroyContext + aclrtDestroyStream + aclFinalize
};
```

#### 4.5.3 TensorAllocator

```cpp
class TensorAllocator {
public:
    TensorAllocator(atb::Context* ctx, aclrtStream stream);

    /// 分配 NPU 内存并设置 TensorDesc（512 字节对齐）
    Status AllocFloat16(atb::Tensor& tensor, std::vector<int64_t> shape);
    Status AllocFloat32(atb::Tensor& tensor, std::vector<int64_t> shape);
    Status AllocInt64(atb::Tensor& tensor, std::vector<int64_t> shape);

    /// Host ↔ Device 拷贝（内部调用 aclrtMemcpy）
    Status CopyToDevice(atb::Tensor& tensor, const void* host_data, size_t size);
    Status CopyToHost(void* host_data, const atb::Tensor& tensor, size_t size);

    /// 释放 NPU 内存（aclrtFree）
    void Free(atb::Tensor& tensor);
    /// 释放所有已分配的 tensor（模型析构时调用）
    void FreeAll();
};
```

#### 4.5.4 GraphBuilder（RAII 封装）

```cpp
class GraphBuilder {
public:
    explicit GraphBuilder(const std::string& name);
    ~GraphBuilder();

    /// 初始化图算子（对应 ATB GraphOpBuilder::Init）
    Status Init(const std::string& op_name,
                const atb::InferShapeFunc& infer_func,
                const atb::SVector<std::string>& in_names,
                const atb::SVector<std::string>& out_names);

    /// 添加 ATB 操作（自动调用 CreateOperation + AddOperation）
    template <typename OpParam>
    Status AddOp(const OpParam& param,
                 const atb::SVector<std::string>& in_names,
                 const atb::SVector<std::string>& out_names);

    /// 添加 Reshape（改变中间 tensor 的 shape）
    Status Reshape(const std::string& src_name,
                   const atb::ReshapeFunc& func,
                   const std::string& view_name);

    /// 构建图算子（返回 OperationHandle，自动管理生命周期）
    OperationHandle Build();

    /// 执行图算子（完整 Setup → GetWorkspace → Execute 流程）
    Status Execute(atb::Context* ctx, const atb::VariantPack& pack,
                   uint8_t* workspace, uint64_t workspace_size);
};
```

#### 4.5.5 BufferPool（per-runtime，非全局单例）

```cpp
// BufferPool 是每个 IRuntime 实例的成员，非全局单例
// 这样多引擎实例之间互不干扰，线程安全
class BufferPool {
public:
    Status SetBufferSize(int64_t size);
    /// 获取 workspace 指针（内部按需扩容，扩容时持有 mutex）
    uint8_t* GetWorkspace(uint64_t required_size);
    /// 释放 workspace 内存
    void Free();
};
```

#### 4.5.6 ATB 执行模式（关键约定）

所有 ATB Operation 执行必须遵循 Setup → Workspace → Execute 三步：

```cpp
// 标准执行模式（模型 Forward() 内部使用）
Status RunOperation(atb::Operation* op, atb::Context* ctx,
                    const atb::VariantPack& pack) {
    uint64_t ws_size = 0;
    // Step 1: Setup — 计算所需 workspace 大小
    ATB_LLM_CHECK(op->Setup(pack, ws_size, ctx));
    // Step 2: 获取 workspace 内存
    uint8_t* workspace = runtime_->GetWorkspace(ws_size);
    // Step 3: Execute — 执行算子
    ATB_LLM_CHECK(op->Execute(pack, workspace, ws_size, ctx));
    // Step 4: 同步（如需要）
    ATB_LLM_CHECK(runtime_->Synchronize());
    return OK;
}
```

**VariantPack 构建约定**（最容易出错的部分）：

```cpp
// VariantPack 包含输入输出 Tensor 列表，顺序必须与 Graph 定义一致
atb::VariantPack pack;
pack.inTensors = {input_tensor};          // SVector<Tensor>
pack.outTensors = {output_tensor};        // 顺序 = Graph 的 inTensorNum/outTensorNum
// inTensors 中每个 Tensor 的 shape/dtype/format 必须与 Graph 定义匹配
// 否则 Setup 阶段会报 ERROR_INVALID_TENSOR_* 错误
```

### 4.6 IO 层设计

```cpp
// WeightLoader — 从 safetensors 加载权重到 NPU
class WeightLoader {
public:
    Status LoadFromFile(const std::string& path);
    Status GetTensor(const std::string& key, WeightInfo& info);
    std::vector<std::string> GetKeysByPrefix(const std::string& prefix);
    Status CopyToNPU(const std::string& key, atb::Tensor& dst, TensorAllocator& alloc);
};

// JsonConfig — cJSON 的类型安全封装（内部管理 cJSON 生命周期）
class JsonConfig {
public:
    ~JsonConfig();  // 内部调用 cJSON_Delete
    static JsonConfig Load(const std::string& path);
    int GetInt(const std::string& key, int default_val = 0) const;
    float GetFloat(const std::string& key, float default_val = 0.0f) const;
    std::string GetString(const std::string& key, const std::string& default_val = "") const;
    std::vector<int> GetIntArray(const std::string& key) const;
    JsonConfig GetSubConfig(const std::string& key) const;
    bool HasKey(const std::string& key) const;
};
```

---

## 5. 组件层设计（可复用的图组件）

组件层是**可选工具**，不是强制基类。模型可以选择使用，也可以完全自己构建。

### 5.1 组件列表

```
src/components/
├── attention/
│   ├── self_attention_graph.h/cpp    # 标准 SelfAttention (GQA/MHA)
│   ├── mla_attention_graph.h/cpp     # Multi-Latent Attention (DeepSeek)
│   └── cross_attention_graph.h/cpp   # Cross-Attention（预留）
├── mlp/
│   ├── swiglu_mlp_graph.h/cpp        # SwiGLU MLP (Qwen/Llama/...)
│   ├── gelu_mlp_graph.h/cpp          # GELU MLP (Vision encoder)
│   └── moe_mlp_graph.h/cpp           # MoE MLP (DeepSeek/Mixtral)
├── norm/
│   ├── rms_norm_graph.h/cpp          # RMSNorm
│   └── layer_norm_graph.h/cpp        # LayerNorm
├── position/
│   ├── rope_graph.h/cpp              # RoPE 旋转位置编码
│   └── mrope.h/cpp                   # 多维 RoPE（Qwen-VL 系列）
├── vision/
│   ├── patch_embed_graph.h/cpp       # Patch Embedding
│   ├── vision_block_graph.h/cpp      # ViT Block 通用封装
│   └── vision_merger_graph.h/cpp     # Vision Merger MLP
└── fusion/
    ├── token_inject.h/cpp            # Token 注入（Qwen-VL 风格）
    └── deepstack_graph.h/cpp         # Deepstack 融合（Qwen3VL 特有）
```

### 5.2 组件使用方式

模型可以选择使用组件，也可以不用：

```cpp
// 方式 1：使用组件层（推荐，减少重复代码）
Status Qwen3VLModel::Load(const std::string& model_dir, IRuntime* rt) {
    auto builder = rt->CreateGraphBuilder("TextDecoderLayer");
    components::RmsNormGraph input_norm(builder.get(), "input_norm", hidden_size_, eps_);
    components::SelfAttentionGraph attention(builder.get(), "attn", nh_, nkv_, hd_);
    components::RmsNormGraph post_norm(builder.get(), "post_norm", hidden_size_, eps_);
    components::SwiGluMLPGraph mlp(builder.get(), "mlp", hidden_size_, interm_size_);
    // ... 连接 residual 等
    text_layer_op_ = builder->Build();
}

// 方式 2：直接使用 ATB 算子（完全控制）
Status CustomModel::Load(const std::string& model_dir, IRuntime* rt) {
    auto builder = rt->CreateGraphBuilder("MyLayer");
    atb::infer::RmsNormParam norm_param;
    norm_param.layerType = atb::infer::RmsNormParam::RMS_NORM_NORM;
    norm_param.normParam.epsilon = 1e-6;
    builder->AddOperation(norm_param, {"input"}, {"normed"});
    atb::infer::MultiLatentAttentionParam mla_param;  // MLA 算子
    mla_param.headNum = 128;
    // ...
    my_layer_op_ = builder->Build();
}
```

### 5.3 MoE 组件示例

```cpp
// src/components/mlp/moe_mlp_graph.h
class MoEMLPGraph {
public:
    struct Config {
        int hidden_size;
        int intermediate_size;
        int num_experts;
        int top_k;
        int num_shared_experts;    // DeepSeek 的 shared expert
        bool use_expert_parallel;
    };
    MoEMLPGraph(atb::GraphOpBuilder* builder, const std::string& name,
                const Config& cfg);
    // 内部创建：Router(gate) → TopK → Expert MLPs → Weighted Sum
    // + optional SharedExpert → Add
};
```

---

## 6. 模型适配示例

### 6.1 Qwen3VL Embedding

```cpp
class Qwen3VLModel : public IModel {
public:
    Status Load(const std::string& model_dir, IRuntime* rt) override;
    Status Forward(const InferRequest& req, InferResult& res) override;
    const char* GetName() const override { return "qwen3vl_embedding"; }
    bool HasVision() const override { return true; }

private:
    IRuntime* rt_;
    int n_layer_, nh_t_, nkv_t_, hd_t_, hidden_t_, interm_t_;
    int nh_v_, hd_v_, v_depth_;
    std::vector<int> ds_indexes_;

    // 权重（NPU-resident，析构时由 TensorAllocator::FreeAll 释放）
    std::vector<std::vector<atb::Tensor>> t_layer_weights_;
    std::vector<std::vector<atb::Tensor>> v_block_weights_;
    atb::Tensor embed_w_, norm_w_;

    // ATB 图（OperationHandle RAII，析构时自动 DestroyOperation）
    OperationHandle g_t_layer_;
    OperationHandle g_t_norm_;
    OperationHandle g_v_first_;
    OperationHandle g_v_block_;
    OperationHandle g_v_merger_;
    OperationHandle g_v_ds_;

    // 辅助 tensor
    atb::Tensor cached_mask_;
    int cached_seq_len_ = -1;

    Status LoadConfig(const std::string& model_dir);
    Status LoadWeights(const std::string& model_dir);
    Status BuildGraphs();
    Status EnsureTextGraph(int seq_len);
};
```

**Forward() 完整执行流程**（对应 Python engine.py:_run_text + _run_vision）：

```cpp
Status Qwen3VLModel::Forward(const InferRequest& req, InferResult& res) {
    auto* ctx = rt_->GetContext();

    // 1. Text embedding（CPU 查表）
    auto* alloc = rt_->GetAllocator();
    atb::Tensor input_embeds;
    alloc->AllocFloat16(input_embeds, {1, req.text.seq_length, hidden_t_});
    // CPU: for each token, memcpy embed_w_[token_id] → input_embeds
    CpuEmbedLookup(embed_w_, req.text.input_ids, req.text.seq_length, input_embeds);

    // 2. Vision（如有图片输入）
    std::vector<atb::Tensor> ds_feats;
    if (req.mode == InputMode::IMAGE_AND_TEXT || req.mode == InputMode::IMAGE_ONLY) {
        // 预处理 → pixel_values + grid_thw
        // Run vision: FirstLayer → 23x Block → Merger
        // Inject vision tokens into input_embeds at image_token positions
        RunVision(req.preprocessed, input_embeds, ds_feats);
    }

    // 3. 计算位置编码
    atb::Tensor cos_npu, sin_npu;
    ComputeMRoPE(req.text.input_ids, req.text.seq_length, cos_npu, sin_npu);

    // 4. Text decoder layers（28 层循环，split-graph 策略）
    EnsureTextGraph(req.text.seq_length);
    atb::Tensor hidden = input_embeds;

    for (int i = 0; i < n_layer_; i++) {
        // 构建 VariantPack（顺序必须与 Graph 定义一致）
        atb::VariantPack pack;
        pack.inTensors = {hidden, t_layer_weights_[i][0], ..., cos_npu, sin_npu, cached_mask_};
        pack.outTensors = {hidden_out};

        // ATB 标准执行模式：Setup → Workspace → Execute
        uint64_t ws_size = 0;
        ATB_LLM_CHECK(g_t_layer_.get()->Setup(pack, ws_size, ctx));
        uint8_t* ws = rt_->GetWorkspace(ws_size);
        ATB_LLM_CHECK(g_t_layer_.get()->Execute(pack, ws, ws_size, ctx));

        // Deepstack 融合（如有）
        if (!ds_feats.empty() && i < ds_feats.size()) {
            // hidden[vis_mask] += ds_feats[i]  (Elewise ADD)
        }
        hidden = hidden_out;
    }

    // 5. FinalNorm
    atb::VariantPack norm_pack;
    norm_pack.inTensors = {hidden, norm_w_};
    norm_pack.outTensors = {norm_out};
    uint64_t ws_size = 0;
    ATB_LLM_CHECK(g_t_norm_.get()->Setup(norm_pack, ws_size, ctx));
    ATB_LLM_CHECK(g_t_norm_.get()->Execute(norm_pack, rt_->GetWorkspace(ws_size), ws_size, ctx));

    // 6. 池化 + 拷贝到 host
    // Pooling: 取最后一个有效 token 的 hidden state
    // aclrtMemcpy NPU → host (res.data)
    alloc->CopyToHost(res.data.data(), norm_out_pooled, res.data.size());
    return OK;
}
```

### 6.2 DeepSeek-V2 MoE

```cpp
class DeepSeekV2Model : public IModel {
public:
    Status Load(const std::string& model_dir, IRuntime* rt) override;
    Status Forward(const InferRequest& req, InferResult& res) override;
    const char* GetName() const override { return "deepseek_v2"; }
    bool IsMoE() const override { return true; }

private:
    IRuntime* rt_;
    int num_experts_, top_k_, num_shared_experts_, moe_layer_freq_;

    // 不同类型的 decoder layer 图
    atb::Operation* g_dense_layer_ = nullptr;   // Dense layer
    atb::Operation* g_moe_layer_ = nullptr;      // MoE layer
    atb::Operation* g_norm_ = nullptr;
};
```

### 6.3 纯文本 Qwen3

```cpp
class Qwen3Model : public IModel {
public:
    Status Load(const std::string& model_dir, IRuntime* rt) override;
    Status Forward(const InferRequest& req, InferResult& res) override;
    const char* GetName() const override { return "qwen3"; }
    // HasVision() 默认返回 false

private:
    atb::Operation* g_layer_ = nullptr;
    atb::Operation* g_norm_ = nullptr;
};
```

---

## 7. 模型自动检测与注册

### 7.1 检测逻辑

```cpp
std::unique_ptr<IModel> CreateModel(const std::string& model_dir) {
    auto cfg = JsonConfig::Load(model_dir + "/config.json");
    std::string model_type = cfg.GetString("model_type", "");

    auto it = g_registry.find(model_type);
    if (it != g_registry.end()) return it->second();

    // 前缀匹配兜底
    if (model_type.find("qwen3") != std::string::npos) {
        if (cfg.HasKey("vision_config"))
            return std::make_unique<Qwen3VLModel>();
        return std::make_unique<Qwen3Model>();
    }
    if (model_type.find("deepseek") != std::string::npos)
        return std::make_unique<DeepSeekV2Model>();

    return nullptr;
}
```

### 7.2 注册宏

```cpp
#define REGISTER_MODEL(type_name, factory_fn)                          \
    static bool _reg_##type_name = []() {                              \
        RegisterModelFactory(#type_name, factory_fn);                  \
        return true;                                                   \
    }();

// 使用：
REGISTER_MODEL(qwen3vl_embedding, []() {
    return std::make_unique<Qwen3VLModel>();
})
```

---

## 8. 目录结构（修订版）

```
atb_cpp_llm_engine/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── design.md                     # 本文档
│   └── model_adaptation_guide.md     # 新模型适配指南
│
├── include/atb_llm/                  # 公共头文件（4 个）
│   ├── engine.h                      # LLMEngine 对外接口
│   ├── model.h                       # IModel 接口 + 注册宏
│   ├── runtime.h                     # IRuntime 接口
│   └── types.h                       # 公共类型定义
│
├── src/
│   ├── core/                         # ATB 资源管理
│   │   ├── context_manager.h/cpp
│   │   ├── tensor_allocator.h/cpp
│   │   ├── graph_builder.h/cpp
│   │   ├── buffer_pool.h/cpp
│   │   └── stream_guard.h/cpp
│   │
│   ├── io/                           # 权重 + 配置
│   │   ├── weight_loader.h/cpp
│   │   ├── json_config.h/cpp
│   │   └── safetensors_reader.h/cpp
│   │
│   ├── log/
│   │   └── logger.h/cpp
│   │
│   ├── ops/                          # ATB 算子封装（1:1）
│   │   ├── linear_op.h/cpp
│   │   ├── linear_parallel_op.h/cpp  # Linear + AllReduce 融合
│   │   ├── rms_norm_op.h/cpp
│   │   ├── layer_norm_op.h/cpp
│   │   ├── self_attention_op.h/cpp
│   │   ├── paged_attention_op.h/cpp
│   │   ├── mla_attention_op.h/cpp
│   │   ├── mla_preprocess_op.h/cpp   # 融合 rmsNorm+quant+matmul+rope+cache
│   │   ├── rope_op.h/cpp
│   │   ├── elewise_op.h/cpp
│   │   ├── activation_op.h/cpp
│   │   ├── split_op.h/cpp
│   │   ├── concat_op.h/cpp
│   │   ├── transpose_op.h/cpp
│   │   ├── softmax_op.h/cpp
│   │   ├── reduce_op.h/cpp
│   │   ├── gather_op.h/cpp
│   │   ├── set_value_op.h/cpp
│   │   ├── reshape_and_cache_op.h/cpp # KV cache 写入
│   │   ├── gating_op.h/cpp           # MoE token↔expert 映射
│   │   ├── group_topk_op.h/cpp       # MoE 分组 top-k
│   │   ├── grouped_matmul_op.h/cpp   # MoE 融合 expert matmul
│   │   └── transdata_op.h/cpp        # ND↔NZ 格式转换
│   │
│   ├── components/                   # 可复用图组件
│   │   ├── attention/
│   │   │   ├── self_attention_graph.h/cpp
│   │   │   ├── mla_attention_graph.h/cpp
│   │   │   └── cross_attention_graph.h/cpp
│   │   ├── mlp/
│   │   │   ├── swiglu_mlp_graph.h/cpp
│   │   │   ├── gelu_mlp_graph.h/cpp
│   │   │   └── moe_mlp_graph.h/cpp
│   │   ├── norm/
│   │   │   ├── rms_norm_graph.h/cpp
│   │   │   └── layer_norm_graph.h/cpp
│   │   ├── position/
│   │   │   ├── rope_graph.h/cpp
│   │   │   └── mrope.h/cpp
│   │   ├── vision/
│   │   │   ├── patch_embed_graph.h/cpp
│   │   │   ├── vision_block_graph.h/cpp
│   │   │   └── vision_merger_graph.h/cpp
│   │   └── fusion/
│   │       ├── token_inject.h/cpp
│   │       └── deepstack_graph.h/cpp
│   │
│   ├── preprocess/
│   │   └── image_processor.h/cpp
│   │
│   └── engine/
│       ├── llm_engine.h/cpp
│       ├── runtime_impl.h/cpp        # IRuntime 实现
│       └── model_registry.h/cpp
│
├── adapters/
│   ├── qwen3vl_embedding/
│   │   ├── qwen3vl_model.h/cpp
│   │   ├── qwen3vl_weights.h/cpp
│   │   ├── qwen3vl_config.h/cpp
│   │   ├── qwen3vl_preprocess.h/cpp
│   │   └── register.cpp
│   ├── qwen3/
│   │   ├── qwen3_model.h/cpp
│   │   └── register.cpp
│   └── deepseek_v2/                  # 预留
│       ├── deepseek_model.h/cpp
│       └── register.cpp
│
├── tests/
│   ├── unit/
│   │   ├── test_ops/
│   │   ├── test_components/
│   │   └── test_engine/
│   └── benchmark/
│       └── benchmark.cpp
│
└── utils/
    ├── safetensors.hh
    ├── cJSON.h / cJSON.c
    └── plog_simple.h
```

---

## 9. 错误处理

```cpp
// 错误码定义在 types.h 中（见 4.1）

// ATB 错误检查宏
#define ATB_LLM_CHECK(expr)                                             \
    do {                                                                \
        atb::Status _s = (expr);                                       \
        if (_s != atb::NO_ERROR) {                                     \
            LOG_ERROR("%s:%d ATB error: %d", __FILE__, __LINE__, _s);  \
            return atb_llm::ERROR_ATB_BASE + _s;                       \
        }                                                              \
    } while (0)

#define ATB_LLM_CHECK_ACL(expr)                                        \
    do {                                                                \
        auto _s = (expr);                                              \
        if (_s != ACL_SUCCESS) {                                       \
            LOG_ERROR("%s:%d ACL error: %d", __FILE__, __LINE__, _s);  \
            return atb_llm::ERROR_NPU_MEMORY;                          \
        }                                                              \
    } while (0)
```

---

## 10. 构建系统

```cmake
cmake_minimum_required(VERSION 3.16)
project(atb_cpp_llm_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(CANN REQUIRED)

add_library(thirdparty STATIC utils/cJSON.c)
target_include_directories(thirdparty PUBLIC utils)

file(GLOB_RECURSE SRCS src/*.cpp)
add_library(atb_llm_engine SHARED ${SRCS})
target_include_directories(atb_llm_engine PUBLIC include)
target_link_libraries(atb_llm_engine PRIVATE thirdparty ascendcl atb)

file(GLOB_RECURSE ADAPTER_SRCS adapters/*.cpp)
add_library(atb_llm_adapters SHARED ${ADAPTER_SRCS})
target_link_libraries(atb_llm_adapters PRIVATE atb_llm_engine)

enable_testing()
add_subdirectory(tests)
```

编译命令：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

---

## 11. 使用示例

```cpp
#include "atb_llm/engine.h"

int main() {
    atb_llm::EngineConfig config;
    config.model_dir = "/path/to/Qwen3-VL-Embedding-2B";
    config.buffer_size = 5LL * 1024 * 1024 * 1024;
    config.device_id = 0;
    config.normalize = true;

    std::unique_ptr<atb_llm::LLMEngine> engine;
    atb_llm::LLMEngine::Create(config, engine);

    // 纯文本
    atb_llm::InferRequest req;
    req.mode = atb_llm::InputMode::TEXT_ONLY;
    req.text = {input_ids.data(), 1, seq_len};
    atb_llm::InferResult result;
    engine->Encode(req, result);

    // 图文混合（原始图片）
    req.mode = atb_llm::InputMode::IMAGE_AND_TEXT;
    req.raw_image = {img_data.data(), 3, 480, 640};
    engine->Encode(req, result);

    // 图文混合（已预处理）
    req.mode = atb_llm::InputMode::PREPROCESSED;
    req.preprocessed = {pv_data, num_patches, patch_dim, grid_data, ACL_FLOAT16};
    engine->Encode(req, result);
}
```

---

## 12. 新模型适配指南

### 步骤

1. **创建目录** `adapters/your_model/`
2. **实现 `IModel`**：
   - `Load()` — 解析配置 + 加载权重 + 构建 ATB 图
   - `Forward()` — 定义完整的推理流程
3. **注册** — 用 `REGISTER_MODEL` 宏
4. **测试** — 参照 `tests/` 验证

### 什么时候用组件层

| 场景 | 建议 |
|---|---|
| 标准 GQA/MHA Attention | 用 `SelfAttentionGraph` |
| SwiGLU MLP | 用 `SwiGluMLPGraph` |
| MoE MLP | 用 `MoEMLPGraph` |
| MLA Attention | 用 `MlaAttentionGraph` |
| 非标准结构 | 直接用 Ops 层自建 |
| 完全自定义 | 直接用 ATB `GraphOpBuilder` |

### ATB 算子速查

| 用途 | ATB 参数结构体 | 备注 |
|---|---|---|
| 线性层 | `atb::infer::LinearParam` | 所有 Linear |
| RMSNorm | `atb::infer::RmsNormParam` | |
| LayerNorm | `atb::infer::LayerNormParam` | |
| SelfAttention | `atb::infer::SelfAttentionParam` | GQA/MHA，encoder/decoder |
| PagedAttention | `atb::infer::PagedAttentionParam` | 生成模型 KV cache |
| MLA | `atb::infer::MultiLatentAttentionParam` | DeepSeek MLA |
| MLA 预处理 | `atb::infer::MlaPreprocessParam` | 融合 rmsNorm+quant+matmul+rope+cache |
| RoPE | `atb::infer::RopeParam` | |
| 逐元素运算 | `atb::infer::ElewiseParam` | add/mul/cast/quant 等 |
| 激活函数 | `atb::infer::ActivationParam` | SiLU/GELU/FAST_GELU |
| Split | `atb::infer::SplitParam` | |
| Concat | `atb::infer::ConcatParam` | |
| Transpose | `atb::infer::TransposeParam` | |
| Softmax | `atb::infer::SoftmaxParam` | |
| Reduce | `atb::infer::ReduceParam` | |
| Gather | `atb::infer::GatherParam` | |
| SetValue | `atb::infer::SetValueParam` | 原地写入（token 注入） |
| MoE Gating | `atb::infer::GatingParam` | token↔expert 映射，TP/EP |
| MoE GroupTopk | `atb::infer::GroupTopkParam` | DeepSeek 分组 top-k |
| MoE 路由融合 | `atb::infer::FusedAddTopkDivParam` | DeepSeek sigmoid+add+topk |
| MoE 分组MatMul | `atb::infer::GroupedMatmulWithRoutingParam` | 融合 expert matmul + routing |
| MoE 融合计算 | `atb::infer::GmmDeqSwigluQuantGmmDeqParam` | GMM+dequant+swiglu+quant+GMM |
| KV Cache | `atb::infer::ReshapeAndCacheParam` | 写入 KV cache |
| 线性+通信 | `atb::infer::LinearParallelParam` | TP 融合 linear+AllReduce |
| AllReduce | `atb::infer::AllReduceParam` | 多卡通信 |
| AllGather | `atb::infer::AllGatherParam` | 多卡通信 |
| ReduceScatter | `atb::infer::ReduceScatterParam` | 多卡通信 |
| 格式转换 | `atb::infer::TransdataParam` | ND↔NZ 格式转换 |

---

## 13. v1 vs v2 对比

| 维度 | v1 | v2 |
|---|---|---|
| 扩展点 | `ModelAdapter` 接口（~20 个虚函数） | `IModel` 接口（3 个核心方法） |
| 推理流程 | Engine 硬编码 | Model 自主控制 |
| 图构建 | Engine 调用 adapter 的 Build* 方法 | Model 在 Load() 内自行构建 |
| 权重管理 | Engine 通过 Get*Weights 访问 | Model 自行管理 |
| 组件复用 | 无（各模型重复实现） | 组件层可选使用 |
| MoE 支持 | 不支持 | MoE 组件 + 模型自定义 |
| MLA 支持 | 不支持 | MLA 算子封装 |
| 新模型成本 | 实现 ~20 个虚函数 | 实现 3 个方法 + 按需使用组件 |
| Engine 复杂度 | 高（了解模型结构） | 低（只管运行时） |

---

## 14. 性能优化要点

1. **权重预加载**：所有权重在 Load() 阶段一次性拷贝到 NPU，推理时零拷贝
2. **Graph 缓存**：相同 shape 的 Graph 只构建一次，后续复用
3. **Workspace 池化**：避免每次推理反复 malloc/free workspace
4. **整图下发**：使用 `GRAPH_LAUNCH_MODE` 减少算子调度开销
5. **Stream 并行**：视觉和文本计算可使用不同 stream 并行（未来扩展）

---

## 15. 实施计划

> 关键原则：Phase 1 必须产出一个可运行的端到端流程（哪怕只跑 RMSNorm），尽早暴露 API 理解错误。

### Phase 1: 基础框架 + 最小 E2E
- [ ] 目录结构 + CMakeLists.txt
- [ ] RAII 包装器：OperationHandle, GraphOpBuilderPtr, ContextHandle
- [ ] Core 层：ContextManager, TensorAllocator, BufferPool(per-runtime)
- [ ] IO 层：WeightLoader(safetensors), JsonConfig(cJSON RAII), Logger
- [ ] types.h, runtime.h, model.h 接口定义
- [ ] **里程碑**：一个只包含 RMSNorm 的最简模型，端到端跑通 Load → Forward → 输出

### Phase 2: 基础 Ops + Text Decoder Layer
- [ ] Ops：LinearOp, RmsNormOp, ElewiseOp, ActivationOp, SplitOp, ConcatOp, RopeOp
- [ ] Ops：SelfAttentionOp, TransposeOp, SoftmaxOp, ReduceOp
- [ ] 组件：SelfAttentionGraph, SwiGluMLPGraph, RmsNormGraph
- [ ] **里程碑**：完整的 Text Decoder Layer 图构建 + 执行，与 Python 参考实现精度对比

### Phase 3: Vision + Qwen3VL 完整路径
- [ ] Ops：LayerNormOp, TransdataOp, SetValueOp, GatherOp
- [ ] 组件：VisionBlockGraph, PatchEmbedGraph, VisionMergerGraph, DeepstackGraph
- [ ] **里程碑**：Qwen3VL 完整 encoder 路径（vision + text），E2E 精度验证

### Phase 4: 引擎层 + 适配器
- [ ] LLMEngine 实现（PImpl）
- [ ] RuntimeImpl 实现（IRuntime）
- [ ] ModelRegistry + REGISTER_MODEL 宏
- [ ] Qwen3VLModel 适配器完整实现
- [ ] 性能 Benchmark

### Phase 5: 扩展模型（持续）
- [ ] MoE Ops：GatingOp, GroupTopkOp, GroupedMatmulOp, FusedAddTopkDivOp
- [ ] MLA Ops：MlaAttentionOp, MlaPreprocessOp, ReshapeAndCacheOp
- [ ] 组件：MoEMLPGraph, MlaAttentionGraph
- [ ] Qwen3 纯文本适配器
- [ ] DeepSeek-V2 MoE 适配器
- [ ] 多设备支持：LinearParallelOp, AllReduceOp, AllGatherOp

---

## 16. 已知限制与未来规划

当前设计针对 **embedding 模型（encoder-only）** 优化，以下能力暂不在首期范围内，但架构已预留扩展空间：

| 能力 | 当前状态 | 扩展方案 |
|---|---|---|
| KV Cache / 自回归解码 | 不支持 | 新增 `KVCacheManager` 类 + `PagedAttentionOp`，模型在 `Forward()` 中按 prefill/decode 模式切换 |
| 多设备并行 (TP/EP/PP) | 不支持 | `EngineConfig` 扩展 `ParallelConfig`，新增 `LinearParallelOp` + `AllReduceOp` 等通信算子 |
| 动态 batch | 不支持 | `InferRequest` 扩展为支持 batch > 1，模型需处理 padding/mask |
| 量化推理 | 不支持 | ATB 已支持 INT8 量化（`ElewiseParam::ELEWISE_QUANT`），需新增量化权重加载路径 |
| 在线推理服务 | 不支持 | 在 `LLMEngine` 上层封装 request queue + response callback |
| Tensor 格式自动转换 | 手动 | 新增 `FormatGuard` 自动检测 ATB op 所需格式并插入 `TransdataOp` |

**ATB tensor 格式注意事项**：
- `atb::TensorDesc::format` 字段控制内存布局（`ACL_FORMAT_ND` vs `ACL_FORMAT_FRACTAL_NZ`）
- 不同 ATB 算子可能要求不同格式，Python 版由 torch_atb 自动处理，C++ 版需显式管理
- 常见模式：ND 用于通用计算，NZ 用于矩阵乘优化
- 使用 `TransdataParam` 进行格式转换，注意转换前后 shape 会变化
