# C++ 框架重构计划

## 目标
使 ATB C++ 推理引擎能方便地适配更多模型（Qwen3、Qwen2VL、DeepSeek-V2/V3、GLM-4V、Mixtral 等），
解决当前代码中 components/layers 断裂、适配器过重、缺乏多模型参数化等问题。

## 审查修正要点（来自 subagent 审查）
1. **components/ 扁平化改为 text/vision/common 三目录**（非完全扁平） ✅
2. **参数化组件改为策略模式 + 独立 Builder**（非统一 switch-case） ⚠️ 部分完成
3. **放弃 4 层继承，改用 1 层基类 + 组合**（BaseModel : IModel，适配器直接组合 Runner） ⚠️ 基类完成，组合未完成
4. **Runner 只管图生命周期，不持有执行逻辑** ✅
5. **InferRequest 用 void* metadata 替代 dynamic_cast 多态** ✅
6. **适配器瘦身目标 790→300-400 行**（非 50 行） ❌ 未达标（790→775）

---

## 审查目标达成度评估

### ✅ 已达标（5/8 修正要点 + 3/8 遗漏项）

| 审查要点 | 状态 | 说明 |
|----------|------|------|
| 1. text/vision/common 三目录 | ✅ 完成 | Phase 1 |
| 2. 策略模式 if-else 分发 | ⚠️ 可接受 | 当前只有 GQA 实现，if-else 足够；**MLA 必须拆 Builder** |
| 4. Runner 只管图生命周期 | ✅ 完成 | Phase 2 |
| 5. void* metadata | ✅ 完成 | Phase 4 |
| 7c. buffer_size 上提 Runtime | ✅ 完成 | Phase 8 |
| 7d. WeightHelpers 复用 | ✅ 完成 | Phase 7 |
| 7e. ICrossModalFusion 接口 | ✅ 完成 | Phase 6 |
| 7f. Registry 增强 | ✅ 完成 | Phase 4 |

### ❌ 未达标（3 项关键差距）

| 差距 | 当前状态 | 目标 | 影响 |
|------|----------|------|------|
| **3. 适配器未组合 Runner** | Qwen3VLModel 直接持有 OperationHandle | 适配器组合 unique_ptr\<TextRunner\> + unique_ptr\<VisionRunner\> | 新模型无法复用 Runner；图构建逻辑仍散落在适配器 |
| **6. 适配器行数 790→775** | 几乎没瘦 | 300-400 行 | RunVision(180行)、ComputePosEmbedInterp(100行)、RunTextDecoder(70行) 仍在适配器中 |
| **遗漏项未处理** | 无 | 见下方 Phase 9-13 | 影响长期可维护性 |

---

## 验证流程（每个 Phase 必须全部通过）
1. **C++ 编译**：0 error
2. **C++ 单元测试**：test_core, test_io_adapters, test_text_ops, test_vision_ops, test_text_model 全部 SUCCESS
3. **C++ vs Python 精度对比**（核心验证：用相同输入，对比 C++ 和 Python 的推理输出）：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
   source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1 2>/dev/null; source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 2>/dev/null
   export ATB_BUILD_DEPENDENCY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/developer/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
   cd /mnt/workspace/gitCode/atb_llm/atb_cpp_qwen3vl_embedding/build
   # Step A: C++ 生成推理结果到 /tmp/cpp_embedding.bin
   ./test_consistency
   # Step B: Python 读 C++ 结果 + 跑 Python engine → 比较余弦相似度
   cd /mnt/workspace/gitCode/atb_llm
   python atb_cpp_qwen3vl_embedding/tests/test_consistency.py
   # 期望: cosine similarity > 0.99
   ```
4. **C++ vs Python 多模式精度对比**（text-only / image-only / image+text 三种模式）：
   ```bash
   cd /mnt/workspace/gitCode/atb_llm/atb_cpp_qwen3vl_embedding/build
   ./test_accuracy
   cd /mnt/workspace/gitCode/atb_llm
   python atb_cpp_qwen3vl_embedding/tests/test_accuracy.py
   ```
5. **提交 git，push 到远端**

---

## TODO 列表

### Phase 0: 修复 components/layers 断裂 ✅ DONE
- [x] 给 SelfAttentionGraph 增加 use_qk_norm 和 rotary_dim 参数
- [x] TextDecoderLayerGraph 改为调用 SelfAttentionGraph::Build() + SwiGluMlpGraph::Build() + RmsNormGraph::Build()
- [x] C++ 单元测试通过
- Commit: 3d3a9a8

### Phase 1: 目录整理 ✅ DONE
- [x] layers/ → components/text/; attention/norm/mlp/position/ → components/common/; vision/ 保留
- [x] 更新 CMakeLists.txt, #include, 命名空间
- [x] C++ 单元测试通过
- Commit: 868e899

### Phase 2: Runner 层重构 ✅ DONE
- [x] models/ → runners/; TextModel→TextRunner, VisionModel→VisionRunner
- [x] EnsureBuilt(seq_len) + seq_len 缓存; Config 含 use_qk_norm/rotary_dim/use_mask
- [x] Runner 只管图生命周期
- [x] C++ 单元测试通过
- Commit: 0d998a0

### Phase 3: 策略模式引入 + LayerDescriptor 参数化 ✅ DONE
- [x] include/atb_llm/layer_desc.h: AttnConfig/MlpConfig/NormConfig + 枚举
- [x] SelfAttentionGraph/SwiGluMlpGraph/RmsNormGraph: 新增 Config-based Build 重载
- [x] TextRunner::Config 含 LayerDescriptor layer_desc
- [x] C++ 单元测试通过
- [x] C++ vs Python 精度对比: cosine=0.999996 ✅
- Commit: 726fa16

### Phase 4: 接口精简 + Registry 增强 ✅ DONE
- [x] PreprocessedImage 增加 metadata/metadata_size 字段，grid_thw 标注 DEPRECATED
- [x] Qwen3VL PrepareInputs 支持 metadata fallback 读取 grid_thw
- [x] ModelRegistry: map→vector<RegistryEntry>，加 CompatibilityCheck + priority
- [x] 注册宏增加 REGISTER_MODEL_WITH_CHECK 变体
- [x] CreateModel() 匹配策略: 精确匹配 → 兼容性检查（删除硬编码前缀逻辑）
- [x] Qwen3VL 注册改用 REGISTER_MODEL_WITH_CHECK + IsQwen3VLCompatible
- [x] C++ 单元测试通过
- [x] C++ vs Python 精度对比: cosine=0.999996 ✅
- Commit: 8730f33

### Phase 5: BaseModel 基类引入 ✅ DONE
- [x] 新建 src/families/base_model.h/cpp
- [x] BaseModel : IModel，提供 ExecuteGraph()、EmbeddingLookup()、RunPooling()、FindImageTokenPositions
- [x] Qwen3VLModel 改为继承 BaseModel，删除重复的工具方法代码
- [x] C++ 单元测试通过
- [x] C++ vs Python 精度对比: cosine=0.999996 ✅
- Commit: 357d7c7

### Phase 6: ICrossModalFusion 接口提取 ✅ DONE
- [x] 新建 src/components/common/cross_modal_fusion.h（接口）
- [x] 新建 src/components/common/deepstack_fusion.h/cpp（Deepstack 实现）
- [x] 将 InjectDeepstack 逻辑移入 DeepstackFusion::InjectFeatures
- [x] Qwen3VLModel 通过 unique_ptr<DeepstackFusion> 持有
- [x] C++ 单元测试通过
- [x] C++ vs Python 精度对比: cosine=0.999996 ✅
- Commit: 4eeebb4

### Phase 7: WeightHelpers 复用函数 ✅ DONE
- [x] 新建 src/io/weight_helpers.h/cpp
- [x] 提供 CopyWeightToFp16NPU、CopyWeightToFp16Host、LoadLinearWeights 复用函数
- [x] Qwen3VLWeights 中的 static helper 改为调用 io:: 命名空间函数
- [x] C++ 单元测试通过
- [x] C++ vs Python 精度对比: cosine=0.999996 ✅
- Commit: dc97c30

### Phase 8: buffer_size 上提 Runtime ✅ DONE
- [x] IRuntime 接口添加 SetBufferSize(uint64_t) 虚方法
- [x] RuntimeImpl 实现 SetBufferSize，委托给 BufferPool::SetBufferSize
- [x] 验证 BufferPool::SetBufferSize 接口存在且可用
- [x] C++ 单元测试通过
- [x] C++ vs Python 精度对比: cosine=0.999996 ✅
- Commit: fdf2ac0

---

### Phase 9: 适配器组合 Runner 🔲 TODO
**审查修正 #3 的关键差距：适配器应通过组合持有 Runner，而非直接持有 OperationHandle**

当前 Qwen3VLModel 直接持有 `OperationHandle vis_first_layer_graph_` 等 5 个图句柄，
并自行调用 `TextDecoderLayerGraph::Build()` 和 `RmsNormGraph::Build()`。
审查要求：适配器通过 `unique_ptr<TextRunner>` 和 `unique_ptr<VisionRunner>` 组合 Runner。

- [ ] Qwen3VLModel 改为持有 `unique_ptr<TextRunner>` + `unique_ptr<VisionRunner>`
- [ ] 将 `vis_first_layer_graph_`, `vis_block_graph_`, `vis_merger_graph_` 移入 VisionRunner
- [ ] 将 `text_decoder_graph_`, `text_norm_graph_`, `cached_text_seq_len_` 移入 TextRunner
- [ ] Qwen3VLModel 通过 Runner 的 getter 获取图句柄来编排执行
- [ ] C++ 单元测试通过
- [ ] C++ vs Python 精度对比通过 (cosine > 0.99)

### Phase 10: 适配器瘦身 🔲 TODO
**审查修正 #6 的关键差距：适配器 790→775 行，目标 300-400 行**

当前 RunVision(180行)、ComputePosEmbedInterp(100行)、RunTextDecoder(70行)、PrepareInputs(70行)
仍全在适配器中。审查指出约 400 行是 Qwen3VL 独有逻辑，无法被基类吸收，
但约 200-300 行的 Load/BuildGraphs/Forward 骨架可由 Runner 封装。

- [ ] Phase 9 完成后评估：Runner 组合是否已自然减少适配器代码
- [ ] RunVision 中 deepstack 特征提取已移至 DeepstackFusion，评估进一步瘦身空间
- [ ] ComputePosEmbedInterp 是否可提取到 VisionRunner 或独立组件
- [ ] 目标：qwen3vl_model.cpp + .h ≤ 500 行
- [ ] C++ 单元测试通过
- [ ] C++ vs Python 精度对比通过 (cosine > 0.99)

### Phase 11: 独立 Builder 拆分（MLA/MoE 准备）🔲 TODO
**审查修正 #2 的延伸：当添加 MLA/MoE 时，if-else 分发需升级为独立 Builder**

当前 SelfAttentionGraph::Build(config) 内部用 if-else 按 AttnType 分发，GQA 走现有代码，
MHA/MLA 返回 ERROR_UNSUPPORTED。审查指出 MLA 和 MoE 的输入签名差异大
（MLA 需 kv_b_proj、compress_k/v，MoE 需 router + expert 权重），
必须拆为独立 Builder 才能维护。

- [ ] 新建 IAttentionBuilder 接口
- [ ] GqaAttentionBuilder（从现有 SelfAttentionGraph Build 逻辑提取）
- [ ] MhaAttentionBuilder（stub，返回 ERROR_UNSUPPORTED）
- [ ] MlaAttentionBuilder（stub，返回 ERROR_UNSUPPORTED）
- [ ] SelfAttentionGraph::Build(config) 改为工厂分发
- [ ] 同理：ISwiGluBuilder 接口 + MoE Builder stub
- [ ] C++ 单元测试通过
- [ ] C++ vs Python 精度对比通过 (cosine > 0.99)

### Phase 12: VariantPack 输入命名合约 🔲 TODO
**审查遗漏项：VariantPack 组装顺序是隐式合约，Debug 模式下应提供校验**

- [ ] 组件 Build() 返回 std::vector\<std::string\> input_names / output_names
- [ ] Debug 模式下 ExecuteGraph() 按名称匹配校验 VariantPack 大小
- [ ] Release 模式无额外开销
- [ ] C++ 单元测试通过

### Phase 13: InjectDeepstack NPU scatter-add 优化 🔲 TODO
**审查遗漏项：当前 InjectDeepstack 做 NPU→Host→加法→NPU 往返，可用 ATB 算子替代**

- [ ] 用 ATB ElewiseAdd + SetValue/Gather 实现 NPU 端 scatter-add
- [ ] 消除 NPU→Host→NPU 往返拷贝
- [ ] 性能基准测试验证加速
- [ ] C++ 单元测试通过
- [ ] C++ vs Python 精度对比通过 (cosine > 0.99)

### Phase 14: KV Cache 接口预留 🔲 TODO
**审查遗漏项：若目标包含生成式模型，KV Cache 管理必须纳入架构设计**

- [ ] IRuntime 添加 KV Cache 管理接口（纯虚，预留）
- [ ] KVCacheManager stub 类（struct 定义，无实现）
- [ ] IModel 添加 IsGenerative() 虚方法
- [ ] C++ 编译通过

### Phase 15: 批处理接口准备 🔲 TODO
**审查遗漏项：batch_size > 1 是多模型适配的必要条件，当前硬编码为 1**

- [ ] InferRequest::TextInput 支持批量 input_ids
- [ ] 组件 Build() 接受 batch_size 参数（当前默认 1）
- [ ] MakeCausalMask 支持 batch 维度
- [ ] C++ 编译通过，batch=1 测试不退化

---

## 环境变量（运行 Python 测试需要）
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1 2>/dev/null
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 2>/dev/null
export ATB_BUILD_DEPENDENCY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/developer/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
```

## 模型权重路径
- 实际位置: /mnt/workspace/gitCode/models/model.safetensors
- C++ 测试期望: /mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B/model.safetensors
- 修复: `ln -sf /mnt/workspace/gitCode/models/model.safetensors /mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B/model.safetensors`

## 当前目录结构（Phase 8 完成后）
```
include/atb_llm/     → engine.h, model.h(含RegistryEntry+宏), runtime.h(含SetBufferSize), types.h(含metadata), layer_desc.h
src/
├── core/            → RAII, NpuTensor, allocator, context, buffer_pool, graph_builder
├── io/              → safetensors_reader, weight_loader, json_config, weight_helpers(新)
├── ops/             → 14 个 ATB 算子包装
├── components/
│   ├── common/      → self_attention_graph, swiglu_mlp_graph, rms_norm_graph, mrope, rope_1d,
│   │                  cross_modal_fusion(新), deepstack_fusion(新), moe_mlp_graph(stub)
│   ├── text/        → decoder_layer_graph
│   └── vision/      → 6 个视觉组件
├── runners/         → text_runner(含LayerDescriptor), vision_runner
├── families/        → base_model(新, 提供ExecuteGraph/EmbeddingLookup/RunPooling/FindImageTokenPositions)
├── adapters/        → qwen3vl_embedding/ (9 files, Qwen3VLModel 继承 BaseModel)
├── engine/          → llm_engine, runtime_impl, model_registry(含RegistryEntry+compat_check)
└── log/             → logger
```

## 审查目标与当前差距汇总

### 审查修正 6 条：3 条完全达标，2 条部分达标，1 条未达标

| # | 审查修正 | 达标 | 说明 |
|---|----------|------|------|
| 1 | components/ → text/vision/common | ✅ | Phase 1 |
| 2 | 策略模式 + 独立 Builder | ⚠️ | if-else 可用于 GQA；MLA/MoE 需拆 Builder → Phase 11 |
| 3 | 1层基类 + 组合 Runner | ⚠️ | 基类✅；但适配器未组合 Runner → Phase 9 |
| 4 | Runner 只管图生命周期 | ✅ | Phase 2 |
| 5 | void* metadata | ✅ | Phase 4 |
| 6 | 适配器瘦身 790→300-400 | ❌ | 790→775；需 Phase 9+10 协同 |

### 遗漏项 6 条：2 条已完成，4 条待做

| # | 遗漏项 | 状态 | Phase |
|---|--------|------|-------|
| a | WeightLayout 参数化 | ⏸️ 暂缓 | 优先级低，当前 WeightHelpers 已足够 |
| b | VariantPack 输入命名合约 | 🔲 | Phase 12 |
| c | buffer_size 上提 Runtime | ✅ | Phase 8 |
| d | InjectDeepstack NPU 优化 | 🔲 | Phase 13 |
| e | KV Cache 接口预留 | 🔲 | Phase 14 |
| f | 批处理接口准备 | 🔲 | Phase 15 |
