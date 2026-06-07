# C++ 框架重构计划

## 目标
使 ATB C++ 推理引擎能方便地适配更多模型（Qwen3、Qwen2VL、DeepSeek-V2/V3、GLM-4V、Mixtral 等），
解决当前代码中 components/layers 断裂、适配器过重、缺乏多模型参数化等问题。

## 审查修正要点（来自 subagent 审查）
1. **components/ 扁平化改为 text/vision/common 三目录**（非完全扁平）
2. **参数化组件改为策略模式 + 独立 Builder**（非统一 switch-case）
3. **放弃 4 层继承，改用 1 层基类 + 组合**（BaseModel : IModel，适配器直接组合 Runner）
4. **Runner 只管图生命周期，不持有执行逻辑**
5. **InferRequest 用 void* metadata 替代 dynamic_cast 多态**
6. **适配器瘦身目标 790→300-400 行**（非 50 行）

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
   # 若模型路径需修正: ln -sf /mnt/workspace/gitCode/models/model.safetensors /mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B/model.safetensors
   ./test_e2e
   ./test_accuracy
   ./test_consistency
   ```
6. **提交 git，push 到远端**

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
- [x] Python 组件一致性测试通过（cosine > 0.999）
- [x] Python E2E 权重加载测试通过（Text:0.999961, Image:0.999720, Image+Text:0.999916）
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
- [x] BaseModel : IModel，提供 ExecuteGraph()、EmbeddingLookup()、RunPooling()、Synchronize() 等工具方法
- [x] Qwen3VLModel 改为继承 BaseModel，删除重复的工具方法代码
- [x] C++ 单元测试通过
- [x] Python 组件一致性测试通过
- [x] Python E2E 权重加载测试通过

### Phase 6: ICrossModalFusion 接口提取 ✅ DONE
- [x] 新建 src/components/common/cross_modal_fusion.h（接口）
- [x] 新建 src/components/common/deepstack_fusion.h/cpp（Deepstack 实现）
- [x] 将 Qwen3VLModel::InjectDeepstack() 逻辑移入 DeepstackFusion
- [x] Qwen3VLModel 通过 unique_ptr<ICrossModalFusion> 持有
- [x] C++ 单元测试通过
- [x] Python 组件一致性测试通过
- [x] Python E2E 权重加载测试通过

### Phase 7: WeightHelpers 复用函数 ✅ DONE
- [x] 新建 src/io/weight_helpers.h/cpp
- [x] 提供 LoadLinearWeights()、LoadDecoderLayerWeights() 等复用函数
- [x] Qwen3VLWeights 中的重复加载代码改为调用 helpers
- [x] C++ 单元测试通过
- [x] Python 组件一致性测试通过
- [x] Python E2E 权重加载测试通过

### Phase 8: buffer_size 上提 Runtime ✅ DONE
- [x] IRuntime 接口添加 SetBufferSize(uint64_t) 虚方法
- [x] RuntimeImpl 实现 SetBufferSize，委托给 BufferPool::SetBufferSize
- [x] 验证 BufferPool::SetBufferSize 接口存在且可用
- [x] C++ 单元测试通过

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

## 当前目录结构（Phase 3 完成后）
```
include/atb_llm/     → engine.h, model.h, runtime.h, types.h, layer_desc.h(新)
src/
├── core/            → RAII, NpuTensor, allocator, context, buffer_pool, graph_builder
├── io/              → safetensors_reader, weight_loader, json_config
├── ops/             → 14 个 ATB 算子包装
├── components/
│   ├── common/      → self_attention_graph, swiglu_mlp_graph, rms_norm_graph, mrope, rope_1d, moe_mlp_graph
│   ├── text/        → decoder_layer_graph
│   └── vision/      → 6 个视觉组件
├── runners/         → text_runner(含LayerDescriptor), vision_runner
├── adapters/        → qwen3vl_embedding/ (9 files)
├── engine/          → llm_engine, runtime_impl, model_registry
└── log/             → logger
```
