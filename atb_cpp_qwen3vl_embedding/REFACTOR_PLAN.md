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

## TODO 列表

### Phase 0: 修复 components/layers 断裂 ✅ DONE
- [x] 给 SelfAttentionGraph 增加 use_qk_norm 和 rotary_dim 参数
- [x] TextDecoderLayerGraph 改为调用 SelfAttentionGraph::Build() + SwiGluMlpGraph::Build() + RmsNormGraph::Build()
- [x] 所有 NPU 测试通过
- Commit: 3d3a9a8

### Phase 1: 目录整理 ✅ DONE
- [x] layers/ (2文件) → components/text/ (decoder_layer_graph.h/cpp)
- [x] components/attention/ + norm/ + mlp/ + position/ → components/common/
- [x] components/vision/ 保留
- [x] 更新 CMakeLists.txt, #include, 命名空间
- [x] 所有 NPU 测试通过
- Commit: 868e899

### Phase 2: Runner 层重构 ✅ DONE
- [x] models/ → runners/ 目录重命名
- [x] TextModel → TextRunner, VisionModel → VisionRunner
- [x] 命名空间 models → runners
- [x] TextRunner::Build() → EnsureBuilt(seq_len) + seq_len 缓存
- [x] TextRunner::Config 包含 use_qk_norm, rotary_dim, use_mask
- [x] Runner 只管图生命周期，不持有 Run() 逻辑
- [x] 所有 NPU 测试通过
- Commit: 0d998a0

### Phase 3: 策略模式引入 + LayerDescriptor 参数化 🔲 TODO
- [ ] 在 include/atb_llm/layer_desc.h 中定义 AttnConfig/MlpConfig/NormConfig + 枚举
- [ ] AttentionGraph::Build() 内部按 AttnConfig.type 分发到 GqaAttentionBuilder / MlaAttentionBuilder / MhaAttentionBuilder
- [ ] 共享代码提取到 attention_utils.h 自由函数（AddOProjection, ReshapeTo3D 等）
- [ ] MlpGraph::Build() 内部按 MlpConfig.type 分发到 SwiGluBuilder / GeGLUBuilder / MoEBuilder
- [ ] NormGraph 保持 if-else（仅 2 种变体，代码极短）
- [ ] TextRunner::Config 改用 LayerDescriptor 替代分散的参数
- [ ] 编译 + 测试通过

### Phase 4: 接口精简 + Registry 增强 🔲 TODO
- [ ] InferRequest: PreprocessedImage 加 const void* metadata + int64_t metadata_size
- [ ] 移除 PreprocessedImage 中硬编码的 grid_thw（Qwen3VL 适配器改用 metadata）
- [ ] ModelRegistry: 从 map 改为 vector<RegistryEntry>，加 CompatibilityCheck + priority
- [ ] 注册宏增加 REGISTER_MODEL_WITH_CHECK 变体
- [ ] CreateModel() 匹配策略: 精确匹配 → 兼容性检查 → 前缀匹配（降级警告）
- [ ] 编译 + 测试通过

### Phase 5: BaseModel 基类引入 🔲 TODO
- [ ] 新建 src/families/base_model.h/cpp
- [ ] BaseModel : IModel，提供 ExecuteGraph()、EmbeddingLookup()、RunPooling()、Synchronize() 等工具方法
- [ ] Qwen3VLModel 改为继承 BaseModel，删除重复的工具方法代码
- [ ] 编译 + 测试通过

### Phase 6: ICrossModalFusion 接口提取 🔲 TODO
- [ ] 新建 src/components/common/cross_modal_fusion.h（接口）
- [ ] 新建 src/components/common/deepstack_fusion.h/cpp（Deepstack 实现）
- [ ] 将 Qwen3VLModel::InjectDeepstack() 逻辑移入 DeepstackFusion
- [ ] Qwen3VLModel 通过 unique_ptr<ICrossModalFusion> 持有
- [ ] 编译 + 测试通过

### Phase 7: WeightHelpers 复用函数 🔲 TODO
- [ ] 新建 src/io/weight_helpers.h/cpp
- [ ] 提供 LoadLinearWeights()、LoadDecoderLayerWeights() 等复用函数
- [ ] Qwen3VLWeights 中的重复加载代码改为调用 helpers
- [ ] 编译 + 测试通过

### Phase 8: buffer_size 上提 Runtime 🔲 TODO
- [ ] 将 set_atb_buffer_size() 从模型 Load() 中移出
- [ ] Runtime::Init() 时设置一次，模型不再调用
- [ ] 编译 + 测试通过

---

## 每次重构的验证流程
1. 编译通过（0 error）
2. 运行所有可用的 NPU 测试：test_core, test_io_adapters, test_text_ops, test_vision_ops, test_text_model
3. 确认所有测试 Status: SUCCESS!
4. 提交 git，push 到远端

## 当前目录结构（Phase 2 完成后）
```
src/
├── core/              # 基础设施（12 文件）
├── io/                # I/O（6 文件）
├── ops/               # 原子算子（28 文件）
├── components/
│   ├── common/        # 跨域共享组件（10 文件：attention/mlp/norm/rope）
│   ├── text/          # 文本侧（2 文件：decoder_layer_graph）
│   └── vision/        # 视觉侧（12 文件）
├── runners/           # 图生命周期管理（4 文件：text_runner + vision_runner）
├── adapters/          # 模型适配层
│   └── qwen3vl_embedding/  # 9 文件
├── engine/            # 引擎层（6 文件）
└── log/               # 日志（2 文件）
```
