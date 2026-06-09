# C++ 框架重构计划

## 目标

使 ATB C++ 推理引擎能方便地适配更多模型（Qwen3、Qwen2VL、DeepSeek-V2/V3、GLM-4V、Mixtral 等），
解决当前代码中 components/layers 断裂、适配器过重、缺乏多模型参数化等问题。

---

## 1. 架构审查结论

### 1.1 审查修正要点

| # | 修正要求 | 状态 | 说明 |
|---|----------|------|------|
| 1 | components/ 扁平化改为 text/vision/common 三目录 | ✅ | Phase 1 |
| 2 | 参数化组件改为策略模式 + 独立 Builder | ⚠️ | if-else 可用于 GQA；MLA/MoE 需拆 Builder（Phase 11） |
| 3 | 放弃 4 层继承，改用 1 层基类 + 组合 | ✅ | BaseModel : IModel，适配器直接组合 Runner（Phase 5+9） |
| 4 | Runner 只管图生命周期，不持有执行逻辑 | ✅ | Phase 2 |
| 5 | InferRequest 用 void* metadata 替代 dynamic_cast 多态 | ✅ | Phase 4 |
| 6 | 适配器瘦身目标 790→300-400 行 | ⚠️ | 790→588（Phase 10），RunVision/RunTextDecoder/PrepareInputs 的 Qwen3VL 特有逻辑难以进一步提取 |

### 1.2 未完全达标项

| 项目 | 当前状态 | 目标 | 原因 |
|------|----------|------|------|
| 适配器行数 | 588 行 (.h 96 + .cpp 492) | 300-400 行 | Qwen3VL 独有的权重编排逻辑（RunVision 155行 / PrepareInputs 80行 / RunTextDecoder 73行），提取到 Runner 会增加耦合；Phase 17 后已删除 PrepareInputs 和 RunTextDecoder |
| InjectFeatures 纯 NPU 实现 | CPU 侧加法 + partial-copy | 纯 NPU scatter-add | ATB SetValue/Gather 是否支持任意索引 scatter-add 需查文档验证 |

### 1.3 待解决问题

| 问题 | 优先级 | 描述 |
|------|--------|------|
| InjectFeatures 纯 NPU 实现 | 中 | 消除最后一段 NPU↔Host 传输 |
| Debug 模式 -DDEBUG 与 LogLevel::DEBUG 冲突 | 低 | CMakeLists `-DDEBUG` 宏与 logger.h 枚举冲突 |
| 测试加固 P2（18 项） | 低 | 见 [archive/test-hardening-log.md](./archive/test-hardening-log.md) H12-H29 |

---

## 2. 已完成的 Phase 汇总

### Phase 0-8: 基础重构

| Phase | 关键变更 | Commit |
|-------|----------|--------|
| 0. 修复 components/layers 断裂 | SelfAttentionGraph 加 use_qk_norm/rotary_dim；TextDecoderLayerGraph 调用子组件 Build | 3d3a9a8 |
| 1. 目录整理 | layers/ → components/text/ + common/ + vision/ | 868e899 |
| 2. Runner 层重构 | models/ → runners/；TextRunner/VisionRunner 只管图生命周期 | 0d998a0 |
| 3. 策略模式 + LayerDescriptor | AttnConfig/MlpConfig/NormConfig + 枚举；Config-based Build 重载 | 726fa16 |
| 4. 接口精简 + Registry 增强 | PreprocessedImage.metadata；ModelRegistry 优先级；REGISTER_MODEL_WITH_CHECK | 8730f33 |
| 5. BaseModel 基类 | ExecuteGraph/EmbeddingLookup/RunPooling/FindImageTokenPositions 复用 | 357d7c7 |
| 6. ICrossModalFusion 接口 | DeepstackFusion 提取；InjectFeatures 独立 | 4eeebb4 |
| 7. WeightHelpers 复用 | CopyWeightToFp16NPU/CopyWeightToFp16Host/LoadLinearWeights | dc97c30 |
| 8. buffer_size 上提 Runtime | IRuntime::SetBufferSize；委托给 BufferPool | fdf2ac0 |

### Phase 9-15: 架构升级

| Phase | 关键变更 | Commit |
|-------|----------|--------|
| 9. 适配器组合 Runner | Qwen3VLModel 持有 unique_ptr\<TextRunner\> + unique_ptr\<VisionRunner\>，移除 5 个 OperationHandle | e67b26d |
| 10. 适配器瘦身 | ComputePosEmbedInterp 提取为独立组件；790→588 行 | 329c43a |
| 11. 独立 Builder 拆分 | IAttentionBuilder + Gqa/Mha/Mla；IMlpBuilder + SwiGlu/GeGlu/Gelu/MoE；工厂分发 | 845fce3 |
| 12. VariantPack 命名合约 | BuildResult + ValidateVariantPack() + Debug 模式 ExecuteGraphChecked() | 34fc552 |
| 13. InjectDeepstack 优化 | TensorAllocator offset-based copy + partial-copy（~1000x 传输量减少） | 1af0bb4 |
| 14. KV Cache 接口预留 | IRuntime::AllocKVCache/FreeKVCache；KVCacheManager stub；IModel::IsGenerative() | 359dc04 |
| 15. 批处理接口准备 | AttnConfig/MlpConfig 加 batch_size 字段（默认 1，向后兼容） | 359dc04 |

### Phase 16-18: 生产化与一致性

| Phase | 关键变更 | Commit |
|-------|----------|--------|
| 16. Qwen3VLEmbedder 部署层 + Chat Template 对齐 | embedder.h/cpp；LAST_TOKEN_BY_MASK pooling；chat_tokenizer.py；gen_baseline_tokens.py | 7d6fee1 |
| 17. 逐模块精度验证 | 根因定位：GetRopeIndex 缺 vision_start_token_id + LayerNorm epsilon 读错 key；三模式 cos ≥ 0.999 | （多个） |
| 18. 框架一致性整改 | P0: 删除 normalize 死字段；P1: Forward→ForwardWithTiming 收敛（-210行）；P2: 抽出 debug_dump（-125行）；P3: Embedder invariants | 65d8acc~1e039df |

---

## 3. 当前性能基线（Phase 19，2026-06-09）

**配置**: 5 warmup + 3 iter，MM text ~500 tokens（chat-templated），Ascend 910B NPU

### 3.1 三路对比：C++ ATB vs Python ATB vs Transformers

| Mode | S | VisTok | C++ ATB | Py ATB | TF | Py/C++ | TF/C++ |
|------|---|--------|--------:|-------:|----:|------:|------:|
| TEXT 100 | 100 | 0 | **12.20** | 24.57 | 96.78 | 2.01× | 7.93× |
| TEXT 512 | 512 | 0 | **21.47** | 35.05 | 104.79 | 1.63× | 4.88× |
| TEXT 1024 | 1024 | 0 | **35.64** | 49.36 | 106.38 | 1.38× | 2.98× |
| TEXT 2048 | 2048 | 0 | **62.54** | 79.53 | 113.28 | 1.27× | 1.81× |
| TEXT 4096 | 4096 | 0 | **126.36** | 154.48 | 319.88 | 1.22× | 2.53× |
| IO 416×672 | 273 | 273 | **37.49** | 63.36 | 201.71 | 1.69× | 5.38× |
| IO 720×1280 | 880 | 880 | **81.36** | 106.18 | 345.20 | 1.31× | 4.24× |
| IO 1080×1920 | 1222 | 1222 | **115.90** | 143.30 | 552.01 | 1.24× | 4.76× |
| IO 1440×2560 | 1222 | 1222 | **115.97** | 140.60 | 543.27 | 1.21× | 4.68× |
| MM 416×672 | 940 | 273 | **53.11** | 79.88 | 208.10 | 1.50× | 3.92× |
| MM 720×1280 | 1547 | 880 | **101.34** | 125.49 | 350.51 | 1.24× | 3.46× |
| MM 1080×1920 | 1889 | 1222 | **133.97** | 160.06 | 556.52 | 1.19× | 4.15× |
| MM 1440×2560 | 1889 | 1222 | **134.19** | 162.38 | 555.20 | 1.21× | 4.14× |

**关键观察**:
- C++ ATB 全面最快，geomean 领先 Python ATB 1.39×、领先 TF 4.22×
- C++/Python ATB 差距随 S 增大收敛：TEXT 100 (2.01×) → TEXT 4096 (1.22×) — ATB graph launch overhead 在短序列占主导
- TF 对文本长度不敏感（eager execution Python dispatch overhead），E2E 仍慢 ~4×
- IO vs MM 对 TF 几乎无差异（+6ms），C++ +15ms — TF 对 batch=1 attention_mask 有短路优化

### 3.2 精度验证（C++ ATB vs Python ATB，13/13 Cosine ≥ 0.99）

| Mode | Cosine | |
|------|-------:|----|
| TEXT 100 | 0.999946 | PASS |
| TEXT 512 | 0.999946 | PASS |
| TEXT 1024 | 0.999980 | PASS |
| TEXT 2048 | 0.999894 | PASS |
| TEXT 4096 | 0.999985 | PASS |
| IO 416×672 | 0.999832 | PASS |
| IO 720×1280 | 0.999962 | PASS |
| IO 1080×1920 | 0.999770 | PASS |
| IO 1440×2560 | 0.999916 | PASS |
| MM 416×672 | 0.999962 | PASS |
| MM 720×1280 | 0.999943 | PASS |
| MM 1080×1920 | 0.999969 | PASS |
| MM 1440×2560 | 0.999958 | PASS |

最低 cosine 为 IO 1080×1920 (0.999770)，远高于 0.99 阈值。**可用于生产环境。**

> 子阶段性能明细、缓存行为分析、P0-P8 性能演变历史见 [archive/benchmark-report.md](./archive/benchmark-report.md)。

---

## 4. 踩坑经验集（Lessons Learned）

Phase 16-19 期间遇到的 10 个关键问题，按发现顺序排列。

### 4.1 fp16 二进制的"比特重解释" vs "数值 cast"

**症状**: IO/MM 模式下 Python ATB 输出全为 NaN（norm=nan），cosine 校验失败。

**根因**: C++ 将 fp16 值以 uint16 二进制写入 `.bin`，Python 用 `np.array(struct.unpack(...), dtype=np.uint16)` 加载后保持 uint16。`to_npu_half()` 调用 `.half()` 时做了**整数值→float16 的数值 cast**（如 uint16 值 15360 → float16 15360.0），而非**比特重解释**。

正确做法: `np.array(..., dtype=np.uint16).view(np.float16)` — numpy `.view()` 重解释比特位而不改变内存。

> **教训**: 跨越 C++/Python 边界的二进制 fp16 数据，必须确认两边使用相同的位解释。`np.frombuffer(raw, dtype=np.float16)` 也正确，但 `struct.unpack` + `dtype=np.uint16` 之后 `.astype(np.float16)` 就是错的。

### 4.2 Config "可配置"字段从未被读取

**症状**: `EngineConfig::normalize` 被多处设置为 `true`，但 `LLMEngine::Impl::Init` 从未读取它。实际 normalize 由 `Qwen3VLConfig::normalize` 控制。

**根因**: 接口设计时预留了"可能有用"的字段，但实现路径没走它。调用方以为可以控制行为，实际不能。

> **教训**: 每个 config 字段必须有对应的消费代码。新增字段时问："谁读它？怎么验证它确实被读了？"用 `grep -rn "\.field_name"` 验证完整的写→读链路。

### 4.3 同一功能的双路径实现必然分叉

**症状**: `Qwen3VLModel` 维护两条推理路径：`Forward`（fp32 cos/sin/mask）和 `ForwardWithTiming`（fp16 + NPU cache）。各 ~200 行，改 bug 只动一边。

**根因**: 为性能优化开了第二条路径，但没有删除旧路径。

> **教训**: 性能优化后必须把原路径删掉，让 `Forward` 成为 `ForwardWithTiming` 的薄 wrapper。通过 benchmark compare 确认精度无损后立即删旧代码。

### 4.4 debug dump 代码不要混进 production 路径

**症状**: `qwen3vl_model.cpp` 内 8 处 `if (getenv("ATB_DEBUG_VISION"))` 块（每处 ~15 行），分布在关键路径上。

**根因**: 逐层调试时每次加一个 dump 块，调试完没有清理。

> **教训**: debug dump 抽成 `core/debug_dump.{h,cpp}` 中的工具函数，调用方只需一行 `debug::DumpNpuFp16(rt, tensor, count, path)`，不再污染业务逻辑。

### 4.5 DEPRECATED 标签与实现冲突

**症状**: `PreprocessedImage::grid_thw` 注释标了 `DEPRECATED`，但代码实际优先读它，只在它为 null 时才 fallback 到 `metadata`。

**根因**: 注释表达了意图，但代码没跟上。

> **教训**: DEPRECATED 标签一旦写了就必须配套 `LOG_WARN` 或 `[[deprecated]]` 属性，否则就是误导。决定不废弃了就删掉标签。

### 4.6 Wrapper 类必须提供 wrapper 价值

**症状**: `Qwen3VLEmbedder` 最初只是 `LLMEngine` 的 pass-through（三行 `return engine_->XXX()`），没有 embedder 特有的校验。

**根因**: 创建 wrapper 时只考虑了"未来可能加逻辑"，但没加任何实际逻辑。

> **教训**: Wrapper 类从第一天起就应该有区别于底层接口的 contract：`Encode` 校验 batch_size==1、输出 shape=={hidden_size}、头文件注释写明 invariants。

### 4.7 多框架 benchmark 必须使用完全相同的输入

**症状**: C++ benchmark 用裸 token 序列（`[151655] * N`），Python 走完整 chat template pipeline（`processor.apply_chat_template`），cosine 只有 0.2-0.3。

**根因**: 两边各自独立构造输入，没有人确认它们是否等价。

> **教训**: 写统一的 token 生成脚本（`gen_baseline_tokens.py`），所有框架加载同一批 `.bin` 文件。首次搭建时先用最小测试向量确认 pipeline 能跑通。

### 4.8 .bin 文件命名混乱导致静默错配

**症状**: C++ 读 `tokens_chat_mm_{W}x{H}.bin`，Python 读 `tokens_mm_{W}x{H}.bin`。两组文件不同步时 cosine 莫名其妙低，代码表面看不出异常。

**根因**: 多个脚本各自生成 `.bin`，没有统一命名约定和生成入口。

> **教训**: 所有 benchmark .bin 由一个脚本统一生成。文件格式契约：`[int32 count][element_type * count]`。

### 4.9 验证修复时先 git stash 确认基线

**症状**: P4 实验期间加了额外 sync → 精度崩溃（block 23 cosine 0.28），一直以为是原有问题，花大量时间排查 deepstack 注入逻辑。

**根因**: 每次新改动都可能引入新问题。不先确认基线是否正常，排查方向全错。

> **教训**: 精度下降时第一步永远是 `git stash` 回到上一个已知正确的 commit，重新跑 benchmark 确认基线。基线通过→问题在新代码；基线不过→问题在原有代码。

### 4.10 NPU sync 语义：不是"加了更安全"

**症状**: 在 vision pixel CopyToDevice 和 FirstLayer ExecuteGraph 之间加 `runtime_->Synchronize()`，导致 block 1 之后所有输出 cosine 断崖式下降。

**根因**: `aclrtMemcpy`（H2D/D2H）在 Ascend 上是同步的。额外 sync 引入 NPU stream 上的隐式 barrier，打断 ATB graph 内部多 stream 调度。

> **教训**: H2D/D2H 操作已自带同步，不要额外加 sync。`ASCEND_LAUNCH_BLOCKING=1` 不保证 byte-exact 确定性。NPU 上"多加 sync 提高安全性"是反模式。

---

## 5. 缺失测试复盘：已有 Bug → 反推缺失的测试

从踩坑经验 + Phase 16-19 实际 bug 逐一反推：**如果有这个测试，能不能提前发现？**

### 5.1 复盘矩阵

| # | Bug / 教训 | 调试耗时 | 缺失的测试 | 现在有吗？ | 缺口 |
|---|-----------|---------|-----------|----------|------|
| B1 | fp16 .bin 比特重解释错误 | 小时级 | **L1: .bin 格式 round-trip** — C++ 写已知 fp16 → Python 读 → 逐元素对比 | ✅ test_bin_format | — |
| B2 | EngineConfig.normalize 死字段 | 天级 | **L0: Config 布线测试** — 每个 public config 字段验证有消费代码 | ✅ test_config_wiring | — |
| B3 | Forward/ForwardWithTiming 双路径分叉 | 天级 | **L3: Wrapper 一致性** — Forward == ForwardWithTiming（同输入、忽略计时） | ✅ 已消除 | — |
| B4 | Vision LayerNorm epsilon 读错 JSON key | 小时级 | **L0: Config key 校验** — dump C++ 和 Python 加载的所有 config 值 → diff | ✅ test_config_wiring | — |
| B5 | GetRopeIndex 缺 vision_start_token_id | 天级 | **L1: MRoPE chat 模板输入** — 含 vision_start 标记的多模态 token → 比较 C++ vs Python 位置编码 | ✅ test_mrope_cpu T7-9 | — |
| B6 | 跨框架 token 输入不同（裸 token vs chat template） | 天级 | **L4: 输入身份校验** — benchmark 前比较 C++/Python 加载的 token hash | ⚠️ 半缓解（gen_baseline_tokens.py 统一入口，但缺 hash 校验） | 🟡 P1 |
| B7 | .bin 文件命名静默错配 | 天级 | 同 B6 — 输入身份校验 | ⚠️ 半缓解 | 🟡 P1 |
| B8 | 额外 sync 破坏精度 | 小时级 | **L4: Sync 安全性** — ATB_DISABLE_PER_OP_SYNC=0 vs =1，cosine ≥ 0.99 | ❌ | 🟡 P1 |
| B9 | 权重加载双截断 (f32→bf16→fp16) | 小时级 | **L0: 权重精度** — C++ vs Python 加载的 fp16 权重逐元素对比 | ❌ | 🟡 P1 |
| B10 | SmartResize 银行家舍入 | 小时级 | **L1: SmartResize 舍入** — 边界值 (x.5) 对比 C++ vs Python | ✅ test_preprocess_cpu | — |
| B11 | Bf16ToFp16 截断偏差 | 小时级 | **L1: 浮点转换精度** — 所有极端值 C++ vs CANN API | ✅ test_float_utils | — |
| B12 | debug dump 混入 production | — | 代码质量问题，非测试缺口 | ✅ debug_dump 已抽出 | — |

> **统计**（更新于 2026-06-09）: 12 个 bug，其中 **7 个已有测试覆盖**（B1/B2/B3/B4/B5/B10/B11），**3 个半缓解**（B6/B7/B12），**2 个仍无测试**（B8/B9）。

### 5.2 🔴 高优先级缺口

#### ✅ G1: .bin 文件格式 round-trip 测试（对应 B1, B6, B7）— `5cd8d4b`

`tests/level1_cpu_pure/test_bin_format.{cpp,py}` — C++ 写已知 fp16 bit patterns（0, ±1, ±inf, NaN, denorm, min/max, 3.0, 4.0）→ Python 读并验证。**关键反回归**：证明 `.view(np.float16)` ≠ `.astype(np.float16)`。

#### ✅ G2: Config 布线测试（对应 B2, B4）— `5cd8d4b`

`tests/level0_framework/test_config_wiring.{cpp,py}` — C++ 加载 Qwen3VLConfig 并 dump 29 字段到 JSON → Python diff。**关键反回归**：`vis_epsilon < 1e-4`（如果读错成 `initializer_range` 会是 0.02）。

#### 🔲 G3: 预变更回归脚本（对应 B8, B3）

```
脚本内容（benchmark 运行前自动执行）:
  1. 保存当前 commit hash
  2. 如果工作区有未提交修改 → git stash
  3. 跑 benchmark --mode compare 全量 13/13
  4. 验证 cosine ≥ 0.99
  5. git stash pop（如果有 stash）
  6. 只在基线通过后才允许跑新代码的 benchmark

新增文件: scripts/verify_baseline.sh
```

**能预防的 bug**: B8（sync 破坏精度）、B3（双路径分叉）

### 5.3 🟡 中优先级缺口

#### ✅ G4: MRoPE chat 模板多图像测试（对应 B5）— `b9b60ed`

扩展 `test_mrope_cpu.cpp` 6→9 tests：
- **Test 7**: 多图像 — 两幅图各自独立 2D 网格（`[[1,2,2], [1,4,4]]`）
- **Test 8**: Chat 模板 — 真实生产输入 `<|im_start|>` `<|vision_start|>` `<|image_pad|>`×4 文本 `<|im_end|>`
- **Test 9**: 边界 — 5 个 inline 子 case（vision_start at 0/S-2/S、相邻 vision_start、vision_start+非 image token 回退） + 批量参考对比

Python 参考数据生成器新增 `mrope_pid_multi_img` / `mrope_pid_chat_template` / `mrope_pid_boundary` 三个 stage，使用真实的 `engine_utils.get_rope_index`。

**解决了 H29**（test_mrope_cpu: 缺 multi-batch + multi-image 覆盖）。

#### 🔲 G5: Sync 安全性测试（对应 B8）

```
测试内容:
  1. 同一输入，分别以 ATB_DISABLE_PER_OP_SYNC=0 和 =1 运行
  2. 验证 13/13 cosine ≥ 0.999
  3. 验证无全零输出 token

新增文件: tests/level4_e2e/test_sync_safety.cpp
          依赖: 需要环境变量控制 per-op sync
```

#### 🔲 G6: 权重加载精度测试（对应 B9）

```
测试内容:
  1. C++ 加载 Safetensors → dump 每层首个权重的前 100 个 fp16 值
  2. Python 加载相同 Safetensors → dump 同样位置
  3. 逐元素对比，允许 ±1 ULP 差异

新增文件: tests/level0_framework/test_weight_precision.cpp
          tests/level0_framework/test_weight_precision.py
```

### 5.4 缺口与现有 P2 待办项关联

缺口分析发现，[archive/test-hardening-log.md](./archive/test-hardening-log.md) 中 18 个 P2 待办项里有 3 个与上述缺口直接相关：

| P2 ID | 描述 | 对应的缺口 |
|-------|------|-----------|
| **H29** | test_mrope_cpu: 缺 multi-batch + multi-image + t>1 | ✅ 已通过 G4 解决（Test 7+8+9，`b9b60ed`） |
| H15 | test_core: TensorAllocator 全 buffer 比对 | 弱关联 G2 |
| H27 | test_base_model_utils: 缺 RunPooling::LAST_TOKEN CPU 单测 | 弱关联，不影响已有 bug |

### 5.5 立即可执行的行动项

| 优先级 | 行动 | 预计工作量 | 状态 |
|--------|------|----------|------|
| 🔴 P0 | G1: .bin round-trip 测试 | 2h | ✅ `5cd8d4b` |
| 🔴 P0 | G2: Config 布线测试 | 2h | ✅ `5cd8d4b` |
| 🔴 P0 | G3: 预变更回归脚本 | 1h | 🔲 待实现 |
| 🟡 P1 | G4: MRoPE chat 模板多图像测试（解决 H29） | 2h | ✅ `b9b60ed` |
| 🟡 P1 | G6: 权重加载精度测试 | 1h | 🔲 待实现 |
| 🟡 P1 | G5: Sync 安全性测试 | 2h (需 NPU) | 🔲 待实现 |

---

## 6. 开发检查清单

1. **新增 config 字段** → `grep` 确认有消费代码（💡 写 L0 config 布线测试更可靠）
2. **新增函数** → 写 L1 单元测试（纯 CPU，秒级）
3. **修改推理路径** → 先跑 G3 基线确认，再跑 13/13 cosine ≥ 0.99
4. **修改 .bin 格式** → 更新 `gen_baseline_tokens.py`，跑 G1 round-trip 测试
5. **修改后精度异常** → `git stash` 确认基线，二分排查（💡 G3 脚本自动化）
6. **加 debug dump** → 只用 `debug::DumpNpuFp16()`，不手写 fopen/fwrite
7. **新增 DEPRECATED 标签** → 同时加 `[[deprecated]]` 或 `LOG_WARN`，否则不标
8. **C++/Python 跨语言数据** → 跑 G1 bin 格式 round-trip + G2 config diff

---

## 7. 附录

### 7.1 环境配置

参考 CLAUDE.md 中的完整环境配置。核心变量：
```bash
source ~/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source ~/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=~/Ascend/nnal/atb/latest/atb/cxx_abi_1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
```

### 7.2 验证流程

1. **C++ 编译**: `cmake --build build` — 0 error
2. **C++ 单元测试**: `cd build && ctest` — 35/35 全部 SUCCESS（含新增 test_bin_format, test_config_wiring）
3. **C++ vs Python 精度**: `./test_consistency` + `python tests/test_consistency.py` — cosine > 0.99
4. **C++ vs Python 多模式**: `./test_accuracy` + `python tests/test_accuracy.py`
5. **全量基准**: `./benchmark --mode compare` + `python tests/test_embedder_e2e.py --mode both --bench`

### 7.3 测试金字塔

| Level | 测试数 | 覆盖范围 |
|-------|--------|---------|
| L0 基础框架 | 3 | test_core, test_io_adapters, test_config_wiring |
| L1 CPU 纯函数 | 10 | test_mrope_cpu (9 cases), test_vision_rope_cpu, test_preprocess_cpu, test_pos_embed_cpu, test_float_utils, test_base_model_utils, test_causal_mask_fp16, test_embedder_utils, test_embedder_invariants, test_bin_format |
| L2 算子精度 | 20 | 覆盖 RMSNorm/LayerNorm/Linear/Activation/Elewise/SplitConcat/Softmax/GatherReduce/TransposeSetValue/RoPE/SelfAttention/SwiGLU/TextDecoder/VisionAttention/VisionMLP/VisionBlock/PatchEmbed/VisionMerger + text_ops + vision_ops |
| L3 集成 | 6 | test_text_model, test_deepstack, test_deepstack_npu_tensor, test_vision_runner_full, test_text_runner_full, test_vision_stages |
| L4 E2E | 5 | test_e2e, test_consistency, test_accuracy, test_stage_precision, test_forward_error_paths |
| Benchmark | 1 | benchmark |
| **总计** | **45** | 含新增 test_bin_format, test_config_wiring |

### 7.4 当前目录结构

```
include/atb_llm/     → engine.h, model.h, runtime.h, types.h, layer_desc.h, embedder.h, kv_cache.h
src/
├── core/            → RAII, NpuTensor, allocator, context, buffer_pool, graph_builder, debug_dump
├── io/              → safetensors_reader, weight_loader, json_config, weight_helpers
├── ops/             → 14 个 ATB 算子包装
├── components/
│   ├── common/      → self_attention_graph, swiglu_mlp_graph, rms_norm_graph, mrope, rope_1d,
│   │                  cross_modal_fusion, deepstack_fusion, gqa_attention_builder, mlp_builder
│   ├── text/        → decoder_layer_graph
│   └── vision/      → 6 个视觉组件 + pos_embed + vis_rope
├── runners/         → text_runner, vision_runner
├── families/        → base_model
├── adapters/        → qwen3vl_embedding/ (qwen3vl_model, qwen3vl_weights, qwen3vl_preprocess, qwen3vl_config, register)
├── engine/          → llm_engine, embedder, runtime_impl, model_registry
└── log/             → logger
```
