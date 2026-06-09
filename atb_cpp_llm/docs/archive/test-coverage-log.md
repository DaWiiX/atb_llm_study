# C++ 框架测试覆盖 TODO

> **架构师职责声明（请反复阅读，防止越位）**:
> - ✅ 我（architect）**只**负责：(a) 维护本文档；(b) 调度 subagents
> - ❌ 我**绝对不**写代码、改 CMakeLists、改测试、改 Python reference
> - ❌ 我**绝对不**亲自跑 build/test
> - 一切开发工作必须通过 `Agent` 工具委派给 subagent
> - 当 subagent 完成后，**我唯一的动作**是用 Edit 工具更新本文档的 `[x]` 状态

---

## 一、当前测试金字塔状态

| Level | 覆盖率 | 文件数 | 状态 |
|-------|--------|--------|------|
| L0 基础框架 | 100% | 2 | 🟢 完整 |
| L1 CPU 纯函数 | 100% | 6 | 🟢 完整 |
| L2 算子精度 | 100% | 13 | 🟢 完整 |
| L3 集成 | 100% | 4 | 🟢 完整 |
| L4 E2E | 100% | 2 | 🟢 完整 |

**最终结果**: 33 个测试二进制 / 100+ test cases / 800+ assertions，全部通过。

---

## 二、已完成的测试（不要再做）

### Level 0 ✅
- [x] `test_core.cpp` — ContextManager, TensorAllocator, GraphBuilder, BufferPool, JsonConfig, RAII, NpuTensor, CreateRuntime
- [x] `test_io_adapters.cpp` — SafetensorsReader, WeightLoader, Qwen3VLConfig, SmartResize（基本）, BilinearResize, PreprocessImage

### Level 1 ✅
- [x] `test_mrope_cpu.cpp` — GetRopeIndex (5场景) + MRoPE::Compute
- [x] `test_vision_rope_cpu.cpp` — VisionRotaryEmbedding::ComputeFreqTable + ComputeVisionRotPosEmb + 多图像 + 边界
- [x] `test_preprocess_cpu.cpp` — SmartResize vs Python + banker's rounding + 边界

### Level 2 (仅有"创建/不崩溃"测试，不算精度测试)
- [x] `test_text_ops.cpp` — Op 创建 + 极简 execute（仅验证非零）
- [x] `test_vision_ops.cpp` — Op 创建 + 极简 execute（仅验证非零）

### Level 3 部分 ✅
- [x] `test_text_model.cpp` — TextRunner.Build + MakeCausalMask + execute 1 layer

### Level 4 部分 ✅
- [x] `test_e2e.cpp` — TEXT_ONLY 模式 E2E + EmbeddingLookup bounds
- [x] `test_forward_error_paths.cpp` — 错误路径
- [x] `test_consistency.cpp` — C++ vs Python TEXT_ONLY 对比

---

## 三、待完成的测试 TODO（按优先级排序）

### 🔴 P0: Level 1 CPU 纯函数（不依赖 NPU）

| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L1-1 | `test_pos_embed_cpu.cpp` | `ComputePosEmbedInterp` 双线性插值 + spatial merge | [x] PASS 3/3 cos=1.0 | Agent-L1A |
| L1-2 | `test_float_utils.cpp` | `Fp16ToF32`, `Fp32ToFp16`, `Bf16ToFp16` 转换精度 | [x] PASS 5/5 (含 CANN API 文档化) | Agent-L1B |
| L1-3 | `test_base_model_utils.cpp` | `EmbeddingLookup`, `FindImageTokenPositions`, `RunPooling`（CPU部分） | [x] PASS 12 cases / 50 assertions | Agent-L1C |
| L1-4 | 扩展 `gen_cpu_reference.py` | 已有 pos_embed stage，需验证可运行 | [x] | (已完成) |

### 🟡 P1: Level 2 ATB 算子精度（vs PyTorch/NumPy，需 NPU）

#### Batch 2A: 基础算子
| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L2-1 | `test_rms_norm_precision.cpp` | RmsNorm (NORM/PRENORM/POSTNORM) vs PyTorch | [x] PASS 3/3 cos=1.0 | Agent-L2A |
| L2-2 | `test_layer_norm_precision.cpp` | LayerNorm vs PyTorch（验证 axis 参数） | [x] PASS 3/3 cos=1.0 | Agent-L2A |
| L2-3 | `test_linear_precision.cpp` | Linear（bias/transpose 变种）vs PyTorch | [x] PASS 3/3 cos=1.0 | Agent-L2A |
| L2-4 | `test_activation_precision.cpp` | SiLU / GELU / FastGELU vs PyTorch | [x] PASS 3/3 cos=1.0 | Agent-L2A |

#### Batch 2B: 张量操作
| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L2-5 | `test_elewise_precision.cpp` | Add/Mul/Muls/Sub/Cast vs NumPy | [x] PASS 5/5 cos=1.0 | Agent-L2B |
| L2-6 | `test_split_concat_precision.cpp` | Split/Concat vs NumPy | [x] PASS 4/4 cos=1.0 | Agent-L2B |
| L2-7 | `test_softmax_precision.cpp` | Softmax vs PyTorch | [x] PASS 3/3 cos=1.0 | Agent-L2B |
| L2-8 | `test_gather_reduce_precision.cpp` | Gather/Reduce(MAX/MIN/SUM) vs NumPy | [x] PASS 4/4 cos=1.0（MAX/MIN 用 bf16） | Agent-L2B |
| L2-9 | `test_transpose_set_value_precision.cpp` | Transpose/SetValue vs NumPy | [x] PASS 3/3 cos=1.0 | Agent-L2B |

#### Batch 2C: 注意力相关
| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L2-10 | `test_rope_precision.cpp` | RoPE (rotaryCoeff=2/4) vs 手算 | [x] PASS 2/2 cos=1.0 | Agent-L2C |
| L2-11 | `test_self_attention_precision.cpp` | SelfAttention (MHA/GQA, with/without mask) vs PyTorch | [x] PASS 3/3 cos=1.0 | Agent-L2C |

#### Batch 2D: Component 图（粗粒度精度）
| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L2-12 | `test_swiglu_mlp_precision.cpp` | SwiGluMlpGraph vs Python text_mlp | [x] PASS 2/2 cos=1.0 | Agent-L2D |
| L2-13 | `test_text_decoder_layer_precision.cpp` | TextDecoderLayerGraph vs Python text_decoder_layer | [x] PASS 2/2 cos=1.0 | Agent-L2D |

#### Batch 2E: Vision Components
| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L2-14 | `test_vision_attention_precision.cpp` | VisionAttentionGraph vs Python vision_attention | [x] PASS cos=1.0 | Agent-L2E |
| L2-15 | `test_vision_mlp_precision.cpp` | VisionMlpGraph vs Python vision_mlp | [x] PASS cos=1.0 | Agent-L2E |
| L2-16 | `test_vision_block_precision.cpp` | VisionBlockGraph vs Python vision_block | [x] PASS cos=0.999999 | Agent-L2E |
| L2-17 | `test_patch_embed_precision.cpp` | PatchEmbedGraph vs Python patch_embed | [x] PASS cos=1.0 | Agent-L2E |
| L2-18 | `test_vision_merger_precision.cpp` | VisionMergerGraph + DeepstackGraph vs Python | [x] PASS 2/2 cos=1.0 | Agent-L2E |

### 🟡 P2: Level 3 集成测试

| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L3-1 | `test_deepstack.cpp` | DeepstackFusion (IsDeepstackLayer, ExtractFeatures, InjectFeatures) | [x] PASS 4/4 (220 assertions) | Agent-L3A |
| L3-2 | `test_vision_runner_full.cpp` | VisionRunner 完整推理流水线 vs Python | [x] PASS 2/2 (27 assertions) | Agent-L3B |
| L3-3 | `test_text_runner_full.cpp` | TextRunner 28-layer 循环 vs Python | [x] PASS 2/2 (17 assertions) | Agent-L3B |

### 🟢 P3: Level 4 E2E 补全

| # | 测试文件 | 覆盖目标 | 状态 | 负责 Agent |
|---|---------|---------|------|----------|
| L4-1 | 扩展 `test_e2e.cpp` | IMAGE_ONLY 模式 E2E | [x] PASS (681ms) | Agent-L4A |
| L4-2 | 扩展 `test_e2e.cpp` | IMAGE_AND_TEXT 模式 E2E | [x] PASS (677ms) | Agent-L4A |

---

## 四、调度策略

### 并行批次（无 CMakeLists 冲突的方法）

由于多个 agent 同时改 CMakeLists.txt 会冲突，采用**串行批次**：

**Wave 1 (P0 - Level 1)**: 3 个并行 agent
- Agent-L1A: test_pos_embed_cpu.cpp
- Agent-L1B: test_float_utils.cpp
- Agent-L1C: test_base_model_utils.cpp
- 每个 agent **只改自己的 .cpp 文件**，CMakeLists 由 Wave 1 末尾的"合并 agent"完成

**Wave 2 (P1 - Level 2 基础)**: Batch 2A + 2B 并行
**Wave 3 (P1 - Level 2 复杂)**: Batch 2C + 2D + 2E 并行
**Wave 4 (P2 - Level 3)**: 2 个并行
**Wave 5 (P3 - Level 4)**: 1 个 agent

每 Wave 结束后：
1. 我（architect）夹一个 CMakeLists 合并 agent
2. 我（architect）夹一个 build & run agent 验证通过
3. 我（architect）更新本文档的 `[x]` 状态

---

## 五、进度追踪

- 总测试任务: 26
- 已完成: 26 / 26 (全部 Wave 完成)
- 失败/阻塞: 0
- 进行中 Wave: ✅ 全部完成

---

## 六、已修复 Bug 与对应测试

| Bug | 对应测试（防回归） |
|-----|------------------|
| GetRopeIndex 缺 vision_start | `test_mrope_cpu.cpp` Test 4 |
| LayerNorm epsilon/axis | `test_layer_norm_precision.cpp` (L2-2) |
| SmartResize banker's rounding | `test_preprocess_cpu.cpp` Test 2 |
| Bf16ToFp16 截断 | `test_float_utils.cpp` (L1-2) |
| VisionRotary dim/2 columns | `test_vision_rope_cpu.cpp` Test 1 |
| Reduce MAX/MIN dtype 不支持 fp16 | `test_gather_reduce_precision.cpp` (用 bf16) |

---

## 七、最终回归报告

```
═══════ FULL REGRESSION ═══════
SUMMARY: 33 passed, 0 failed
```

### 已完成的所有测试目标 (33 个)

#### Level 0 (基础框架) — 2
- test_core, test_io_adapters

#### Level 1 (CPU 纯函数) — 6
- test_mrope_cpu, test_vision_rope_cpu, test_preprocess_cpu
- test_pos_embed_cpu, test_float_utils, test_base_model_utils

#### Level 2 (算子/组件精度) — 13
- test_rms_norm_precision, test_layer_norm_precision, test_linear_precision, test_activation_precision
- test_elewise_precision, test_split_concat_precision, test_softmax_precision
- test_gather_reduce_precision, test_transpose_set_value_precision
- test_rope_precision, test_self_attention_precision
- test_swiglu_mlp_precision, test_text_decoder_layer_precision
- test_vision_attention_precision, test_vision_mlp_precision, test_vision_block_precision
- test_patch_embed_precision, test_vision_merger_precision

#### Level 2 (旧式 execute 测试，保留) — 2
- test_text_ops, test_vision_ops

#### Level 3 (集成) — 4
- test_text_model, test_deepstack, test_vision_runner_full, test_text_runner_full

#### Level 4 (E2E) — 3
- test_e2e (6 cases: Load, TEXT_ONLY, Multi-inference, Bounds, **IMAGE_ONLY**, **IMAGE_AND_TEXT**)
- test_forward_error_paths, test_consistency

### 总精度水平
- 所有精度测试: cosine ≥ 0.999 (绝大多数 = 1.000000)
- 唯一 0.999999: test_vision_block_precision (fp16 链路深度造成的合理微差)

---

## 八、调度策略回顾（已执行）

成功执行的 5 个 Wave：
- **Wave 1**: 3 个 agent 并行编写 Level 1 → 1 个集成 agent (CMake+build+run)
- **Wave 2**: 2 个 agent 并行编写 Level 2 基础算子 → 1 个集成 agent
- **Wave 3**: 3 个 agent 并行编写 Level 2 复杂组件 → 1 个集成 agent
- **Wave 4**: 2 个 agent 并行编写 Level 3 集成 → 1 个集成 agent
- **Wave 5**: 1 个 agent 完成 Level 4 E2E + 全量回归验证

共 **12 个 subagent**，架构师全程**只做调度和文档更新**，从未亲自写代码。
