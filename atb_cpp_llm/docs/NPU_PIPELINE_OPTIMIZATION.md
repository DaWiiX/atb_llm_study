# NPU Pipeline Optimization Roadmap

> 长期追踪文档：让 Qwen3VL Embedding 推理管线尽可能保持「入口一次 H2D，出口一次 D2H，中间全 NPU 异步流水」。

最后更新：2026-06-08（P5 完成）

---

## 总览

理想目标：

```
  ┌──────────────────────────────────────────────────────────────┐
  │                  CPU (host)                                  │
  ├──────────────────────────────────────────────────────────────┤
  │  Preprocess image  │ tokenize │ build cheap (idx, wt) tables │
  └────────┬──────────────────────────────────────────────────────┘
           │ 一次 H2D
           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                  NPU (device)                                │
  │  ┌─────────────────────────────────────────────────────────┐ │
  │  │  PosEmb graph → VisRoPE graph                           │ │
  │  │       ↓ pos_npu, cos_npu, sin_npu                       │ │
  │  │  Vision FirstLayer → 23× VisionBlock → Merger           │ │
  │  │       │                          ↓                      │ │
  │  │       └─→ Deepstack mergers (NPU-resident features)     │ │
  │  │                                  │                      │ │
  │  │  Token embedding lookup → Scatter image tokens (NPU)    │ │
  │  │                                  ↓                      │ │
  │  │  MRoPE cos/sin (NPU) → 28× TextDecoderLayer             │ │
  │  │  └─ IndexAdd ds_feat at deepstack layers (all on NPU)   │ │
  │  │                                  ↓                      │ │
  │  │  FinalNorm                                              │ │
  │  └─────────────────────────────────────────────────────────┘ │
  └────────────┬─────────────────────────────────────────────────┘
               │ 一次 D2H
               ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                  CPU (host)                                  │
  │  Pool last token + L2 normalize → return embedding           │
  └──────────────────────────────────────────────────────────────┘
```

理想的整个管线只有 **2 次 H↔D 同步**：开始的 H2D 和结尾的 D2H。

---

## 现状基线（2026-06-08 起点）

未做任何 NPU 化时（baseline），所有 vision pos_embed、vision rope、deepstack 注入都是 CPU 上做。E2E 在 896×896 下耗时 **581.51 ms**。

---

## 路线图

### ✅ 已完成

#### P1: Vision PosEmbed Interpolation → NPU
- **Commit**: `059f554`
- **CPU 干预点**: `pos_embed_interp.cpp:ComputePosEmbedInterp` 三层 for 循环 + bilinear 插值
- **方案**: Stage A 在 CPU 算 (4 idx + 4 wt) 表 — Stage B NPU graph 做 4× Gather + 4× Mul + 3× Add
- **算子需求**: 已有 GatherOp + ElewiseOp（Add/Mul，原生支持 broadcast）
- **实测收益**:
  | 分辨率 | vision_pos 阶段 | 节省 |
  |--------|---|---|
  | 224×224 | 14.09 → 0.89 ms | -94% |
  | 416×672 | 79.29 → 3.04 ms | -96% |
  | 672×416 | 79.48 → 3.02 ms | -96% |
  | 896×896 | 228.42 → 8.03 ms | -96% |

#### P2: Vision RoPE cos/sin → NPU
- **Commit**: `aae31cb`
- **CPU 干预点**: `mrope.cpp:ComputeVisionRotPosEmb` 嵌套 for + std::cos/sin
- **方案**: Stage A 算 (row_idx, col_idx) + freq_table — Stage B NPU graph 做 2× Gather + 2× Concat + Cos + Sin
- **算子需求**: 新增 ElewiseOp::MakeCos/MakeSin（已添加）；发现 ATB Concat 只支持 fp16/bf16
- **实测收益**:
  | 分辨率 | vision_pos 阶段 (vs P1 后) |
  |--------|---|
  | 224×224 | 0.89 → 0.75 ms |
  | 416×672 | 3.04 → 0.92 ms |
  | 672×416 | 3.02 → 0.91 ms |
  | 896×896 | 8.03 → 1.25 ms |

#### P3: Deepstack InjectFeatures → NPU IndexAdd
- **Commit**: `395b6b8`
- **CPU 干预点**: 每个 image token 单行 NPU→CPU→add→NPU 往返；text decoder 每层之间触发显式 sync
- **方案**: 用 ATB IndexAdd 算子在 NPU 上一次性 scatter-add
- **算子需求**: 新增 IndexAddOp（4 输入: var, indices, updates, alpha——通过试错发现）
- **意外发现**: text_model 从 300ms 跌到 42ms（远超直接节省）。原因是 InjectFeatures 内的逐行同步在每层都阻塞了 stream，移除后整个 28 层 decoder 异步流水起来了。
- **实测收益（端到端）**:
  | 分辨率 | E2E |
  |--------|---|
  | 224×224 | 39.32 → 23.37 ms |
  | 416×672 | 133.70 → 44.73 ms |
  | 672×416 | 133.27 → 43.57 ms |
  | 896×896 | 355.85 → 100.98 ms |

---

### ✅ 已完成

#### P5: Deepstack 端到端 device-resident
- **Commit**:（待 git commit）
- **CPU 干预点**: vision 阶段每个 deepstack 层 1 次显式 sync + D↔H 往返（896×896 下 9.4 MB）
- **方案**: `std::vector<std::vector<uint16_t>>` → `std::vector<NpuTensor>`，删 sync/CopyToHost/CopyToDevice。ExtractFeatures 直接把 NPU tensor move 到容器，InjectFeatures 直接消费。
- **算子需求**: 无（IndexAdd 已在 P3 就绪）
- **实测收益**:
  | 分辨率 | P3 E2E | P5 E2E | 节省 |
  |--------|-------:|-------:|-----:|
  | 224×224 | 23.37 ms | 22.36 ms | **-4.3%** |
  | 416×672 | 44.73 ms | 39.88 ms | **-10.8%** |
  | 672×416 | 43.57 ms | 40.48 ms | **-7.1%** |
  | 896×896 | 100.98 ms | 88.63 ms | **-12.2%** |
- **全部超越预期**（89 ms vs 93 ms 预期），收益随分辨率增长——数据量越大，减少 D↔H 往返的收益越大
- **T8 精度验证**: cosine: TEXT 1.000000 / IMAGE 0.999489 / BOTH 0.999857（与 P3 基线完全一致，零回归）

#### 🛠️ Benchmark 框架修补（T8c，同步完成）
- Preprocess 时间现在正确计入 `StageTimings`（之前永远为 0）
- 新增 `--mode bench`，一键跑 4 个固定分辨率

---

### ⏳ 候选未来项

#### P4 (验证): ExecuteGraph 内部 Synchronize 的真实开销
- **位置**: `src/families/base_model.cpp:57` 每个 ATB op Execute 后都强制 Synchronize
- **疑问**: P3 已证明 InjectFeatures 内的 sync 移除后 text decoder 跳跃 7×。但 ExecuteGraph 自带的 sync 仍在每个 op 后调一次（~70 次/inference）。
- **理论**: 单一 stream 上 sync 已排空 stream 时几乎免费（< 1μs）；但如果有其他 sync 源同时存在，叠加可能可观。
- **方案**: 加 `ATB_FORCE_SYNC` 环境变量开关，benchmark 对比。如果有显著收益 → 改默认行为。
- **优先级**: 低（先做 P5）

#### P6: Vision Merger → image token 注入端到端 NPU 化
- **位置**: `qwen3vl_model.cpp:411-451` (ForwardWithTiming) 和 `qwen3vl_model.cpp:641-647` (PrepareInputs)
- **当前路径**: merger 输出 `CopyToHost` → `vis_embeds_host` → CPU memcpy scatter → `inputs_embeds` → `CopyToDevice`
- **数据量**: 896×896 下 vis_embeds (~3.2 MB) + inputs_embeds (~3.6 MB) = 6.8 MB 往返
- **方案**: 需要 NPU **ScatterUpdate / IndexPut**（写入而非累加）算子
  - 已知 ATB 有 IndexAdd（累加），但还没确认有 IndexPut（覆盖）
  - 候选 workaround: `inputs_embeds[positions] = 0; IndexAdd(inputs_embeds, positions, vis_embeds)` 用两步 IndexAdd
- **算子需求**: 调研 ATB 是否有 ScatterUpdate；若无则用 workaround
- **预期收益**: ~2-3 ms
- **优先级**: 中

#### P7: MRoPE cos/sin → NPU
- **位置**: `mrope.cpp:MRoPE::Compute` + `qwen3vl_model.cpp:481-512`
- **当前耗时**: 896×896 下 ~1.5 ms（包含 fp32→fp16 + H2D）
- **方案**: 仿照 P2 拆 Stage A (host) + Stage B (NPU graph)。复杂度高于 P2 因为 interleaved（按 mrope_section 隔几位换值）。
- **算子需求**: 已有 (Gather, Mul, Concat, Cos, Sin)
- **预期收益**: ~1.5 ms
- **优先级**: 低（收益较小）

#### P8: Causal Mask → NPU 生成或隐式
- **位置**: `runners/text_runner.cpp:MakeCausalMask` + 上传
- **当前耗时**: 896×896 下 ~0.5 ms
- **方案 A**: NPU graph 生成（用 Arange + 比较 + Cast）
- **方案 B**: 用 ATB self-attention 的 implicit causal mask 选项（如果支持）
- **预期收益**: ~0.5 ms
- **优先级**: 低

#### Positions 复用
- **位置**: `deepstack_fusion.cpp:InjectFeatures` 每次都把 positions 上传一次（28 decoder 层 × 3 deepstack = 84 次小拷贝，每次 ~kb）
- **方案**: 在 PrepareInputs 时一次上传 positions 到 NPU，DeepstackFusion 直接消费
- **预期收益**: 微小（< 0.5 ms）
- **优先级**: 极低

#### Text Embedding Table → NPU?
- **位置**: `qwen3vl_model.cpp:EmbeddingLookup`
- **数据**: vocab=151936 × hs=2048 × 2B = **622 MB**（**比所有其他权重加起来还大**）
- **结论**: ❌ 不可行（NPU 内存太宝贵，不值得占用）
- **保持现状**: CPU lookup 然后 H2D。当前已经只 H2D 一次。

---

## 实测性能演变表

| 分辨率 | Baseline | After P1 | After P2 | After P3 | After P5 (实测) |
|--------|---------:|---------:|---------:|---------:|---------------:|
| 224×224 |  60.08 ms |  39.75 |  39.32 |  23.37 |  **22.36** |
| 416×672 | 213.15 ms | 134.59 | 133.70 |  44.73 |  **39.88** |
| 672×416 | 212.32 ms | 135.27 | 133.27 |  43.57 |  **40.48** |
| 896×896 | 581.51 ms | 362.49 | 355.85 | 100.98 |  **88.63** |

最终加速（vs baseline）: 896×896 约 **6.6×**（超越预期 6.3×）

---

## 精度回归基线

`tests/test_accuracy.py` 三模式 cosine（vs Python ATB 参考）:

| Mode | Baseline | P1 | P2 | P3 | P5 | 阈值 |
|------|---------:|---:|---:|---:|---:|-----:|
| TEXT_ONLY | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ≥ 0.99 |
| IMAGE_ONLY | 0.999594 | 0.999594 | 0.999489 | 0.999489 | 0.999489 | ≥ 0.99 |
| IMAGE_AND_TEXT | 0.999844 | 0.999844 | 0.999857 | 0.999857 | 0.999857 | ≥ 0.99 |

**绝不通过降低阈值来"通过"测试。** 每个优化项必须保持 ≥ 0.99。

---

## 测试金字塔（每个 P 都遵循）

- **Level 1 (Host pure-function)**: 若 Stage A 可独立测，与 Python reference byte-exact 比对
- **Level 2 (NPU op precision)**: 新增的 NPU graph 与 Python reference cosine ≥ 0.999
- **Level 3 (E2E)**: `test_accuracy.py` 三模式 ≥ 0.99

参考: `docs/TEST_STRATEGY_GUIDE.md`
