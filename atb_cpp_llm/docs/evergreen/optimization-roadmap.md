# NPU Pipeline Optimization Roadmap

> 长期追踪文档：让 Qwen3VL Embedding 推理管线尽可能保持「入口一次 H2D，出口一次 D2H，中间全 NPU 异步流水」。

最后更新：2026-06-09（P4: NPU sync 语义实验结论 + P8 提交 + 13/13 精度基线）

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

## 🔥 逐阶段性能热点（2026-06-08 精确计时）

> 方法：在 `ForwardWithTiming` 中每个 NPU 阶段用 `runtime_->Synchronize()` 夹住，确保阶段时间反映真实 GPU 执行时间（而非仅 CPU 下发时间）。E2E 不计额外 sync 开销。

### S=4096 纯文本 (399.5 ms)

```
┌─────────────────┬──────────┬─────────┬──────────────────────────────────┐
│ 阶段             │ 时间(ms) │ 占比     │ 详情                              │
├─────────────────┼──────────┼─────────┼──────────────────────────────────┤
│ Text Embed+Inj  │    4.19  │   1.0%  │ CPU EmbeddingLookup 4096 tokens   │
│ Position IDs    │   59.51  │  14.9%  │ MRoPE + MakeCausalMask (CPU)      │
│ Text Model      │  335.63  │  84.0%  │ ↓ 详见子阶段                      │
│  ├ H2D prep     │ 211.72   │  53.0%  │ 🔥 52MB拷贝 + 17.8M次fp32→fp16    │
│  ├ 28 layers    │ 119.95   │  30.0%  │ 平均 4.28ms/层 @ S=4096           │
│  └ Norm+D2H     │   3.93   │   1.0%  │ FinalNorm + D2H (33.6MB mask)     │
│ Pooling         │    0.15  │   0.0%  │ Last token + L2 norm              │
├─────────────────┼──────────┼─────────┼──────────────────────────────────┤
│ E2E             │  399.49  │  100%   │                                  │
└─────────────────┴──────────┴─────────┴──────────────────────────────────┘
```

**H2D prep 内部分析**（211.72 ms）：
- Causal mask: 4096² = 16.8M float32 → 分配 + 填充 + fp32→fp16 逐个转换 → 约 170ms
- cos/sin: 4096×128 = 524K × 2 = 1M float32 → fp32→fp16 逐个转换 → 约 10ms
- inputs_embeds CopyToDevice: 4096×2048×2B = 16.8MB → 约 5ms
- mask CopyToDevice: 4096²×2B = 33.6MB → 约 10ms
- cos/sin CopyToDevice: 各 1MB → 约 2ms
- 其余：EnsureBuilt（首次）+ AllocNpuFloat16 开销

### 896×896 纯图像 (86.8 ms, S=784, patches=3136)

```
┌─────────────────┬──────────┬─────────┬──────────────────────────────────┐
│ 阶段             │ 时间(ms) │ 占比     │ 详情                              │
├─────────────────┼──────────┼─────────┼──────────────────────────────────┤
│ Vision PosEmb   │    1.56  │   1.8%  │ ✅ P1+P2 NPU化后，占比可忽略      │
│ Vision Model    │   41.53  │  47.9%  │ ↓ 详见子阶段                      │
│  ├ FirstLayer   │    1.67  │   1.9%  │ patch_embed + pos_embed + block 0 │
│  ├ Blocks(1..23)│   35.14  │  40.5%  │ 23 blocks × 1.53ms/block          │
│  └ Merger+D2H   │    3.94  │   4.5%  │ merger + D2H 拷贝                 │
│ Text Embed+Inj  │    0.79  │   0.9%  │ CPU memcpy 注入 (273 tokens)     │
│ Position IDs    │    2.69  │   3.1%  │ MRoPE + mask (S=784, 小)          │
│ Text Model      │   40.06  │  46.2%  │ ↓ 详见子阶段                      │
│  ├ H2D prep     │   10.06  │  11.6%  │ fp32→fp16 + 拷贝 ~5MB             │
│  ├ 28 layers    │   29.51  │  34.0%  │ 平均 1.05ms/层 @ S=784            │
│  └ Norm+D2H     │    0.49  │   0.6%  │ FinalNorm + D2H                   │
│ Pooling         │    0.14  │   0.2%  │                                  │
├─────────────────┼──────────┼─────────┼──────────────────────────────────┤
│ E2E             │   86.76  │  100%   │                                  │
└─────────────────┴──────────┴─────────┴──────────────────────────────────┘
```

### 4 分辨率 Scaling 对比

| 阶段 | 224² (S=49) | 416×672 (S=273) | 672×416 (S=273) | 896² (S=784) |
|------|-----------:|----------------:|----------------:|-------------:|
| Vision PosEmb | 0.78 ms | 0.88 ms | 0.91 ms | 1.21 ms |
| Vision Model | 9.42 ms | 16.73 ms | 17.22 ms | 40.34 ms |
| Text Embed+Inj | 0.18 ms | 0.62 ms | 0.60 ms | 2.56 ms |
| Position IDs | 0.12 ms | 0.60 ms | 0.61 ms | 2.54 ms |
| Text Model | 11.83 ms | 20.48 ms | 20.82 ms | 40.68 ms |
| **E2E** | **22.42 ms** | **39.39 ms** | **40.26 ms** | **87.42 ms** |

### 🎯 三大核心发现

1. **🔥 S=4096 H2D prep = 212ms (53%)** — Causal mask 是 4096²=16.8M float32，分配+填充+fp32→fp16 逐个转换+上传占了超过一半时间。Python 快 2.7× 主要受益于 mask 处理策略不同。

2. **Position IDs = 59.5ms (14.9%)** — `MakeCausalMask` 分配 16.8M float32 并填充 + `MRoPE::Compute` 计算 cos/sin。纯 CPU 瓶颈。

3. **28 text layers = 120ms (30%)** — 每层 4.28ms @ S=4096，与 Python ATB 基本持平（~120ms）。说明 NPU compute 部分两边一致，差距全在 H2D 准备阶段。

### Python vs C++ 差距根因（S=4096: 152ms vs 415ms）

| 组件 | C++ (ms) | Python (ms) | 差距分析 |
|------|---------:|------------:|---------|
| H2D prep (mask+cos+sin) | **212** | ~5 | Python mask 直接 fp16 生成/复用 |
| 28 text layers | 120 | ~120 | 持平（同 ATB） |
| Position IDs + Embed | 64 | ~27 | MRoPE + Mask 生成 |
| **Total** | **399** | **152** | |

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
- **Commit**: `60db5bb`
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
- 新增 `--mode compare`，13 组合完整对比矩阵 + 输出 .bin 文件

#### 🛠️ 逐阶段精确计时（2026-06-08）
- `ForwardWithTiming` 中每个 NPU 阶段边界添加 `runtime_->Synchronize()` 确保阶段时间反映真实 GPU 执行时间
- 新增 Vision 子阶段计时：FirstLayer / Blocks(1..23) / Merger+D2H
- 新增 Text 子阶段计时：H2D prep / 28 layers / Norm+D2H

#### P8 (P0): Causal Mask fp16 直出 + NPU 缓存复用
- **Commit**: `9a73444`
- **CPU 干预点**: `ForwardWithTiming` 中 mask/cos/sin 的 fp32→fp16 转换循环 + 每次推理重新生成
- **方案**:
  - A: `MakeCausalMaskFp16` 直接输出 fp16 mask（跳过 float32 中间体 + 转换循环）
  - B: `MRoPE::ComputeFp16` 直接输出 fp16 cos/sin（跳过转换循环）
  - C: 按 seq_len 缓存 mask_npu / cos_npu / sin_npu NPU tensors，同 seq_len 直接复用
- **算子需求**: 无（纯 CPU 优化 + NPU tensor 生命周期管理）
- **实测收益（S=4096 TEXT-ONLY）**:
  | 指标 | Before | After | 节省 |
  |------|-------:|------:|-----:|
  | H2D prep | 211.72 ms | 1.37 ms | **-99.4%** |
  | Position IDs | 59.51 ms | 0.02 ms | **-99.9%** |
  | E2E | 399.49 ms | **127.75 ms** | **-68.0% (3.13×)** |
- **C++ vs Python S=4096**: 128ms vs 152ms → **C++ 反超 16%**
- **S=784 (896×896)**: H2D prep 10.06ms → 0.35ms，E2E 86.76ms → 75.33ms (-13.2%)
- **测试**: `test_causal_mask_fp16` (24 assertions PASS — byte-exact vs fp32→fp16 转换)
- **精度**: 零回归（mask 生成逻辑不变，只是跳过中间 float32 分配）

#### ✅ P4: NPU Per-Op Sync 语义实验
- **Commit**: Phase 17（P0-P3 连续提交链 `65d8acc`~`1e039df`）
- **实验目标**: 验证每个 ATB ExecuteGraph 后强制 Synchronize 的开销，以及是否可以安全移除
- **实验方法**:
  - 在 `base_model.cpp` 中加 `ATB_DISABLE_PER_OP_SYNC` 环境变量控制 per-op sync
  - 对比 sync=1 和 sync=0 两种模式的 benchmark（13/13 cosine + 性能）
- **关键发现 — sync 不是"加了更安全"**:
  1. **H2D/D2H 自带同步**: `aclrtMemcpy`（H2D/D2H）在 Ascend 上是同步的，额外 sync 引入 NPU stream 上隐式 barrier 打断 ATB graph 内部多 stream 调度
  2. **额外 sync 会破坏精度**: 在 vision pixel `CopyToDevice` 和 FirstLayer `ExecuteGraph` 之间加 `runtime_->Synchronize()`，导致 block 1 之后所有输出 cosine 断崖式下降（block 23 cosine → 0.28）。多次确认该现象稳定复现
  3. **NPU 上"多加 sync 提高安全性"是反模式**: 单 stream 上已排空时 sync 几乎免费（< 10μs），59 次 × 10μs ≈ 0.6ms 可忽略。但如果在多 stream 调度区域加 sync，会破坏 graph compiler 的异步流水线优化
- **结论**: 
  - deepstack_fusion.cpp 中的 per-op sync 必须保留（P5 性能收益依赖正确调度）
  - 其他位置**不加**额外 sync，除非有明确的同步需求且经过 13/13 精度验证
- **经验教训**: 见 [lessons-learned.md 主题1第3条](../lessons-learned.md#主题-1精度调试)（NPU sync 不是"加了更安全"）
- **相关测试**: ✅ G5 `test_sync_safety.{cpp,py}`（2026-06-09）— per-op sync 和 timing sync 均可安全移除（cosine=1.000 bit-exact），deepstack 和 D2H sync 必须保留

> **后续演进（Batch 1, 2026-06-21）**：上述结论已部分超越。
> - **H1**：per-op sync 默认从 ON 翻转为 **OFF**。env 名从 opt-out `ATB_DISABLE_PER_OP_SYNC` 改为 opt-in `ATB_ENABLE_PER_OP_SYNC`（旧名不再读取）。A/B 实测默认 async 拿到 12–13% e2e 收益（text 65.5→57.1ms，io/mm 115.7→100.6ms，stddev 0.71→0.04）。
> - **H4**：deepstack `InjectFeatures` 的 `sync=true` **已移除**——IndexAdd 与其生产者同在单 ATB stream，FIFO 排序保证输入就绪，forward 末尾 FinalNorm sync 兜底正确性。结论中「deepstack per-op sync 必须保留」不再成立。
> - **D2H sync 仍必须保留**且强化：ATB 跑在独立 stream，同步 `aclrtMemcpy` D2H **不**等待 ATB 待决 work，故 vision merger / text FinalNorm 的 `CopyToHost` 前必须有显式 `Synchronize()`（否则 IMAGE_ONLY cos 崩至 ~0.19）。`RunVision` 同步补齐；`debug::DumpNpuFp16` 的 sync 移到 D2H 之前修复 debug 保真度。

---

### ⏳ 候选未来项（按优先级排序）

#### 🟡 P8-C: Causal Mask → NPU 生成
- **当前状态**: P8-A + P8-B 已完成，mask 已经在 CPU 上 fp16 直出 + NPU 缓存
- **方案**: NPU graph 生成 mask（用 Arange + 比较 + Cast），完全消除 CPU 参与和 H2D
- **剩余收益**: 首次生成本身 ~43ms（仅第一帧），后续缓存命中已 0ms
- **优先级**: 🟢 低（P8-A/B 已解决最痛的点）

#### 🟡 P6: Vision Merger → image token 注入端到端 NPU 化
- **位置**: `qwen3vl_model.cpp:453-469` (ForwardWithTiming)（PrepareInputs 已在 Phase 17 删除，其逻辑已内联到 ForwardWithTiming）
- **当前路径**: merger 输出 `CopyToHost` → `vis_embeds_host` → CPU memcpy scatter → `inputs_embeds` → `CopyToDevice`
- **数据量**: 896×896 下 vis_embeds (~3.2 MB) + inputs_embeds (~3.6 MB) = 6.8 MB 往返
- **方案**: 需要 NPU **ScatterUpdate / IndexPut**（写入而非累加）算子
  - 已知 ATB 有 IndexAdd（累加），但还没确认有 IndexPut（覆盖）
  - 候选 workaround: `inputs_embeds[positions] = 0; IndexAdd(inputs_embeds, positions, vis_embeds)` 用两步 IndexAdd
- **算子需求**: 调研 ATB 是否有 ScatterUpdate；若无则用 workaround
- **预期收益**: ~2-3 ms
- **优先级**: 🟡 中

#### 🟢 P7: MRoPE cos/sin → NPU
- **位置**: `mrope.cpp:MRoPE::Compute` + `qwen3vl_model.cpp:500-504`
- **当前耗时**: S=4096 下 ~59ms（包含 `GetRopeIndex` + `MRoPE::Compute` + `MakeCausalMask`）。纯 MRoPE::Compute 约 15-20ms
- **方案**: 仿照 P2 拆 Stage A (host) + Stage B (NPU graph)。复杂度高于 P2 因为 interleaved（按 mrope_section 隔几位换值）
- **算子需求**: 已有 (Gather, Mul, Concat, Cos, Sin)
- **预期收益**: ~15 ms
- **优先级**: 🟢 低（先做 P8 mask 优化，mask 消除后 Position IDs 耗时会降到 ~30ms，MRoPE 占比上升）

#### P9: Text Embedding H2D 异步化
- **位置**: `qwen3vl_model.cpp:517-518` inputs_embeds CopyToDevice
- **当前耗时**: S=4096 下 16.8MB H2D 约 5ms（P8 后已缓存，仅首次发生）
- **方案**: H2D 操作已在 Ascend 上自带同步（P4 实验确认）。当前 CopyToDevice 已异步于后续 NPU 计算
- **预期收益**: 极小（< 1ms，P8 缓存已解决重度问题）
- **优先级**: 极低

#### Positions 复用
- **位置**: `deepstack_fusion.cpp:InjectFeatures` 每次都把 positions 上传一次（28 decoder 层 × 3 deepstack = 84 次小拷贝，每次 ~kb）
- **方案**: 在 ForwardWithTiming 中 early stage 一次上传 positions 到 NPU，DeepstackFusion 直接消费（PrepareInputs 已在 Phase 17 删除）
- **预期收益**: 微小（< 0.5 ms）
- **优先级**: 极低

#### Text Embedding Table → NPU?
- **位置**: `qwen3vl_model.cpp:EmbeddingLookup`
- **数据**: vocab=151936 × hs=2048 × 2B = **622 MB**（**比所有其他权重加起来还大**）
- **结论**: ❌ 不可行（NPU 内存太宝贵，不值得占用）
- **保持现状**: CPU lookup 然后 H2D。当前已经只 H2D 一次。

#### 🟡 P10: Preprocess 加速（Batch 2，2026-06-22 启动）

**权威基准源**（任何预处理改动对齐这个，不要自己定语义）：
- 官方实现 `/mnt/workspace/gitCode/Qwen3-VL-Embedding`（模型权重/推理脚本）调 transformers + qwen_vl_utils。
- 图像 resize 权威链路在 `/mnt/workspace/gitCode/Qwen3-VL/qwen-vl-utils/src/qwen_vl_utils/vision_process.py:140`：`Image.open → to_rgb → smart_resize → image.resize((w,h))`，**`image.resize` 不传 resample 参数 → PIL 默认 `Image.BICUBIC`**。这就是官方预处理 resize 的真值来源。
- transformers Qwen3VL `image_processing_qwen3_vl` / `video_processing_qwen3_vl` 同样 `resample = PILImageResampling.BICUBIC`。
- Python 参考侧 `atb_python_qwen3vl_embedding/preprocess.py` 用 torch `F.interpolate(mode='bicubic', align_corners=False)`，与 PIL 在 boundary handling 上有可观察差异（自然图像上 cosine 仍 ≥0.999）。**PIL（`vision_process.py:140`）是最权威基准**。
- PIL resize 内部实现参考：`docs/archive/Resample.c`（PIL 官方 libImaging/Resample.c）。

**当前瓶颈**（910B 实测，1080×1920）：
| 阶段 | 耗时 | 占比 |
|---|---|---|
| bicubic | 516ms | 78.5% |
| patch extraction | 116ms | 17.6% |
| normalize | 25ms | 3.8% |
| smartresize | ~0 | 0% |
| **total** | **657ms** | |

根因：当前 `qwen3vl_preprocess.cpp:BicubicResize` 不可分离（4×4=16 tap/像素）+ 每像素 16 次浮点 `CubicWeight`（含 fabs/floor/分支）+ 运行时 clamp 分支。PIL Resample.c 的优化正好是反面：两阶段可分离（16→8 tap）+ 系数预算表（运行时零三角函数）+ 8bpc 定点整数 + clip8 查找表。

**数据搬运不是瓶颈**：当前 H2D fp16 pixel_values 最大 14MiB（<1ms），ATB 流程 H2D uint8 原图更小（1.36–4×减少）。657ms 是 CPU compute，不是搬运。

**ATB 化方向约束**（重要，勿走偏）：
- ATB 层（`atb/infer_op_params.h`）**无 resize/interpolate 算子**。
- aclnn 有 `aclnnUpsampleBicubic2d`（语义=torch，a=-0.75 Mitchell、无 antialias、align_corners=False）和 `aclnnResize`。**【2026-06-22 纠正】** 早先记录"仅 910b kernel、无 310P"是错的——回查 CANN 9.0.0 官方文档产品支持表：Atlas 训练系列（910B）√ **与** Atlas 推理系列（310P）√ **都支持**。aclnn 路径跨平台可用，不再是移植性硬阻塞。（教训见 lessons-learned 主题 7 第 11 条：文档里的平台/能力结论是易过期事实，使用前回查权威源。）
- **ATB 版预处理应优先用 ATB 基础算子组合计算图**（reshape/transpose/concat/elewise 等，引擎 `src/ops/` 已封装），而非直接调 aclnn——这是项目既有方向。aclnn 作为跨平台加速候选（P10-B 已验证精度可用）。

**三阶段计划**：
- **P10-A（CPU PIL 式重写，零移植风险，确定收益）** ✅：移植 Resample.c 三大手法到 `qwen3vl_preprocess.cpp`——可分离两阶段卷积 + 系数预算表 + normalize 融合进 vertical pass 输出 + patch `f` 帧去重。实测 657→141ms@1080×1920（4.7×），x86/ARM 通用、零新库、数学等价（edge-clamp Catmull-Rom，对齐现有 test_preprocess_cpu 阈值）。**作为后续 ATB 化的 baseline。**
- **P10-B（aclnn 官方插值，双平台，AA 910B 特化）** ✅ 精度闸口+工程化+性能实测完成（Developer→Reviewer→Re-review 闭环，多轮）：`aclnnUpsampleBicubic2d`（非 AA）在降采样 1080/1440→832 时 vs PIL 仅 cos=0.987/0.958 ❌，**`aclnnUpsampleBicubic2dAA`（含抗混叠预滤波）rescues P10-B**，降采样从 0.958 拉回 0.999996 ✅。AA 仅支持 910B（CANN 商用版产品表：Atlas 推理系列=310P ×）。**双路径**：910B → `NpuBicubicResizeAA`（4/4 ≥0.99998）；310P → 非 AA（2/4 通过，降采样 case P10-A CPU 兜底）。**（310P 降采样已被 P10-C 取代为 small-op AA 拼装，见下。）**`PreprocessImageNpu` 6 步管线实现（SmartResize→H2D→AA/非AA Bicubic→3×Elewise normalize→D2H→CPU patch），goto-cleanup 内存安全，AA 降采样条件守卫（恒等/上采样走非 AA）。**性能实测 NPU vs P10-A CPU geomean 1.4×**（416×672 1.7× / 720×1280 1.9× / 1080×1920 1.4× / 1440×2560 0.9×）。
- **P10-B 后续优化（待启动，按收益/难度排序）**：
  - **#2 跳过恒等 resize（已证伪，放弃）**：原假设 `new_h==height && new_w==width` 时跳过 bicubic 可省 aclnn 调用 + 避免 Mitchell 核恒等平滑。**A/B 实测证伪**：bicubic 核（Mitchell/Catmull-Rom）满足 `W(0)=1, W(±1)=0, W(±2)=0`，恒等尺度下源采样映射为整数、核退化为单位冲激，`aclnnUpsampleBicubic2d` 精确返回输入（cos/max_diff 字节相同），无平滑可避免；省的 ~0.1ms aclnn 调用被噪声淹没。**根因**：假设 bicubic 核在恒等下会平滑图像是错的。聚焦 #1。
  - **#1 Patch extraction NPU 化（最大收益，消除 D2H 搬运）**：当前归一化 fp16 tensor D2H 回 CPU 做 patch 重排，是 1440×2560 退化 0.9× 的根因（D2H 搬 7.5MB + 引擎后续 H2D 再搬 7.5MB）。NPU 化后 pixel_values 留 device，引擎下游直接消费，消除往返搬运。
    - **闸口全过（2026-06-23 spike）**：① ATB `TransposeOp` 7 维 permute `[1,4,2,5,0,3,6]` bit-exact；② **8 维 perm `[2,5,3,6,1,0,4,7]`（含 tp 维）bit-exact**，Python 交叉验证等价 9 维 `.permute(0,3,6,4,7,2,1,5,8)`；③ **`AsStridedParam` stride=0 广播 tp 帧确认可用**（`infer_op_params.h:133`，`atb::CreateOperation` 模板创建）；④ 完整 patch 管线 `[C,new_h,new_w] → AsStrided(stride=0 tp) → 8维 Transpose → Reshape` vs CPU **bit-exact（cos=1.0）**，64 assertions PASS。`test_patch_transpose_spike.cpp`。
    - **NPU 方案确立**：AsStrided(stride=0 广播 tp) + Transpose(8 维 perm `[2,5,3,6,1,0,4,7]`) + Reshape，全程 device 内、零 D2H。
    - **约束（已查 ATB 源码 `ascend-transformer-boost-v9.0.0`）**：ATB `TransposeOp` 的 `Dims.dimNum ∈ (0,8]`（`types.h:31 MAX_DIM=8`）。Patch 9 维 permute squeeze grid_t=1 后 8 维（含 tp）正好 ≤8 限制。
    - 验收：patch 输出 cos vs CPU 路径 ≥0.999（spike 已 bit-exact）；4 分辨率性能，1440×2560 从 0.9× 拉到 >1.0×。
- **P10-C（ATB 基础算子组合 PIL resample）** ✅（2026-06-29，Batch A/B/C，Developer→Reviewer 破坏者→Re-review 闭环）：原为 aclnn 不可用时的 long-shot 储备；因 **310P 上 `aclnnUpsampleBicubic2dAA` 不支持**（aclnnStatus=561103），降采样需要 AA 才能对齐 PIL（非 AA 1080/1440 仅 0.987/0.958），故落地为 310P 降采样生产路径。
  - **实现**：`src/components/vision/smallop_bicubic_aa.{h,cpp}` `NpuBicubicResizeAASmallOp`。separable filtering 摊成稠密 fp16 矩阵 → 两次 MatMul（H pass `[C*H,Win]@W_h^T[out_w,in_w]` → Transpose `{0,1,3,2}` → V pass `[C*W',H]@W_v^T[out_h,H]` → 末 Transpose 回 NCHW）。权重 host 端按 PIL `precompute_coeffs`（Resample.c，bicubic a=-0.5，support=2.0×max(1,in/out)）算 fp32 系数（不做 8bpc 整数化/clip8），fp32→fp16 RNE。**per-axis skip**（`need_h=out_w!=in_w`、`need_v=out_h!=in_h`，恒等→D2D memcpy）。单算子用共享 `ExecuteOperation`（`base_model.h`），end-of-pipeline 单次 sync-before-free。
  - **精度**：spike vs PIL ground truth 416 恒等 1.0 / 720→704 0.999984 / 1080→992×1792 0.999980 / 1440 0.999995；H-only 64×128→64×64 直测 0.999998；端到端 small-op AA full-pipeline vs CPU PIL PreprocessImage 4 分辨率 cos=1.0/0.999924/0.999878/0.999950。Reviewer 独立 Python 复刻权重 + 篡改 bin fail-closed 验证证明与 910B aclnn AA 数值等价。
  - **性能基线（910B 实测，separable 5 算子单图，含 H2D 后→输出、不含 D2H，warmup 3 + 10 次均值）**：720×1280→704×1280（V-only）0.81 ms / 1080×1920→992×1792（双轴）3.72 ms / 1440×2560→992×1792（双轴）4.53 ms / 416×672 恒等 memcpy 0.01 ms。
  - **生产分发**（`qwen3vl_preprocess.cpp:438`）：`downsample` 时 910B→`NpuBicubicResizeAA`（aclnn 硬件 AA）、310P→`NpuBicubicResizeAASmallOp`（small-op AA）；非降采样→`NpuBicubicResize`（非 AA aclnn，跨平台）。
  - **后续：组图版性能对比**：当前 separable 是 5 个独立 `Setup/Execute`（无融合）。计划写 `NpuBicubicResizeAASmallOpGraph`（同算法，ATB GraphBuilder 串联 Linear×2+Transpose×2+Reshape 进一张图让 ATB 调度/融合优化），与 separable 基线 + CPU 版（已有 `PreprocessImage`）三方在 910B 对比性能，选最优接入生产分发；三者精度均应 cos≥0.99（组图与 separable 数值应一致，只是调度不同）。

**验收**：cosine ≥ 0.99 vs PIL（P10-B spike 已在 4 生产分辨率达标）；4 分辨率性能实测完成（vs P10-A CPU，geomean 1.4×）：416×672 1.7×/720×1280 1.9×/1080×1920 1.4×/1440×2560 0.9×（退化根因 H2D 开销，待 P10-B 后续 #1 优化）。全管线精度测试 cos≥0.999（非降采样/非 AA 路径）。

---

## 实测性能演变表

| 分辨率 | Baseline | After P1 | After P2 | After P3 | After P5 | After P8 |
|--------|---------:|---------:|---------:|---------:|---------:|---------:|
| 224×224 |  60.08 ms |  39.75 |  39.32 |  23.37 |  22.36 |  **21.05** |
| 416×672 | 213.15 ms | 134.59 | 133.70 |  44.73 |  39.88 |  **37.11** |
| 672×416 | 212.32 ms | 135.27 | 133.27 |  43.57 |  40.48 |  **37.65** |
| 896×896 | 581.51 ms | 362.49 | 355.85 | 100.98 |  88.63 |  **75.33** |

### S=4096 Text-Only

| 指标 | Before P8 | After P8 | 节省 |
|------|----------:|---------:|-----:|
| H2D prep | 211.72 ms | 1.37 ms | -99.4% |
| Position IDs | 59.51 ms | 0.02 ms | -99.9% |
| 28 Layers | 119.95 ms | 119.70 ms | — |
| **E2E** | **399.49 ms** | **127.75 ms** | **-68.0%** |

最终加速（vs baseline）: 896×896 约 **7.7×**，S=4096 text 约 **3.1×**

**生产性能**（2026-06-09，chat-templated inputs，C++ ATB vs Python ATB）:
- C++ ATB geomean 领先 Python ATB **1.39×**、领先 Transformers **4.22×**
- 详见 [archive/refactoring-plan.md §3 当前性能基线](../archive/refactoring-plan.md#3-当前性能基线phase-192026-06-09)

---

## 精度回归基线

### 最新 13/13 全矩阵（2026-06-09，C++ ATB vs Python ATB，chat-templated inputs）

| Mode | S | VisTok | Cosine | Status |
|------|---|--------|-------:|--------|
| TEXT 100 | 100 | 0 | 0.999946 | PASS |
| TEXT 512 | 512 | 0 | 0.999946 | PASS |
| TEXT 1024 | 1024 | 0 | 0.999980 | PASS |
| TEXT 2048 | 2048 | 0 | 0.999894 | PASS |
| TEXT 4096 | 4096 | 0 | 0.999985 | PASS |
| IO 416×672 | 273 | 273 | 0.999832 | PASS |
| IO 720×1280 | 880 | 880 | 0.999962 | PASS |
| IO 1080×1920 | 1222 | 1222 | 0.999770 | PASS |
| IO 1440×2560 | 1222 | 1222 | 0.999916 | PASS |
| MM 416×672 | 940 | 273 | 0.999962 | PASS |
| MM 720×1280 | 1547 | 880 | 0.999943 | PASS |
| MM 1080×1920 | 1889 | 1222 | 0.999969 | PASS |
| MM 1440×2560 | 1889 | 1222 | 0.999958 | PASS |

最低 cosine: IO 1080×1920 = 0.999770，远高于 0.99 阈值。**可用于生产环境。**

### 历史三模式 reference 对比（Phase 16-17 期间）

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

参考: `docs/archive/testing-guide.md`（历史 310P 验证日志）、`docs/evergreen/platform-310p.md`（平台适配现况）
