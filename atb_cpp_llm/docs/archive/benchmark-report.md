# C++ vs Python ATB — P8 子阶段性能明细

> 此报告的最终基准数据已整合至 [refactoring-plan.md](../refactoring-plan.md) §3（Phase 19）。
> 本文档保留子阶段性能明细、缓存行为分析、性能演变历史作为补充参考。
> 基准日期: 2026-06-08 | 优化状态: P1+P2+P3+P5+P8 全部完成

---

## C++ 子阶段性能明细

### TEXT_ONLY

| 阶段 | S=100 | S=512 | S=1024 | S=2048 | S=4096 |
|------|------:|------:|-------:|-------:|-------:|
| Text Embed+Inj | 0.10 | 0.49 | 1.01 | 2.04 | 4.11 |
| Position IDs | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 |
| Text Model | 11.98 | 20.93 | 33.61 | 60.12 | 122.02 |
| Pooling | 0.12 | 0.13 | 0.13 | 0.15 | 0.14 |
| **E2E** | **12.20** | **21.56** | **34.76** | **62.32** | **126.29** |

### IMAGE_ONLY (warm cache)

| 阶段 | 416×672 | 720×1280 | 1080×1920 | 1440×2560 |
|------|--------:|---------:|----------:|----------:|
| Preprocess | 129.25 | 475.94 | 676.15 | 689.08 |
| Vision PosEmb | 0.92 | 1.28 | 1.46 | 1.50 |
| Vision Model | 17.08 | 47.27 | 71.74 | 71.98 |
| Text Embed+Inj | 0.26 | 0.87 | 1.27 | 1.24 |
| Position IDs | 0.20 | 0.62 | 0.85 | 0.71 |
| Text Model | 19.49 | 32.73 | 40.03 | 39.87 |
| Pooling | 0.13 | 0.14 | 0.14 | 0.14 |
| **E2E (no preprocess)** | **38.08** | **82.90** | **115.50** | **115.44** |

### Vision/Text 子阶段（Image-Only）

| 子阶段 | 416×672 | 720×1280 | 1080×1920 | 1440×2560 |
|--------|--------:|---------:|----------:|----------:|
| Vision FirstLayer | 0.75 | 1.95 | 2.54 | 2.50 |
| Vision Blocks (1..23) | 14.81 | 40.96 | 64.77 | 65.28 |
| Vision Merger+D2H | 1.35 | 3.05 | 4.43 | 4.20 |
| Text H2D prep (warm) | 0.22 | 0.31 | 0.37 | 0.41 |
| Text 28 layers | 18.37 | 30.68 | 38.62 | 38.47 |
| Text Norm+D2H | 0.27 | 0.48 | 0.46 | 0.48 |

---

## 缓存行为分析

P8 引入了按 seq_len 的 mask/cos/sin NPU 缓存。以下对比冷启动（首次推理，cache miss）和热缓存（后续推理，cache hit）的性能差异：

| Mode | S | Warm E2E | Cold H2D | Cold Frame≈ | Cold/Warm |
|------|---|---------:|---------:|-----------:|----------:|
| TEXT 100 | 100 | 12.20 | 0.36 | ~12.56 | 1.03× |
| TEXT 512 | 512 | 21.56 | 0.81 | ~22.37 | 1.04× |
| TEXT 1024 | 1024 | 34.76 | 1.17 | ~35.93 | 1.03× |
| TEXT 2048 | 2048 | 62.32 | 3.67 | ~66.00 | 1.06× |
| **TEXT 4096** | **4096** | **126.29** | **43.13** | **~169** | **1.34×** |
| IO 1080×1920 | 1222 | 115.50 | 3.37 | ~118.87 | 1.03× |

- 短序列（S < 2048）：冷启动额外开销可忽略（< 5ms）
- **S=4096 冷启动额外 ~43ms**（mask 16.8M 元素 fp16 生成 + 33.6MB H2D），约为 warm 的 34%
- 同 seq_len 连续推理时完全命中缓存，H2D prep < 0.5ms

---

## 性能演变（vs Baseline）

| 分辨率 | Baseline | P1 | P2 | P3 | P5 | **P8** | 总加速 |
|--------|---------:|---:|---:|---:|---:|------:|-------:|
| 224×224 | 60.08 ms | 39.75 | 39.32 | 23.37 | 22.36 | **21.05** | **2.9×** |
| 416×672 | 213.15 ms | 134.59 | 133.70 | 44.73 | 39.88 | **38.08** | **5.6×** |
| 896×896 | 581.51 ms | 362.49 | 355.85 | 100.98 | 88.63 | **75.33** | **7.7×** |
| S=4096 Text | — | — | — | — | 399.49 | **126.29** | **—** |

---

## 三方对比（P8 跑分，完整数据见 refactoring-plan.md Phase 19）

> P8 跑分的三方对比表已整合至 [refactoring-plan.md §3](../refactoring-plan.md)。Phase 19 在此基础上更新了 ~500-token MM 文本、修复了 fp16 比特重解释 bug，数据更准确。

**P8 关键发现**:
1. C++ 全部 13/13 ≤ Python（speedup 0.54×–0.86×）
2. S=4096: C++ 126ms vs Python ATB 153ms vs TF 321ms — P8 mask 缓存消除了原来 212ms 的 H2D 瓶颈
3. 大分辨率 IO/MM: C++ 快 14-17% — C++ 优势来自更轻的 CPU 路径
4. 1080×1920 和 1440×2560 经 SmartResize 归一化后 S 相同（1222）— 两者 E2E 一致
5. Cold H2D 在 S=4096 时 43ms（一次性开销，同 seq_len 后续推理无此偏移）

---

## 测试方法

- C++ ATB: `./benchmark --mode compare --iter 5 --warmup 3`
- Python ATB: `python tests/test_embedder_e2e.py --mode atb --bench --iter 5 --warmup 3`
- Python TF: `python tests/test_embedder_e2e.py --mode tf --bench --iter 5 --warmup 3`
- 对比: `python tests/test_embedder_e2e.py --mode both --bench`
- 相同输入保证: C++ `SavePixelValues` → Python load; Python `--save-tokens` → C++ `LoadTokenIds`
- Cosine: 比较两边的 pooler output（last token L2-normalized embedding）
- 阈值: cosine ≥ 0.99（绝不降低标准）
