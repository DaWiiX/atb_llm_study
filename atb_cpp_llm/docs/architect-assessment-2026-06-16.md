# 架构师评估报告：性能与生产就绪审查

**日期**: 2026-06-16  
**分支**: `fix/tests`  
**范围**: `atb_python_qwen3vl_embedding/` + `atb_cpp_llm/`  
**模型**: Qwen3VL-Embedding-2B on Ascend 910B  
**评估人**: Claude Code Architect (3 Explore + 4 Action + 1 Verify agent)

---

## 1. 执行摘要

对 Qwen3VL-Embedding-2B 推理框架进行了全面的性能、鲁棒性和生产就绪评估。派遣了 8 个 subagent 完成代码审查、基准测试和验证工作。

**总体评级：🟡 有条件可部署**

- **性能**: ✅ 优秀 — ATB 比 Transformers 快 **3.0-3.6x**，余弦相似度 ≥ 0.999
- **C++ 端**: 🟡 良好 — RAII 正确、错误传播完整，但有 3 个轻量修复项
- **Python 端**: 🔴 需改进 — 4 个 P0 问题（sync 竞态、无资源清理）和 52 个其他发现
- **Benchmark 代码**: 🟡 已修复 — 发现并修复了 E2E 对比的根因（cos 从 0.87 → 0.999）

---

## 2. 性能评估

### 2.1 Benchmark 运行结果 (2026-06-16)

| 分辨率 | VisTok | S | ATB E2E | TF E2E | 加速比 | Cosine |
|--------|--------|---|---------|--------|--------|--------|
| 416×672 | 273 | 299 | 59.9ms | 186.4ms | **3.1×** | 0.9999 |
| 720×1280 | 880 | 906 | 102.9ms | 307.0ms | **3.0×** | 0.9998 |
| 1080×1920 | 1222 | 1248 | 135.4ms | 486.7ms | **3.6×** | 0.9997 |
| 1440×2560 | 1222 | 1248 | 136.2ms | 486.1ms | **3.6×** | 0.9997 |

### 2.2 各阶段耗时分布 (ATB Staged, 1440×2560)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Preprocess (CPU) | 26.6ms | 15.6% |
| Vision Position Embedding | 4.1ms | 2.4% |
| **Vision Model (24 blocks)** | **78.3ms** | **45.9%** |
| Text Embedding + Injection | 3.4ms | 2.0% |
| Position IDs (MRoPE) | 1.9ms | 1.1% |
| Text Model (28 layers) | 56.1ms | 33.0% |
| **Staged Sum** | 170.3ms | 100% |
| **E2E (no internal sync)** | **136.2ms** | — |

### 2.3 性能结论

- **Vision Model 是主要瓶颈** — 占 staged time 的 46%，应作为后续优化的重点
- **加速比稳定在 3.0-3.6×** — 跨分辨率表现一致
- **1080p 和 1440p 的 grid 相同** `[1, 94, 52]` — smart_resize 限制了最大像素数
- **Text Model 28 层耗时 56ms** — 平均每层 2.0ms，优秀

---

## 3. Benchmark 代码质量

### 3.1 已修复问题

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| TF 对比调用 `ref()` 导致 deepstack 不对齐 | cos 0.87-0.91 | cos ≥ 0.999 |
| 使用 `ref.language_model()` 做公平对比 | ❌ | ✅ |

### 3.2 待改进项

| # | 严重程度 | 描述 |
|---|----------|------|
| B1 | P1 | 缺少 P50/P95/P99 尾延迟百分位（C++ 有 min/max/median 但无百分位） |
| B2 | P1 | 缺少 NPU 内存使用量测量（peak HBM、分配模式） |
| B3 | P2 | 缺少冷启动 benchmark（首次推理的 H2D 拷贝和 graph 编译开销） |
| B4 | P2 | 缺少吞吐量 benchmark（多 stream/批量并发） |
| B5 | P2 | 无自动化性能回归检测（无阈值告警） |
| B6 | P2 | C++ benchmark preprocess 时间只测一次并回填，未捕获迭代间差异 |
| B7 | P2 | `compare_py_cpp.py` 包含硬编码的旧 timing 数据（会随时间腐烂） |

---

## 4. C++ 端生产就绪评估

### 4.1 逐类别评估

| 类别 | 评级 | 关键发现 |
|------|------|----------|
| **内存管理** | ✅ PASS | RAII 正确（NpuTensor, OperationHandle, ContextHandle），移动语义正确，无双释放风险 |
| **错误处理** | ⚠️ FAIL | ATB 操作和 NPU 分配检查完整，但 13 处 `CopyToDevice`/`CopyToHost` 返回值未检查 |
| **输入验证** | ⚠️ FAIL | batch_size/seq_len/null 已验证，但缺 seq_len 上限、缺 gthw null guard (UB)、缺 output dim 验证 |
| **线程安全** | ✅ PASS | BufferPool 有 mutex，模型缓存非线程安全但有充分文档说明 |
| **文档完整性** | ⚠️ FAIL | 构建/API/故障排除/性能调优齐全，但**缺部署上线 checklist** |
| **测试覆盖** | ✅ PASS | 50 个 C++ 测试，5 级金字塔（L0-L4），51/51 CTest 通过 |

### 4.2 待修复项

| # | 严重程度 | 文件 | 问题 | 工作量 |
|---|----------|------|------|--------|
| G1 | MEDIUM | `qwen3vl_model.cpp:302` | `gthw` 为 null 且无 metadata 时 UB（生产环境罕见但应修复） | 1 行 |
| G2 | LOW | `qwen3vl_model.cpp:372,474,...` | 13 处 `CopyToDevice`/`CopyToHost` 返回值未检查 | ~13 行 |
| G3 | LOW | `docs/` | 缺少部署 checklist（健康检查、优雅关闭、容量规划） | 新文档 |

---

## 5. Python 端生产就绪评估

### 5.1 P0 问题（部署前必须修复）

| # | 文件:行号 | 描述 |
|---|-----------|------|
| P0-1 | `engine.py:250` | **Sync 竞态条件** — vision block 循环最后没有 sync，`run_merger_npu` 可能在 block N 完成前读取 `h`，产生静默错误 |
| P0-2 | `engine.py:54-133` | **无 NPU 资源清理** — 缺少 `close()`/`__del__`/context manager，NPU 内存和 ATB graph 会在异常或引擎销毁时泄漏 |
| P0-3 | `engine_utils.py:18-42` | **配置/权重加载无错误处理** — `load_config()`, `load_weights()`, `load_preprocessor_config()` 均无 try/except，文件缺失/损坏时崩溃无诊断信息 |
| P0-4 | `engine.py:305` | **forward() 无输入校验** — 不检查 input_ids dtype/shape、不检查 pixel_values/image_grid_thw 一致性、不检查 token_id 范围 |

### 5.2 P1/P2 统计

| 文件 | P1 | P2 | 主要类别 |
|------|----|----|----------|
| `engine.py` | 9 | 6 | 错误处理、输入校验、日志 |
| `engine_utils.py` | 8 | 4 | 错误处理、硬编码值、输入校验 |
| `utils.py` | 5 | 7 | 错误处理、输入校验、算法效率 |
| `preprocess.py` | 6 | 3 | 错误处理、硬编码常量、输入校验 |
| **合计** | **28** | **20** | |

完整清单见 Explore Agent 报告（52 项，含具体行号）。

---

## 6. 关键发现与建议

### 6.1 Benchmark 修复（已完成 ✅）

Benchmark.py 的 TF 精度对比曾使用 `ref()`（完整 Qwen3VLModel.forward），内部会重新运行 vision encoder 产生不同的 deepstack features，导致与 ATB 路径不对齐，余弦相似度仅 0.87-0.91。已修改为使用 `ref.language_model()` + 预计算的 vision embeddings，余弦相似度恢复至 ≥ 0.999。与 `test_vision_diagnostics.py` 的修复模式相同。

### 6.2 部署前建议优先级

```
P0 (必须修复，估计 4 小时):
├── engine.py:250 sync 竞态 → 取消注释或在循环后添加同步
├── engine.py:54-133 资源清理 → 添加 __del__ + context manager
├── engine_utils.py:18-42 错误处理 → 添加 try/except + 清晰错误消息
├── engine.py:305 输入校验 → 添加 shape/dtype/range 检查
└── G1: qwen3vl_model.cpp:302 gthw null guard

P1 (强烈建议，估计 8 小时):
├── benchmark: 添加 P50/P95/P99 百分位
├── benchmark: 添加 NPU 内存使用量测量
├── Python: 28 个 P1 项（错误处理、日志、硬编码值）
├── G2: 13 处 CopyToDevice 返回值检查
└── 部署 checklist 文档

P2 (改善项，按需):
├── benchmark: 冷启动/吞吐量/回归检测
├── Python: 20 个 P2 项（文档、代码清理）
└── 结构化日志（logging 替代 print）
```

### 6.3 积极亮点

1. **C++ RAII 实现无可挑剔** — NpuTensor, OperationHandle, ContextHandle 均正确实现移动语义和析构释放
2. **测试金字塔完整** — Python 7 测试文件 + C++ 50 测试文件，5 级金字塔设计
3. **文档详尽** — 1600+ 行的 design.md 覆盖架构、API、算子参考、适配指南
4. **性能优秀** — 3.0-3.6× 加速比，各阶段计时精细
5. **审计文化良好** — audit-fix-plan.md (61 发现) 和 test-fix-plan.md (21 发现) 均有详细跟踪

---

## 7. 总体裁决

| 维度 | 评级 | 说明 |
|------|------|------|
| **功能正确性** | 🟢 | 余弦相似度 ≥ 0.999，51/51 CTest 通过 |
| **性能表现** | 🟢 | 3.0-3.6× 加速比，Vision Model 为主要瓶颈（可优化） |
| **C++ 鲁棒性** | 🟡 | RAII/错误传播优秀，3 个轻量 gap |
| **Python 鲁棒性** | 🔴 | 4 P0 问题，缺少错误处理和资源管理 |
| **Benchmark 可信度** | 🟢 | 修复后精度验证可靠，计时方法正确 |
| **可维护性** | 🟢 | 文档齐全，测试覆盖广 |
| **生产就绪** | 🟡 | **有条件可部署** — 修复 P0 项后可上线 |

**结论**: 框架核心计算引擎（C++ 端）已达到生产质量。Python 端作为推理 API 层需要补充错误处理、输入校验和资源管理。Benchmark 经过本轮修复后精度验证可信。建议在修复 4 个 P0 项后进入生产部署。

---

*报告生成: 2026-06-16 | Agent 总数: 8 (3 Explore + 4 Action + 1 Verify)*
*详细发现日志: 见各 agent output file*

---

## 8. 已完成修复清单（第 1 批：G1, G2, G3, B1）

**分支**: `fix/benchmark`（基于 `fix/tests`），已推送至 `origin`。

| # | 目标 | 提交 | 状态 | 审查轮次 | 说明 |
|---|------|------|------|----------|------|
| G1 | gthw null guard | `c697bc7` | ✅ 已完成 | 1 轮通过 | `qwen3vl_model.cpp:302` 添加 `gthw` 空指针检查，防止 UB |
| G2 | CopyToDevice/CopyToHost 返回值检查 | `cd529f1` | ✅ 已完成 | 1 轮通过 | 修复 16 处（超预估的 13 处）未检查返回值的调用点 |
| G3 | 部署 checklist 文档 | `e6aa3ab` | ✅ 已完成 | 2 轮（首轮 25 问题→全部修复） | 新增 `docs/deployment-checklist.md`（692 行），所有命令与源码一致 |
| B1 | P50/P95/P99 百分位 | `7a755ed` | ✅ 已完成 | 2 轮（7 问题→全部修复） | `benchmark.py` 添加线性插值百分位函数，集成至所有输出行和对比表 |

### 8.1 G1 详情：gthw null guard

在 `ForwardWithTiming` 入口添加防御性检查：
```cpp
if (!gthw) {
    LOG_ERROR("ForwardWithTiming: grid_thw and metadata are both null");
    return ERROR_INVALID_PARAM;
}
```
- 位置：`qwen3vl_model.cpp` 第 306-309 行
- 影响：防止 `gthw` 和 metadata 同时为空时的未定义行为（生产环境罕见但理论上可能）

### 8.2 G2 详情：CopyToDevice/CopyToHost 返回值检查

覆盖了 `qwen3vl_model.cpp` 中全部 `CopyToDevice`（15 处）和 `CopyToHost`（3 处）调用，统一使用模式：
```cpp
Status s = alloc->CopyToDevice(tensor, data, size);
if (s != STATUS_OK) {
    LOG_ERROR("FuncName: copy xxx to device failed");
    return s;
}
```
- 总计 88 行新增、28 行删除
- 审查 agent 逐一验证了每个调用点的日志信息和错误返回路径

### 8.3 G3 详情：部署 checklist 文档

新增 `atb_cpp_llm/docs/deployment-checklist.md`（692 行），涵盖：

| 章节 | 内容 |
|------|------|
| 1. 部署前验证 | 硬件检查、软件依赖、模型完整性、构建验证 |
| 2. 运行时配置 | 环境变量、ATB buffer pool 大小表、日志路径、内存限制 |
| 3. 健康检查 | 启动检查 `--mode all --iter 1`、运行时 cron job、监控阈值 |
| 4. 优雅关闭 | 信号处理表、drain timeout、RAII 析构链 |
| 5. 运维流程 | 日志轮转、模型权重更新 SOP、回滚、事故响应表 |
| 6. 容量规划 | 吞吐量参考、并发策略、扩展指导 |
| 附录 A | 汇总 checkbox |
| 附录 B | `atb_diag.sh` 一键诊断脚本 |
| 附录 C | ATB 错误码速查表 |

**首轮审查 25 个问题（5 严重）的根因**：文档中虚构了不存在的 CLI 参数（`--mode load-only`、`--mode health`、`--model`、`./server` 二进制等）。第二轮通过阅读 `benchmark.cpp` 和 `CMakeLists.txt` 源码获取真实接口（`--mode all`、`QWEN3VL_EMB_MODEL_DIR` 环境变量等），全部命令重写验证。

### 8.4 B1 详情：P50/P95/P99 百分位

在 `benchmark.py` 中添加线性插值百分位函数：
```python
def percentile(vals, p):
    """Compute p-th percentile via linear interpolation."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be between 0.0 and 1.0, got {p}")
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    if n == 0:
        return float('nan')
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * p
    f = int(k)
    c = k - f
    if f + 1 < n:
        return float(sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f]))
    return float(sorted_vals[f])
```
- 与 NumPy `np.percentile(method='linear')` 对齐
- 集成至 E2E、vision model、text model 输出行
- 在 ATB-vs-TF 对比表中新增 P95 列
- 总计 49 行新增、6 行删除

**首轮审查 7 个问题（2 bug）**：
- 返回类型不一致（`numpy.float64` vs Python `float`）→ 用 `float(...)` 包裹
- 空数据返回 `0.0` 而非 `NaN` → 改为 `float('nan')`
- 要求预排序输入 → 内部调用 `sorted(vals)`
- 无 p 范围校验 → 添加 `ValueError`
- 缩进不一致和表格宽度问题 → 格式化修复

---

## 9. 工作方法：Developer → Reviewer → Re-review 循环

### 9.1 方法论

每个修复项执行以下流程，**直到审查 agent 找不到任何问题为止**：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Developer      │────▶│  Reviewer       │────▶│  Re-review      │
│  Agent          │     │  Agent          │     │  Agent          │
│                 │     │                 │     │                 │
│ · 上下文干净    │     │ · 上下文干净    │     │ · 上下文干净    │
│ · 按计划执行    │     │ · 辩证挑刺      │     │ · 验证修复      │
│ · 思考全面      │     │ · 逐项对照计划  │     │ · 确认零问题    │
│ · 自省不糊弄    │     │ · 不放过小问题  │     │ · 最终裁决      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         │   发现 N 个问题       │                       │
         │◀──────────────────────│                       │
         │   修复后重新提交      │                       │
         │──────────────────────▶│   0 个问题            │
         │                       │──────────────────────▶│
         │                       │                       │  确认通过 ✅
```

### 9.2 各修复项审查统计

| 修复项 | 首轮发现问题 | 第二轮发现问题 | 总轮次 | 问题类型 |
|--------|-------------|---------------|--------|----------|
| G1 | 0 | — | 1 | — |
| G2 | 0 | — | 1 | — |
| G3 | 25（5 严重） | 0 | 2 | 虚构 CLI 参数、缺失验证步骤、文档结构 |
| B1 | 7（2 bug） | 0 | 2 | 返回类型不一致、空数据行为、参数校验缺失 |

### 9.3 方法论洞察

**为什么需要审查循环**：

1. **Developer agent 天然带有"建设者偏见"** — 倾向于验证自己的代码能工作，而非寻找边界情况
2. **Reviewer agent 天然带有"破坏者视角"** — 会从相反方向思考，寻找一切可能出错的地方
3. **首轮即通过的项（G1、G2）说明问题定义清晰、修改范围小** — 单点修复不需要多轮
4. **需要 2 轮的项（G3、B1）说明初始需求有歧义空间** — 文档创作和新功能实现容易引入假设性错误

**关键原则**：
- Developer 和 Reviewer **必须使用不同的 agent 实例**（上下文隔离、视角独立）
- Reviewer agent 的 prompt 应明确要求"辩证性地挑毛病"，而非"验证正确性"
- 每个审查轮次结束后，**architect 只负责汇总发现并派发下一轮**，不参与具体修复
- Reviewer 不会给出"这段代码写得不错"的评价 — 它只输出问题清单；零问题输出 = 通过

### 9.4 ⚠️ 强制性运行时测试（2026-06-16 新增）

**静态代码审查有根本性局限。** 第 2 批审查的 B2 项（NPU 内存测量）虽经过 3 轮静态审查，但所有 reviewer 均未在 NPU 硬件上实际运行 benchmark。以下类型的问题仅靠阅读代码无法发现：

| 问题类型 | 为什么静态分析发现不了 |
|----------|----------------------|
| API 行为误解 | `reset_max_memory_allocated()` 后立即读取返回 0 — 只有在 NPU 上执行才能确认 |
| 运行时竞态 | ATB graph 异步执行在哪个 stream、何时完成 — 依赖硬件调度 |
| 性能回归 | 新增 sync 是否导致可测量的延迟 — 必须实际计时 |
| 兼容性 | torch_npu 版本差异、ATB 图编译失败 — 只有实际运行才能发现 |
| 数值精度 | 修改后余弦相似度是否仍然 ≥ 0.99 — 必须用真实模型推理 |

### 9.5 测试执行规范：使用项目工具而非手动设置环境变量

**严格遵守下述测试执行方式，不可手动设置环境变量或绕过项目工具。**

#### Python 测试

**环境变量加载**：所有测试脚本通过 `from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR` 自动加载 `.env` 配置。`env.py` 的查找逻辑：
1. 检查 `ATB_DOTENV_PATH` 环境变量（可选覆盖）
2. 从 `env.py` 所在目录向上遍历 6 级查找 `.env` 文件
3. 优先级：`os.environ > .env > default`

**推荐运行方式**：
```bash
# 全量测试（自动使用 .env 中的 QWEN3VL_EMB_MODEL_DIR）
python atb_python_qwen3vl_embedding/tests/run_all.py

# 带 benchmark
python atb_python_qwen3vl_embedding/tests/run_all.py --benchmarks

# 仅特定测试
python atb_python_qwen3vl_embedding/tests/run_all.py --include '*e2e*'

# 排除 310P 特定测试（在 910B 上运行）
python atb_python_qwen3vl_embedding/tests/run_all.py --exclude '*310p*'
```

**❌ 禁止**：
```bash
# 不要手动 export 模型路径
export QWEN3VL_EMB_MODEL_DIR=/some/path   # ❌ 绕过 .env
# 不要手动设置 ASCEND_PLATFORM
export ASCEND_PLATFORM=910B               # ❌ 绕过 .env
```

#### C++ 测试

**必须使用 `build_and_test.sh`**，它会自动：
1. 加载 repo root 的 `.env`（`set -a; source .env; set +a`）
2. 依次 source Ascend 环境脚本（`ascend-toolkit/set_env.sh` → `cann/set_env.sh` → `atb/set_env.sh --cxx_abi=1`）
3. CMake configure + build（自动设置 `ATB_BUILD_DEPENDENCY_PATH`）
4. 生成 Python 参考数据（`gen_all.py`）或复用已有数据
5. 检测 NPU（`npu-smi`），有 NPU 时运行 CTest

**推荐运行方式**：
```bash
# 完整构建 + 测试
bash atb_cpp_llm/build_and_test.sh

# 仅构建（不运行测试）
bash atb_cpp_llm/build_and_test.sh --no-test

# 复用已有参考数据（不重新生成 /tmp/*.bin）
bash atb_cpp_llm/build_and_test.sh --no-refresh-refdata

# 跳过参考数据生成和依赖它的测试（28 个 needs_refdata 测试）
bash atb_cpp_llm/build_and_test.sh --no-refdata

# 仅测试（跳过构建，需已有 build/）
bash atb_cpp_llm/build_and_test.sh --test-only

# Debug 构建
bash atb_cpp_llm/build_and_test.sh --debug

# 仅运行特定级别（level1_cpu_pure / level2_op_precision / level3_integration / level4_e2e）
bash atb_cpp_llm/build_and_test.sh level1_cpu_pure

# 仅运行特定测试名
bash atb_cpp_llm/build_and_test.sh test_text_model

# 列出所有注册的测试
bash atb_cpp_llm/build_and_test.sh --list
```

**❌ 禁止**：
```bash
# 不要手写 cmake 命令
cmake -S atb_cpp_llm -B atb_cpp_llm/build -DATB_DIR=...  # ❌ 绕过 build_and_test.sh
# 不要手写 ctest 命令
ctest --test-dir atb_cpp_llm/build                         # ❌ 绕过 build_and_test.sh
# 不要手动 source Ascend 脚本后再手动 cmake
source /usr/local/Ascend/.../set_env.sh                    # ❌ build_and_test.sh 已处理
cmake ...                                                   # ❌
```

#### 架构师运行时验证流程（修订版）

```
审查流程（修订版 v2）：

1. Developer Agent 完成代码修改
2. Developer Agent 必须运行相关单元测试，证明修改不破坏现有功能
   - Python 修改 → python atb_python_qwen3vl_embedding/tests/run_all.py
   - C++ 修改 → bash atb_cpp_llm/build_and_test.sh
3. Reviewer Agent 进行静态代码审查（逐行阅读 + 对照参考实现）
4. Reviewer Agent 必须运行相关单元测试，验证输出与预期一致
5. Architect 在每个批次完成后运行全量测试套件
   - Python 全量: python atb_python_qwen3vl_embedding/tests/run_all.py
   - C++ 全量: bash atb_cpp_llm/build_and_test.sh
   - 两者均须 PASS（0 失败）
```

**审查通过 = 静态审查零问题 + 单元测试全部 PASS**

> ⚠️ 此规则为 2026-06-16 第 2 批审查后新增。第 1 批（G1-G3, B1）和第 2 批（P0-1~P0-3, B2）均未执行运行时审查。这 8 项修改需要在硬件可用时进行运行时回溯验证。

### 9.6 ⚠️ 310P Benchmark 失败事件：流程缺陷暴露（2026-06-16）

**事件描述**：在完成 `test-fix-plan.md` 的 21 项全面代码审查后，用户在 310P 上单独运行 benchmark 时发现：

- **`--mode all text io mm`** 全部报错 "setup fail with status 4"（ATB "call operation setup fail"）
- **`--mode compare`** 报错 `/tmp/tokens_chat_text_only_100.bin` 文件缺失

**后续发现（910B）**：C++ benchmark 在 910B 上同样存在严重问题：
- **`--mode all` 和 `--mode text`**：**静默崩溃** — 进程在 `main()` 之前被 `std::abort()` 杀死，无任何日志输出
- **`--mode compare`**：同样报 token 文件缺失

这意味着 **C++ benchmark 在所有平台上都已损坏，但测试套件从未运行它**。

**两个问题均非 test-fix-plan.md 审查范围内发现的新问题** — 它们都是代码中已知但未修复的缺陷。

#### 问题 1: Token 文件命名不匹配（已知但未修复）

**症状**: C++ `benchmark --mode compare` 期望 `tokens_chat_*` 前缀的文件，Python `benchmark --save-tokens` 生成 `tokens_*` 前缀（无 `chat`、无 seq_len 后缀）。

| 来源 | TEXT_ONLY | IMAGE_ONLY | IMAGE_AND_TEXT |
|------|-----------|------------|----------------|
| C++ 期望 | `tokens_chat_text_only_{seq}.bin` | `tokens_chat_io_{W}x{H}.bin` | `tokens_chat_mm_{W}x{H}.bin` |
| Python 生成 | `tokens_text_only.bin` ❌ | **不生成** ❌ | `tokens_mm_{w}x{h}.bin` ❌ |

**此问题在 `refactoring-plan.md:194` 中已明确记录**：
> *"症状: C++ 读 `tokens_chat_mm_{W}x{H}.bin`，Python 读 `tokens_mm_{W}x{H}.bin`。两组文件不同步时 cosine 莫名其妙低，代码表面看不出异常。教训: 所有 benchmark .bin 由一个脚本统一生成。"*

然而该问题从未被纳入 `test-fix-plan.md` 的可操作修复项 — **知识存在但未转化为行动**。

#### 问题 2: 310P "call operation setup fail"（代码注释中已记录）

**症状**: 310P 上所有模式（text/io/mm）的 benchmark 均失败，错误为 ATB "call operation setup fail"。

**根因**: 310P 要求 causal mask 使用 FRACTAL_NZ 格式（format=29），若使用 ND 格式，ATB SelfAttention 内部 Transdata 操作失败。此行为在代码中已有明确注释：
- `utils.py:245-246`: *"without it, ATB SelfAttention sees ND format, tries internal ND→NZ TransdataOperation, and fails on 310P with 'call operation setup fail'"*
- `engine.py:286-288`: *"310P: causal mask in NZ layout... fails on 310P with 'call operation setup fail'"*

`engine.py:_ensure_text_graph()` 已通过 `is_310p()` 判断处理此问题，但 `is_310p()` 依赖 `.env` 中的 `ASCEND_PLATFORM=310P` 变量。**如果 310P 机器上 `.env` 未正确设置此变量，`is_310p()` 返回 False，静默 fallback 到 910B 的 ND 格式 mask，导致 ATB 图编译/运行失败。**

#### 问题 3: C++ benchmark `GetModelDir()` 的 `std::abort()` 导致静默崩溃（910B + 310P 均受影响）

**症状**: 用户直接运行 `./build/benchmark --mode all` 或 `--mode text` 时，进程静默退出，无任何 LOG_INFO 或 LOG_ERROR 输出。`--mode compare` 能输出错误信息仅因为偶然的环境变量设置差异。

**根因**: `benchmark.cpp:44` 中：
```cpp
static const std::string MODEL_DIR = GetModelDir();
```

这是一个**静态全局变量**，在 `main()` 之前初始化。而 `test_env.h:209-220` 中的 `GetModelDir()` 实现：
```cpp
inline std::string GetModelDir() {
    std::string val = GetEnv("QWEN3VL_EMB_MODEL_DIR");
    if (!val.empty()) return val;
    std::fprintf(stderr, "...");
    std::abort();  // ← 在 main() 之前杀死进程！
}
```

当用户直接运行 benchmark 二进制（而非通过 `build_and_test.sh`）时，`QWEN3VL_EMB_MODEL_DIR` 可能未在 shell 环境中设置。此时：
1. `GetEnv()` 检查 shell 环境 → 未设置
2. `GetDotEnvCache()` 搜索 `.env` 文件 → 取决于 CWD，不一定能找到
3. `GetModelDir()` 调用 `std::abort()` → **进程在静态初始化阶段被杀死，main() 从未执行**
4. 所有 `LOG_INFO`/`LOG_ERROR` 宏从未有机会执行
5. 只有 ATB 库自身的静态初始化日志（`mki_log delete old file:...`）被打印

**为什么 CTest 和 `build_and_test.sh` 没发现**：
- `build_and_test.sh` 在运行 ctest 前 `source .env`，将 `QWEN3VL_EMB_MODEL_DIR` 导出到 shell 环境
- CTest 继承 shell 环境，所以所有 C++ 测试都能正常获取 model_dir
- **但 C++ benchmark 未被注册为 CTest 测试** — `CMakeLists.txt` 中没有 `add_test(NAME benchmark ...)` 
- 因此 `build_and_test.sh → ctest` 从不运行 benchmark 二进制

**这是一个双重失败**：
1. benchmark 在无 env 设置时静默崩溃（可用性问题）
2. benchmark 完全不在自动化测试中（覆盖缺口）

#### 为什么 test-fix-plan.md 的全面审查没有发现

| 盲区类型 | 具体表现 | 严重度 |
|----------|----------|--------|
| **静态审查 ≠ 运行时测试** | 命名不匹配只有执行 `--mode compare` 才会暴露；mask 格式错误只有 310P 上运行才触发；`std::abort()` 只有直接运行二进制才触发 | 🔴 严重 |
| **已知问题未追踪到修复** | `refactoring-plan.md` 记录的问题未纳入 `test-fix-plan.md` 可操作项，知识孤岛导致问题持续 | 🔴 严重 |
| **跨平台测试缺失** | 全部测试在 910B 完成，310P 特定代码路径（NZ mask 格式）仅做目视检查 | 🔴 严重 |
| **benchmark "通过" 具误导性** | `run_all.py --benchmarks` 仅用默认 `--mode mm` 运行；C++ benchmark 根本不在 CTest 中注册 | 🔴 严重 |
| **工作流端到端测试缺失** | compare 模式全流程（生成tokens→C++ benchmark→Python 对比）从未端到端验证 | 🟡 中等 |
| **环境依赖隐式假设** | `is_310p()` 依赖 `.env` 配置；`GetModelDir()` 的 `std::abort()` 在无 env 时杀死进程 | 🟡 中等 |

#### 🆕 为什么架构师也没有第一时间发现（元认知缺陷）

**我在派发修复 agent 之前，犯了和 test-fix-plan.md 审查相同的错误：只分析代码，不运行验证。**

具体失误链：
1. **假设基线正确** — 看到 51/51 CTest 通过，就假设 C++ benchmark 也能工作。事实是 benchmark **不在 CTest 中注册**。
2. **跳过了复现步骤** — 直接进入代码分析模式（读 6 个文件），试图通过阅读代码推断 "status 4" 含义，而不是先在 910B 上运行 `./build/benchmark --mode all` 确认基线。
3. **信任间接信号** — 用 Python benchmark 的通过来推断 C++ benchmark 的正常，两者是独立二进制。
4. **未检查测试注册** — 没有验证 `CMakeLists.txt` 中 benchmark 是否被 `add_test()` 注册。

**如果我先在 910B 上运行 `./build/benchmark --mode all`，30 秒内就能发现静默崩溃**，然后追踪到 `GetModelDir() → std::abort()` 根因。这会立即暴露三个问题（abort + 命名不匹配 + 未注册到 CTest），而非用户逐步提供信息后才逐个发现。

**架构师教训**：
> **收到 bug 报告后，第一步永远是复现问题、建立基线，而非直接分析代码。** "测试全部通过"不能替代"实际运行出问题的那个命令"。

#### 暴露的系统性缺陷

1. **审计文档之间的知识孤岛**：`refactoring-plan.md` 记录的已知问题未与 `test-fix-plan.md` 联动，导致已知缺陷持续存在。需要一个统一的缺陷追踪机制。

2. **审查覆盖 ≠ 验证覆盖**：代码结构审查（覆盖矩阵、阈值合理性、冗余分析）无法替代运行时行为验证。benchmark 有 4 mode × 4 分辨率 × `--save-tokens` × `--load-pixel-values` × `--no-ref` 等多种组合，仅默认路径通过测试不等于全路径正确。

3. **平台假设未显式化**：`ASCEND_PLATFORM` 环境变量的正确配置是正确性的前提条件，但无自动检测机制验证其与实际 NPU 硬件是否一致。正确行为隐式依赖人工配置。

4. **"通过"的语义膨胀**：`run_all.py --benchmarks` 中 benchmark.py 退出码 0 被解读为 "benchmark 功能正常"，但实际上仅验证了默认参数的 `--mode mm` 路径。测试通过 ≠ 功能完整。

#### 修正措施（已派发修复 agent）

| # | 措施 | 说明 |
|---|------|------|
| **F1** | 修复 Python benchmark.py `--save-tokens` 命名 | 改用 `chat_tokenizer` + `tokens_chat_*` 命名，对齐 C++ 期望 |
| **F2** | 创建 `gen_compare_tokens.py` | 一键生成 C++ compare 模式所需的全部 13 个 token 文件 |
| **F3** | 平台自检（后续） | benchmark/engine 初始化时验证 `ASCEND_PLATFORM` 与实际 NPU 硬件一致 |
| **F4** | 扩展 benchmark 测试覆盖（后续） | `run_all.py` 增加 `--mode all --save-tokens` 等非默认路径的测试 |

### 9.7 🆕 架构师工作流 v2：经验提炼（2026-06-16）

基于 §9.6 事件中暴露的流程缺陷，将工作流升级为 v2。

#### 核心原则（新增/强化）

| 原则 | 来源 | 含义 |
|------|------|------|
| **复现优先于分析** | §9.6 元认知缺陷 | 收到 bug 报告 → 先在可用硬件上运行出问题的命令 → 建立基线 → 再读代码 |
| **审查必须复现** | 用户反馈 + §9.4 | Reviewer 的第一项工作是运行出问题的命令确认问题存在，然后再审查修复 |
| **交叉验证** | F3 发现（910B 也失败） | C++ 修改必须验证 Python 端不受影响，反之亦然；一个平台通过 ≠ 全平台正常 |
| **测试注册检查** | F3 发现（benchmark 不在 CTest） | 确认修改涉及的二进制/脚本已被自动化测试覆盖，未覆盖则同步添加 |
| **已知问题搜索** | F1 发现（refactoring-plan 已记录） | 修复前搜索 `docs/*.md` 中的已知问题记录，避免重复劳动或忽视已知缺陷 |
| **禁止假设** | 用户明确要求 | Architect 不清楚需求时，使用 grill-me skill 向用户彻底问清楚，严禁猜测 |

#### 修订后的 Developer → Reviewer 循环 (v2)

```
┌──────────────────────────────────────────────────────────────────┐
│                    Architect 前置步骤（新增）                      │
│  1. 收到任务/问题报告                                              │
│  2. 复现问题，建立基线（先跑命令，不先读代码）                       │
│  3. 搜索 docs/*.md 中的已知问题记录                                │
│  4. 检查目标二进制/脚本是否在自动化测试中注册                        │
│  5. 确定问题范围（不止用户报告的 mode，检查所有 mode/组合）          │
│  6. 如有不明确 → grill-me skill 向用户问清楚 → 严禁猜测            │
│  7. 编写详细的工作范围和目标（developer agent 不能自己猜测）         │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Developer      │────▶│  Reviewer       │────▶│  Re-review      │
│  Agent          │     │  Agent          │     │  Agent          │
│                 │     │                 │     │                 │
│ · 上下文干净    │     │ · 上下文干净    │     │ · 上下文干净    │
│ · 按计划执行    │     │ · 辩证挑刺      │     │ · 验证修复      │
│ · 思考全面      │     │ · 逐项对照计划  │     │ · 确认零问题    │
│ · 自省不糊弄    │     │ · 不放过小问题  │     │ · 最终裁决      │
│ · 有疑问→汇报  │     │ · 🆕 先复现问题 │     │                 │
│ · 🆕 先运行测  │     │ · 🆕 运行测试   │     │                 │
│   试验证基线   │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         │   发现 N 个问题       │                       │
         │◀──────────────────────│                       │
         │   修复后重新提交      │                       │
         │──────────────────────▶│   0 个问题            │
         │                       │──────────────────────▶│
         │                       │                       │  确认通过 ✅
```

#### Architect 审查通过标准 (v2)

**审查通过 = 静态审查零问题 + 单元测试全部 PASS + 复现确认问题已消失**

| 阶段 | 检查项 |
|------|--------|
| Developer 提交后 | 相关单元测试 PASS（Python: `run_all.py` / C++: `build_and_test.sh`） |
| Reviewer 审查时 | 复现原问题 → 确认修复后问题消失 → 静态代码审查零问题 |
| Architect 批次后 | 全量测试 PASS（Python + C++ 双端） |

> ⚠️ "全量测试 PASS" 仅涵盖**已注册到测试套件的测试**。对于未注册的二进制/脚本（如 C++ benchmark），architect 必须单独验证。

#### Architect 派发 agent 的 briefing 模板

每个 developer agent 的 prompt 必须包含：
1. **问题描述**：症状、影响范围、根因（如已知）
2. **工作范围**：哪些文件需要修改，哪些**不需要**修改（防止过度修改）
3. **验收标准**：具体的、可验证的通过条件
4. **禁止事项**：明确列出不允许的操作（如"不要修改其他函数"）
5. **测试要求**：需要运行的具体测试命令
6. **上报路径**：遇到不明确的情况 → 汇报给 architect，不要自己猜测

---

## 10. 预防措施：如何避免类似问题

### 10.1 根因分析

本轮发现的各类问题可归纳为以下根因类别：

| 根因类别 | 示例 | 出现频次 |
|----------|------|----------|
| **接口假设错误** | G3: 虚构 `--mode load-only`/`--model`；B1: 假设输入已排序 | 高 |
| **边界条件遗漏** | G1: `gthw` 空指针未检查；B1: 空列表返回 0.0 而非 NaN | 高 |
| **返回值忽略** | G2: 16 处 `CopyToDevice`/`CopyToHost` 返回值未检查 | 中 |
| **类型不一致** | B1: 返回 `np.float64` 而非 Python `float` | 低 |
| **🆕 文件命名约定不一致** | C++ `tokens_chat_*` vs Python `tokens_*` — 跨语言接口契约未同步 | 高 |
| **🆕 环境配置隐式依赖** | `is_310p()` 正确性依赖 `.env`，错误配置导致静默 fallback 和运行时失败 | 高 |
| **🆕 知识孤岛** | `refactoring-plan.md` 记录的已知问题未纳入 `test-fix-plan.md` 可操作项 | 高 |
| **🆕 多模式测试覆盖不完整** | benchmark 4 种 mode × 多种选项，仅默认路径被测试覆盖 | 中 |

### 10.2 预防机制

#### A. 开发阶段预防

**1. 接口契约检查清单**（适用于任何新功能/文档）：
- [ ] 所有引用的 CLI 参数/API 是否在源码中真实存在？
- [ ] 所有引用的文件路径/二进制名称是否在构建系统中存在？
- [ ] 所有配置项是否有对应的环境变量或配置文件定义？
- **验证方式**: `grep` 源码确认，不可凭记忆或推断

**2. 边界条件检查清单**（适用于任何函数/方法）：
- [ ] 所有指针/引用参数是否检查了 `nullptr`/`None`？
- [ ] 所有容器输入是否处理了空容器情况？
- [ ] 所有数值计算是否检查了除零/空集/越界？
- [ ] 所有返回值是否检查了错误码？

**3. 代码审查检查清单**（Reviewer 视角）：
- [ ] 是否存在"看起来合理但实际错误的假设"？（最常见陷阱）
- [ ] 错误路径是否可达且有测试覆盖？
- [ ] 文档中的命令是否可以直接复制粘贴执行？
- [ ] ⚠️ **是否已运行相关单元测试并全部 PASS？**（纯静态审查不可接受 — 见 §9.4）
- [ ] 修改的 API 返回值/副作用是否在 NPU 硬件上验证过？
- [ ] 🆕 **C++ 和 Python 之间的文件命名/接口契约是否一致？**（跨语言接口必须对照检查）
- [ ] 🆕 **所有 benchmark/test mode 组合是否都经过测试？**（不只默认路径）
- [ ] 🆕 **环境变量依赖是否在代码中显式验证？**（不应隐式依赖外部配置的正确性）

#### B. CI/CD 阶段预防

**4. 自动化检查建议**（推荐加入 CI pipeline）：

| 检查项 | 工具/方法 | 覆盖的问题类别 |
|--------|----------|---------------|
| 文档-二进制一致性 | 解析文档中的 `` `--flag` `` 模式，与 `--help` 输出对比 | 接口假设错误 |
| 返回值检查 lint | C++ 扫描 `CopyToDevice`/`CopyToHost` 调用，确保返回值被消费 | 返回值忽略 |
| 空指针检查 lint | C++ 扫描函数参数使用，确保有 null guard | 边界条件遗漏 |
| Python 类型检查 | 对公开 API 添加类型注解，CI 中运行 `mypy` | 类型不一致 |

#### C. 流程阶段预防

**5. Developer-Reviewer 双人流程**（即本报告第 9 节的方法论）：
- 所有非平凡修改（>20 行或涉及新文件）必须经过独立的 reviewer agent
- Reviewer 必须从"这段代码哪里会出问题"的角度审阅，而非"这段代码对不对"
- 发现问题后 developer 修复并重新提交，reviewer 重新审阅直到零问题

**6. "真值验证"原则**：
- 文档中的任何命令/参数/路径必须能追溯到源码（不可推断）
- 性能数据的对比基线必须与生产环境一致（不可用不同配置）
- 测试的参考实现必须是同输入下的黄金标准（本项目中为 transformers）

**7. 🆕 跨语言接口契约检查**（适用于 C++/Python 互操作）：
- [ ] C++ 和 Python 是否使用相同的文件命名约定？逐文件对照检查
- [ ] 所有 `.bin` 文件的生成者和消费者是否使用同一份格式定义？
- [ ] 新增跨语言文件依赖时，是否同时更新生成脚本和消费代码？
- **验证方式**: 运行完整的跨语言工作流（生成→消费→对比），不只分别测试两端

**8. 🆕 审计文档联动机制**：
- 任何审计/回顾文档中标记的已知问题必须同步到 `test-fix-plan.md` 或对应的可操作追踪文档
- 定期扫描 `docs/*.md` 中的"症状"/"根因"/"教训"关键词，提取未修复项并排优先级
- **教训来源**: `refactoring-plan.md:194` 记录了 `tokens_chat_*` 命名不匹配 6 个月以上但从未修复

**9. 🆕 平台自检机制**：
- 关键环境变量（如 `ASCEND_PLATFORM`）在引擎初始化时与实际 NPU 硬件对比验证
- 不匹配时发出明确警告（而非静默 fallback 到错误行为）
- **教训来源**: 310P 上 benchmark 全部失败但 910B 上全 PASS，环境配置错误被静默掩盖

**10. 🆕 Benchmark 多模式测试**：
- `run_all.py --benchmarks` 应测试非默认 mode（至少 `--mode all`），不只是默认 `--mode mm`
- 关键功能组合（`--save-tokens`、`--load-pixel-values`）应有独立测试用例
- 每个 benchmark mode 至少有一条"冒烟测试"（1 warmup + 1 iter + 1 resolution）

#### D. 🆕 架构师自我检查清单

**11. 🆕 收到 bug 报告后的第一步：复现，不是分析代码**：
- [ ] **在分析任何代码之前**，先在当前可用硬件上运行用户报告的确切命令
- [ ] 如果当前硬件与用户不同（如用户在 310P，我在 910B），先在可用硬件上运行同等命令建立基线
- [ ] "测试全部通过"不等于"出问题的那个命令能工作" — 验证用户实际执行的命令
- [ ] 检查目标二进制/脚本是否在自动化测试中注册（CTest / run_all.py）
- [ ] **只有复现问题后，才开始阅读代码分析根因**

**12. 🆕 派发 agent 前的验证**：
- [ ] 是否已确认问题的完整范围（不只是用户报告的症状，还包括同一二进制/脚本的其他 mode）
- [ ] 是否已验证问题在多个 mode/平台上的表现（如 text + io + mm + compare + all）
- [ ] 是否已检查相关二进制是否被自动化测试覆盖
- **教训来源**: 本次事件 — 架构师直接分析代码而跳过复现，导致第三个根因（`std::abort()`）完全遗漏，需用户提供额外信息才发现

### 10.3 与现有 CLUADE.md 测试精度原则的关系

本项目 CLAUDE.md 已规定：
> 绝不通过降低验收标准来"通过"测试。如果 C++ 和 Python 在相同输入下余弦相似度低于 0.99，说明存在 bug，必须定位并修复根因，而不是放宽阈值。

本报告的预防措施是对该原则的**补充和扩展** — 精度原则覆盖"怎么验证正确性"，本报告的清单覆盖"怎么在第一时间避免写错"。

---

## 11. 后续工作计划

### 11.1 当前状态

| 维度 | 已完成 | 进行中 | 待开始 |
|------|--------|--------|--------|
| C++ 端鲁棒性 | G1, G2, G3 ✅ | — | — |
| Benchmark 增强 | B1, B2 ✅ | — | B3-B7 |
| Python 端 P0 | P0-1, P0-2, P0-3 ✅ | — | P0-4 |
| Python 端 P1/P2 | — | — | 48 项 |
| ⚠️ 运行时验证 | — | **待执行** | — |

### 11.2 第 2 批完成清单

| # | 目标 | 提交 | 状态 | 审查轮次 | 运行时测试 |
|---|------|------|------|----------|------------|
| P0-3 | engine_utils.py 错误处理 | `b5f6d8b` | ✅ | 2 轮（7→0 问题） | ⚠️ 未执行 |
| B2 | NPU 内存使用量测量 | `394acc5` | ✅ | 3 轮（7→6→0 问题） | ⚠️ 未执行 |
| P0-1+P0-2 | sync 竞态修复 + 资源清理 | `2cb1dbe` | ✅ | 2 轮（2+5→0 问题） | ⚠️ 未执行 |

### 11.3 第 3 批目标建议（6 项 — 含紧急修复 F1-F2）

| # | 目标 | 严重程度 | 预估工作量 | 说明 |
|---|------|----------|------------|------|
| **F1** | 🔴 修复 benchmark.py token 命名不匹配 | P0 | 1h | 改用 chat_tokenizer + `tokens_chat_*` 命名，对齐 C++ 期望 |
| **F2** | 🔴 创建 `gen_compare_tokens.py` | P0 | 0.5h | 一键生成 C++ compare 模式所需的 13 个 token 文件 |
| **P0-4** | engine.py forward() 输入校验 | P0 | 1h | 检查 input_ids dtype/shape、pixel_values/image_grid_thw 一致性、token_id 范围 |
| **B3** | 冷启动 benchmark | P2 | 1.5h | 测量首次推理的 H2D 拷贝和 graph 编译开销 |
| **B4** | 吞吐量 benchmark | P2 | 2h | 多 stream/批量并发吞吐量测试 |
| **回溯验证** | 运行全量测试验证第 1-2 批修改 | P0 | 1h | `python tests/run_all.py` + `bash atb_cpp_llm/build_and_test.sh` |

### 11.4 后续批次概览

| 批次 | 内容 | 触发条件 |
|------|------|----------|
| 第 3 批 | P0-4（输入校验）+ B3（冷启动）+ B4（吞吐量）+ **回溯验证** | 当前 |
| 第 4 批 | Python P1 项（28 项）+ B5（回归检测） | 第 3 批完成 |
| 第 5 批 | Python P2 项（20 项）+ B6/B7（preprocess 计时 + 硬编码数据清理） | 第 4 批完成 |

每完成 4 项目标后进行状态汇报和 plan 更新。

---

## 12. 修订记录

| 日期 | 修订内容 |
|------|----------|
| 2026-06-16 | 初始评估报告（8 agent：3 Explore + 4 Action + 1 Verify） |
| 2026-06-16 | 第 1 批修复完成：G1, G2, G3, B1（4/4 ✅）；新增第 8-12 节 |
| 2026-06-16 | 第 2 批修复完成：P0-1, P0-2, P0-3, B2（4/4 ✅）；新增 §9.4 强制性运行时测试规则；新增 §10 审查清单中的运行时验证要求 |
| 2026-06-16 | 🔴 **310P+910B benchmark 失败事件**：新增 §9.6（3 个根因分析 + 元认知缺陷分析）；更新 §10.1 根因表（+4 行）；更新 §10.2 审查清单（+5 项）+ 架构师自我检查清单（+2 项）；更新 §11.3 第 3 批计划（含 F1-F3 紧急修复） |
