# 工作流（方法论）

> 本文件规定**怎么推进工作**——architect 编排、Developer→Reviewer→Re-review 循环、审查通过标准、agent 派发规范。这是正面流程；踩坑视角见 [lessons-learned.md](./lessons-learned.md) 主题 7。两份互补，本文件讲"该怎么做"，lessons 讲"别再怎么错"。
>
> 来源：architect-assessment §9.1–§9.7（提炼为常青流程，原始事件细节见 `archive/`）。

---

## 1. 总体模型

```
┌──────────────────────────────────────────────────────────────────┐
│                    Architect 前置步骤（§2）                        │
│  1. 收到任务/问题报告                                              │
│  2. 复现问题，建立基线（先跑命令，不先读代码）                       │
│  3. 搜索 docs 中的已知问题（STATUS.md + lessons-learned.md + archive）│
│  4. 检查目标二进制/脚本是否在自动化测试中注册                        │
│  5. 确定问题范围（不止用户报告的症状，检查所有 mode/组合）           │
│  6. 如有不明确 → grill-me skill 向用户问清楚 → 严禁猜测            │
│  7. 编写详细工作范围和目标（developer agent 不能自己猜测）          │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Developer      │────▶│  Reviewer       │────▶│  Re-review      │
│  · 上下文干净    │     │  · 上下文干净    │     │  · 上下文干净    │
│  · 按计划执行    │     │  · 辩证挑刺      │     │  · 验证修复      │
│  · 思考全面      │     │  · 逐项对照计划  │     │  · 确认零问题    │
│  · 自省不糊弄    │     │  · 不放过小问题  │     │  · 最终裁决      │
│  · 有疑问→汇报  │     │  · 先复现问题    │     │                 │
│  · 先运行测试    │     │  · 运行测试      │     │                 │
│    验证基线      │     │                  │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │   发现 N 个问题              │   0 问题          │ 确认通过 ✅
         │◀─────────────────────────────│──────────────────▶│
```

**循环直到 Reviewer 输出零问题。** Reviewer 只输出问题清单，不给"写得不错"的评价；零问题输出 = 通过。

---

## 2. Architect 前置步骤（七步，不可跳过）

1. **收到任务/问题报告** —— 记录症状、影响范围、用户期望。
2. **复现优先于分析** —— 先在可用硬件上运行用户报告的确切命令建立基线，再读代码。"测试全部通过"不能替代"实际运行出问题的那个命令"。若当前硬件与用户不同（如用户 310P、architect 910B），先在可用硬件跑同等命令建立基线。
3. **搜索已知问题** —— grep `docs/` 的"症状/根因/教训"关键词，查 [STATUS.md](./STATUS.md) 待办和 [lessons-learned.md](./lessons-learned.md)。避免重复劳动或忽视已记录缺陷（知识孤岛是 §9.6 的根因之一）。
4. **测试注册检查** —— 确认修改涉及的二进制/脚本已被自动化测试覆盖（CTest / `run_all.py`）；未覆盖则同步添加测试。
5. **确定问题范围** —— 不止用户报告的 mode，检查同一二进制/脚本的所有 mode/组合（如 text+io+mm+compare+all）。
6. **禁止假设** —— 不清楚时用 grill-me skill 向用户彻底问清楚，严禁猜测。developer agent 遇到不明确汇报给 architect，不自己猜。
7. **编写工作范围和目标** —— 见 §5 briefing 模板，developer agent 不能自己猜测范围。

---

## 3. Developer → Reviewer → Re-review 循环

### 3.1 三阶段职责

| 阶段 | 视角 | 职责 |
|------|------|------|
| **Developer** | 建设者 | 上下文干净、按计划执行、思考全面、自省不糊弄、有疑问即汇报、先运行测试验证基线 |
| **Reviewer** | 破坏者 | 上下文干净、辩证挑刺、逐项对照计划、不放过小问题、**先复现问题再审查修复**、运行测试 |
| **Re-review** | 裁决者 | 上下文干净、验证修复、确认零问题、最终裁决 |

### 3.2 为什么需要审查循环

- Developer 有**建设者偏见**——倾向于验证代码能工作，而非找边界情况。
- Reviewer 有**破坏者视角**——从相反方向思考，找一切可能出错的地方。
- 首轮即通过（如 G1/G2）= 问题定义清晰、修改范围小，单点修复不需多轮。
- 需多轮（如 G3/B1）= 初始需求有歧义空间，文档创作和新功能易引入假设性错误。

### 3.3 关键原则

- Developer 和 Reviewer **必须用不同 agent 实例**（上下文隔离、视角独立）。
- Reviewer 的 prompt 明确要求"辩证性地挑毛病"，而非"验证正确性"。
- 每轮审查后，**architect 只负责汇总发现并派发下一轮**，不参与具体修复。

---

## 4. 审查通过标准（v2）

**审查通过 = 静态审查零问题 + 单元测试全部 PASS + 复现确认问题已消失**

| 阶段 | 检查项 |
|------|--------|
| Developer 提交后 | 相关单元测试 PASS（Python: `run_all.py` / C++: `build_and_test.sh`） |
| Reviewer 审查时 | 复现原问题 → 确认修复后问题消失 → 静态代码审查零问题 |
| Architect 批次后 | 全量测试 PASS（Python + C++ 双端） |

> ⚠️ "全量测试 PASS" 仅涵盖**已注册到测试套件的测试**。未注册的二进制/脚本（如 C++ benchmark）architect 必须单独验证——见 [lessons-learned.md](./lessons-learned.md) 主题 3。

### 4.1 强制性运行时测试

静态审查有根本局限，以下五类问题**只有实际运行才能发现**：

| 问题类型 | 为什么静态分析发现不了 |
|----------|----------------------|
| API 行为误解 | 如 `reset_max_memory_allocated()` 后立即读取返回 0，只有 NPU 上执行才能确认 |
| 运行时竞态 | ATB graph 异步执行的 stream/完成时机，依赖硬件调度 |
| 性能回归 | 新增 sync 是否导致可测延迟，必须实际计时 |
| 兼容性 | torch_npu 版本差异、ATB 图编译失败，只有实际运行才能发现 |
| 数值精度 | 修改后余弦相似度是否仍 ≥ 0.99，必须用真实模型推理 |

---

## 5. Agent 派发 briefing 模板

每个 developer/reviewer agent 的 prompt 必须包含 6 要素：

1. **问题描述**：症状、影响范围、根因（如已知）。
2. **工作范围**：哪些文件需修改，哪些**不需要**修改（防止过度修改）。
3. **验收标准**：具体的、可验证的通过条件。
4. **禁止事项**：明确列出不允许的操作（如"不要修改其他函数"）。
5. **测试要求**：需要运行的具体测试命令。
6. **上报路径**：遇到不明确的情况 → 汇报给 architect，不要自己猜测。

---

## 6. 核心原则速查

| 原则 | 含义 |
|------|------|
| **复现优先于分析** | 收到 bug → 先跑命令建基线，再读代码 |
| **审查必须复现** | Reviewer 第一项工作是运行出问题的命令确认问题存在，再审查修复 |
| **交叉验证** | C++ 修改验证 Python 不受影响，反之亦然；一个平台通过 ≠ 全平台正常 |
| **测试注册检查** | 修改涉及的二进制/脚本若未在自动化测试中，同步添加 |
| **已知问题搜索** | 修复前搜 docs，避免重复劳动或忽视已记录缺陷 |
| **禁止假设** | 不清楚用 grill-me 问清楚，严禁猜测 |

---

## 7. 测试执行规范（用项目工具，不手设环境变量）

### 7.1 Python 测试

环境变量通过 `from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR` 自动加载 `.env`（`env.py` 查 `ATB_DOTENV_PATH` → 向上遍历 6 级 → 优先级 `os.environ > .env > default`）。

```bash
# 全量测试（自动用 .env 的 QWEN3VL_EMB_MODEL_DIR）
python atb_python_qwen3vl_embedding/tests/run_all.py
# 带 benchmark
python atb_python_qwen3vl_embedding/tests/run_all.py --benchmarks
# 仅特定测试
python atb_python_qwen3vl_embedding/tests/run_all.py --include '*e2e*'
# 排除 310P 特定测试（在 910B 上运行）
python atb_python_qwen3vl_embedding/tests/run_all.py --exclude '*310p*'
```

❌ 禁止 `export QWEN3VL_EMB_MODEL_DIR=...` / `export ASCEND_PLATFORM=910B`（绕过 .env）。

### 7.2 C++ 测试

必须用 `build_and_test.sh`，它自动：加载 .env → source Ascend 环境 → CMake configure+build → 生成/复用参考数据 → 检测 NPU 跑 CTest。

```bash
bash atb_cpp_llm/build_and_test.sh                    # 完整构建+测试
bash atb_cpp_llm/build_and_test.sh --no-test          # 仅构建
bash atb_cpp_llm/build_and_test.sh --no-refresh-refdata  # 复用已有参考数据
bash atb_cpp_llm/build_and_test.sh --no-refdata       # 跳过参考数据及依赖测试
bash atb_cpp_llm/build_and_test.sh --test-only        # 仅测试（需已有 build/）
bash atb_cpp_llm/build_and_test.sh --debug            # Debug 构建
bash atb_cpp_llm/build_and_test.sh level1_cpu_pure    # 仅特定级别
bash atb_cpp_llm/build_and_test.sh test_text_model    # 仅特定测试名
```

❌ 禁止手写 cmake/ctest 命令、手动 source Ascend 脚本后再手动 cmake。

---

## 关联

- 踩坑视角（先复现、同型复发必闭环、修复前搜已知问题等教训）：[lessons-learned.md](./lessons-learned.md) 主题 7
- 预防机制检查清单（13 项）：`archive/architect-assessment-2026-06-16.md` §10.2
- 当前状态/待办：[STATUS.md](./STATUS.md)
