# 文档索引

> 本目录是 C++ 引擎 + Python 引擎的文档主页。**查现状先看 [STATUS.md](./STATUS.md)，查坑先看 [lessons-learned.md](./lessons-learned.md)。**

## 三层结构

文档按「时效 × 用途」分层，不是按时间堆叠：

| 层 | 性质 | 维护方式 |
|----|------|----------|
| 根级 | 索引 + 真相源 + 工作流 + 教训 | 持续更新 |
| `evergreen/` | 常青参考（现在仍有效的知识） | 持续维护 |
| `archive/` | 已冻结快照（历史记录，**不代表现状**） | 只读 |

---

## 根级（持续更新）

| 文件 | 描述 |
|------|------|
| [STATUS.md](./STATUS.md) | **当前状态单一真相源** —— 做完什么 / 还剩什么。查进度只看这里 |
| [WORKFLOW.md](./WORKFLOW.md) | **工作方法论** —— architect 编排、Developer→Reviewer→Re-review 循环、审查通过标准、agent briefing 模板 |
| [lessons-learned.md](./lessons-learned.md) | **踩坑经验集** —— 按主题组织，带触发关键词，做事前检索 |
| [dev-dispatch-log.md](./dev-dispatch-log.md) | **派单开发历程** —— 每次 Developer/Reviewer 派单的派法 + subagent 结果，复盘传承素材 |

## evergreen/（常青参考）

| 文件 | 描述 |
|------|------|
| [design.md](./evergreen/design.md) | 核心架构设计 v2 — 分层架构、IModel 接口、组件图、多模型适配 |
| [platform-310p.md](./evergreen/platform-310p.md) | 310P 平台适配 — NZ mask 策略、兼容性矩阵、运维指南、实测经验 |
| [deployment-checklist.md](./evergreen/deployment-checklist.md) | 生产部署逐项核查清单 + 一键诊断脚本 + ATB 错误码速查 |
| [optimization-roadmap.md](./evergreen/optimization-roadmap.md) | NPU 性能优化长期路线 — P0 到 P8 逐阶段收益、冷热缓存分析 |

## archive/（已冻结快照）

已完成或被超越的计划/评估，**只读历史记录，不代表现状**（现状看 STATUS.md）：

| 文件 | 描述 |
|------|------|
| [architect-assessment-2026-06-16.md](./archive/architect-assessment-2026-06-16.md) | 架构师评估报告 — 性能/鲁棒性/生产就绪审查 + 工作流 v2 + 预防机制 13 项 |
| [audit-fix-plan.md](./archive/audit-fix-plan.md) | 2026-06-14 架构审计 61 项修复计划（已 100% 修复） |
| [test-fix-plan.md](./archive/test-fix-plan.md) | 2026-06-15 测试套件审计 P1–P21（20/21 已修复，P4 待办） |
| [refactoring-plan.md](./archive/refactoring-plan.md) | C++ 框架重构 Phase 0–20 + 踩坑经验 + 基准报告 + 开发检查清单 |
| [cpp11-compat-todo.md](./archive/cpp11-compat-todo.md) | C++17→C++14 降级完成报告 |
| [testing-guide.md](./archive/testing-guide.md) | 2026-06-14 310P 验证执行日志（310P 现况看 evergreen/platform-310p.md） |
| [benchmark-report.md](./archive/benchmark-report.md) | P8 子阶段性能明细 + 缓存行为分析 + 性能演变历史 |
| [test-coverage-log.md](./archive/test-coverage-log.md) | 测试覆盖扩展记录 — 33 个测试二进制 / 100+ cases 构建过程 |
| [test-hardening-log.md](./archive/test-hardening-log.md) | 测试加固审计 — P0/P1 修复记录 + P2 (18 项低优先级待办) |
| [P5-execution-log.md](./archive/P5-execution-log.md) | P5 Deepstack 端到端 NPU 化 — 10 个原子任务派发/验证记录 |

---

## 阅读路线

- **第一次了解项目** → [evergreen/design.md](./evergreen/design.md)
- **查当前进度/待办** → [STATUS.md](./STATUS.md)
- **要派发/审查工作** → [WORKFLOW.md](./WORKFLOW.md)（方法论 + agent briefing 模板）
- **遇到坑/做事前** → [lessons-learned.md](./lessons-learned.md)（按主题 + 触发关键词）
- **要部署** → [evergreen/deployment-checklist.md](./evergreen/deployment-checklist.md)
- **310P 适配/验证** → [evergreen/platform-310p.md](./evergreen/platform-310p.md)
- **性能优化方向** → [evergreen/optimization-roadmap.md](./evergreen/optimization-roadmap.md)
- **了解某次审计/重构的历史细节** → `archive/` 对应文件

---

## 记录纪律（防止文档再乱）

文档乱的三个根因：状态真相分散漂移、踩坑按事件记录导致同类复发、一次性任务不退役。以下纪律针对性消除。

### 完成一批工作后的 3 步

1. **更新 [STATUS.md](./STATUS.md)**：勾掉完成项，加新待办。**不新建文档记录进度。**
2. **抽取教训**：若这次踩了/学到了新坑，归并进 [lessons-learned.md](./lessons-learned.md) 对应主题（没有就加主题），检查同主题是否有更弱的旧条目可替换。跨会话需主动 recall 的规则同步到 `.claude/.../memory/`。
3. **快照超越即冻结**：若某次审计/评估被新的工作超越，把它移入 `archive/`，不再当作现状参考。现状只在 STATUS.md。

### 何时新建文档（只有两种情况）

- **常青参考**：新子系统、新平台适配、长期路线 —— 放 `evergreen/`。
- **跨多会话的长期计划**：放根级或 `evergreen/`，并在 STATUS.md 链接。

**一次性任务不单独成文**：用 STATUS.md 的 todo 条目 + git 提交信息记录。否则就是 `archive/cpp11-compat-todo.md` 那样的孤岛（已完成却混在活动区）。

### 何时归档

- 计划/评估的所有项都完成或转入 STATUS.md 待办 → 移入 `archive/`。
- 文档内容已被另一篇完全覆盖 → 移入 `archive/`，在 README 注明被谁取代。

### 教训记录规范（写入 lessons-learned.md）

- 按**主题**归类，不按时间。同主题只留最强的一条，去重。
- 每条带**触发关键词**（脑海中冒出这些词时先查这里）。
- 标注**来源**（哪次事件/哪篇 archive），便于追溯上下文。
- **同型复发必须合并**：若新坑是旧坑的变体，合并为一条并注明"已复发 N 次"，泛化为通用规则而非针对单次事件。
