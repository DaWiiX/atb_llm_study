# 文档索引

## 核心设计文档

| 文件 | 描述 |
|------|------|
| [design.md](./design.md) | 核心架构设计 v2 — 分层架构、IModel 接口、组件图、多模型适配 |
| [refactoring-plan.md](./refactoring-plan.md) | 重构计划（Phase 0–19）、踩坑经验集、基准报告、开发检查清单 |
| [optimization-roadmap.md](./optimization-roadmap.md) | NPU 性能优化长期路线 — P0 到 P8 逐阶段收益、冷热缓存分析 |
| [testing-guide.md](./testing-guide.md) | 测试策略指南 — Level 0~4 测试金字塔、精度标准、反模式 |
| [cpp11-compat-todo.md](./cpp11-compat-todo.md) | C++17→C++11 降级待办 — `make_unique` / 泛型 lambda 改造清单 |

## 历史归档

已完成任务的执行记录，按时间倒序保留在 [archive/](./archive/)：

| 文件 | 描述 |
|------|------|
| [archive/P5-execution-log.md](./archive/P5-execution-log.md) | P5 Deepstack 端到端 NPU 化 — 10 个原子任务派发/验证记录 |
| [archive/benchmark-report.md](./archive/benchmark-report.md) | P8 子阶段性能明细 + 缓存行为分析 + 性能演变历史（基准数据见 refactoring-plan §3） |
| [archive/test-coverage-log.md](./archive/test-coverage-log.md) | 测试覆盖扩展记录 — 33 个测试二进制 / 100+ cases 构建过程 |
| [archive/test-hardening-log.md](./archive/test-hardening-log.md) | 测试加固审计 — P0/P1 修复记录 + P2 (18 项低优先级待办) |

## 快速导航

- 想了解整体架构 → [design.md](./design.md)
- 想了解当前状态、已完成的 Phase、踩坑经验 → [refactoring-plan.md](./refactoring-plan.md)
- 想了解性能优化历程和下一步方向 → [optimization-roadmap.md](./optimization-roadmap.md)
- 想写测试或理解测试标准 → [testing-guide.md](./testing-guide.md)
