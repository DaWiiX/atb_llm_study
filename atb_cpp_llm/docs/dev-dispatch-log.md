# 派单开发历程（Dev Dispatch Log）

> 每次 Developer/Reviewer/Re-review 派单的**派法**（architect briefing 要点）+ **subagent 结果**（做了什么、验收数据、发现的问题）。按时间倒序追加，单一文件不拆批。
>
> 记录纪律见 [WORKFLOW.md](./WORKFLOW.md) §5.1。Reviewer 发现归档进 [lessons-learned.md](./lessons-learned.md)（§3.3 审查发现归档纪律）。

---

## 2026-06-23 ｜ benchmark 接入路径C ｜ Developer（重派）

**派法**：
- 角色：Developer。把 benchmark.cpp 接入路径 C（引擎内 NPU preprocess），让 e2e 用上 NPU preprocess。
- 工作范围：`tests/benchmark.cpp` 三处 PreprocessImage 调用点（行 543/671/873）。
- 关键约束：① bench 模式改 raw_image + IMAGE_AND_TEXT（触发路径 C），删 pixel_values host buffer、删 preprocessed 填充；② **compare 模式保留 CPU PreprocessImage**（需存 `/tmp/cpp_pv_*.bin` 给 Python 对比，路径 C 下 device tensor 不出引擎拿不到 CPU 副本）；③ actual_patches/grid_thw/vis_tokens 几何仍引擎外用 SmartResize 算（构造 input_ids 需要，纯几何不依赖 preprocess 实现）；④ 计时从引擎外 pre_ms 迁到 StageTimings.preprocess_ms（RunBenchmark 已返回 results[i].timings）。
- 验收：构建零 warning；bench 模式走 raw_image（日志有 path C no D2H）、preprocess_ms 从 timings 取、e2e 数字产出；compare 模式不破；4 分辨率 e2e 对比改前。

**结果**：（待 subagent 返回后填）
