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

**结果**：
- 改 `tests/benchmark.cpp` 四处(两个函数 RunMultimodalBenchmark/RunImageOnlyBenchmark):预处理+请求构造按 cmp_mode 分支(!cmp_mode 走 raw_image 路径C,cmp_mode 保留 CPU);计时迁移(!cmp_mode 用 MeanStage(preprocess_ms),cmp_mode 用墙钟)。第三处(RunCompareMode)未改,始终 compare 保留 CPU。
- 关键修正:`actual_patches = grid_h*grid_w`(grid_t=1),非原 `num_patches`(grid_t=2 仅 buffer 超分配)。路径 C 不调 PreprocessImage,误用 num_patches 会导致 vis_tokens 翻倍。
- 验收数据(910B,5iter/3warmup),preprocess_ms CPU→路径C:416 27.68→12.46(-55%)、720 93.23→38.67(-59%)、1080 137.01→83.24(-39%)、1440 155.88→150.17(-3.7%)。staged_sum 公平对比 4 分辨率路径C全更优。
- ⚠️ e2e_mean 不可直接对比:路径C把 preprocess 折入引擎内 e2e(PREPROCESSED 模式引擎置 preprocess_ms=0,外置 CPU preprocess 不计入 e2e),口径迁移非回归。公平对比看 staged_sum。
- 验证:构建零 warning;bench 走路径C(日志 path C no D2H);compare 保留 CPU 存 /tmp/cpp_pv_*.bin;test_path_c_raw_image cos=0.999956 不破。
- 坑:grid_t 不一致(benchmark 原用2算num_patches,引擎用1);路径C日志是 LOG_INFO 需 LOG_LEVEL=INFO 可见;--mode all/throughput/cold 顺带用上路径C。
- 遗留:e2e 口径迁移需 Reviewer 确认 staged_sum 对比公平性;grid_t 不一致是既有代码隐患(路径C暴露),可单独记 lessons。

## 2026-06-23 ｜ benchmark 接入路径C ｜ Reviewer

**派法**：破坏者审查 benchmark.cpp 路径C接入。重点：e2e 口径迁移公平性(Developer 说看 staged_sum 不看 e2e)、grid_t 不一致坑、compare 完整性、计时正确性、--mode all/throughput/cold raw_image 生命周期、内存所有权。

**结果**：
- 零 BLOCKER。核心结论成立：e2e 口径迁移**公平且正确**(路径C e2e 含 preprocess、PREPROCESSED 不含,直接比 e2e 让路径C看着慢 46-147% 是口径差非真慢)；staged_sum 口径一致(两路径都含 pre_ms)，非 preprocess 阶段两路径差 <1.2ms 证明**路径C无额外开销**；staged_sum 4 分辨率路径C全更优因 NPU preprocess 更快。
- staged_sum 实测(路径C vs CPU)：416 38→54、720 104→161、1080 181→235、1440 247→256(ms)。
- [MAJOR] pre_ms cold-vs-warm 口径不对称：路径C pre=MeanStage(warm per-iter 均值),cmp_mode pre=墙钟单次 cold + 含 host 编排。3/4 分辨率 margin 大结论稳健,1440 margin 仅 8ms(3.6%)需确认 cold 量级。建议 cmp_mode 加 warmup 或标注口径。
- [MINOR] grid_t 不一致：既有代码(main 已有,grid_t=2 算 num_patches 超 buffer,引擎用 grid_t=1)，路径C 逼其显式化(actual_patches=grid_h*grid_w)。当前单图无 bug,视频/多帧 grid_t=2 会静默错算 vis_tokens。应归 lessons。
- compare 完整、计时正确、--mode all/throughput/cold 通过(raw_image 生命周期 OK)、内存所有权无泄漏(Adopt 置空源 deviceData)。
- 待办：benchmark.cpp 未 commit；MAJOR 处理；grid_t 补 lessons。

## 2026-06-23 ｜ benchmark 接入路径C ｜ Re-review(修MAJOR)

**派法**：Developer 修 Reviewer MAJOR(cmp_mode pre_ms cold-vs-warm 口径不对称)+ 确认 benchmark.cpp 提交状态。方案:cmp_mode 加 PreprocessImage warmup 再计时,且计时收窄为仅 PreprocessImage(排除 SmartResize/host 编排),与路径C StageTimings.preprocess_ms 口径对齐。

**结果**：
- cmp_mode 两函数(RunMultimodalBenchmark/RunImageOnlyBenchmark)加 warmup(临时 buffer 不污染正式 pixel_values),pre_start/pre_end 移入 cmp_mode 分支只包裹 PreprocessImage。
- 修后对比(路径C NPU warm均值 vs cmp CPU warm单次)：416 12.33 vs 20.89、720 37.30 vs 74.20、1080 82.92 vs 121.65、1440 146.55 vs 133.63。
- 1440 方向翻转(修前 cmp 偏高+8ms,修后 cmp 反低 12.9ms)——证明原 margin 是 cold+host 编排假象,口径对称成功。
- grid_t 坑已归 lessons 主题1第6条。
- 构建零 warning;bench/compare/test_path_c 全 EXIT=0;未 commit 等 architect 统一提交。
