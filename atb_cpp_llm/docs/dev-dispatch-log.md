# 派单开发历程（Dev Dispatch Log）

> 每次 Developer/Reviewer/Re-review 派单的**派法**（architect briefing 要点）+ **subagent 结果**（做了什么、验收数据、发现的问题）。按时间倒序追加，单一文件不拆批。
>
> 记录纪律见 [WORKFLOW.md](./WORKFLOW.md) §5.1。Reviewer 发现归档进 [lessons-learned.md](./lessons-learned.md)（§3.3 审查发现归档纪律）。

---

## 2026-06-25 ｜ 测试体系整顿 阶段3 testing-guide 文档 ｜ Developer

**派法**：
- 角色：Developer。只改 docs，编写 testing-guide / testing-architecture，并更新索引、流程纪律、状态、教训、派单日志。
- 工作范围：`atb_cpp_llm/docs/evergreen/testing-guide-dev.md`、`atb_cpp_llm/docs/evergreen/testing-architecture.md`、`README.md`、`WORKFLOW.md`、`STATUS.md`、`lessons-learned.md`、`dev-dispatch-log.md`。
- 关键约束：不改代码/测试；不要把 benchmark compare 写成 official gate；310P limitation 必须诚实；文档要可执行。
- 验收：新文档能被 README 找到；WORKFLOW 有测试覆盖矩阵纪律；STATUS 有阶段3记录；dev-dispatch-log 有记录；grep 能看到 `test_engine_vs_official` / `testing-guide-dev` / `testing-architecture`。

**结果**：
- 新增 `evergreen/testing-guide-dev.md`：写明 official gate 优先、修改类型必跑矩阵、910B 推荐命令顺序、Reviewer checklist、假阳性 grep 自检、310P AA limitation。
- 新增 `evergreen/testing-architecture.md`：写明 L0–L4/benchmark 分层、official `Qwen3VLEmbedder` 真相源原则、覆盖矩阵、测试语义分类、决策树、refdata discipline。
- 更新 `README.md` evergreen 索引和阅读路线；`WORKFLOW.md` 增加“测试覆盖矩阵纪律”；`STATUS.md` 记录阶段3；`lessons-learned.md` 主题4补充 official gate / self-consistency / diagnostic 区分；本文件追加派单记录。
- 文档未运行 NPU 测试；验收以静态 grep 和 diff 审查为主。

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

## 2026-06-24 ｜ 910B PreprocessImageNpu 精度对标官方 ｜ Developer(实验探查)

**派法**:
- 角色:Developer。闸口任务——用实验测清 910B AA/非AA NPU 全管线 vs 官方 transformers 参考的精度,决定 AA 去留。不改生产代码,不降阈值。
- 背景前提(architect 给的,后被实验推翻):architect 认为"官方 PIL BICUBIC 是非 AA(image.resize((w,h)) 无 pre-filter)",据此怀疑 910B AA 路径偏离官方、应去掉。要求用实验说话验证。
- 关键约束:参考必须是官方 transformers(qwen-vl-utils + Qwen3VLProcessor do_resize=False),不是 Python ATB、不是 PIL 单算子。4 个生产分辨率。

**结果**:
- ⚠️ 推翻 architect 前提:官方 PIL 降采样**本身是 AA**(Pillow BICUBIC 降采样内置 support 扩展)。architect 用来判定"非 AA"的 max_diff=0 测试在比两次都带 AA 的 PIL 调用,无效。用 upscale 隔离法证明:PIL 核=Catmull-Rom(=CPU 核,upscale cos 0.99999),但降采样 PIL 走 AA、CPU 不走,同核降到 0.80-0.95(噪声 1440)。所以 spike "非 AA vs PIL 0.987" 是对的,PIL 参考没带错 antialias。
- 数据表(全管线 cosine vs 官方):416 非降采样 NPU非AA=1.0;720/1080/1440 降采样 NPU-AA vs 官方=0.99998/0.99999/0.999996(达标),NPU-非AA vs 官方=0.9991/0.9537/0.8022(破 0.99),CPU-非AA vs 官方同 NPU-非AA。
- 结论:**AA 路径是对的,不该去掉**。910B NPU AA 才匹配官方。
- 实验反而暴露 3 个真问题:① CPU PreprocessImage(非AA)降采样偏离官方,而 benchmark compare/test_accuracy/test_stage_precision 全走 CPU 参考→这些"精度验证"参考本身错,两个错参考互比 cos=1.0 掩耳盗铃;② 910B(NPU-AA)与 310P(NPU-非AA)降采样分歧~0.80;③ NPU 核 Mitchell a=-0.75 ≠ 官方/CPU Catmull-Rom a=-0.5(upscale 就 0.999)。
- 产出:/tmp/measure_aa*.py + /tmp/spike_out.txt。未改 repo。
- 决定留给 architect:精度目标口径(对齐官方高频图 vs 自然图分布)。

## 2026-06-24 ｜ max_pixels 对齐官方 Qwen3VLEmbedder(1310720→1843200) ｜ Developer

**派法**:
- 角色:Developer。把引擎 resize max_pixels 从 config(1310720)改为 embedder 常量 1843200,对齐官方 qwen3vl_embedding.py:28。终极目标:引擎输出 vs 官方 Qwen3VLEmbedder 推理 cosine≥0.99。
- 闸口(已验明):官方 Qwen3VLEmbedder 推理 max_pixels 恒=1843200(do_resize=False + process_vision_info 注入 content dict),preprocessor_config.json 的 1310720 在 do_resize=False 时不参与 resize。我们 C++ 读 config 1310720、Python preprocess_image 默认 1310720(Python smart_resize 默认 1843200,自身分裂)。
- 真相源方案:embedder 常量 kQwen3VLEmbeddingMaxPixels=1800*32*32,config.cpp 删除 max_pixels 读取,所见即所做(避免 footgun 复发)。Python 镜像。
- 关键约束:只有 1080×1920/1440×2560 grid 变(832×1504→992×1792,4888→6944 patches);416×672/720×1280 及 toy/stage/e2e 图(≤800×600)grid 不变。MAJOR-1:gen_official_pixel_values.py 重写为 do_resize=False+process_vision_info 真官方链(之前 do_resize=True+1310720 是 processor-config 链,虚假宣称 official)。.bin 读取全动态尺寸变大无需改读逻辑。不降阈值。单批不拆(真相源+参考重生+测试断言原子耦合)。

**结果**:
- 生产代码:qwen3vl_config.h 新增 kQwen3VLEmbeddingMaxPixels 常量 + struct 默认;config.cpp 删除 max_pixels 读取(注释说明 embedder 覆盖);engine.py pp_max_px=1800*32*32 不读 config;preprocess.py:52 默认 1843200。
- 参考生成:gen_official_pixel_values.py 重写为真官方链(Qwen3VLProcessor.from_pretrained + conversation 注入 min/max_pixels + process_vision_info(image_patch_size=16) + processor(do_resize=False));gen_cpu_reference.py 4 处 1310720→1843200。
- 测试:test_aclnn_bicubic_spike 3 处 cfg.pp_max_pixels→1843200 + 注释/LOG 诚实标注官方链;test_preprocess_cpu/test_io_adapters/test_config_wiring/test_preprocess 断言更新。io_adapters "自定义 config max_pixels=1000000" 用例转为覆盖守卫(断言 ==1843200 + min_pixels==3136 证明 config 加载成功)。
- test_config_wiring 偏差处理:C++ 删 config 读取后 pp_max_pixels 恒 1843200,故 gen_cpu_reference:2233 + test_config_wiring.py:92 用常量 1843200(不读 config)三方一致(若按计划字面 pp.get 会读到 config 的 1310720 破三方)。
- 终极验收(910B):4 分辨率 NPU 全管线 vs 官方 cos = 1.0/0.999924/0.999878/0.999951,全过 0.99。AA spike 4 分辨率 vs PIL cos≥0.9999 不回归。official_pv count:416/720 不变(1092/3520 patches),1080/1440→6944 patches(992×1792)。
- C++ 全量(test_preprocess_cpu/test_io_adapters/test_config_wiring/test_vision_stages/test_path_c/test_e2e)全 PASS。Python 抽查(test_preprocess/test_e2e/test_embedder_e2e)全 PASS。grep 1310720 仅剩 3 处解释性历史注释。
- 显存:vision PA_ENCODER flash 线性,6944 grid 910B 64GB + buffer_pool +25% 自动扩容,无 OOM。

## 2026-06-24 ｜ max_pixels 对齐官方 ｜ Reviewer(破坏者)

**派法**:破坏者审查。重点:gen_official_pixel_values.py 真官方链(MAJOR-1 核心)、真相源彻底性、覆盖守卫、残留 1310720、grid 变大显存、阈值未降、范围合规。

**结果**:
- 零 BLOCKER、零 MAJOR(MAJOR-1 彻底解决)。
- **决定性 probe**:用真实 Qwen3VLEmbedder 的 unbound 方法(format_model_input + _preprocess_inputs)+ stand-in(只带必要属性不加载 2B 权重)对 4 分辨率产出 pixel_values,与 gen_official bin 对比——**4 分辨率全 bit-exact(max_diff=0.0)**。证明 gen_official 是真官方链,比 cos 阈值更硬。
- 真相源彻底:config.cpp GetInt("max_pixels") 已删除,无遗漏读取点。覆盖守卫有效(io_adapters min_pixels==3136 证明 config 加载成功,max_pixels 被故意忽略)。test_config_wiring 三方一致。残留 1310720 仅 3 处解释注释。grid 变大 910B 无 OOM。阈值全 0.99 未降。
- 1080 cos=0.999878 低于计划预估 0.99999 的根因:参考链 bit-exact(已证),差异是 NPU fp16+bicubic-AA vs 官方 fp32 在 worst-case 随机噪声输入下的固有差异(自然图会更高)。非参考链瑕疵。
- [MINOR-1] run_all.py + test_text_diagnostics.py 的 shutdown 修复混在同一 working tree(先前任务遗留,与 max_pixels 无关),建议拆提交。
- [MINOR-2] AA spike TEST_CASE 2 加 310P skip(防御性,aclnnUpsampleBicubic2dAA 310P 不支持会 crash),910B 全跑,无害。
- 结论:可合并(APPROVE)。

## 2026-06-24 ｜ max_pixels 对齐官方 ｜ Re-review

**派法**:验证 Reviewer 清单零问题 + 全量测试不回归 + 终极 cos 落地 + MINOR-1 提交拆分。

**结果**:
- Reviewer 清单零 BLOCKER/MAJOR,2 MINOR(MINOR-1 提交拆分已执行;MINOR-2 防御性 skip 保留)。
- C++ 全量 7 测试 PASS:test_config_wiring/test_e2e/test_path_c_raw_image/test_vision_stages/test_aclnn_bicubic_spike/test_preprocess_cpu/test_io_adapters,零回归(298s)。
- 提交拆分:① fix(tests) test_text_diagnostics shutdown + run_all timeout(独立);② max_pixels 对齐 + official gate 基建 + MAJOR-1 参考链修复(原子合并,避免破窗中间态)。
- 终极目标达成:910B 4 生产分辨率 PreprocessImageNpu 全管线 vs 官方 Qwen3VLEmbedder 推理 cos≥0.99(最低 0.999878)。

## 2026-06-25 ｜ 测试体系整顿 阶段0 精度盲区实测 ｜ Developer

**派法**:
- 角色:Developer。只跑实验,不改 repo。脚本/输出放 `/tmp`。
- 目标:在 max_pixels=1843200 后,实测官方 Qwen3VLEmbedder 链 vs CPU PreprocessImage 链的 pixel_values 差距,并查 path C full embedding vs 官方是否有现成验证入口。
- 输入:4 个生产分辨率(416/720/1080/1440),noise(seed=2026 shared input bin) + natural(`/mnt/workspace/gitCode/Qwen3-VL-Embedding/data/examples/0.jpeg` resize 到目标尺寸)。
- 参考链:官方链为 Qwen3VLProcessor + conversation 注入 min_pixels=4096/max_pixels=1843200 + process_vision_info(image_patch_size=16) + processor(do_resize=False);CPU 链为 `atb_python_qwen3vl_embedding.preprocess.preprocess_image`。

**结果**:
- 产物:`/tmp/measure_blind_spots.py`, `/tmp/measure_blind_spots_A_results.json`, `/tmp/test_aclnn_bicubic_spike_phase0_with_official.log`。repo 未修改。
- CPU(非AA) vs 官方(AA) pixel_values cosine:
  - noise:416=1.000000000,720=0.999102734,1080=0.995964392,1440=0.902562175(1440 破 0.99)
  - natural:416=1.000000000,720=0.999994070,1080=0.999991300,1440=0.999992786(全过)
- NPU pv vs official pv(复用 spike):416=1.000000,720=0.999924,1080=0.999878,1440=0.999951;NPU pv vs CPU pv:416=1.000000,720=0.999786,1080=0.994999,1440=0.899187。
- 结论:910B NPU AA 路径对官方是对的;CPU 非AA 是偏离官方的路径,尤其 high-frequency/noise 1440。`benchmark compare`/`test_accuracy` 用 CPU/Python 自家链互比,对"官方精度"是弱信号/假信号。
- path C full embedding vs 官方未完成:现有入口不能直接保存 path C raw_image full engine embedding bin。`benchmark --mode compare` 可保存 bin 但走 PREPROCESSED/CPU;bench/all/cold/throughput 走 raw_image path C 但 `RunBenchmark(..., nullptr, ...)` 不保存输出;`test_path_c_raw_image.cpp` 只 96×96 且只自比对。需要新增 `test_engine_vs_official.cpp` 或给 benchmark 增加 path C 保存输出能力。

## 2026-06-25 ｜ 测试体系整顿 阶段0 精度盲区实测 ｜ Reviewer

**派法**:
- 角色:Reviewer。破坏者审查阶段0数据可信性,避免基于错数据推进阶段1/2。
- 审查重点:`/tmp/measure_blind_spots.py` 是否真用官方 Qwen3VLEmbedder 链(do_resize=False + process_vision_info + per-item 1843200)、CPU 链是否调用当前 `preprocess_image`、noise/natural 输入是否同源、cos/max_diff 是否有 shape/grid/flatten 错误、path C full embedding blocker 是否真实。

**结果**:
- APPROVE。零 BLOCKER、零 MAJOR。
- 独立复核 1440×2560 noise:official/cpu shape 均 `(6944,1536)`,grid 均 `[[1,62,112]]`,cos=0.9025621751299795,max_diff=1.0823148488998413,与 JSON 一致。确认这是 CPU 非AA vs 官方 AA 的真实差异,非脚本误差。
- 官方链确认:Qwen3VLProcessor + process_vision_info(image_patch_size=16)+processor(do_resize=False)+per-item max_pixels=1843200。CPU 链确认调用当前 repo `preprocess_image` 默认 1843200。noise 输入来自 fixed seed shared bin,natural 输入同一 uint8 图喂 A/B。
- path C full embedding blocker 成立:benchmark 非 compare 的 raw_image path C 调 `RunBenchmark(..., nullptr, ...)` 不保存 output;`--mode compare` 能保存但走 PREPROCESSED/CPU;`test_path_c_raw_image.cpp` 只 96×96 自比对且不保存 bin。未发现现成保存入口。
- [MINOR] 阶段0脚本 system prompt 用 `Represent the given image.`,官方默认是 `Represent the user's input.`;但阶段0只比 pixel_values,文本不影响结论。阶段2 full embedding gate 必须严格对齐官方 prompt/chat template/token。
- 影响判断:benchmark compare/test_accuracy 的 CPU/Python 自家链互比是弱信号,不能证明 vs 官方;阶段2 必须新增 path C raw_image full embedding vs official gate。

## 2026-06-25 ｜ 测试体系整顿 阶段1 假阳性退码修复 ｜ Developer

**派法**:
- 角色:Developer。只修已知"已经计算 cosine 但失败不退码"的假阳性,不新增官方参考、不重写 e2e。
- 范围:`test_vision_stages.cpp` L2a/L3 等已有 LOG_ERROR 不 CHECK 的位置;`test_pipeline_trace.py` trace 打 PASS/FAIL 但 main 恒 return 0。
- 禁止:不改阈值、不改输入、不改参考、不碰 `test_e2e.cpp`(阶段2 official full gate 处理)。

**结果**:
- `test_vision_stages.cpp`:原 manual main 改为 `RunVisionStages()` + doctest `TEST_CASE`;所有已有 cosine failure log 后加 `CHECK`(L0/L1 0.999,L2a 0.99,L2b 0.999,L3a/L3b 0.99),失败会让 CTest FAIL。
- `test_pipeline_trace.py`:trace() 返回 bool,main 聚合 all_ok,最后 `return 0 if all_ok else 1`;Step 8+ alternate transformers call path 标为 diagnostic/non-gated(已知低 cosine 观测,非 ATB-vs-reference gate);保留 del ref + torch.npu.empty_cache()。
- 验证:build test_vision_stages + ctest PASS;python test_pipeline_trace.py exit 0;grep 确认 LOG_ERROR.*cos 后均接 CHECK,main 不再无条件 return 0。

## 2026-06-25 ｜ 测试体系整顿 阶段1 假阳性退码修复 ｜ Reviewer

**派法**:破坏者审查 worktree diff。重点:manual main→doctest 是否破坏 CTest;所有 LOG_ERROR.*cos 是否接 CHECK;pipeline trace 是否把核心 gate 纳入 all_ok;Step 8+ diagnostic 是否合理;阈值/输入/参考是否未改。

**结果**:
- APPROVE。零 BLOCKER、零 MAJOR、零必须修复 MINOR。
- Diff 范围只改预期两个文件。`test_vision_stages.cpp` 单源 target,`DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN` 不会重复 main;`RunVisionStages()` 语义等价,`CHECK(RunVisionStages()==0)` 接入退码。所有 `LOG_ERROR(...cos...)` failure 分支后均有 CHECK,阈值未降低。实测 L2/L3 margin 充足(L2a/L2b 1.0,L3a 0.999999,L3b 0.999997)。
- `test_pipeline_trace.py`:Step 1-7 核心 gate 均参与 all_ok,Step6 原周期性 layer cosine 也 gate,Step7 final hidden_state gate;Step 8+ alternate transformers call path 明确 diagnostic/non-gated 合理;manual vs ref-captured inputs 数值对比仍 gate;main 不再无条件 return 0;清理保留。
- 复现:build/ctest test_vision_stages PASS;python test_pipeline_trace.py exit 0。

## 2026-06-25 ｜ 测试体系整顿 阶段1 假阳性退码修复 ｜ Re-review

**派法**:合回主工作区后复跑关键测试并确认退码路径。

**结果**:
- 主工作区合回两个文件后,`cmake --build atb_cpp_llm/build --target test_vision_stages -j8 && ctest --test-dir atb_cpp_llm/build -R test_vision_stages --output-on-failure` PASS。
- `python atb_python_qwen3vl_embedding/tests/test_pipeline_trace.py` exit 0。输出显示 Step 1-7 gated 全 PASS,Step 8+ 低 cosine 均标 `(diagnostic, not gated)`,manual vs ref-captured inputs gate 全 PASS。
- 阶段1目标达成:两个已知假阳性测试现在能把核心 cosine failure 传递为测试失败。

## 2026-06-25 ｜ 测试体系整顿 阶段2 path C full embedding vs 官方 gate ｜ Developer

**派法**:
- 角色:Developer。重新派发 fresh worktree,新增 910B path C raw_image 全 engine embedding vs 官方 Qwen3VLEmbedder pooled embedding gate。
- 关键约束:真实 Qwen3VLEmbedder public API 或等价链;默认 prompt 必须是 `Represent the user's input.`;tokens 必须从同一次官方 preprocess 捕获,不能手写;4 个生产分辨率;`CHECK cos>=0.99`;310P skip;refdata/CTest 登记完整。不改 benchmark 性能路径。

**结果**:
- 新增 `gen_official_embedding.py`:实例化真实 `Qwen3VLEmbedder(MODEL_DIR)`,调用 public `embedder.process(inputs, normalize=True)` 生成官方 pooled embedding;wrapper 捕获同一次 `_preprocess_inputs` 的 `input_ids` 写 token bin;guard 默认 prompt/max_pixels;public embedding vs captured chain `max_diff==0`;用 `torch_npu.contrib.transfer_to_npu` 让官方 cuda 设备映射到 NPU(仅生成器进程内)。
- 新增 `test_engine_vs_official.cpp`:910B 执行,非 910B skip;读取 shared input bin + official tokens/embed bins;构造 `InferRequest mode=IMAGE_AND_TEXT` + `raw_image` 触发 path C;校验 image token 数==SmartResize 后 merged_tokens;`CHECK cos>=0.99`;检查 `timings.preprocess_ms>0`。
- 注册:CMake target + RESOURCE_LOCK npu_e2e + needs_refdata;gen_all 注册;build_and_test sentinel 加 official_embed/tokens。
- 生成参考:416 tokens=295 image_tokens=273 grid[[1,26,42]],720 tokens=902 image_tokens=880 grid[[1,44,80]],1080/1440 tokens=1758 image_tokens=1736 grid[[1,62,112]],dim=2048,public_vs_internal_max_diff=0。
- 验证:新 target build PASS;`ctest -R test_engine_vs_official` PASS(43.9s),cos=416 0.999882 / 720 0.999235 / 1080 0.999469 / 1440 0.999690;`test_aclnn_bicubic_spike` 仍 PASS。

## 2026-06-25 ｜ 测试体系整顿 阶段2 path C full embedding vs 官方 gate ｜ Reviewer/Re-review

**派法**:
- Reviewer agent 因 5-hour quota 429 未产出有效审查。主 agent 接手执行破坏者审查 + Re-review。
- 审查重点:official generator 真伪(prompt/max_pixels/public API/token同源),C++ 是否真 raw_image path C,token/image token 校验,refdata/CTest/RESOURCE_LOCK,310P skip,阈值未降。

**结果**:
- APPROVE。零 BLOCKER、零 MAJOR。
- 静态审查:diff 范围符合预期;`gen_official_embedding.py` 使用真实 Qwen3VLEmbedder public API,默认 prompt/max_pixels guard,token 从同一次 `_preprocess_inputs` 捕获,public vs internal max_diff=0;`transfer_to_npu` 仅生成器进程内可接受。`test_engine_vs_official.cpp` 确认为 `IMAGE_AND_TEXT + raw_image` path C,不是 PREPROCESSED;tokens 读 official bin;image token 数校验=merged_tokens;embedding loader/ cosine 维度检查严格;阈值 0.99 未降。
- 注册完整:CMake target,RESOURCE_LOCK npu_e2e,needs_refdata,gen_all,built_and_test sentinels 均已登记。
- Re-review 主工作区:先运行 `gen_official_embedding.py` 成功(4 分辨率 public_vs_internal_max_diff=0),首次 build 因 build 目录未 reconfigure 找不到新 target;重新 `cmake -S atb_cpp_llm -B atb_cpp_llm/build` 后 build `test_engine_vs_official` PASS,`ctest -R test_engine_vs_official` PASS(44.8s)。`ctest -R test_aclnn_bicubic_spike` PASS(22.2s)。
- 阶段2目标达成:新增 CTest official full embedding gate,910B 4 生产分辨率 path C raw_image 全 engine embedding vs 官方 pooled embedding cos≥0.99。

## 2026-06-25 ｜ 测试体系整顿 阶段3 testing guide ｜ Reviewer

**派法**:
- 角色:Reviewer。破坏者审查阶段3 docs-only 改动。重点:official gate / self-consistency / diagnostic 是否分清;benchmark compare 是否被误写 official gate;310P limitation 是否诚实;命令是否可执行;README/WORKFLOW/STATUS/lessons/dev-dispatch-log 是否索引正确;确认 docs-only 范围。

**结果**:
- 审查结论:不 approve 直接合回,需修 2 BLOCKER + 2 MAJOR。
- [BLOCKER-1] 不能整条 worktree branch 合回:full branch diff 含代码/测试,但阶段3工作区自身是 docs-only。必须只拷 7 个 docs 文件。
- [BLOCKER-2] 两个核心新文档 `testing-guide-dev.md` / `testing-architecture.md` 是 untracked,普通 diff/提交可能漏掉。合回必须显式包含。
- [MAJOR-1] `benchmark --mode compare` 命令链不完整:只跑 C++ benchmark + compare_py_cpp 会缺 `/tmp/py_*.bin`;必须补 Python benchmark `--load-pixel-values` 生成 py 输出。
- [MAJOR-2] official gates 命令缺 CMake configure/reconfigure;干净 checkout 或旧 build cache 下新 target 不存在。必须补 `cmake -S ... -B ...`。
- 准确性方面通过:文档没有把 benchmark/test_accuracy 误写 official;`test_engine_vs_official`/`test_aclnn_bicubic_spike` 语义准确;310P limitation 诚实。

## 2026-06-25 ｜ 测试体系整顿 阶段3 testing guide ｜ Re-review

**派法**:
- 合回主工作区时只显式拷贝 7 个 docs 文件,不合整条 worktree branch;补 Reviewer 指出的 MAJOR 命令问题;运行文档静态验证。

**结果**:
- 合回范围仅 docs:`testing-guide-dev.md`,`testing-architecture.md`,`README.md`,`WORKFLOW.md`,`STATUS.md`,`lessons-learned.md`,`dev-dispatch-log.md`。
- 修 MAJOR-1:benchmark 矩阵与命令链补 `python atb_python_qwen3vl_embedding/tests/benchmark.py --mode all --load-pixel-values`,并说明 C++ compare 只生成 `/tmp/cpp_*.bin`,Python benchmark 生成 `/tmp/py_*.bin`,最后 compare_py_cpp 才能完成 13 case。
- 修 MAJOR-2:official gates 命令前补 `cmake -S atb_cpp_llm -B atb_cpp_llm/build -DCMAKE_BUILD_TYPE=Release`。
- 验证:`git diff --check` 通过;README/WORKFLOW/STATUS grep 可见 testing docs / official gate;guide 中可见 cmake configure 与 load-pixel-values 步骤。
- 阶段3 docs 目标达成。
