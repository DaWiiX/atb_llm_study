# 派单开发历程（Dev Dispatch Log）

> 每次 Developer/Reviewer/Re-review 派单的**派法**（architect briefing 要点）+ **subagent 结果**（做了什么、验收数据、发现的问题）。按时间倒序追加，单一文件不拆批。
>
> 记录纪律见 [WORKFLOW.md](./WORKFLOW.md) §5.1。Reviewer 发现归档进 [lessons-learned.md](./lessons-learned.md)（§3.3 审查发现归档纪律）。

## 2026-06-25 ｜ OFFICIAL_EMBED_CASES 统一 official embedding case 选择 ｜ Developer

**派法**：
- 角色：Developer。在 `.env` 新增 `OFFICIAL_EMBED_CASES` 配置项，统一 generator / C++ gate / gen_all sentinels / build_and_test sentinels 的 case 选择。
- 工作范围：`.env.example`、`atb_python_qwen3vl_embedding/env.py`、`atb_cpp_llm/tests/_tests_env.py`、`gen_official_embedding.py`、`test_engine_vs_official.cpp`、`gen_all.py`、`build_and_test.sh`。
- 关键约束：不引入 small/auto/profile —— 用户直接编辑 `.env` 里的逗号列表；`test_engine_vs_official` 不再整测 skip 310P；不降阈值。
- 验收：python -m py_compile 全通过、build_and_test C++ test_engine_vs_official 编译通过、git diff --check 通过。

**结果**：
- `.env.example`：新增 `OFFICIAL_EMBED_CASES=416x672,720x1280,1080x1920,1440x2560`；修 `QWEN3VL_EMB_SRC` 模板语义为 repo root。
- `env.py`：新增 `OFFICIAL_EMBED_CASES` 默认四个；修 `QWEN3VL_EMB_SRC` 注释。
- `_tests_env.py`：re-export `OFFICIAL_EMBED_CASES`。
- `gen_official_embedding.py`：新增 `parse_cases()` 解析逗号分隔 HxW；`main()` 按 `OFFICIAL_EMBED_CASES` 控制 case；保留 NPU first/CPU fallback；保留 `resolve_official_src()` 兼容 root/src。
- `test_engine_vs_official.cpp`：删除 `if (!Is910B()) return 0;`；改用 `GetEnv("OFFICIAL_EMBED_CASES", ...)` + `std::istringstream` 动态 case；编译通过。
- `gen_all.py`：`--skip-fresh` 的 official embedding sentinel 按 `OFFICIAL_EMBED_CASES` 展开首个 case；新增 `_parse_cases()`。
- `build_and_test.sh`：官方 embedding/token sentinel 按 `${OFFICIAL_EMBED_CASES}` 展开。
- 已有改动保留：`data_utils.empty_npu_cache_safe()`、test_e2e/test_embedder_e2e safe cleanup、test_stage_reference TEXT_ONLY fallback、visrope 注释、docs。
- 全 16 文件 diff，py_compile 全通过，git diff --check 通过，C++ 编译通过。
- 遗留：需 910B/310P 真机复测确认 `OFFICIAL_EMBED_CASES=416x672,720x1280` 行为。

## 2026-06-25 ｜ OFFICIAL_EMBED_CASES Reviewer 修复 ｜ Re-review

**结果**：全部修复通过，闭环。

- BLOCKER-1 ✅ `test_engine_vs_official.cpp`: 增加 `if (cases.empty()) { LOG_ERROR; return 1; }` 守卫空 case 列表假阳性。
- MAJOR-1 ✅ `gen_official_embedding.py:parse_cases()`: 移除 bracket 剥离 `strip('[]')`，统一为纯 `HxW,HxW` 格式。docstring 同步移除 bracket 示例。
- MAJOR-3 ✅ `test_engine_vs_official.cpp`: 无效 token 改为 `LOG_WARN` + skip，不再静默丢弃。
- MAJOR-4 ✅ `testing-guide-dev.md`: "CPU fallback 只生成 416/720" 改为 "CPU fallback 也按 OFFICIAL_EMBED_CASES 生成；310P 用户应自行在 .env 中只保留 416x672,720x1280"。
- py_compile 通过、git diff --check 通过。
- 遗留：C++ 构建需在 NPU 环境验证（当前环境无 CANN 工具链）。
- lessons-learned 追加 #15 空 case 守卫、#16 跨语言解析一致两条教训。


## 2026-06-25 ｜ 310P official reference fallback + 配置模板修复 ｜ Developer

**派法**：
- 角色：Developer。修 310P 上 `gen_official_embedding.py` 的 official reference 生成失败：先修 `.env` 模板 root/src 误导，再给官方 `Qwen3VLEmbedder` NPU→CPU fallback，顺手同步核心 transformers reference cleanup 和测试语义文档。
- 工作范围：`.env.example`、`atb_python_qwen3vl_embedding/env.py`、`atb_cpp_llm/tests/python_reference/gen_official_embedding.py`、`atb_python_qwen3vl_embedding/tests/data_utils.py`、`test_e2e.py`、`test_embedder_e2e.py`、`test_stage_reference.py`、visrope 注释、testing guide、lessons。
- 关键约束：official reference 不是 910B-only；不能把 310P official gate skip 当最终目标；不改生产 preprocess/算子逻辑，不降阈值。

**结果**：
- 修正 `QWEN3VL_EMB_SRC` 模板语义为官方 repo root，并在 official embedding generator 中兼容 repo root / src 两种输入。
- `gen_official_embedding.py` 改为优先 official NPU，遇到 310P unsupported op 后 fallback official CPU float32；支持 `OFFICIAL_EMBED_FORCE_CPU=1` 强制验证 fallback。后续改为由 `.env` 的 `OFFICIAL_EMBED_CASES` 控制生成哪些分辨率（不再硬编码 CPU fallback 只生成 416/720）。
- 核心 TF reference 测试改用 safe NPU cache cleanup 并打印实际 ref device/dtype；`test_stage_reference.py` 在 engine 初始化失败时仍可生成 TEXT_ONLY transformers CPU reference。
- `896x896` visrope case 标注为 Level2 op stress grid，避免误认为 production/e2e resolution。
- lessons 追加 official reference 全平台原则和 `.env` root/src 语义教训。

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

## 2026-06-25 ｜ CPU bicubic 对齐官方 PIL 8bpc Batch A ｜ Developer

**背景**:CPU `BicubicResize`/`PreprocessImage` 是 non-AA 固定 4-tap,与官方 PIL `Image.resize` 默认 BICUBIC(8bpc 定点 AA)分叉,noise 大幅降采样 cos→0.90。用户要求 bit-exact 复刻 PIL 8bpc,精度是底线。计划见 `.claude/plans/eager-marinating-hippo.md`。

**派法**:
- 角色:Developer。只改生产代码 `qwen3vl_preprocess.cpp`,逐行对照 `docs/archive/Resample.c`。给出完整实现规范(kPrecisionBits=22/Clip8/BicubicFilter 分组式/PrecomputeCoeffs8bpc/Horizontal+VerticalPass8bpc uint8→uint8/ResampleCore8bpc per-axis skip)+ 11 项 bit-exact 雷区清单。要求只验证编译+雷区自检,不跑测试(参考还是 non-AA,待 B/C)。禁动 BankersRound/SmartResize/NPU/patchify。不用 worktree。

**结果**:
- 编译零告警;11 项雷区自检全过。新增 PIL 8bpc 定点 helper,改写 BicubicResize(.h 签名不变)/PreprocessImage(resize→uint8→normalize→保留patchify),删旧 4 个 non-AA 函数。

## 2026-06-25 ｜ CPU bicubic 对齐官方 PIL 8bpc Batch A ｜ Reviewer

**派法**:
- 角色:破坏者 Reviewer。默认怀疑有 bug,逐条复核 11 项必查(重点 per-axis skip 维度传参、(int)截断方向、ss偏置、normalize等价、orphan include),并独立编译复现。

**结果**:
- 结论:bit-exact 正确,**无 BLOCKER/无 MAJOR**。把移植 helper 抽出独立编译,对照真实 PIL 12.1.1 跑 408/408 用例(8定向+400随机fuzz)逐像素零差异 maxdiff=0。11 项必查全 PASS。
- MINOR M1:`test_io_adapters.cpp:750-791/827-852` 硬编码旧 float 期望值会失败 → 已补入 Batch C 范围(原计划漏列)。MINOR M2:未跑完整 ATB build,Batch C 断言更新后做 CPU 冒烟。
- Batch A 无需 Re-review 修复,通过。

## 2026-06-25 ｜ CPU bicubic 对齐官方 PIL 8bpc Batch B ｜ Developer + Reviewer + Re-review

**派法**:
- Developer:迁移 `gen_cpu_reference.py` 参考从 non-AA 到 PIL（删 `_cpp_bicubic_resize`、新增 `_pil_preprocess_image`、新增降采样用例）。强调 patchify 次序必须与 C++ 逐字一致、smart_resize 全限定 import 不重写。
- Reviewer 破坏者:逐条查 patchify 次序 bit-exact、tp 帧复制、PIL 逐通道、normalize 值/顺序、fp16 RNE、降采样真触发 AA、删除遗留；亲写独立循环复刻 patchify 比对。
- Re-review（SendMessage 续 Dev 上下文）:修 MAJOR-1。

**结果**:
- Dev:新增 `_pil_preprocess_image`（patchify permute(0,3,6,4,7,2,1,5,8) 对齐 C++ :323-352）+ 降采样用例 `down_144x272`（→128x256 双轴 AA），generator exit 0。
- Reviewer:patchify 次序经独立循环复刻确认 bit-exact（Batch C 闸口成立），必查 1-6/8 全 PASS；1 个 MAJOR——删 `bicubic_*_output.bin` 生产者但 test_preprocess_cpu.cpp:359 Test4 仍读它致失败。
- Re-review:MAJOR-1 修复——`_gen_bicubic_case` 重新 emit `bicubic_*_output.bin` 作为 PIL 值的 byte-identical alias，Test4 保持绿，gate exit 0。归档 lessons 主题6第4条。

## 2026-06-25 ｜ CPU bicubic 对齐官方 PIL 8bpc Batch C ｜ Developer + Reviewer

**派法**:
- Developer:迁移测试断言到 PIL bit-exact（Test4/5 改 PIL、删 Test6、新 Test7 prod bit-exact、test_io_adapters 硬编码期望迁移）+ 新增 CPU-vs-official gate（无 Is910B 守卫）。要求实跑验证、bit-exact 失败不许放宽阈值。
- Reviewer 破坏者:对抗性验证断言真 fail-closed（篡改 bin 看是否 EXIT=1）、SKIP 不计 PASS、official gate 无误守、独立 PIL 复算 io_adapters 期望值、**实跑全量回归判定 Batch D**。

**结果**:
- Dev:全绿。test_preprocess_cpu Test4/5/7 max_diff=0（416 恒等/720→704 混合轴/1080,1440 双轴降采样）；CPU-vs-official 4 分辨率 cos=1.0（max_diff=2.43e-4 纯 fp16 量化差）；test_io_adapters 2 case PIL 期望（0/255/255/0/84、39/57/113/131）。
- Reviewer:Batch C 3 文件 **0 BLOCKER/0 MAJOR/0 MINOR**——对抗实证篡改/隐藏 bin 均 EXIT=1（fail-closed），official gate 无守卫真退码，io_adapters 期望经独立 PIL 复算一致。
- **关键回归发现**:窄口径 vision_stages L0=0.999908（>0.999）未命中，但 **test_stage_precision IMAGE_ONLY cos=0.978341 FAIL**；engine-only 诊断 cos=0.999929 坐实根因 = Python 引擎 `preprocess.py:85` 仍用 torch F.interpolate（non-AA）未对齐 PIL。**Batch D 必须触发**。归档 lessons 主题6第4条。

## 2026-06-25 ｜ CPU bicubic 对齐官方 PIL 8bpc Batch D ｜ Developer + Reviewer + MINOR 修复

**背景**:Batch C Reviewer 回归发现 Python 引擎 `preprocess.py:85` 未对齐 PIL，致 test_stage_precision IMAGE_ONLY FAIL。用户确认继续 Batch D（精度底线，plan 已含此条件触发项）。

**派法**:
- Developer:先调研 blast radius（谁依赖 preprocess_image 输出），再把 `F.interpolate(bicubic non-AA)` 换成逐通道 PIL `Image.resize(BICUBIC)` 8bpc，重生 Family B 参考，跑全量回归 + Python 包测试验证无回归。
- Reviewer 破坏者:重点查 CHW→uint8 转换（[0,1] vs [0,255] 偏移）、CPU/NPU 数据流、重生完整性（stage 参考时间戳）、standalone 参考遗漏、gate 真退码；亲写独立 PIL 复刻全链比对。
- MINOR 修复（SendMessage 续 Dev）:test_preprocess.py 假阳性退码。

**结果**:
- Dev:preprocess.py resize 改逐通道 PIL 8bpc AA（与 `_pil_preprocess_image` bit-exact cos=1.0），移除 orphan torch.nn.functional import。重生 stage_L0/stage_final_* 等。test_stage_precision IMAGE_ONLY 0.978341→PASS（≥0.99），test_vision_stages PASS。Python 包回归全改善:test_preprocess cos≈1.0、embedder_e2e Image 0.999978、e2e Image 0.999845。自曝环境坑:ASCEND_RT_VISIBLE_DEVICES=3 致 device_count==0/aclInit 107001，移除后正常。
- Reviewer:无 BLOCKER/MAJOR。CHW→uint8 风险点不存在（输入契约恒为 uint8，硬校验）；数据流 CPU→NPU 正确；stage 参考时间戳确认晚于生产改动无 stale；standalone 参考（test_first_layer_ref/test_vision_block_ref）磁盘无 bin、无 C++ 消费者、不在 ctest，隔离无假绿；672x476 独立复刻 bit-exact；vs 官方 AutoProcessor cos=1.0。1 MINOR:test_preprocess.py 用 return 非 assert（预存假阳性）。
- MINOR 修复:`return all_ok`→`assert all_ok`，__main__ 改由 assert 驱动退码。验证 forced FAIL→EXIT=1、restored→EXIT=0，临时改动完全恢复。

**全任务成果**:C++ CPU + Python 引擎预处理均 bit-exact 对齐官方 PIL 8bpc；1440 noise cos 0.90→1.0；CPU vs 官方 4 分辨率 cos=1.0；全管线 IMAGE_ONLY 0.978→PASS。lessons 归档主题6第4条（重实现参考链需同步同源生产者 + 窄口径是钝指标）。

## 2026-06-29 ｜ 310P 降采样 small-op AA bicubic 拼装 Batch A/B/C ｜ Developer + Reviewer 破坏者 + Re-review

**背景**:310P 上 `aclnnUpsampleBicubic2dAA` 不支持（aclnnStatus=561103），非 AA 降采样 1080/1440 仅 cos 0.987/0.958 不达 0.99 闸口；原 310P 降采样降级 CPU/skip。目标:用 ATB 小算子拼装 PIL AA bicubic（separable 双轴 dense MatMul），让 310P 降采样端到端对齐 910B aclnn AA。三 Batch 派单，全程 Dev→Reviewer 破坏者→Re-review 纪律。

### Batch A ｜ Spike 单分辨率算法验证
**派法**:
- Developer:新增 `smallop_bicubic_aa.{h,cpp}`（`PrecomputeCoeffsFloat` 复刻 Resample.c + `BuildDenseWeightFp16` 稠密化 + `NpuBicubicResizeAASmallOp` H/V pass + per-axis skip + goto-cleanup）+ `test_aa_smallop_spike.cpp` TC1（1080×1920→992×1792 最劣 case vs PIL），注册 REFDATA_DEPENDENT_TESTS。
- Reviewer 破坏者:独立 Python 复刻 `PrecomputeCoeffsFloat` 逐元素对照权重、篡改 bin 验 cos 跌破、lifetime/cleanup grep、雷区清单（`(int)` 截断/xmin-xmax clamp/filterscale/归一化）逐条核。

**结果**:
- Dev:TC1 cos=0.999980。
- Reviewer:命中 **BLOCKER-1**——cleanup 在 async 算子未 drain 前 `alloc->Free`（`aclrtFree` 非 stream-ordered，可能 free 仍被读的 w_h/h_out/h_tr/w_v/v_out）。
- Re-review（SendMessage 续 Dev）:BLOCKER-1 修复——cleanup 前加 end-of-pipeline `runtime->Synchronize()`，再逆序 Free；H-only 提前 goto 也经此 sync。

### Batch B ｜ 扩 4 分辨率 + 端到端 vs CPU PIL 闸口
**派法**:
- Developer:TC2（4 生产分辨率 vs PIL，含 416 恒等/720 V-only/1080,1440 双轴）+ TC3（full NPU pipeline small-op AA + normalize + patchify vs CPU PIL PreprocessImage，无 Is910B 守卫），手串 `RunNpuPipelineSmallOpAA` 不碰生产分发。**用 `ExecuteOperation`（`base_model.h`）替换 anonymous `RunAtbOp`，删重复**（后续工作 DRY 项）。
- Reviewer 破坏者:独立 Python PIL+numpy 端到端 vs TC3 dump 对照、篡改 bin、核 per-axis skip 真生效、4 分辨率轮跑无泄漏。

**结果**:
- Dev:TC2 416=1.0/720=0.999984/1080=0.999980/1440=0.999995；TC3 端到端 cos=1.0/0.999924/0.999878/0.999950。
- Reviewer:证明 small-op AA 与 910B 硬件 AA **数值等价**（独立 Python 交叉验证），无 BLOCKER；**MINOR-1**——H-only 路径（`need_h && !need_v`，独有直写 output_view + 提前 goto cleanup）无测试覆盖。

### Batch C ｜ 工程化分发 + 端到端回归 + 性能基线 + 补盲区 + 文档
**派法**:Developer surgical 改 `qwen3vl_preprocess.cpp:438` 分发（910B→aclnn AA / 310P→small-op AA / 非降采样→非 AA）+ 补 H-only TEST_CASE（MINOR-1）+ perf 基线 TEST_CASE + 910B 默认回归 + 文档 5 处。

**结果**:
- 分发改造:include `smallop_bicubic_aa.h` + 嵌套 if/else（`aclError`→`Status` 转换），:434 注释更新。0 新 warning。
- H-only（TC4 64×128→64×64）cos=0.999998（补 MINOR-1 盲区，gen_cpu_reference.py 加 `honly_64x128` case）。
- perf 基线（separable 5 算子单图，910B 实测）:720→704 0.81 ms / 1080→992×1792 3.72 ms / 1440 4.53 ms / 416 恒等 0.01 ms（后续组图版对比基线）。
- 910B 默认回归:`test_path_c_raw_image` cos=0.999956、`test_engine_vs_official` 4 分辨率 cos=0.999882/0.999235/0.999469/0.999690、`ctest -L needs_refdata` **31/31 PASS**（aclnn AA 路径未被改坏）。spike 全 5 TC pass。
- `ASCEND_PLATFORM=310P` 评估:test_aa_smallop_spike（不涉 attention）310P 配置下 5/5 PASS；test_path_c_raw_image 在 910B 硬件 + 310P 配置下 Path A Encode FAIL（`ExecuteOperation` Setup status 4 = SelfAttention NZ mask 平台模拟副作用，非 small-op 问题，记录不阻塞——small-op 正确性已由 TC3 + spike 310P 配置证）。

**关键发现归档**:BLOCKER-1 async free（`aclrtFree` 非 stream-ordered，cleanup 须先 drain）、TC3 与 910B aclnn AA 数值等价、`ExecuteOperation` 复用消除 RunAtbOp 重复（DRY）、H-only 盲区补测（MINOR-1）。
