# 项目状态（单一真相源）

> **本文件是"做完什么 / 还剩什么"的唯一参考。** 其余快照计划（`archive/` 下的 audit-fix-plan、test-fix-plan、architect-assessment、refactoring-plan）一旦被超越即冻结，**不再代表现状**——查现状只看这里。
>
> 维护规则：每完成一批工作，更新本文件对应行；不新建文档记录进度。最后更新见文末。

---

## 1. 总体裁决

🟡 **有条件可部署**（功能 + 精度 + benchmark 线已完成；代码质量加固剩 43 项非阻塞）

- **性能**：✅ ATB 比 Transformers 快 4.1–4.8×（Batch 1 异步优化后），余弦相似度 ≥ 0.999（13/13 组合）。Batch 1 异步流水恢复净赚 12–13% e2e（text 65.5→57.1ms, io/mm 115.7→100.6ms），stddev 0.71→0.04
- **C++ 端**：✅ RAII 正确、错误传播完整、61 项审计 100% 修复
- **Python 端**：🟡 P0/P1-HIGH 已修复；P1-MEDIUM(23) + P2(20) 待办
- **Benchmark**：✅ B1–B7 + F1–F3 + M4/M5/M6 完成；310P 闭环已复验通过

---

## 2. 已完成工作

### 2.1 C++ 框架重构（refactoring-plan Phase 0–20）
全部 ✅。components 三目录扁平化、BaseModel+组合、Runner 只管图生命周期、Registry 增强、适配器瘦身 790→588 行、独立 Builder 拆分、KV Cache/批处理接口预留、逐模块精度验证、Warning 清零 + 一键构建 + 硬编码路径消除。

### 2.2 架构审计修复（audit-fix-plan，61 项）
全部 ✅（2026-06-14）。1 CRITICAL + 12 HIGH + 24 MEDIUM + 24 LOW，全量回归 51/51 PASS。详见 `archive/audit-fix-plan.md`。

### 2.3 测试修复（test-fix-plan，P1–P21）
20/21 ✅。仅 **P4 待办**（见 §3.3）。详见 `archive/test-fix-plan.md`。

### 2.4 生产就绪修复（architect-assessment 批次）

| 工作线 | 项 | 状态 |
|--------|----|------|
| C++ 鲁棒性 | G1 gthw null guard、G2 CopyToDevice/Host 返回值检查、G3 部署 checklist | ✅ |
| Benchmark 增强 | B1 百分位、B2 NPU 内存测量、B3 冷启动、B4 吞吐量、B5 回归检测、B6 preprocess 计时、B7 硬编码清理 | ✅ |
| Python P0 | P0-1 sync 竞态、P0-2 资源清理、P0-3 engine_utils 错误处理、P0-4 forward 输入校验 | ✅ |
| Benchmark 紧急修复 | F1 token 命名、F2 gen_compare_tokens.py、F3 GetModelDir std::abort | ✅ |
| 测试基础设施 | R1 CTest 注册修复、R2 注释修正、Python 并行化(33m→19m)、910B 自动排除 310P 测试 | ✅ |
| Benchmark 输出质量 | M1 compare human 表格、M2 bench 分辨率对齐、M3 io seq_len、M4 冒烟测试、M5 分辨率单一数据源、M6 统一输出策略 | ✅ |
| 运行时验证 | 910B 全量测试 PASS | ✅ |
| 平台检测根治 | `Is310P()`/`Is910B()` 改读 `.env`（抽 `src/utils/dotenv.h` 共享头，消除生产代码依赖 test 头）+ benchmark 启动期 `aclrtGetSocName()` 硬件探针自检（不匹配 `LOG_ERROR` 提醒，不 abort）。修裸启动 `./benchmark` 读不到 `ASCEND_PLATFORM` 静默走 910B ND mask 致 310P Transdata 崩溃 | ✅ |
| Benchmark 输出修复 | `ReportStages`/`ReportColdStart`/`ReportThroughput` 的 `LOG_INFO`→`printf`(stdout)，修复默认 WARN log level 吞掉 human-readable 报告（只剩 `BENCH_RESULT` machine 行） | ✅ |
| Python 测试参考 fallback | `load_tf_ref` 加 NPU→CPU eager probing + 薄代理（`_TFRef`），310P 上 transformers 参考撞不支持算子时自动退 CPU float32，对齐 C++ 参考生成器设计。910B 回归 7 文件全 PASS | ✅ (910B) / ⏳ 310P 待复验 |

### 2.5 平台适配
910B + 310P 双平台。310P NZ mask 策略、GQA 原生支持（cos=1.0）、平台检测 API（`is_310p()`/`Is310P()`）均完成。详见 `evergreen/platform-310p.md`。

### 2.6 性能优化 Batch 1（异步流水恢复）
全部 ✅（2026-06-21）。翻转 per-op sync 默认 OFF（env 名 opt-out→opt-in `ATB_ENABLE_PER_OP_SYNC`）、移除 deepstack `InjectFeatures` 硬编码 sync + idx/alpha 缓存复用、补齐 async 模式 D2H 前置 sync、修复 debug dump 保真度。验收：构建零 warning、op 9/9 + e2e 6/6 全过（含 test_sync_safety 5 配置 cos≥0.999）、12–13% e2e 收益（text 65.5→57.1ms / io 115.7→102.6ms / mm 115.7→100.6ms，stddev 0.71→0.04）、100-iter 无 HBM 泄漏、compare 13/13 裸跑正常。M1（ws_size 缓存 + GRAPH_LAUNCH_MODE）暂缓为后续批次。详见 `evergreen/optimization-roadmap.md` P4 后续演进注记。

### 2.7 性能优化 Batch 2-A（CPU 预处理 PIL 式重写）
全部 ✅（2026-06-22）。`qwen3vl_preprocess.cpp` 移植 PIL `Resample.c` 三大手法：可分离两阶段卷积（16→8 tap）+ 系数预算表（`PrecomputeCoeffs`，运行时零 `CubicWeight` 重算）+ normalize 融合进 vertical pass + patch `f` 帧去重。验收：构建零 warning；`test_preprocess_cpu` 6/6（TestBicubicVsPython/TestPreprocessImageVsPython cos=1.000000）+ `test_io_adapters` magic number 不变 + e2e/accuracy/patch_embed 全过；preprocess 4 分辨率 4.3–5.0× 加速（416×672 129→27ms、720×1280 460→92ms、1080×1920 657→141ms、1440×2560 657→154ms）；compare 13/13 正常。数学等价（edge-clamp Catmull-Rom 语义不变，double 累加精度更高）。Reviewer 零阻塞 bug，独立非二进制维度验证 max_diff≤9.2e-5。详见 `evergreen/optimization-roadmap.md` P10。

### 2.8 P10-B 精度 spike（aclnnUpsampleBicubic2d / AA 闸口验证）
✅ P10-B 闸口通过（2026-06-22，Developer→Reviewer→Re-review 闭环）。验证 aclnn 与 PIL BICUBIC 在 **引擎真实 smart_resize 参数**（factor=32/min_pixels=4096/max_pixels=1310720）下的 cosine：

| 分辨率 | 引擎输出 | 非 AA cos | AA cos (仅910B) |
|--------|----------|-----------|-----------------|
| 416×672 | 416×672 | 1.000000 ✅ | 1.000000 ✅ |
| 720×1280 | 704×1280 | 0.999680 ✅ | 0.999984 ✅ |
| 1080×1920 | 832×1504 | 0.986983 ❌ | **0.999993** ✅ |
| 1440×2560 | 832×1504 | 0.958426 ❌ | **0.999996** ✅ |

**关键发现**：①非 AA（`aclnnUpsampleBicubic2d`，torch a=-0.75 Mitchell 无 antialias）在降采样（1080/1440→832）时严重偏离 PIL，闸口失败；②**AA 版本（`aclnnUpsampleBicubic2dAA`，含抗混叠预滤波）rescues P10-B**，降采样从 0.958 拉回 0.999996。③**AA 仅支持 910B**（CANN 商用版产品表：Atlas 推理系列=310P 为 ×，`@domain aclnn_ops_train`），310P 上 AA 不可用。**P10-B 双路径**：910B → `NpuBicubicResizeAA`（4/4 全部 ≥0.99998）；310P → 非 AA（2/4 通过）+ 降采样 case 降级 P10-A CPU 兜底。

修复 wrapper：非 AA + AA 两版均使用 `ACL_FORMAT_ND` + 显式 strides（官方模式）、Execute 后 `aclrtSynchronizeStream`、**不**手动 `aclDestroyAclOpExecutor`。新增 `test_aclnn_bicubic_spike`（level2，4 个 TEST_CASE：非 AA 基线 + AA 闸口 + 全管线精度 + 性能实测）+ gen 脚本生产分辨率 case。审查过程发现并修复 4 BLOCKER（gen 参数错误→闸口翻转 / REFDATA 未登记假阳性 / header 合约不一致 / 死代码）、5 MAJOR（含 sync 热路径声明）、4 MINOR。过程教训 5+1 项（含 solo 开发复发）归并进 `lessons-learned.md` 主题 7 第 0/7–11 条。

**P10-B 工程化完成**：`PreprocessImageNpu` 全 NPU 管线（SmartResize→H2D→AA/非AA Bicubic→3×Elewise normalize broadcast→AsStrided+8维Transpose patch→D2H），goto-cleanup 内存安全，AA 降采样条件守卫（恒等/上采样走非 AA，避免 AA 平滑破坏精度）。全管线精度 bit-exact vs CPU（cos/max_diff 6 位小数一致）。**性能实测 NPU vs P10-A CPU geomean 1.7×**（416×672 2.1× / 720×1280 2.3× / 1080×1920 1.6× / 1440×2560 1.0×）。**Patch NPU 化**（#1，AsStrided stride=0 广播 tp + 8 维 Transpose perm `[2,5,3,6,1,0,4,7]`，全程 device 内零 D2H 往返）使 geomean 从 1.4×→1.7×，1440×2560 从 0.9×→1.0×。#2 跳过恒等 resize 经 A/B 实测证伪放弃（bicubic 核恒等下为单位冲激无平滑）。后续可优化：device tensor 输出 API 彻底消除 D2H（#1 已把 D2H 降到 15MB 一次，但 pixel_values 仍需回 CPU 给调用方）。

---

## 3. 待办（按优先级）

### 3.1 🟡 310P 硬件闭环验证（C++ 已通过 / Python fallback 待复验）
310P C++ 全量 + benchmark compare 已在真实 310P 复验通过（2026-06-20）：C++ 全量过、裸启动 `./benchmark --mode compare` 不再崩 Transdata（平台检测根治后读 `.env` 的 `ASCEND_PLATFORM=310P`）、human 表正常输出。**Python 参考侧 fallback（§2.4 末行）910B 回归通过,但原本在 310P FAIL 的 `test_e2e`/`test_embedder_e2e`/`test_deepstack_integration`/`test_pipeline_trace` 待在 310P 上复验转 PASS**。310P 机器的回归复验步骤：
1. `git pull` → `grep ASCEND_PLATFORM .env` 为 `310P`
2. `bash atb_cpp_llm/build_and_test.sh`（全量）
3. `python atb_python_qwen3vl_embedding/tests/run_all.py`（Python 全量，确认 fallback 生效、cosine ≥ 0.99）
4. `./atb_cpp_llm/build/benchmark --mode compare`（裸启动即可，自动读 `.env`）
5. `python atb_cpp_llm/tests/compare_py_cpp.py`（cosine ≥ 0.99）

> 注：全量测试**不覆盖** benchmark 二进制（benchmark 不在 CTest），故 4–5 步是独立的必要验证。
>
> 310P 遗留（非本次范围）：`test_310p_combinations` 5/18 `CreateOperation failed` 是 platform-310p.md 实验矩阵已标的 310P 不支持 attention 配置组合，属预期（13/18 正常），应改测试标 expected-skip；`test_text_diagnostics` 卡死是子进程僵尸，非算子。二者单独修。

### 3.2 🟡 Python 代码质量加固（43 项，非阻塞）
- **P1 MEDIUM：23 项** — 硬编码值、`print`→`logging`、额外输入校验等健壮性改进
- **P2：20 项** — 文档补全、代码清理

不影响功能正确性和精度，可现在做也可 310P 验证后做。详见 `archive/architect-assessment-2026-06-16.md` §11.1。

### 3.3 🟡 P4：DeepSeek/Mixtral/Qwen3(non-VL) 零测试覆盖
当前引擎支持多模型族，但仅 Qwen3VL-Embedding-2B 有测试。分阶段：
- Phase 1：Qwen3 纯文本 L2/L3 测试
- Phase 2：Mixtral MoE MLP 精度测试
- Phase 3：DeepSeek-V2/V3（MLA + MoE）完整测试

详见 `archive/test-fix-plan.md` P4。

### 3.4 🟢 已知低优先遗留
- 适配器行数 588（目标 300–400）—— Qwen3VL 独有权重编排逻辑难以进一步提取，接受现状
- Debug 构建下 `-DDEBUG` 宏与 `LogLevel::DEBUG=0` 枚举冲突（Release 不受影响）

---

## 4. 性能基线（2026-06-09，910B，5 warmup + 3 iter）

C++ ATB 全面最快：geomean 领先 Python ATB **1.39×**、领先 Transformers **4.22×**。13/13 组合 cosine ≥ 0.99（最低 IO 1080×1920 = 0.999770）。完整数据见 `archive/refactoring-plan.md` §3 与 `archive/benchmark-report.md`。

---

## 5. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-06-18 | 建立本文件作为单一真相源，归并自 architect-assessment §11.1 + audit-fix-plan + test-fix-plan + refactoring-plan §1 |
| 2026-06-18 | 310P 全量结果：C++ 50/50 全过；Python 参考侧撞 ArgMaxWithValue/Conv3d FAIL。根因：transformers 参考在 310P NPU 跑撞不支持算子，C++ 端有 fallback 而 Python `load_tf_ref` 无。修复：`load_tf_ref` 加 NPU→CPU eager probing + `_TFRef` 薄代理，910B 回归 7 文件全 PASS，310P 待复验 |
| 2026-06-20 | 平台检测根治 + benchmark human 表修复 + 310P 真机复验通过 |
| 2026-06-21 | 性能优化 Batch 1：异步流水恢复（H1 per-op sync 默认 OFF + H4 deepstack sync 移除 + idx/alpha 缓存 + async D2H sync 补齐）。12–13% e2e 收益，精度无损，Dev→Reviewer→Re-review 闭环 |
| 2026-06-22 | 性能优化 Batch 2-A：CPU 预处理 PIL 式重写（可分离两阶段卷积 + 系数预算表 + normalize 融合 + patch f 帧去重）。preprocess 4.3–5.0× 加速（657→141ms@1080×1920），cos=1.0 精度无损，Dev→Reviewer→Re-review 闭环 |
| 2026-06-22 | P10-B 精度 spike：非 AA 降采样 1080/1440→832 vs PIL cos=0.987/0.958 闸口失败 → **aclnnUpsampleBicubic2dAA rescues P10-B**，降采样从 0.958 拉回 0.999996。AA 仅 910B（310P ×）。双路径确立。全程 Developer→Reviewer→Re-review 闭环：4 BLOCKER 修复 + 5 项过程失误归并 lessons（含 solo 复发主题 7 第 0 条）|
| 2026-06-22 | P10-B 工程化：`PreprocessImageNpu`（6 步 NPU 管线：SmartResize→H2D→AA/非AA Bicubic→3×Elewise normalize→D2H→CPU patch）实现，goto-cleanup 内存安全，AA 降采样条件守卫。全管线精度测试 cos≥0.999（非降采样/非 AA）。性能实测 NPU vs CPU geomean 1.4×（1.7×/1.9×/1.4×/0.9×），1440×2560 退化根因 H2D 开销 |
| 2026-06-23 | P10-B #1 Patch extraction NPU化：AsStrided(stride=0广播tp)+8维Transpose perm[2,5,3,6,1,0,4,7]+Reshape，全程device内零D2H往返。vs CPU bit-exact。geomean 1.4×→1.7×，1440×2560 0.9×→1.0×。#2 跳过恒等resize A/B实测证伪放弃(bicubic核恒等下单位冲激)。Dev→Reviewer→Re-review闭环 |
| 2026-06-23 | 路径C：NPU preprocess接入引擎dispatch。PreprocessImageNpu加device tensor重载(跳过D2H)+Internal重构DRY；NpuTensor::Adopt+TensorAllocator::Detach所有权转移原语；ForwardWithTiming激活raw_image分支(device tensor直喂pixels_npu,无H2D)；PREPROCESSED旁路不动(公共契约不变)；StageTimings.preprocess_ms引擎内回填。e2e测试raw_image vs PREPROCESSED embedding cos=0.999956。零BLOCKER/零MAJOR,4 MINOR修复 |
| 2026-06-23 | 流程纪律：Reviewer探查的BLOCKER/MAJOR必须归档lessons-learned(用户要求)。WORKFLOW §3.3加「审查发现归档纪律」+ lessons主题7第12条元规则 + 路径C Reviewer两条泛化发现归主题8第7/8条(调用方sync空操作/Detach静默no-op) |
| 2026-06-23 | benchmark 接入路径C:bench模式改 raw_image(引擎内NPU preprocess),compare保留CPU。staged_sum 公平对比 4 分辨率路径C全更优(416 38→54/720 104→161/1080 181→235/1440 247→256ms),preprocess_ms CPU→NPU 降 3.7-59%。e2e 口径迁移(路径C含preprocess、PREPROCESSED不含)经 Reviewer 确认公平。MAJOR(pre_ms cold口径)修+warmup,1440翻转证cold假象。grid_t隐式领域事实归 lessons 主题1第6条 |
| 2026-06-24 | Python 测试参考 NPU→CPU fallback 合入(cherry-pick fix/python-ref-fallback): `load_tf_ref` 加 eager probing + `_TFRef` 薄代理 + 7 个测试文件 device 透传改写 + bf16 权重 CPU float32 dtype 修复。910B 回归全 PASS,310P 待复验 |
