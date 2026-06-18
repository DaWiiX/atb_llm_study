# 项目状态（单一真相源）

> **本文件是"做完什么 / 还剩什么"的唯一参考。** 其余快照计划（`archive/` 下的 audit-fix-plan、test-fix-plan、architect-assessment、refactoring-plan）一旦被超越即冻结，**不再代表现状**——查现状只看这里。
>
> 维护规则：每完成一批工作，更新本文件对应行；不新建文档记录进度。最后更新见文末。

---

## 1. 总体裁决

🟡 **有条件可部署**（功能 + 精度 + benchmark 线已完成；代码质量加固剩 43 项非阻塞）

- **性能**：✅ ATB 比 Transformers 快 3.0–3.6×，余弦相似度 ≥ 0.999（13/13 组合）
- **C++ 端**：✅ RAII 正确、错误传播完整、61 项审计 100% 修复
- **Python 端**：🟡 P0/P1-HIGH 已修复；P1-MEDIUM(23) + P2(20) 待办
- **Benchmark**：✅ B1–B7 + F1–F3 + M4/M5/M6 完成；310P 闭环待硬件复验

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
| Python 测试参考 fallback | `load_tf_ref` 加 NPU→CPU eager probing + 薄代理（`_TFRef`），310P 上 transformers 参考撞不支持算子时自动退 CPU float32，对齐 C++ 参考生成器设计。910B 回归 7 文件全 PASS | ✅ (910B) / ⏳ 310P 待复验 |

### 2.5 平台适配
910B + 310P 双平台。310P NZ mask 策略、GQA 原生支持（cos=1.0）、平台检测 API（`is_310p()`/`Is310P()`）均完成。详见 `evergreen/platform-310p.md`。

---

## 3. 待办（按优先级）

### 3.1 🔴 310P 硬件闭环验证（部署前必须）
310P 全量已跑：**C++ 50/50 全过**；Python 侧参考 fallback 已修（见 §2.4），**待在 310P 复验**原本 FAIL 的 `test_e2e`/`test_embedder_e2e`/`test_deepstack_integration`/`test_pipeline_trace` 转 PASS。到 310P 机器按此顺序：
1. `git pull` → `grep ASCEND_PLATFORM .env` 必须为 `310P`（配错会静默 fallback，§9.6 根因）
2. `bash atb_cpp_llm/build_and_test.sh`（全量，推理核心 + 平台算子可信）
3. `python atb_python_qwen3vl_embedding/tests/run_all.py`（Python 全量，确认 fallback 生效、cosine ≥ 0.99）
4. `ctest -R test_benchmark_modes`（benchmark 外壳在 310P 能跑）
5. `python atb_cpp_llm/scripts/gen_compare_tokens.py` → `./benchmark --mode compare`
6. `python atb_cpp_llm/tests/compare_py_cpp.py`（cosine ≥ 0.99）

> 注：全量测试**不覆盖** benchmark 二进制（benchmark 不在 CTest，仅 `test_benchmark_modes` 冒烟），故 4–6 步是独立的必要验证。
>
> 310P 遗留（非 fallback 范围）：`test_310p_combinations` 5/18 `CreateOperation failed` 是 platform-310p.md 实验矩阵已标的 310P 不支持 attention 配置组合，属预期（13/18 正常），应改测试标 expected-skip；`test_text_diagnostics` 卡死是子进程僵尸，非算子。二者单独修。

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
