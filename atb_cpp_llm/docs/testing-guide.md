# 310P SelfAttention 测试指南

## 概述

本文档记录 310P 平台适配的完整测试策略。三个问题已在 commit `a23d173` 中修复：

| 问题 | 修复 | 状态 |
|------|------|------|
| Device ID 硬编码 | `ASCEND_DEVICE_ID` 环境变量 | ✅ 已修复 |
| GQA 错误跳过 | 移除所有 `Is310P()` GQA guard | ✅ 已修复 |
| Graph 层 mask ND 格式 | 共享 `test::UploadMask()` helper | ✅ **6/14 实测 23/23 通过** |
| Python ATB mask | `nd_to_nz_fp16()` + engine.py | ✅ 已修复 |
| C++ mask 直接 NZ 生成 | `MakeCausalMaskNzFp16()` | ✅ 已优化 |

**最新 310P 实测结果（6/14）**：
- 阶段 1: 10/10 通过 (cos=1.0) ✅
- 阶段 2 (无参考数据): 23/23 通过 ✅
- Python 侧: ⏳ 待 310P 重新验证（本次修复后）

**下一步**：在 310P 上重新跑 Python E2E 测试。

---

## 第一步：拉取最新代码

```bash
cd /path/to/atb_llm
git checkout feat/310p-support
git pull origin feat/310p-support
# 确认 HEAD 是 a23d173
git log --oneline -1
```

## 第二步：环境准备

```bash
export ASCEND_PLATFORM=310P
# 多卡环境可额外指定 device（默认卡0）：
# export ASCEND_DEVICE_ID=0

source ~/Ascend/ascend-toolkit/set_env.sh
source ~/Ascend/cann/set_env.sh
source ~/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source ~/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=~/Ascend/nnal/atb/latest/atb/cxx_abi_1
```

---

## 阶段 1：原子算子精度测试

**测试二进制**: `test_self_attention_precision`

**范围**：仅 SelfAttention 单算子（Q/K/V/mask → output），不含 QKV projection、RoPE、O-projection。

**参考数据生成**:
```bash
cd atb_cpp_llm
python3 tests/python_reference/gen_cpu_reference.py --stage op_self_attention
# 在 /tmp/ 下生成 cpu_op_sa_*.bin（10 个 case，含 gqa_causal）
```

**运行**:
```bash
cd atb_cpp_llm/build
cmake --build . --target test_self_attention_precision -j8
ASCEND_PLATFORM=310P ./test_self_attention_precision 2>&1 | tee /tmp/310p_phase1.log
```

### 测试用例矩阵

| # | CASE 名称 | S | nh | kvh | hd | mask | 类型 | 310P 预期 | 关键验证点 |
|---|----------|---|---|-----|-----|------|------|----------|-----------|
| 1 | `mha_nomask` | 8 | 4 | 4 | 32 | ❌ | MHA | ✅ 通过 | 无 mask 基线 |
| 2 | `gqa_nomask` | 8 | 12 | 4 | 64 | ❌ | GQA | ✅ 通过（保守） | 待验证 GQA 可用性 |
| 3 | `mha_causal` | 8 | 4 | 4 | 32 | ✅ | MHA | ❓ NZ mask, S=8 非16对齐 | NZ mask + 非对齐 S |
| 4 | `mha_causal_s4` | 4 | 4 | 4 | 32 | ✅ | MHA | ❓ NZ mask, S=4 非16对齐 | 最小 S，padding 比例最大 |
| 5 | `mha_causal_s16` | 16 | 4 | 4 | 32 | ✅ | MHA | ✅ 16对齐，最可能通过 | 理想对齐 + NZ mask |
| 6 | `mha_causal_s32` | 32 | 4 | 4 | 32 | ✅ | MHA | ✅ 16对齐，应通过 | 较大对齐 + NZ mask |
| 7 | `mha_causal_hd128_s4` | 4 | 16 | 16 | 128 | ✅ | MHA | ❓ 真实hd=128, S=4 | 真实 MHA + 非对齐 S |
| 8 | `mha_causal_hd128_s16` | 16 | 16 | 16 | 128 | ✅ | MHA | ✅ 真实hd=128, 16对齐 | 真实 MHA + 理想对齐 |
| 9 | `mha_nomask_hd128_s16` | 16 | 16 | 16 | 128 | ❌ | MHA | ✅ 通过 | 真实 MHA 无 mask 基线 |

**分组说明**：
- **Group 1**（#1-3）：基础功能，覆盖 MHA/GQA、有/无 mask
- **Group 2**（#4-6）：不同 S 值 — 测试 16 对齐约束（S=4/8 非对齐 vs S=16/32 对齐）
- **Group 3**（#7-9）：真实模型参数 — hd=128, nh=16（Qwen3VL-Embedding-2B MHA 权重展开后）

**精度标准**: 余弦相似度 ≥ 0.99，理想值 = 1.0

**判断逻辑**：
- 如果 **所有带 mask case 都失败** → NZ mask 方案不兼容，切手动 attention
- 如果 **仅非16对齐 S 失败**（#3/4/7 失败，#5/6/8 通过）→ 310P 严格要求 S 16对齐，需要 padding
- 如果 **全部通过** → NZ mask 方案可行，继续阶段 2

---

## 阶段 2：组件/图层级测试

**前提**：阶段 1 中至少 case #5/#6/#8（16对齐 MHA causal）通过。

### 2a. Graph 构建测试 (`test_text_ops`)

**范围**：ATB graph 构建 + NPU 执行。不含精度对比（无参考数据），只验证 graph 能 build + execute。

```bash
cd atb_cpp_llm/build
cmake --build . --target test_text_ops -j8
ASCEND_PLATFORM=310P ./test_text_ops 2>&1 | tee /tmp/310p_phase2a.log
```

| # | CASE | 内容 | 类型 | 310P 预期 | 说明 |
|---|------|------|------|----------|------|
| 1 | Op Creation | Linear/RMSNorm/Rope/SA 创建 | — | ✅ | 基础算子 |
| 2 | GraphBuilder AddOp | 往 graph 添加算子 | — | ✅ | 基础设施 |
| 3 | RmsNormGraph | RMSNorm graph | — | ✅ | |
| 4 | SwiGluMlpGraph | SwiGLU MLP graph | — | ✅ | |
| 5 | SelfAttentionGraph | **MHA** 完整 attention graph | MHA+mask | ✅ **含 NZ mask** | 关键：SA 接收 NZ mask |
| 6 | SelfAttentionGraph | **GQA** attention graph (12,2) | GQA+mask | ✅ 通过 | 待验证 |
| 7 | TextDecoderLayerGraph | **MHA** decoder layer | MHA+mask | ✅ **含 NZ mask** | 关键：完整 layer |
| 8 | TextDecoderLayerGraph | **GQA** decoder layer | GQA+mask | ✅ 通过 | 待验证 |
| 9 | Execute on NPU | 综合执行 | MHA+mask | ✅ | |

### 2b. DecoderLayer 精度测试 (`test_text_decoder_layer_precision`)

**范围**：完整 TextDecoderLayer（attention + MLP + residual），有参考数据精度对比。

```bash
cmake --build . --target test_text_decoder_layer_precision -j8
ASCEND_PLATFORM=310P ./test_text_decoder_layer_precision 2>&1 | tee /tmp/310p_phase2b.log
```

| # | CASE | S | nh | kvh | hd | mask | 类型 | 310P 预期 | 说明 |
|---|------|---|----|-----|-----|------|------|----------|------|
| 1 | `small no-mask` | 8 | 4 | 4 | 32 | ❌ | MHA | ✅ | 无 mask 基线 |
| 2 | `GQA with mask` | 8 | 12 | 4 | 64 | ✅ | GQA | ✅ 通过 | 待验证 |

**⚠️ 缺失覆盖**：目前没有 **MHA + causal mask** 的 DecoderLayer 精度测试。Case 1 无 mask，Case 2 是 GQA（被 skip）。阶段 1 通过后应补充。

### 2c. TextModel 精度测试 (`test_text_model`)

**范围**：完整 text model（28 层 decoder layers + FinalNorm），有参考数据精度对比。

```bash
cmake --build . --target test_text_model -j8
ASCEND_PLATFORM=310P ./test_text_model 2>&1 | tee /tmp/310p_phase2c.log
```

| # | CASE | 内容 | 类型 | 310P 预期 | 说明 |
|---|------|------|------|----------|------|
| 1 | TextModel Build | MHA 28层 model 构建 | MHA+mask | ✅ | |
| 2 | Causal Mask | mask 生成验证 | — | ✅ NZ mask | 验证 NZ 格式正确 |
| 3 | TextModel Execute | MHA 28层执行+精度 | MHA+mask | ✅ **含 NZ mask** | **关键** |
| 4 | TextModel GQA | GQA 28层 (12,2) | GQA+mask | ✅ 通过 | 待验证 |

---

## 阶段 3：端到端测试

**前提**：阶段 2 全部通过（cos ≥ 0.99）。

### 3a. C++ E2E

```bash
cd atb_cpp_llm/build
cmake --build . --target test_e2e -j8
ASCEND_PLATFORM=310P ./test_e2e 2>&1 | tee /tmp/310p_phase3a.log
```

**范围**：完整 Qwen3VL-Embedding-2B 推理（vision + text），使用 GQA→MHA 展开权重。

| 测试内容 | 类型 | 说明 |
|---------|------|------|
| 全模型推理 + 精度对比 | MHA（展开后） | Vision + Text，NZ mask |

### 3b. Python E2E

```bash
cd atb_python_qwen3vl_embedding
ASCEND_PLATFORM=310P python tests/test_engine.py 2>&1 | tee /tmp/310p_phase3b_engine.log
ASCEND_PLATFORM=310P python tests/test_e2e_full_pipeline.py 2>&1 | tee /tmp/310p_phase3b_full.log
```

| 测试 | 内容 | 类型 |
|------|------|------|
| `test_engine.py` | Engine 级别：text-only, image-only, image+text | GQA→MHA 展开 |
| `test_e2e_full_pipeline.py` | 全流程 vs transformers 参考 | GQA→MHA 展开 |

---

## 测试结果记录表

### 阶段 1 结果

| # | CASE | 上次 (6/12) | 本次 |
|---|------|------------|------|
| 1 | mha_nomask | cos=1.0 ✅ | ⬜ |
| 2 | gqa_nomask | cos=1.0 ✅ | ⬜ |
| 3 | mha_causal | cos=1.0 ✅ | ⬜ |
| 4 | mha_causal_s4 | cos=1.0 ✅ | ⬜ |
| 5 | mha_causal_s16 | cos=1.0 ✅ | ⬜ |
| 6 | mha_causal_s32 | cos=1.0 ✅ | ⬜ |
| 7 | mha_causal_hd128_s4 | cos=1.0 ✅ | ⬜ |
| 8 | mha_causal_hd128_s16 | cos=1.0 ✅ | ⬜ |
| 9 | mha_nomask_hd128_s16 | cos=1.0 ✅ | ⬜ |
| 10 | **gqa_causal (新增)** | — | ⬜ |

### 阶段 2 结果

| 二进制 | # | CASE | 上次 (6/12) | 本次 |
|--------|---|------|------------|------|
| test_text_ops | all | Graph 构建 | 7/7 passed ✅ | ⬜ |
| test_text_ops | — | GQA cases | ❌ SKIP (已修复) | ⬜ |
| test_decoder_layer | 1 | small no-mask | ✅ | ⬜ |
| test_decoder_layer | 2 | GQA with mask | ❌ SKIP (已修复) | ⬜ |
| test_text_model | 1-3 | Build/Mask/Execute | ❌ Setup 失败 | ⬜ |
| test_text_model | 4 | TextModel GQA | ❌ SKIP (已修复) | ⬜ |
| test_text_runner_full | 1 | MHA pipeline | ❌ Setup 失败 | ⬜ |
| test_text_runner_full | 2 | GQA pipeline | ❌ SKIP (已修复) | ⬜ |

### 阶段 3 结果

| 测试 | 上次 (6/12) | 本次 |
|------|------------|------|
| C++ test_e2e | ❌ Graph Setup 失败 | ⬜ |
| Python test_engine (text-only) | ❌ | ⬜ |
| Python test_engine (image-only) | ✅ | ⬜ |
| Python test_engine (image+text) | ❌ | ⬜ |
| Python test_e2e_full_pipeline | ❌ | ⬜ |

---

## 失败诊断：需要收集的信息

```bash
# 1. 测试输出（完整日志）
cat /tmp/310p_phase1.log   # 或对应阶段日志

# 2. ATB 内部日志（最重要！）
cat /home/developer/ascend/log/atb/$(ls -rt /home/developer/ascend/log/atb/ | tail -n 1)

# 3. 硬件确认
npu-smi info -t board -i 0
echo "ASCEND_PLATFORM=$ASCEND_PLATFORM"

# 4. 测试摘要
grep -E "cosine|FAILED|ERROR|Skipping|test cases|PASSED" /tmp/310p_phase1.log

# 5. ATB 版本
strings ~/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so \
  | grep -iE "version|build" | head -5
```

---

## 回退计划

| 情况 | 方案 |
|------|------|
| Graph 层 mask Setup 仍失败 | 反馈 ATB 日志，分析具体哪个 runner 失败 |
| E2E 精度不达标 (cos < 0.99) | 逐层对比，定位精度丢失点 |

---

## 如何反馈结果

按以下模板整理结果，**附带原始日志文件**，发给我：

### 反馈模板

```
【环境】
- 芯片型号: (npu-smi info -t board -i 0 的输出)
- ATB 版本: (strings ~/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so | grep -i version | head -3)
- ASCEND_PLATFORM=310P

【阶段 1 — 原子算子】全部/部分 通过
- 如果全部通过，贴一行: "10/10 passed, all cos=1.0" 即可
- 如果有失败，贴完整输出:
  grep -E "cosine|FAILED|ERROR|test cases" /tmp/310p_phase1.log

【阶段 2a — Graph 构建】全部/部分 通过
- 贴: grep "test cases" /tmp/310p_phase2a.log

【阶段 2b — DecoderLayer 精度】全部/部分 通过
- 贴: grep -E "cosine|test cases" /tmp/310p_phase2b.log

【阶段 2c — TextModel 精度】全部/部分 通过
- 贴: grep "test cases" /tmp/310p_phase2c.log

【阶段 3 — E2E】全部/部分 通过
- 贴: grep -E "cosine|test cases" /tmp/310p_phase3a.log (C++)
- 贴: grep -E "cosine|PASS|FAIL" /tmp/310p_phase3b_*.log (Python)

【ATB 日志】（如果任何阶段失败，必须提供）
- 附件: /home/developer/ascend/log/atb/<最新日志文件>
```

### 快速收集脚本

把以下脚本保存为 `collect_results.sh`，运行后把生成的 `310p_results.tar.gz` 发给我：

```bash
#!/bin/bash
OUTDIR=/tmp/310p_feedback
mkdir -p $OUTDIR

# 环境信息
npu-smi info -t board -i 0 > $OUTDIR/env.txt 2>&1
echo "ASCEND_PLATFORM=$ASCEND_PLATFORM" >> $OUTDIR/env.txt
strings ~/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so | grep -iE "version|build" | head -5 >> $OUTDIR/env.txt 2>/dev/null

# 测试日志
cp /tmp/310p_phase1.log $OUTDIR/ 2>/dev/null
cp /tmp/310p_phase2a.log $OUTDIR/ 2>/dev/null
cp /tmp/310p_phase2b.log $OUTDIR/ 2>/dev/null
cp /tmp/310p_phase2c.log $OUTDIR/ 2>/dev/null
cp /tmp/310p_phase3a.log $OUTDIR/ 2>/dev/null
cp /tmp/310p_phase3b_*.log $OUTDIR/ 2>/dev/null

# ATB 日志
ATBLOG=$(ls -rt /home/developer/ascend/log/atb/ 2>/dev/null | tail -n 1)
if [ -n "$ATBLOG" ]; then
    cp /home/developer/ascend/log/atb/$ATBLOG $OUTDIR/atb_latest.log 2>/dev/null
fi

# 摘要
echo "=== Test Summary ===" > $OUTDIR/summary.txt
for f in $OUTDIR/*.log; do
    echo "--- $(basename $f) ---" >> $OUTDIR/summary.txt
    grep -E "cosine|FAILED|ERROR|test cases|PASSED" "$f" 2>/dev/null | head -30 >> $OUTDIR/summary.txt
done

cd /tmp && tar czf 310p_results.tar.gz 310p_feedback/
echo "Done: /tmp/310p_results.tar.gz"
```

---

## 已知已验证项（无需重新测试）

| 已验证 | 结果 | 时间 |
|--------|------|------|
| NZ mask 原子级 SelfAttentionOp | cos=1.0 (9/9) | 6/12 |
| GQA 在 310P 上支持 | cos=1.0 | 6/12 |
| S 非16对齐可用 | cos=1.0 (S=4,8) | 6/12 |
| isTriuMask 不需要 | B4/B5 对比验证 | 6/12 |

## 本次待验证项（优先级排序）

| # | 问题 | 验证方式 | 优先级 |
|---|------|---------|--------|
| 1 | **NZ mask 在 graph 中正确传播？** | 阶段 2a test_text_ops case #5/#7 | **最高** |
| 2 | **DecoderLayer + NZ mask 精度？** | 阶段 2b GQA with mask | **最高** |
| 3 | **28 层全流程 NZ mask 精度？** | 阶段 2c test_text_model | 高 |
| 4 | **TextRunner GQA Full Pipeline** | 阶段 2 末尾 | 高 |
| 5 | E2E 精度 | 阶段 3 | 中 |
