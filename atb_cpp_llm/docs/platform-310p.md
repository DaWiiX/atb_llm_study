# 310P (Atlas推理系列产品) 平台适配文档

## 概述

ATB LLM 项目支持 **Ascend 310P** (Atlas推理系列产品) 和 **Ascend 910B** (Atlas A2推理系列产品) 两个平台，数值结果等价。

两个平台的差异在于 ATB 算子支持范围不同。本文档记录适配策略、架构设计、测试方法、已发现问题和运维要点。

## ATB 文档关键发现（2026-06-12 重新审查）

### 文档来源

| 页面 | 内容 | 关键发现 |
|------|------|---------|
| [0261 定义](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0261.html) | SelfAttentionParam 结构体 | 完整枚举值列表 |
| [0262 参数列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0262.html) | 每个参数的含义和默认值 | isTriuMask 只对倒三角 mask 有效 |
| [0264 输入输出](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0264.html) | 通用输入输出列表 | attentionMask 支持 6 种 shape |
| [0265 PA_ENCODER](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0265.html) | **PA_ENCODER 输入输出** | **310P mask 格式为 NZ！** |
| [0266 PREFIX_ENCODER](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0266.html) | PREFIX_ENCODER 输入输出 | 需要 blockTables + kvCache |
| [0267 CAUSAL_MASK](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0267.html) | PREFIX_ENCODER + CAUSAL_MASK | 无 mask 输入，仅 seqLen + kvSeqLen |
| [0268 约束说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0268.html) | **310P 特殊约束** | headSize∈(0,256]、16对齐、nTokens 向上对齐16 |
| [0010 TensorDesc](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendtbapi/ascendtb_01_0010.html) | Tensor 描述符 | format 字段可指定 ACL_FORMAT_FRACTAL_NZ |

### 关键发现 1：PA_ENCODER 在 310P 上 mask 格式是 NZ（不是 ND！）

文档 0265 页**明确**指出 PA_ENCODER 下 attentionMask 的格式：

| 平台 | Mask 格式 |
|------|----------|
| Atlas A2 训练/推理系列产品 (910B) | **ND** |
| Atlas 推理系列产品 (310P) | **NZ** |
| Atlas 训练系列产品 | NZ |

**这意味着**：在 310P 上，传给 PA_ENCODER SelfAttention 的 mask tensor **必须是 NZ (FRACTAL_NZ) 格式**，不是 ND 格式。这正是我们之前用 TransdataOp ND→NZ 预转换的方向！

### 关键发现 2：310P 约束汇总（0268 页）

从约束说明页面提取的 **310P 专属约束**：

| 约束项 | 说明 |
|--------|------|
| batch 范围 | `0 < batch <= 2000` |
| headSize 范围 | `(0, 256]`（开启量化/logN缩放/SWA/BNSD时）；否则 `(0, 576]` |
| headSize 对齐 | **必须为 16 的倍数** |
| max_seq_len 对齐 | **应 16 对齐** |
| qSeqLen == kvSeqLen | PA_ENCODER 下必须相等 |
| headSize == vHeadSize | 量化/logN/SWA/BNSD 场景下 |
| nTokens 计算 | PA_ENCODER 下：所有 batch seqLen 之和向上对齐到 16 的整数倍 |
| GQA 支持 | PA_ENCODER 下 `headNum > kvHeadNum` 即可 |
| headNum % kvHeadNum == 0 | 必须整除 |

### 关键发现 3：TensorDesc format 字段

从 0010 页确认：`TensorDesc.format` 字段类型为 `aclFormat`，默认 `ACL_FORMAT_UNDEFINED`。创建 tensor 时可以指定 format。但 **数据必须按照指定 format 的物理布局写入**，不能以 ND 布局数据写入 NZ 格式 tensor。

### 关键发现 4：之前文档的错误

原 platform-310p.md 说 "BNSD 布局（所有 calcType）：CreateOp failed" — 这是在 910B 上测试的结果。文档 0262 页显示 `inputLayout` 可设为 `TYPE_BNSD`（枚举值 1），310P 产品支持页面显示 BNSD 在 310P 上部分支持。需要在真实 310P 上重新验证。

## 平台差异

### 已知兼容性

| 特性 | 910B | 310P |
|------|------|------|
| SelfAttention GQA (`kv_head_num < head_num`) | ✅ 支持 | ✅ 支持（实测 cos=1.0）|
| RopeOperation | ✅ 支持 | ✅ 支持 |
| Linear / RMSNorm / LayerNorm | ✅ 支持 | ✅ 支持 |
| 所有 Vision 路径算子 | ✅ 支持 | ✅ 支持 |
| SelfAttention PA_ENCODER mask=ND | ✅ 支持 | ❌ 不支持（需 NZ 格式）|
| SelfAttention PA_ENCODER mask=NZ | ❓ 未测试 | ✅ 实测通过 (cos=1.0) |
| S 非16对齐 (NZ mask) | — | ✅ 实测通过 (S=4/8 cos=1.0) |

### Mask 格式需求（基于文档 0265）

```
310P PA_ENCODER mask 要求:
  格式: FRACTAL_NZ (ACL_FORMAT=29)
  逻辑 shape: [maxSeqLen, maxSeqLen] 等（与 910B 相同的逻辑 shape）
  物理 shape: [1, ceil(S/16), ceil(S/16)*16, 16]（float16 下）
  
910B PA_ENCODER mask 要求:
  格式: ND (ACL_FORMAT=2)
  shape: [maxSeqLen, maxSeqLen]
```

## 实验矩阵：SelfAttentionOp 原子级参数组合

> **实验原则**：先基础功能，后高级功能。每次只改变一个变量。记录所有结果。

### 基础功能矩阵（必须通过）

| # | S | nh | kvh | hd | mask | mask格式 | isTriuMask | 预期 | 状态 |
|---|----|----|-----|-----|------|---------|------------|------|------|
| B1 | 16 | 4 | 4 | 32 | none | — | 0 | ✅ PASS | **待310P实测** |
| B2 | 16 | 4 | 4 | 32 | causal | ND | 0 | ❌ Transdata fail | **待310P实测** |
| B3 | 16 | 4 | 4 | 32 | causal | ND | 1 | ❓ | **待310P实测** |
| B4 | 16 | 4 | 4 | 32 | causal | NZ | 0 | ✅ 应通过(文档) | **待310P实测** |
| B5 | 16 | 4 | 4 | 32 | causal | NZ | 1 | ❓ | **待310P实测** |
| B6 | 4 | 4 | 4 | 32 | causal | NZ | 0 | ❓ S非16对齐 | **待310P实测** |
| B7 | 8 | 4 | 4 | 32 | causal | NZ | 0 | ❓ S非16对齐 | **待310P实测** |
| B8 | 32 | 4 | 4 | 32 | causal | NZ | 0 | ✅ 应通过 | **待310P实测** |

### 真实模型参数矩阵

| # | S | nh | kvh | hd | mask | mask格式 | 说明 | 状态 |
|---|----|----|-----|-----|------|---------|------|------|
| R1 | 16 | 16 | 16 | 128 | causal | NZ | 真实MHA+hd=128+16对齐 | **待310P实测** |
| R2 | 4 | 16 | 16 | 128 | causal | NZ | 真实MHA+hd=128+S=4 | **待310P实测** |
| R3 | 880 | 16 | 16 | 128 | causal | NZ | 真实MHA+hd=128+S=880 | **待310P实测** |

### 高级功能矩阵（基础功能通过后再测）

| # | 功能 | 参数 | 依赖 | 状态 |
|---|------|------|------|------|
| A1 | 高精度 | `kernelType=KERNELTYPE_HIGH_PRECISION` | 基础mask通过 | 未开始 |
| A2 | clamp缩放 | `clampType=CLAMP_TYPE_MIN_MAX` | 基础mask通过 | 未开始 |
| A3 | 压缩mask | `maskType=MASK_TYPE_NORM_COMPRESS` | 基础mask通过 | 未开始 |
| A4 | kv-bypass | `kvcacheCfg=K_BYPASS_V_BYPASS` | 基础mask通过 | 未开始 |
| A5 | logN缩放 | `scaleType=SCALE_TYPE_LOGN` | 基础mask通过 | 未开始 |
| A6 | BNSD维度输入 | `inputLayout=TYPE_BNSD` | 基础mask通过 | 未开始 |
| A7 | kv tensorlist | CPU侧kv cache输入 | 需要kv cache | 未开始 |
| A8 | SWA | `cacheType=CACHE_TYPE_SWA` | 基础mask通过 | 未开始 |

## 已尝试方案及结果

### 方案 1: 原生 MASK_TYPE_NORM + ND mask（910B 默认）

```cpp
param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
// mask: 2D ND [S, S]
```

**310P 结果**：❌ 失败
- 错误：`TransdataOperation mki node infer shape fail, inDims is not support`
- 原因：310P SelfAttentionEncoderFusionOpsRunner 内部做 ND→NZ 转换，TransdataOperation 在 310P 上失败
- ATB 日志显示：融合 runner 内部调用的 Transdata 不接受输入的维度

### 方案 2: isTriuMask=1 + MASK_TYPE_NORM + ND mask

```cpp
param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
param.isTriuMask = 1;
// mask: 2D ND [S, S]
```

**910B 结果**：✅ PASS (cos=1.0)
**310P 结果**：❌ 失败（与前次对话汇总一致）
- 错误：同样的 TransdataOperation 内部转换失败
- 说明：isTriuMask=1 不改变内部 Transdata 路径，只是优化标志

### 方案 3: MASK_TYPE_CAUSAL_MASK（内部生成 mask）

```cpp
param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK;
param.calcType = atb::infer::SelfAttentionParam::PREFIX_ENCODER;
```

**结果**：❌ 失败
- 错误：`shape of seqlen should be [batch] or [2, batch]`（错误码 8）
- 原因：MASK_TYPE_CAUSAL_MASK 仅与 PREFIX_ENCODER 配对（文档 0267 页确认），需要 blockTables + kvCache 输入
- PREFIX_ENCODER 需要 `[numBlocks, blockSize, ...]` 格式的 K/V 和 blockTables，与我们的无 KV cache 全量 attention 场景不兼容

### 方案 4: MASK_TYPE_NORM_COMPRESS（压缩 mask）

```cpp
param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
```

**910B 结果**：❌ CreateOp failed
**310P 结果**：❌ CreateOp failed（与前次对话汇总一致）
- 原因：压缩 mask 需要特殊格式（固定 128×128 上三角 tile），在 PA_ENCODER 下不受支持
- 文档 0266 页：MASK_TYPE_NORM_COMPRESS 仅在 PREFIX_ENCODER 下列出（shape `[128, 128]`）

### 方案 5: NZ 预转换（TransdataOp ND→NZ）+ MASK_TYPE_NORM

```
TransdataOp(ND_TO_FRACTAL_NZ): mask [S,S] → mask_nz [1, ceil(S/16), ceil(S/16)*16, 16]
SelfAttention: 接收 NZ 格式 mask
```

**310P 结果**：
- TransdataOp 独立运行：✅ PASS（ND→NZ 转换成功）
- Graph 构建（含 TransdataOp）：✅ PASS（test_text_ops 7/7 全部通过）
- Graph Setup：❌ ERROR_RT_FAIL (error 4)
- SelfAttentionOp 独立测试（含 NZ mask）：❌ 待确认

**分析**：
- 文档 0265 页明确说 310P PA_ENCODER mask 格式是 NZ，所以 NZ mask 方向是正确的
- Graph 构建成功但 Setup 失败 → 中间 tensor 的 format 标记可能有问题
- 可能原因：Graph builder 自动创建的中间 tensor 默认 format=ND，但 TransdataOp 输出的数据是 NZ 布局 → 不一致

### 方案 6: 直接创建 NZ 格式 mask（绕过 TransdataOp）

**思路**（用户建议）：在 `qwen3vl_model.cpp` 的 `cached_mask_npu_` 创建时直接指定 NZ format，数据按 NZ 物理布局写入。这样 SelfAttention 收到的就是 NZ 格式 mask，无需 TransdataOp。

**状态**：⏳ 待实施

**关键问题**：
1. `AllocNpuFloat16({seq_len, seq_len})` 默认创建 ND 格式 tensor
2. 要创建 NZ 格式：需要分配 NZ 物理 shape `{1, ceil(S/16), ceil(S/16)*16, 16}` 并设置 `format = ACL_FORMAT_FRACTAL_NZ`
3. **数据必须按 NZ 布局写入**：FRACTAL_NZ 将矩阵按 16×16 块重排，不能直接写入 ND 布局数据
4. 需要实现 CPU 侧 ND→NZ 数据重排，或在 NPU 上用 TransdataOp 做一次性转换后保存 NZ 格式的 mask 复用

## 310P 实测指南

> **当前状态 (2026-06-12)**：代码已准备好，待 310P 实测验证。
> 
> **核心改动**：310P 上 mask 在 `qwen3vl_model.cpp` 创建时直接转为 NZ (FRACTAL_NZ) 格式，不再在 graph builder 中插入 TransdataOp。

### 前置准备

```bash
# 1. 进入项目目录
cd /path/to/atb_llm

# 2. 确认 310P 环境
echo "ASCEND_PLATFORM=310P" >> .env   # 如果还没有设置
export ASCEND_PLATFORM=310P

# 3. 加载环境
source ~/Ascend/ascend-toolkit/set_env.sh
source ~/Ascend/cann/set_env.sh
source ~/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source ~/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=~/Ascend/nnal/atb/latest/atb/cxx_abi_1

# 4. 验证是真实 310P（非 910B）
npu-smi info -t board -i 0 2>/dev/null | grep -i "310P\|Chip"
# 应该看到 Ascend310P 或类似字样
```

### 阶段 1：原子级测试（SelfAttentionOp 单算子）

**这是最关键的测试**。如果原子级测试都无法通过，不要浪费时间做 E2E。

```bash
cd atb_cpp_llm

# Step 1: 生成参考数据（如果还没有，或重新生成）
# 参考数据在 /tmp/cpu_op_sa_*.bin
# 如果已有，可以跳过此步
python3 tests/python_reference/gen_cpu_reference.py --stage op_self_attention
# 预期输出 9 个 cases：mha_nomask, gqa_nomask, mha_causal, 
#   mha_causal_s4, mha_causal_s16, mha_causal_s32,
#   mha_causal_hd128_s4, mha_causal_hd128_s16, mha_nomask_hd128_s16

# Step 2: 构建（如果还没有 build 目录或代码有改动）
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target test_self_attention_precision -j8

# Step 3: 运行 310P 原子级测试
ASCEND_PLATFORM=310P ./test_self_attention_precision 2>&1 | tee /tmp/310p_sa_test.log

# Step 4: 查看测试结果摘要
grep -E "cosine|PASSED|FAILED|ERROR|test cases|Skipping" /tmp/310p_sa_test.log
```

**原子级测试用例清单：**

| # | CASE | S | nh | kvh | hd | mask | 说明 | 预期 (310P) |
|---|------|---|----|-----|-----|------|------|------------|
| 1 | mha_nomask | 8 | 4 | 4 | 32 | ❌ | 基础无mask基线 | ✅ PASS |
| 2 | gqa_nomask | 8 | 12 | 4 | 64 | ❌ | GQA (310P跳过) | ⏭ SKIP |
| 3 | mha_causal | 8 | 4 | 4 | 32 | ✅ | S=8, 非16对齐, NZ mask | ❓ 关键测试 |
| 4 | mha_causal_s4 | 4 | 4 | 4 | 32 | ✅ | S=4, 非16对齐, NZ mask | ❓ 关键测试 |
| 5 | mha_causal_s16 | 16 | 4 | 4 | 32 | ✅ | **S=16, 16对齐, NZ mask** | ✅ 最可能通过 |
| 6 | mha_causal_s32 | 32 | 4 | 4 | 32 | ✅ | S=32, 16对齐, NZ mask | ✅ 应通过 |
| 7 | mha_causal_hd128_s4 | 4 | 16 | 16 | 128 | ✅ | 真实MHA, S=4, 非16对齐 | ❓ |
| 8 | mha_causal_hd128_s16 | 16 | 16 | 16 | 128 | ✅ | **真实MHA, S=16, 16对齐** | ✅ 最可能通过 |
| 9 | mha_nomask_hd128_s16 | 16 | 16 | 16 | 128 | ❌ | 真实MHA, 无mask | ✅ PASS |

**关键指标：**
- 所有通过的 CASE：余弦相似度必须 **≥ 0.99**（理想情况下 = 1.0）
- GQA CASE 应显示 `SKIP`（310P 不支持 GQA）
- 如果 case 5/6/8（16对齐 S）全部通过但 case 3/4/7（非16对齐 S）失败 → 说明 310P 严格要求 S 16对齐

### 阶段 2：Graph 组件测试

**仅在阶段 1 全部相关 case 通过（cos ≥ 0.99）后执行。**

```bash
cd atb_cpp_llm/build

# 构建所有测试目标
cmake --build . -j8

# Step 1: SelfAttentionGraph + TextDecoderLayerGraph 构建测试
ASCEND_PLATFORM=310P ./test_text_ops 2>&1 | tee /tmp/310p_text_ops.log
# 预期：7/7 passed
# 如果 Graph Build 失败，记录哪个 graph 失败和错误信息

# Step 2: TextDecoderLayer 精度测试（含参考数据）
ASCEND_PLATFORM=310P ./test_text_decoder_layer_precision 2>&1 | tee /tmp/310p_dec.log
# 检查 cos 值

# Step 3: TextModel 精度测试
ASCEND_PLATFORM=310P ./test_text_model 2>&1 | tee /tmp/310p_text_model.log
```

### 阶段 3：E2E 测试

**仅在阶段 1 和 2 全部通过后执行。**

```bash
cd atb_cpp_llm

# 生成 E2E 参考数据（需要模型文件）
python3 tests/python_reference/gen_cpu_reference.py --stage e2e 2>&1

# 运行 E2E 测试
cd build
ASCEND_PLATFORM=310P ./test_qwen3vl_embedding_e2e 2>&1 | tee /tmp/310p_e2e.log

# 检查 cos 值
grep -E "cosine|similarity|PASSED|FAILED" /tmp/310p_e2e.log
```

### 失败诊断：需要反馈的信息

如果任何步骤失败，请收集以下信息并反馈：

```bash
# === 必收集信息 ===

# 1. 测试输出（完整日志）
cat /tmp/310p_sa_test.log
# 或对应阶段的日志文件

# 2. ATB 内部日志（最重要！）
# C++ 测试的日志：
cat /root/atb/log/$(ls -rt /root/atb/log/ 2>/dev/null | tail -n 1) 2>/dev/null
# 或
cat /home/developer/ascend/log/atb/$(ls -rt /home/developer/ascend/log/atb/ 2>/dev/null | tail -n 1) 2>/dev/null

# 3. 硬件信息
npu-smi info -t board -i 0 2>/dev/null
echo "ASCEND_PLATFORM=$ASCEND_PLATFORM"

# 4. ATB 版本
ls ~/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so
strings ~/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so | grep -E "SelfAttention.*version\|ATB.*version\|build.*time" | head -5

# 5. 哪些 case 失败、哪些通过
grep -E "cosine|FAILED|ERROR|Skipping|test cases" /tmp/310p_sa_test.log
```

**反馈时请同时提供原始日志文件**，不要只粘贴摘要。

### 如果 310P 上 NZ mask 方案全部失败

不要慌。如果所有带 mask 的 case 都失败（但无 mask case 通过），说明 310P 的 SelfAttention fusion kernel 不支持我们提供的 NZ mask 格式。此时：

1. **保存所有日志**（ATB 日志尤其重要，会显示 kernel 内部不接受什么）
2. **反馈给我**，我会立即切换到**手动 attention 方案**（MatMul → scale → add_mask → Softmax → MatMul，全用基础 ATB 算子）
3. 手动 attention 方案**保证可行**，只是性能可能略低于融合算子

## 架构设计：310P 逻辑收敛原则

**核心原则**：Graph builder 和 Model 层不感知 310P/910B 差异。所有平台差异仅在对应的 Op 创建函数中处理。

但有一个例外：**mask 的 format 差异**。因为 310P 和 910B 对 PA_ENCODER mask 的 format 要求不同（NZ vs ND），且 tensor 的 format 在创建时就需要确定，这个差异需要在 mask 创建处（`qwen3vl_model.cpp` 或 engine 层）处理。

### 当前实现中的平台差异位置

| 文件 | Is310P() 判断点 | 用途 |
|------|----------------|------|
| `src/ops/self_attention_op.cpp` | 1 处 | isTriuMask 参数（可能不需要）|
| `src/adapters/qwen3vl_embedding/qwen3vl_model.cpp` | 1 处 | GQA→MHA 权重展开 |
| `src/components/common/self_attention_graph.cpp` | 1 处 | TransdataOp ND→NZ（临时方案）|
| `src/components/common/gqa_attention_builder.cpp` | 1 处 | TransdataOp ND→NZ（临时方案）|
| 测试文件 | N 处 | GQA 测试 SKIP guard |

**目标**：将 self_attention_graph.cpp 和 gqa_attention_builder.cpp 中的 TransdataOp 逻辑移除，改为在 mask 创建时直接指定 NZ 格式。

## 适配策略：GQA→MHA 权重展开

### 原理

GQA (Grouped Query Attention) 中多个 query head 共享同一组 KV head。Qwen3VL-Embedding-2B 的配置是 `nh=32, kv_nh=4, ratio=8`。

将 KV 权重从 GQA 展开到 MHA 是**数学精确**的变换：复制每个 KV head `ratio` 次，使得 `kv_nh == nh`。展开后 MHA SelfAttention 产生与原始 GQA 完全相同的 attention 输出。

```
原始 GQA:  K = (4, hd, hidden)     →  每个 KV head 被 8 个 Q head 共享
展开 MHA:  K = (32, hd, hidden)    →  每个 KV head 复制 8 次
```

### Python 侧实现

位置：`atb_python_qwen3vl_embedding/engine.py:141-194`

```python
def _expand_kv_weights_to_mha(self):
    """权重加载时展开 K/V/K-norm，使 nkv_t == nh_t"""
    ratio = self.nh_t // self.nkv_t
    for li in range(self.n_layer):
        # K weight: (kv_nh*hd, hidden) → (nh*hd, hidden)
        k_w_exp = k_w.reshape(kv_nh, hd, -1) \
            .repeat_interleave(ratio, dim=0) \
            .reshape(nh * hd, -1)
        # V weight: 同理
        # K-norm: (hd,) per-head → 无需展开；(kv_nh*hd,) → 展开
    self.nkv_t = self.nh_t  # 更新后图使用 MHA
```

触发条件（`engine.py:109`）：
```python
if is_310p() and self.nkv_t < self.nh_t:
    self._expand_kv_weights_to_mha()
```

### C++ 侧实现

位置：`atb_cpp_llm/src/adapters/qwen3vl_embedding/qwen3vl_model.cpp:78-176`

逻辑与 Python 完全一致：从 NPU CopyToHost → CPU 上 memcpy 展开 → AllocFloat16 新 NPU tensor → CopyToDevice。最后更新 `config_.text_num_kv_heads = config_.text_num_heads`。

## 配置

### 环境变量

```bash
# .env 中设置（默认 910B）
ASCEND_PLATFORM=310P
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ASCEND_PLATFORM` | `910B` | `"910B"` 或 `"310P"` |

### 平台检测 API

| 语言 | 函数 | 位置 |
|------|------|------|
| Python | `is_310p()` → `bool` | `atb_python_qwen3vl_embedding/utils.py:27` |
| Python | `get_platform()` → `str` | `atb_python_qwen3vl_embedding/utils.py:22` |
| C++ | `Is310P()` → `bool` | `atb_cpp_llm/src/utils/cpp11_compat.h:82` |
| C++ | `Is910B()` → `bool` | `atb_cpp_llm/src/utils/cpp11_compat.h:87` |

## 测试策略

### 测试分层与 310P 兼容性

| 层级 | 310P 行为 | 说明 |
|------|-----------|------|
| Level 0 框架 | 全部通过 | 无 NPU 依赖 |
| Level 1 CPU 纯函数 | 全部通过 | 无 NPU 依赖 |
| Level 2 算子精度 | 全部通过（含 GQA）| GQA 实测 cos=1.0 |
| Level 3 集成 | 全部通过（含 GQA）| GQA 实测通过 |
| Level 4 E2E | 全部通过（走 GQA→MHA 展开路径） | engine 层自动处理 |

### 310P 上无跳过的测试

2026-06-12 实测确认 GQA 在 310P 上完全支持（cos=1.0），所有 GQA skip guard 已移除。

## 精度保证

- **绝不降低阈值**：所有精度测试阈值保持 `cos > 0.99`（多数情况下 > 0.9999）
- **数值等价**：GQA→MHA 展开是精确变换，910B 原路径和 310P 展开路径的输出余弦相似度 > 0.99998
- **逐阶段验证**：Stage reference 测试覆盖 Vision 和 Text 的每个中间阶段

## 排查问题

### ATB 日志

```bash
# Python
cat /root/ascend/log/atb/$(ls -rt /root/ascend/log/atb/ | tail -n 1)

# C++
cat /root/atb/log/$(ls -rt /root/atb/log/ | tail -n 1)
```

### ATB 错误码

| 错误码 | 含义 |
|--------|------|
| 0 | NO_ERROR |
| 4 | ERROR_RT_FAIL（运行时失败）|
| 8 | ERROR_INVALID_TENSOR_DIM（无效 tensor 维度）|
| 9 | ERROR_GRAPH_INFERSHAPE_FUNC_FAIL（图形状推导失败）|
| 13 | ERROR_INVALID_TENSOR_INI_MATCH（tensor 初始化不匹配）|

## 新增 310P 相关代码时注意事项

1. **Python**：使用 `from .utils import is_310p` 检测平台
2. **C++**：使用 `#include "utils/cpp11_compat.h"` 然后 `atb_llm::Is310P()` 检测平台
3. **平台差异收敛**：Graph builder 和 Model 层不应包含 `Is310P()` 判断。所有平台差异仅在对应的 Op 创建函数（如 `self_attention_op.cpp`）中处理。确保两个平台使用相同的输入输出布局（tensor 数量和 shape）。
4. **新增 GQA 测试**：必须加上 `Is310P()` 守卫，避免在 310P 上失败
5. **新增 ATB 算子**：先在 310P 诊断脚本中验证可用性
6. **精度不变**：310P 路径的输出必须与 910B 路径数值等价（cos > 0.99）

## FRACTAL_NZ 格式参考

### NZ 物理 shape 计算（float16）

```
ND shape:  [m, n]
NZ shape:  [1, ceil(n/16), ceil(m/16)*16, 16]

其中:
  ceil(n/16) = (n + 15) / 16
  ceil(m/16)*16 = ((m + 15) / 16) * 16
```

### NZ 数据布局

FRACTAL_NZ (ACL_FORMAT=29) 将矩阵按 16×16 的块重新组织：
- 每个 16×16 block 内部是连续存储的
- block 之间按 (block_row, block_col) 顺序排列
- 如果 m 或 n 不是 16 的倍数，padding 区域填 0

### 示例

S=8 的 causal mask：
- ND: `[8, 8]`，64 个 fp16 元素
- NZ: `[1, ceil(8/16), ceil(8/16)*16, 16]` = `[1, 1, 16, 16]`，256 个 fp16 元素
- NZ 数据：用 padding 填充到 16×16 后在 block 内连续存储

## 310P 实测经验（2026-06-14 更新）

### 已验证事实

| 事实 | 来源 | 结论 |
|------|------|------|
| NZ mask 是唯一可行的 mask 方案 | 6/12 实测 | MASK_TYPE_NORM + NZ mask, cos=1.0 |
| GQA 在 310P 上完全支持 | 6/12 实测 | `gqa_nomask` / `gqa_causal` cos=1.0 |
| S 不需要 16 对齐 | 6/12 实测 | S=4/8 在 NZ mask 下也通过 |
| MASK_TYPE_CAUSAL 不可用 | 6/12 前实验 | 需要 PREFIX_ENCODER + KV Cache |
| MASK_TYPE_NORM_COMPRESS 不可用 | 6/12 前实验 | CreateOp 失败 |
| isTriuMask 无效果 | 6/12 前实验 | 310P 上 TransdataOperation 仍然失败 |
| BNSD 布局不可用 | 6/12 前实验 | CreateOp 失败 |

### 310P 开发注意事项

1. **910B 模拟不可信** — GQA 在 910B 模拟 310P 模式下失败，但真实 310P 上完全正常。必须在真实硬件上验证。
2. **C++ 和 Python 必须同步更新** — C++ 侧 `qwen3vl_model.cpp` 做了 NZ mask 转换后，Python `engine.py` 也必须做。两边不一致会导致一方通过一方失败。
3. **`AllocNpuFloat16` 默认 ND 格式** — 310P 上创建 mask tensor 时必须显式设置 `format = ACL_FORMAT_FRACTAL_NZ`，否则 SelfAttention fusion runner 内部 ND→NZ Transdata 会失败。
4. **mask 创建时机** — mask 只创建一次（缓存复用），所以 CPU 侧 NZ 转换的一次性开销可忽略。
5. **Graph builder 层不创建 mask** — 经代码审计确认，C++ graph builder（`self_attention_graph.cpp` 等）只传递 mask 输入，不分配 mask tensor。ND mask 的来源全在 model 层或测试层。

### 经典错误模式

**症状**: `TransdataOperation mki node infer shape fail, inDims is not support`
**原因**: SelfAttention 收到 ND 格式 mask → fusion runner 内部尝试 ND→NZ 转换 → 310P 上 Transdata 不支持
**修复**: 在 mask 创建处（model 层/测试层）直接生成 NZ 格式 mask，让 SelfAttention 跳过内部转换

### 当前状态 (6/14)

| 组件 | 状态 | 备注 |
|------|------|------|
| C++ 原子级测试 | ✅ 10/10 | cos=1.0 |
| C++ Graph 构建测试 | ✅ 7/7 | 含 GQA |
| C++ Graph 精度测试 | ✅ 23/23 | NZ mask 正确传播 |
| Python ATB mask | ✅ 已修复 | `engine.py` + `text_model.py` NZ 支持 |
| Python E2E | ⏳ 待 310P 验证 | |
| GQA→MHA 展开 | ⏳ 可移除 | GQA 原生支持，不再需要展开 |

## 相关文件索引

| 文件 | 角色 |
|------|------|
| `atb_python_qwen3vl_embedding/utils.py:22-29` | Python 平台检测 |
| `atb_python_qwen3vl_embedding/utils.py:132-176` | make_self_attention() 参数灵活化 |
| `atb_python_qwen3vl_embedding/text_attention.py:19-49` | build_attention/add_attention_graph |
| `atb_python_qwen3vl_embedding/engine.py:105-194` | Python GQA→MHA 展开 |
| `atb_python_qwen3vl_embedding/tests/test_310p_diag.py` | 310P 快速诊断 |
| `atb_python_qwen3vl_embedding/tests/test_310p_combinations.py` | 310P 参数深度扫描 |
| `atb_cpp_llm/docs/platform-310p.md` | 本文档 |
| `atb_cpp_llm/src/utils/cpp11_compat.h:74-89` | C++ 平台检测 |
| `atb_cpp_llm/src/adapters/qwen3vl_embedding/qwen3vl_model.cpp:78-176` | C++ GQA→MHA 展开 |
| `atb_cpp_llm/src/ops/self_attention_op.cpp` | C++ SelfAttentionOp 创建 |
| `atb_cpp_llm/src/components/common/self_attention_graph.cpp` | GQA graph builder（含临时 TransdataOp）|
| `atb_cpp_llm/src/components/common/gqa_attention_builder.cpp` | MHA graph builder（含临时 TransdataOp）|
| `atb_cpp_llm/CMakeLists.txt:197-208` | ctest ENVIRONMENT 属性 |
| `atb_cpp_llm/build_and_test.sh:336-341` | gen_all 容错 fallback |
| `.env.example:24-29` | ASCEND_PLATFORM 配置说明 |
