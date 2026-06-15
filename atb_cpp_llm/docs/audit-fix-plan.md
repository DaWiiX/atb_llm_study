# 架构审计修复计划

> **审计日期**: 2026-06-14  
> **审计范围**: C++ 架构合规、Python 架构合规、代码鲁棒性、测试套件完整性  
> **发现问题**: 61 项 (1 CRITICAL, 12 HIGH, 24 MEDIUM, 24 LOW)

🏗️  架构审计综合报告

审计范围

┌───────────────────────┬─────────┬────────────────────────────────────────────────┐
│         维度          │ 子代理  │                    发现问题                    │
├───────────────────────┼─────────┼────────────────────────────────────────────────┤
│ C++ 架构合规          │ Agent 1 │ 7 项 (1 HIGH, 3 MEDIUM, 3 LOW)                 │
├───────────────────────┼─────────┼────────────────────────────────────────────────┤
│ Python 架构 & .env    │ Agent 2 │ 14 项 (5 HIGH, 2 MEDIUM, 7 LOW)                │
├───────────────────────┼─────────┼────────────────────────────────────────────────┤
│ C++ 代码质量 & 鲁棒性 │ Agent 3 │ 29 项 (4 HIGH, 15 MEDIUM, 10 LOW)              │
├───────────────────────┼─────────┼────────────────────────────────────────────────┤
│ 测试套件完整性        │ Agent 4 │ 11 项 (1 CRITICAL, 2 HIGH, 4 MEDIUM, 4 LOW)    │
├───────────────────────┼─────────┼────────────────────────────────────────────────┤
│ 合计                  │         │ 61 项 (1 CRITICAL, 12 HIGH, 24 MEDIUM, 24 LOW) │
└───────────────────────┴─────────┴────────────────────────────────────────────────┘

---
🔴 CRITICAL（必须立即修复）

C1. test_io_adapters 未被标记为 needs_refdata

位置: CMakeLists.txt REFDATA_DEPENDENT_TESTS 列表

test_io_adapters.cpp:970-972 读取 3 个 /tmp/preprocess_* .bin 文件，但不在 needs_refdata 标签列表中。当 --no-refdata 时，该测试静默跳过精度验证（WARN + return），CTest 仍报告 PASS。这意味着 CI
可能在无参考数据的情况下假阳性通过。

根因: gen_cpu_reference.py 产生的 preprocess_* 文件使用了与 cpu_* 不同的命名前缀，grep 模式未覆盖。

---
🟠 HIGH（应在当前迭代修复）

H1. utils.py:get_platform() 绕过 .env 文件 — Python 侧平台检测分裂

位置: atb_python_qwen3vl_embedding/utils.py:24

# utils.py:24 — 只看 os.environ，不看 .env
def get_platform() -> str:
    return os.getenv("ASCEND_PLATFORM", "910B")

# env.py:111 — 看 os.environ → .env → default（正确）
ASCEND_PLATFORM = get_env("ASCEND_PLATFORM", default="910B")

影响: 用户只在 .env 中设置 ASCEND_PLATFORM=310P（未 export 到 shell），则 env.ASCEND_PLATFORM = "310P"，但 is_310p() = False。所有生产代码通过 is_310p() 判断平台，.env 中的设置被静默忽略。

H2. 4 个 Python 测试文件未做 310P NZ mask 适配

┌────────────────────────────┬──────┬───────────────────────────────────────────────────────────┐
│            文件            │ 行号 │                           问题                            │
├────────────────────────────┼──────┼───────────────────────────────────────────────────────────┤
│ test_pipeline_trace.py     │ 161  │ make_causal_mask(S).half().npu() — ND mask，310P 上会失败 │
├────────────────────────────┼──────┼───────────────────────────────────────────────────────────┤
│ test_text_attention.py     │ 49   │ 同上                                                      │
├────────────────────────────┼──────┼───────────────────────────────────────────────────────────┤
│ test_text_decoder_layer.py │ 51   │ 同上                                                      │
├────────────────────────────┼──────┼───────────────────────────────────────────────────────────┤
│ test_text_diagnostics.py   │ 315  │ 同上                                                      │
└────────────────────────────┴──────┴───────────────────────────────────────────────────────────┘

这些是之前 NZ mask 修复遗漏的调用点。它们直接创建 ND 格式 mask 传给 ATB graph，310P 上会触发 "call operation setup fail"。

H3. C++ AllocNpuFloat16 失败后无 null 检查 — 未定义行为

位置: qwen3vl_model.cpp:561-566, 576-586

// AllocNpuFloat16 可能失败返回空 NpuTensor (owns_=false)
cached_cos_npu_ = rt_->AllocNpuFloat16({...});  // 可能失败
cached_sin_npu_ = rt_->AllocNpuFloat16({...});  // 可能失败
// 直接解引用，无 null 检查！
alloc->CopyToDevice(*cached_cos_npu_.Get(), cos_host.data(), byte_size);
//                   ^^^^^^^^^^^^^^^^^^^^^^ — 如果失败，Get() 返回 nullptr，解引用 = UB/crash

H4. ATB_LLM_CHECK 宏误用于非 ATB 状态码

位置: weight_loader.cpp:45-71

ATB_LLM_CHECK 宏假设返回值是 atb::Status，但 alloc.AllocFloat16() 和 alloc.CopyToDevice() 返回 atb_llm::Status。失败时宏计算 ERROR_ATB_BASE + _s，产生无意义的错误码。不过该函数 (CopyToNPU) 本身就是死代码——活跃路径使用
io::CopyWeightToFp16NPU。

H5-H6. C++ 图构建逻辑完全重复

H5: self_attention_graph.cpp 与 gqa_attention_builder.cpp — 190 行完全重复的 Q/K/V→RoPE→Attention→O 逻辑

H6: swiglu_mlp_graph.cpp 与 mlp_builder.cpp — 50 行完全重复的 gate→SiLU→up→Mul→down 逻辑

两处都是 Phase 11（Builder 模式拆分）后遗留的旧路径。任何 bug 修复需要在两处同步进行，必然会漂移。

H7. build_result.h 泄露私有头文件到公共 API

位置: include/atb_llm/build_result.h:3

#include "core/raii.h"  // 私有头文件，外部消费者无法访问

BuildResult::graph 成员类型是 OperationHandle（定义在 src/core/raii.h）。外部消费者必须将 src/ 加入 include path 才能编译。

H8-H9. model_registry 和 pos_embed_interp 零测试覆盖

model_registry.cpp 没有任何测试——如果注册静默失败，E2E 测试仍可能通过（因为默认模型硬编码）。pos_embed_interp.cpp 的 resize/truncate 边界情况未独立测试。

---
🟡 MEDIUM（应在下一迭代修复）

架构层面

┌─────┬────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────┐
│  #  │                                          问题                                          │                                   位置                                   │
├─────┼────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ M1  │ src/util/ (1 文件) vs src/utils/ (1 文件) vs 顶层 utils/ (第三方库) — 命名混乱         │ 目录结构                                                                 │
├─────┼────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ M2  │ DeepstackFusion 重复 Setup→GetWorkspace→Execute 模式（应复用 BaseModel::ExecuteGraph） │ deepstack_fusion.cpp:79-113, 194-216                                     │
├─────┼────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ M3  │ add_op lambda 在 14 个文件中完全相同的 6 行定义                                        │ 14 个 graph builder 文件                                                 │
├─────┼────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ M4  │ 设计文档目录结构与实际不符（components 子目录、runners/ 不存在于设计文档）             │ design.md vs 实际                                                        │
├─────┼────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ M5  │ EncodeWithTiming / StageTimings / ForwardWithTiming 泄露到公共 API                     │ include/atb_llm/                                                         │
├─────┼────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
│ M6  │ 3 个文件 import 了 cpp11_compat.h 但未使用任何符号                                     │ self_attention_op.cpp, self_attention_graph.cpp, decoder_layer_graph.cpp │
└─────┴────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────┘

鲁棒性层面

┌─────┬────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
│  #  │                                  问题                                  │                         位置                         │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M7  │ embed_vocab_size 整除无余数检查 — 恶意/损坏的 safetensors 导致 OOB 读  │ qwen3vl_weights.cpp:128-132                          │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M8  │ Reshape lambda 不检查整除性 — 错误尺寸导致 ATB 静默失败或 OOB          │ patch_embed_graph.cpp:36, vision_merger_graph.cpp:43 │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M9  │ 模型缓存字段无 mutex 保护 — 并发 Forward 有 data race                  │ qwen3vl_model.h:89-92                                │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M10 │ 硬编码 5GB buffer_size — 短序列浪费、长序列不够                        │ embedder.cpp:42                                      │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M11 │ 生产路径中散布 debug dump 调用（虽被 env var 门控，但视觉噪音大）      │ qwen3vl_model.cpp 多处                               │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M12 │ 默认日志级别 INFO，生产高吞吐下日志量过大                              │ logger.h:22-30                                       │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M13 │ gen_python_reference.py 是孤儿生成器 — 不注册在 gen_all.py，无消费者   │ tests/python_reference/                              │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M14 │ bf16 ReduceOp 测试无 Is310P() 守卫 — 310P 不支持 bf16                  │ test_gather_reduce_precision.cpp:249,268             │
├─────┼────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ M15 │ test_index_add_npu 用自定义 main() 而非 doctest — CTest 无法逐用例报告 │ test_index_add_npu.cpp:174                           │
└─────┴────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

---
🔵 LOW（技术债务，择机处理）

- EncodeWithTiming / StageTimings 应在 benchmark 专用 header 中
- nd_to_nz_fp16() 是死代码（utils.py:199-224）
- NpuTensor::Release() 设计危险但无调用者
- 3 处 new 应替换为 atb_llm::make_unique
- ATB_LLM_CHECK_ACL 宏未使用
- TensorAllocator 重复 AllocTensor 可能泄漏
- set_value_op 不验证 starts/ends 长度一致
- 多个 size_t 乘法溢出风险
- ModelRegistry::Register 无 mutex
- fprintf 日志线程安全性依赖实现定义
- grid_thw metadata 无大小验证
- ATB_SKIP_TIMING_SYNCS 环境变量未文档化
- .env sourcing 只加载一个文件
- gen_all.py 等文件用直接 os.getenv 做诊断打印
- test_vision_block_ref.py 重复 sys.path.insert
- 等多个 LOW 级别项目

---
📋 设计文档本身的缺陷

审计过程中也发现了设计文档的问题：

D1. design.md 目录结构与实际脱节

设计文档 Section 8 列出：
src/preprocess/   ← 不存在（预处理器在 adapters/qwen3vl_embedding/）
src/components/{attention,mlp,norm,position,vision,fusion}/  ← 实际是 {common,text,vision}/
缺少 src/runners/、src/families/、src/util/、src/utils/ 的说明。

D2. design.md 未区分 src/util/ 和 src/utils/

src/util/cpp11_compat.h（C++11 兼容 shim）和 src/utils/float_utils.h（浮点转换）是两个不同的目录，但设计文档没有说明何时用哪个。建议合并为一个 src/support/ 或明确定义各自的职责边界。

D3. refactoring-plan.md 缺少当前状态的同步更新

Phase 20 之后仍然有 "Remaining Known Issues" 但部分已过时。应该添加 "Current State (as of 2026-06)" 章节。

D4. platform-310p.md 缺少 Python 侧的同步说明

文档只记录了 C++ 侧 NZ mask 需求，但 Python 侧同样需要 FRACTAL_NZ format tag。应添加 Python 侧的对等要求。

---
🎯 建议的修复优先级

第一阶段：阻塞生产上线的问题（本周）

1. H1: 修复 utils.py:get_platform() 使其读取 .env
2. H3: qwen3vl_model.cpp 加 null 检查
3. H4: 删除 weight_loader.cpp 死代码 CopyToNPU
4. H7: 修复 build_result.h 的私有头文件泄露
5. C1: 将 test_io_adapters 加入 needs_refdata 标签
6. H2: 4 个 Python 测试文件加 NZ mask 适配

第二阶段：消除重复、加固测试（下周）

7. H5-H6: SelfAttention/SwiGLU 旧路径委托到 Builder
8. M3: 提取共享 add_op 到 GraphBuilder
9. H8-H9: 补充 model_registry 和 pos_embed_interp 测试
10. M14: bf16 测试加 310P 守卫

第三阶段：设计文档更新、技术债务清理（持续）

11. D1-D4: 更新设计文档
12. M1: 统一 src/util/ 和 src/utils/ 目录
13. 各 MEDIUM/LOW 项目按优先级处理

---

## 修复原则

1. **统一性优先**: 所有开发者遵循同一种设计规则，不引入不同风格
2. **测试驱动验证**: 每项修复由独立的、无上下文的新 subagent 进行测试验证
3. **最小变更**: 只修改问题相关代码，不触及无关部分
4. **记录完整**: 每项修复记录实现方式、测试结果、完成时间

---

## 第一阶段：阻塞生产上线（本周）

### Fix 1.1: H1 — `utils.py:get_platform()` 绕过 `.env` 文件

**严重级别**: HIGH  
**文件**: `atb_python_qwen3vl_embedding/utils.py:24`  
**问题**: `get_platform()` 使用 `os.getenv("ASCEND_PLATFORM", "910B")` 直接读环境变量，不读 `.env` 文件。而 `env.py:111` 的 `ASCEND_PLATFORM = get_env("ASCEND_PLATFORM", default="910B")` 正确实现了 `os.environ → .env → default` 优先级链。导致 `is_310p()` 在用户仅通过 `.env` 设置平台时返回错误结果。

**修复方案**: `utils.py` 中 `get_platform()` 改为返回 `env.py` 的模块级常量 `ASCEND_PLATFORM`。

**实现**: 
1. 移除 `import os`（不再需要）
2. 新增 `from .env import ASCEND_PLATFORM`
3. `get_platform()` 简化为 `return ASCEND_PLATFORM`
4. 修改文件: `utils.py:14-25`

**测试验证**: 
- ✅ Test 1 (直接导入): 无 import error，输出正确反映平台
- ✅ Test 2 (源码检查): `import os` 已删除，`from .env import ASCEND_PLATFORM` 已添加，`os` 在文件中无其他使用
- ✅ Test 3 (一致性检查): `env.py` 的 `get_env()` 正确实现了 os.environ → .env → default 优先级
- ✅ Test 4 (回归检查): 30 个消费者不受影响，`get_platform()` 签名未改变
- ✅ Test 5 (循环导入): `env.py` 不导入 `utils.py`，无循环导入风险

**验证代理**: Agent `a47c5a20` (独立测试)  
**完成时间**: 2026-06-14 (第一阶段)

---

### Fix 1.2: H3 — `AllocNpuFloat16` 失败后无 null 检查

**严重级别**: HIGH  
**文件**: `src/adapters/qwen3vl_embedding/qwen3vl_model.cpp:561-566, 576-586`  
**问题**: `AllocNpuFloat16` 可能返回空 `NpuTensor`（`owns_=false`），后续 `*cached_cos_npu_.Get()` 解引用 nullptr 导致未定义行为（crash）。

**修复方案**: 每次 `AllocNpuFloat16` 调用后检查 `if (!cached_xxx_npu_) return ERROR_NPU_MEMORY;`

**实现**: 
1. 初始修复（4 处）: `cached_cos_npu_`, `cached_sin_npu_`, `cached_mask_npu_`（310P 路径）, `cached_mask_npu_`（910B 路径）
2. 补充修复（17 处）: `ForwardWithTiming` 中 10 处 + `RunVision` 中 7 处
3. 总共 21 处 `AllocNpuFloat16` 调用均添加了 `if (!var) { LOG_ERROR("..."); return ERROR_NPU_MEMORY; }` 模式检查
4. 同时覆盖了 3 处 `AllocNpuInt32` 调用的检查

**测试验证**: 
- ✅ Test 1 (识别所有 AllocNpuFloat16): 23 个调用点，分布在 4 个函数
- ✅ Test 2 (null 检查存在): 22/23 检查紧接分配之后；1 处（ComputeVisionRopeNpu）为组合检查但无提前解引用
- ✅ Test 3 (错误返回类型): ForwardWithTiming/RunVision/ComputePosEmbedNpu 使用 `ERROR_NPU_MEMORY`，ComputeVisionRopeNpu 使用 `-1`（匹配 `int64_t` 返回类型）
- ✅ Test 4 (1:1 配比): 23 处 AllocNpuFloat16，23 处对应检查
- ✅ Test 5 (编译): 100% 构建成功，零错误

**验证代理**: Agent `a4d7618d` (初始), Agent `a3fd3c1e` (补充)  
**完成时间**: 2026-06-14 (第一阶段)

---

### Fix 1.3: H4 — `weight_loader.cpp` 死代码 `CopyToNPU`

**严重级别**: HIGH  
**文件**: `src/io/weight_loader.cpp:45-71`  
**问题**: `ATB_LLM_CHECK` 宏假设返回值为 `atb::Status`，但 `WeightLoader::CopyToNPU` 中调用的函数返回 `atb_llm::Status`。失败时错误码计算错误。且该函数本身就是死代码——活跃路径使用 `io::CopyWeightToFp16NPU`。

**修复方案**: 删除 `CopyToNPU` 方法及其唯一的调用者 `ATB_LLM_CHECK` / `ATB_LLM_CHECK_ACL` 宏（后者也是死代码）。保留 `CopyWeightToFp16Host` 仍有调用者。

**实现**: 
1. 删除 `WeightLoader::CopyToNPU` 方法声明（weight_loader.h）和定义（weight_loader.cpp）
2. 删除 `ATB_LLM_CHECK` 宏定义（logger.h:87-96）
3. 删除 `ATB_LLM_CHECK_ACL` 宏定义（logger.h:98-106）
4. 清理 weight_loader.cpp 中孤儿 includes（`tensor_allocator.h`, `logger.h`, `safetensors.hh`, `<cstring>`）
5. 清理 weight_loader.h 中孤儿 includes 和 forward declarations

**测试验证**: 
- ✅ Test 1 (CopyToNPU 零调用者): 仅有 `test_io_adapters.cpp` 中有陈旧注释提及（非函数调用）
- ✅ Test 2 (ATB_LLM_CHECK 零调用者): 零结果
- ✅ Test 3 (ATB_LLM_CHECK_ACL 零调用者): 零结果
- ✅ Test 4 (weight_loader.h 清理): CopyToNPU 声明已删除，无孤儿 includes
- ✅ Test 5 (weight_loader.cpp 清理): CopyToNPU 定义已删除，剩余方法完整
- ✅ Test 6 (logger.h 清理): 两个宏已删除，LOG_DEBUG/INFO/WARN/ERROR 完好
- ✅ Test 7 (编译): 100% 构建成功，零错误

**已知遗留**: 
1. `test_io_adapters.cpp:465-475` 测试用例名称和注释提及 `CopyToNPU`（LOW，纯文档性质）  
2. `logger.h:7` 的 `#include <acl/acl.h>` 现已成为孤儿（LOW，无害）

**验证代理**: Agent `adfe109a` (独立测试)  
**完成时间**: 2026-06-14 (第一阶段)

---

### Fix 1.4: H7 — `build_result.h` 泄露私有头文件

**严重级别**: HIGH  
**文件**: `include/atb_llm/build_result.h:3`  
**问题**: `#include "core/raii.h"` 引用了 `src/core/raii.h`（私有头文件），外部消费者编译时必须将 `src/` 加入 include path。`OperationHandle` 类型是 `BuildResult::graph` 成员的公开类型。

**修复方案**: 将 `OperationHandle` 类型定义从 `src/core/raii.h` 移动到 `include/atb_llm/` 下的公共头文件中（新建 `include/atb_llm/operation_handle.h` 或合并到 `build_result.h`），原 `raii.h` 通过 `#include` 引用公共头文件。

**实现**: 
1. 新建 `include/atb_llm/operation_handle.h` — 公共头文件，含 `OperationHandle` 类和 `exchange` 模板
2. `build_result.h`: `#include "core/raii.h"` → `#include "atb_llm/operation_handle.h"`
3. `raii.h`: OperationHandle 类定义移除，改为 `#include "atb_llm/operation_handle.h"`
4. `cpp11_compat.h`: `exchange` 模板移除，改为 `#include "atb_llm/operation_handle.h"`

**测试验证**: 
- ✅ Test 1 (公共头文件完整): `operation_handle.h` 有 `#pragma once`，完整类定义，仅含公共/标准 includes
- ✅ Test 2 (build_result.h 清理): `"core/raii.h"` 已移除，零 `src/` 路径引用
- ✅ Test 3 (raii.h 兼容): OperationHandle 已移除（非重复），ContextHandle 完整
- ✅ Test 4 (cpp11_compat.h 兼容): exchange 已移除（非重复），其他模板完好
- ✅ Test 5 (无重复定义): `class OperationHandle` 唯一定义在 `operation_handle.h:18`
- ✅ Test 6 (消费者编译): 35 个含 `raii.h` 的文件通过传递 include 获取 OperationHandle
- ✅ Test 7 (构建): 100% 编译成功，零错误

**验证代理**: Agent `a4cc0368` (独立测试)  
**完成时间**: 2026-06-14 (第一阶段)

---

### Fix 1.5: C1 — `test_io_adapters` 未标记 `needs_refdata`

**严重级别**: CRITICAL  
**文件**: `CMakeLists.txt` REFDATA_DEPENDENT_TESTS 列表  
**问题**: `test_io_adapters.cpp:970-972` 读取 `/tmp/preprocess_*` .bin 文件但未被标记为 `needs_refdata`。当参考数据缺失时测试静默跳过，CTest 报告假阳性 PASS。

**修复方案**: 在 CMakeLists.txt 的 `REFDATA_DEPENDENT_TESTS` 列表中加入 `test_io_adapters`。

**实现**: 
1. `CMakeLists.txt:342`: 在 REFDATA_DEPENDENT_TESTS 列表中添加 `test_io_adapters`（Level 0 framework tests 组）
2. `CMakeLists.txt:327,338`: 更新注释中的 grep 模式，添加 `preprocess_` 前缀
3. `build_and_test.sh:29,320,348`: 更新测试计数 27→28

**测试验证**: 
- ✅ Test 1 (列表包含): `test_io_adapters` 出现在 REFDATA_DEPENDENT_TESTS 中
- ✅ Test 2 (foreach 循环): 正确使用 `set_property(... APPEND ...)` 添加标签
- ✅ Test 3 (bin 文件引用): `test_io_adapters.cpp:970` 含 WARN+return 跳过模式，仅一个测试用例依赖 refdata
- ✅ Test 4 (其他测试无遗漏): 4 个其他引用 /tmp/*.bin 的文件均为 WRITE 路径（不读 refdata），已正确排除
- ✅ Test 5 (哨兵覆盖): `build_and_test.sh` 中 `cpu_op_rms_norm_medium_input.bin` 哨兵来自同一生成器，覆盖 `preprocess_*` 文件
- ✅ Test 6 (grep 模式一致): 注释中的 grep 模式已包含 `preprocess_`，可正确发现 `test_io_adapters.cpp`

**验证代理**: Agent `ad5c93a0` (独立测试)  
**完成时间**: 2026-06-14 (第一阶段)

---

### Fix 1.6: H2 — 4 个 Python 测试文件缺少 NZ mask 适配

**严重级别**: HIGH  
**文件**:
- `atb_python_qwen3vl_embedding/tests/test_pipeline_trace.py:161`
- `atb_python_qwen3vl_embedding/tests/test_text_attention.py:49`
- `atb_python_qwen3vl_embedding/tests/test_text_decoder_layer.py:51`
- `atb_python_qwen3vl_embedding/tests/test_text_diagnostics.py:315`

**问题**: 这些文件在 310P 路径上使用 `make_causal_mask(S).half().npu()` 创建 ND 格式 mask，未调用 `make_causal_mask_nz_npu()`。310P 上 ATB SelfAttention 无法识别 ND mask，触发 "call operation setup fail"。

**修复方案**: 每个文件中 mask 创建处添加 `is_310p()` 分支，310P 时使用 `make_causal_mask_nz_npu(S)`。

**实现**: 
1. `test_pipeline_trace.py`: 导入添加 `is_310p, make_causal_mask_nz_npu`，拆分为 `causal_mask`（ND, TF 用）和 `causal_mask_atb`（NZ on 310P, ATB 用）
2. `test_text_attention.py`: 添加独立导入行，`inputs.append()` 条件化——310P 用 NZ，否则 ND
3. `test_text_decoder_layer.py`: 同 test_text_attention.py 模式
4. `test_text_diagnostics.py`: 同 test_pipeline_trace.py 双变量模式（ATB 和 TF 路径隔离）

**测试验证**: 
- ✅ Test A (导入): 4/4 文件正确导入 `is_310p` 和 `make_causal_mask_nz_npu`
- ✅ Test B (mask 创建): 4/4 文件有 `if is_310p():` 分支和 `else:` 分支
- ✅ Test C (无死代码): 无未使用变量，两种模式均合理（Pattern A: direct-append；Pattern B: two-variable）
- ✅ Test D (ATB 接收正确 mask): 4/4 文件 ATB graph 接收平台适配的 mask
- ✅ Test E (TF 参考无污染): TF/HF 参考路径一致接收 ND 格式 mask
- ✅ Test F (语法): 4/4 文件 `py_compile` 通过

**验证代理**: Agent `a2ce5953` (独立测试)  
**完成时间**: 2026-06-14 (第一阶段)

---

## 第二阶段：消除重复、加固测试

> **第二阶段完成时间**: 2026-06-14

### Fix 2.1: H5 — SelfAttention 旧路径委托到 Builder

**严重级别**: HIGH  
**文件**: `src/components/common/self_attention_graph.cpp`  
**问题**: 旧 flat-param `Build()` 包含 187 行与 `GqaAttentionBuilder` 完全重复的图构建逻辑（Q/K/V 投影 → RMSNorm → RoPE → SelfAttention → O-projection）。任何 bug 修复需要在两处同步进行，必然会漂移。

**修复方案**: 旧 `Build(OperationHandle&, name, head_num, ...)` 改为构造 `AttnConfig` 并委托给 config-based `Build(OperationHandle&, name, const AttnConfig&)`，消除内联重复。

**实现**: 
1. 删除 ~187 行 inline 图构建代码（投影 param 创建、norm、rope、self-attention、output projection）
2. 替换为 10 行委托代码：构造 AttnConfig → 调用 config-based Build()
3. 清理 5 个孤儿 includes（不再需要的头文件）
4. 修改文件: `src/components/common/self_attention_graph.cpp`

**测试验证**: 
- ✅ Test 1 (self_attention 精度): 10/10 测试用例 PASS，cos=1.0
- ✅ Test 2 (decoder_layer 端到端): 2/2 PASS，cos=1.0
- ✅ Test 3 (text_ops 集成): 全部 PASS
- ✅ Test 4 (旧路径已移除): 旧 Build() 中零 inline 图构建残留

**验证代理**: Agent `a0296f84`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.2: H6 — SwiGLU 旧路径委托到 Builder

**严重级别**: HIGH  
**文件**: `src/components/common/swiglu_mlp_graph.cpp`  
**问题**: 旧 `Build(name, out)` 包含 50 行与 `SwiGluBuilder` 完全重复的图构建逻辑（gate_proj → SiLU → up_proj → Mul → down_proj）。违反 DRY 原则，双路径维护必然导致 bug 漂移。

**修复方案**: 旧 `Build(OperationHandle&, name, out)` 改为构造 `MlpConfig` 并委托给 config-based `Build(OperationHandle&, name, const MlpConfig&)`。

**实现**: 
1. 删除 ~50 行 inline 图构建代码（gate/up/down projection param 创建、activation、elementwise multiply）
2. 替换为 4 行委托代码：构造 MlpConfig → 调用 config-based Build()
3. 清理 4 个孤儿 includes
4. 修改文件: `src/components/common/swiglu_mlp_graph.cpp`

**测试验证**: 
- ✅ Test 1 (swiglu_mlp 精度): 2/2 测试用例 PASS，cos=1.0
- ✅ Test 2 (旧路径已移除): 旧 Build() 中零 inline 图构建残留

**验证代理**: Agent `a0296f84`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.3: M3 — 提取共享 `add_op` lambda

**严重级别**: MEDIUM  
**文件**: `src/core/graph_builder.h`, `src/core/graph_builder.cpp`, 12 个 component/runner 文件  
**问题**: 12 个文件中存在完全相同（仅风格微差）的 `add_op` lambda 定义，共 ~72 行重复代码。每个 lambda 都执行相同的 `OperationHandle(var)` 构造 → `builder->AddOperationHandle()` → `graph.push_back(handle)` 模式。

**修复方案**: 在 `GraphBuilder` 中新增 `AddOp(OperationHandle&&, ...)` 重载，接受右值引用，消除所有 12 个 lambda。

**实现**: 
1. `graph_builder.h`: 新增 `AddOp(OperationHandle&&, std::vector<OperationHandle>&)` 方法声明
2. `graph_builder.cpp`: 新增方法实现（构造 handle → AddOperationHandle → push_back）
3. 12 个文件删除 inline `add_op` lambda 定义，替换调用为 `builder->AddOp(...)`
4. 更新的文件涵盖: self_attention, swiglu_mlp, decoder_layer, vision_block, vision_attention, vision_mlp, vision_patch_embed, vision_model, text_model, deepstack_fusion, 等

**测试验证**: 
- ✅ Test 1 (全部组件): 35+ 调用点全部更新，构建 100% 成功
- ✅ Test 2 (lambda 残留): 零个文件仍定义 inline add_op lambda
- ✅ Test 3 (功能回归): 所有组件测试通过

**验证代理**: Agent `a0296f84`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.4: H8 — 补充 `model_registry` 测试

**严重级别**: HIGH  
**文件**: `tests/level0_framework/test_model_registry.cpp` (NEW), `CMakeLists.txt`  
**问题**: `model_registry.cpp` 零测试覆盖。如果注册静默失败，E2E 测试仍可能通过（默认模型硬编码）。注册顺序、优先级、重复注册等关键路径无任何测试保护。

**修复方案**: 创建包含 10 个测试用例的完整测试文件，覆盖 Singleton 模式、注册/创建、优先级、边界条件。

**实现**: 
1. 创建 `MockModel` — 最小 `IModel` 实现，用于测试注册/创建
2. 10 个 `TEST_CASE` 覆盖:
   - Singleton: 单例正确返回
   - Empty lookup: 空注册表查找返回 nullptr
   - Register + Create: 注册后通过名称创建
   - Has: 注册表存在性检查
   - Duplicate register: 重复注册返回已存在工厂（不覆盖）
   - CreateWithFallback exact match: 精确名称匹配
   - CreateWithFallback no compat: 无兼容模型时返回 nullptr
   - Priority ordering: 工厂按注册顺序返回
   - Null factory: 注册空工厂失败
   - Free functions: `RegisterModel` / `CreateModel` 自由函数正确工作
3. `CMakeLists.txt` line 218: 添加 `test_model_registry` test target

**测试验证**: 
- ✅ Test 1: 10/10 TEST_CASE PASS
- ✅ Test 2: 19/19 assertions PASS
- ✅ Test 3: 构建 100% 成功，零警告

**验证代理**: Agent `a616bfbd`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.5: H9 — 补充 `pos_embed_interp` 测试 + 修复除零

**严重级别**: HIGH  
**文件**: `src/components/vision/pos_embed_interp.cpp`, `tests/level1_cpu_pure/test_pos_embed_cpu.cpp`  
**问题**: 当 h=1 或 w=1 时，`(i * (h-1)) / h` 中 `h-1=0` 导致 float 除零（产生 NaN/inf）。此外 6 个关键边界情况（单像素、num_images=0、非整除 merge 等）未测试。

**修复方案**: 添加 h<=1 或 w<=1 时的 guard（三元运算符直接返回边界坐标），并新增 6 个边界测试用例。

**实现**: 
1. `pos_embed_interp.cpp` lines 29-30: 三元运算符 guard — `coord = (size <= 1) ? 0.0f : (i * (size - 1)) / (size - 1)`
2. 新增 6 个 `TEST_CASE`:
   - h=1 edge: 高度为 1 的插值
   - w=1 edge: 宽度为 1 的插值
   - single pixel: h=1, w=1 全单像素
   - num_images=0: 空输入
   - temporal>1: 多帧输入
   - non-divisible merge: 不能整除的 merge_size

**测试验证**: 
- ✅ Test 1: 9/9 TEST_CASE PASS (3 pre-existing + 6 new)
- ✅ Test 2: cos=1.0（与期望输出完全一致）
- ✅ Test 3: 除零 guard 位置正确（在循环内、除法之前）

**验证代理**: Agent `a616bfbd`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.6: M14 — bf16 测试加 310P 守卫

**严重级别**: MEDIUM  
**文件**: `tests/level2_op_precision/test_gather_reduce_precision.cpp`  
**问题**: bf16 `ReduceOp::MAX` 和 `ReduceOp::MIN` 测试无条件运行。310P 平台不支持 `ACL_BF16`，导致测试在 310P 上崩溃。

**修复方案**: 两个 TEST_CASE 顶部添加 `Is310P()` guard，在 310P 平台跳过。

**实现**: 
1. 添加 `#include "utils/cpp11_compat.h"`（提供 `Is310P()`）
2. MAX bf16 test (line 251): 顶部添加 `if (Is310P()) { GTEST_SKIP() << "bf16 not supported on 310P"; }`
3. MIN bf16 test (line 274): 同上

**测试验证**: 
- ✅ Test 1: 编译通过，零错误
- ✅ Test 2: guard 位置正确（TEST_CASE 第一个语句）
- ✅ Test 3: 910B 上测试正常执行，310P 上正确跳过

**验证代理**: Agent `a616bfbd`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.7: M1 — 统一 `src/util/` 和 `src/utils/` 目录

**严重级别**: MEDIUM  
**文件**: `cpp11_compat.h` 移动，19 个文件的 include 更新，`src/util/` 目录删除  
**问题**: `src/util/cpp11_compat.h`（1 文件）和 `src/utils/float_utils.h`（1 文件）命名混淆——两个同义目录名各含 1 个文件，开发者不清楚何时用哪个。

**修复方案**: 移动 `cpp11_compat.h` 到 `src/utils/`，更新全部 include 引用，删除空 `src/util/` 目录。

**实现**: 
1. `cpp11_compat.h` 从 `src/util/` 移动到 `src/utils/`
2. 12 个源文件 + 7 个测试文件 include 路径从 `"util/cpp11_compat.h"` 更新为 `"utils/cpp11_compat.h"`
3. 删除空 `src/util/` 目录
4. `CMakeLists.txt` 无需修改（`PRIVATE include` 目录已覆盖 `src/`）

**测试验证**: 
- ✅ Test 1: 构建 100% 成功，零错误
- ✅ Test 2: 0 个文件仍引用旧路径 `util/cpp11_compat.h`
- ✅ Test 3: 2 个文档陈旧引用已修复
- ✅ Test 4: 所有测试通过

**验证代理**: Agent `a499ee4d`（发现 2 个文档陈旧引用 → 已修复）  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.8: M2 — DeepstackFusion 复用 ExecuteOperation 共享函数

**严重级别**: MEDIUM  
**文件**: `src/families/base_model.h`, `src/families/base_model.cpp`, `src/components/common/deepstack_fusion.cpp`  
**问题**: `BaseModel::ExecuteGraph` 和 `DeepstackFusion::ExtractFeatures` / `InjectFeatures` 中重复实现了完全相同的 Setup → GetWorkspace → Execute 模式（~80 行重复）。任何 ATB API 变更都需要三处同步修改。

**修复方案**: 提取 `ExecuteOperation()` 自由函数（inline）到 `base_model.h` 并使其返回 `atb::Status`，调用点内部进行错误处理。`BaseModel::ExecuteGraph` 和 `DeepstackFusion` 的两个方法均委托给该函数。

**实现**: 
1. `base_model.h`: 新增 ~70 行 `ExecuteOperation()` inline 实现（Setup → GetWorkspaceSize → Alloc → SetWorkspace → Execute）
2. `BaseModel::ExecuteGraph`: 从 ~50 行减少到 ~10 行（委托给 ExecuteOperation）
3. `DeepstackFusion::ExtractFeatures`: 从 ~35 行减少到 ~10 行
4. `DeepstackFusion::InjectFeatures`: 从 ~35 行减少到 ~10 行

**测试验证**: 
- ✅ Test 1: test_base_model_utils — 12/12 PASS
- ✅ Test 2: test_deepstack — 4/4 PASS
- ✅ Test 3: test_deepstack_npu_tensor — 5/5 PASS
- ✅ Test 4: test_index_add_npu — 4/4 PASS, cos=1.0

**验证代理**: Agent `a0bf075b`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.9: M7 — `embed_vocab_size` 整除检查

**严重级别**: MEDIUM  
**文件**: `src/adapters/qwen3vl_embedding/qwen3vl_weights.cpp`  
**问题**: `embed_vocab_size` 计算中 `text_hidden_size` 可能为 0（除零），tensor shape 可能非 2 维，vocab 大小可能不整除 `text_hidden_size`。恶意或损坏的 safetensors 可触发未定义行为。

**修复方案**: 在除法前添加 3 个 guardrail 检查。

**实现**: 
1. `text_hidden_size > 0` 检查：为 0 时返回 `ERROR_WEIGHT_LOAD`
2. `shape.size() == 2` 检查：非 2 维时返回 `ERROR_WEIGHT_LOAD`
3. `total_elems % text_hidden_size == 0` 整除检查：不整除时返回 `ERROR_WEIGHT_LOAD`
4. 修改文件: `qwen3vl_weights.cpp`

**测试验证**: 
- ✅ Test 1: 代码审查通过，所有 guard 在除法操作之前
- ✅ Test 2: 构建成功
- ✅ Test 3: guard 路径覆盖: 零值、非二维、不整除三种场景

**验证代理**: Agent `a499ee4d`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.10: M8 — Reshape lambda 整除检查

**严重级别**: MEDIUM  
**文件**: `src/components/vision/patch_embed_graph.cpp`, `src/components/vision/vision_merger_graph.cpp`  
**问题**: `patch_embed_graph.cpp` 中 `kernel_size` 可能为 0 导致除零；`vision_merger_graph.cpp` 中 `mer_hs = hidden_size / merge_size` 在 `merge_size=0` 时除零。错误尺寸导致 ATB 静默失败或 OOB 访问。

**修复方案**: 在 `Build()` 顶部添加参数检查，检测到无效值时返回 `ERROR_INVALID_PARAM`。

**实现**: 
1. `patch_embed_graph.cpp`: `if (kernel_size <= 0) { LOG_ERROR("..."); return ERROR_INVALID_PARAM; }` — 在 reshape lambda 之前
2. `vision_merger_graph.cpp`: `if (hidden_size <= 0 || merge_size <= 0) { LOG_ERROR("..."); return ERROR_INVALID_PARAM; }` — 在 mer_hs 计算之前

**测试验证**: 
- ✅ Test 1: 代码审查通过，所有 guard 在除法/reshape 之前
- ✅ Test 2: 构建成功
- ✅ Test 3: 零值和负值均触发 guard

**验证代理**: Agent `a499ee4d`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.11: M9 — 模型缓存字段添加线程安全文档

**严重级别**: MEDIUM  
**文件**: `src/adapters/qwen3vl_embedding/qwen3vl_model.h`, `include/atb_llm/model.h`  
**问题**: 缓存字段（`cached_cos_npu_`, `cached_sin_npu_`, `cached_mask_npu_`）无 mutex 保护，并发 `Forward()` 有 data race 风险。由于当前用例为单线程推理，不需要添加锁（锁会引入性能开销），但必须文档化此约束。

**修复方案**: 添加 Doxygen `@warning` 和 `@note` 文档说明线程安全约束。纯文档变更，无代码逻辑修改。

**实现**: 
1. `qwen3vl_model.h`: 缓存字段添加 `@warning These fields are not mutex-protected. Concurrent Forward() calls cause data races.`
2. `qwen3vl_model.h`: `Forward()` 和 `ForwardWithTiming()` 添加 `@note Not thread-safe. Callers must serialize access.`
3. `model.h`: `IModel::Forward()` 接口添加 `@note Implementations may not be thread-safe. Check concrete model documentation.`

**测试验证**: 
- ✅ Test 1: Doxygen 格式正确（`@warning`, `@note` 语法有效）
- ✅ Test 2: 文档清晰准确（约束和风险明确说明）
- ✅ Test 3: 构建成功（零代码变更）

**验证代理**: Agent `a0bf075b`  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.12: M10 — buffer_size 可配置化

**严重级别**: MEDIUM  
**文件**: `src/engine/embedder.cpp`  
**问题**: 5 GiB `buffer_size` 硬编码——短序列（如 S=64）浪费内存，长序列（如 S=32768）可能不够。不同工作负载需要不同的 buffer 配额。

**修复方案**: 通过 `ATB_BUFFER_SIZE_GB` 环境变量配置，默认 5 GiB，支持 1-1024 GiB 范围。

**实现**: 
1. 添加 `#include <cstdlib>`（`std::getenv`）
2. 读取 `ATB_BUFFER_SIZE_GB` 环境变量
3. 验证值 > 0 且 <= 1024（上限防止误配置）
4. 无效时回退到 5 GiB 默认值（含 WARN 日志）
5. 修改文件: `src/engine/embedder.cpp`

**测试验证**: 
- ✅ Test 1: 解析逻辑正确 — 合法值正确使用
- ✅ Test 2: 边界处理 — 0/负数/非数字均正确处理，回退到默认值
- ✅ Test 3: 上限 1024 GiB 生效

**验证代理**: Agent `a499ee4d`（建议添加上限 → 已添加）  
**完成时间**: 2026-06-14 (第二阶段)

---

### Fix 2.13: M13 — 删除孤儿生成器 `gen_python_reference.py`

**严重级别**: MEDIUM  
**文件**: `tests/python_reference/gen_python_reference.py` (DELETED)  
**问题**: 该文件未注册在 `gen_all.py` 中，且无任何 C++ 消费者读取其输出文件。它是 Phase 10 重构遗留的死代码，存在误导风险（新开发者可能误以为是活跃的参考数据生成器）。

**修复方案**: 确认零消费者后安全删除。

**实现**: 
1. 确认 `gen_all.py` 不导入或调用 `gen_python_reference.py`
2. 确认无 C++ 测试文件读取其输出（grep 零结果）
3. 删除 `tests/python_reference/gen_python_reference.py`

**测试验证**: 
- ✅ Test 1: 0 个引用残留（grep 全仓库零结果）
- ✅ Test 2: `gen_all.py` 不受影响（5/5 生成器正常）
- ✅ Test 3: 构建不受影响

**验证代理**: Agent `a499ee4d`  
**完成时间**: 2026-06-14 (第二阶段)

---

## 第三阶段：技术债务清理

> **第三阶段完成时间**: 2026-06-14

Phase 3 共修复 12 个项目（5 MEDIUM + 4 设计文档 + 3 潜在 bug），派出 8 个开发代理 + 3 个独立测试代理 + 3 个重做修复代理 + 1 个全量测试代理 = 15 个 subagent。

**教训**: 第一轮开发代理中 3/8（WP1、WP2、WP5）报告了修改但未实际写入文件（agent 行为异常）。重做后所有修改均已验证生效。WP6 的 L04 修复引入了 CRITICAL bug（构造函数 public 破坏工厂模式），已通过 WP6-FIX 还原。

---

### Fix 3.1: M5 — `StageTimings` 提取到独立 `timing.h` 头文件

**严重级别**: MEDIUM  
**文件**: `include/atb_llm/timing.h` (NEW), `include/atb_llm/types.h`

**问题**: `StageTimings` 结构体（8 个 timing 字段）定义在 `types.h` 中，污染公共 API 表面。

**修复方案**: 创建 `include/atb_llm/timing.h`，将 `StageTimings` 移动过去，`types.h` 通过 `#include "atb_llm/timing.h"` 保持向后兼容。

**实现**: 
1. 新建 `include/atb_llm/timing.h` — 含 `StageTimings` 结构体（8 个 double 字段）
2. `types.h`: 删除 `StageTimings` 定义（12 行），新增 `#include "atb_llm/timing.h"`
3. 零消费者变更 — `timing.h` 通过 cmake `install(DIRECTORY include/atb_llm ...)` 自动安装

**测试验证**: 
- ✅ Test 1: 100% 编译成功，零错误
- ✅ Test 2: 所有消费者正常（benchmark.cpp, test_embedder_utils.cpp, llm_engine.cpp, embedder.cpp）
- ✅ Test 3: 全量 CTest 51/51 PASS

**验证代理**: Agent `a287de83` (开发), Agent `a754fd1a` (独立测试)  
**完成时间**: 2026-06-14 (第三阶段)

---

### Fix 3.2: M6+M11+M12+M15 — 批量 MEDIUM 项目修复

**M6: 死代码 cpp11_compat.h include**
- `src/ops/self_attention_op.cpp:3` — 删除 `#include "utils/cpp11_compat.h"`（零符号使用）
- `src/components/text/decoder_layer_graph.cpp:8` — 同上
- `src/log/logger.h:7` — 删除孤儿 `#include <acl/acl.h>`（Phase 1 已删除 ATB_LLM_CHECK 宏）

**M11: 生产路径调试代码整理**
- `qwen3vl_model.cpp`: 新增 7 个匿名命名空间辅助函数（DebugDumpVisionRoPE, DebugDumpFirstLayer, DebugDumpBlock1 等），收拢 13 处 inline `debug::Dump*` 调用
- 所有 `ATB_DEBUG_VISION` env var 门控保持不变，零运行时开销

**M12: 默认日志级别 + SetLogLevel API**
- `logger.h`: 默认级别 `INFO` → `WARN`；新增 `SetLogLevel(LogLevel)` API（优先级：SetLogLevel > LOG_LEVEL env var > WARN 默认）
- `qwen3vl_model.cpp`: 2 处 per-inference timing LOG_INFO 降级为 LOG_DEBUG
- `debug_dump.cpp`: 2 处 debug dump LOG_INFO 降级为 LOG_DEBUG

**M15: 测试文件 doctest 化**
- 3 个文件从自定义 `int main()` 转换为 `DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN`：`test_index_add_npu.cpp` (4 TEST_CASE)、`test_pos_embed_npu_graph.cpp` (10 TEST_CASE)、`test_vis_rope_npu_graph.cpp` (10 TEST_CASE)
- 移除手动 `CaseResult` 结构体和 pass/fail 计数，替换为 `CHECK()`/`REQUIRE()`/`CAPTURE()` 宏

**测试验证**: 
- ✅ Test: 全量编译 100% 成功
- ✅ Test: 全量 CTest 51/51 PASS
- ✅ Test (M11): 7 个辅助函数参数与原 inline 调用完全一致
- ✅ Test (M12): `SetLogLevel()` API 存在，默认 WARN 生效
- ✅ Test (M15): 24 个 TEST_CASE 全部编译成功

**验证代理**: Agent `a87a1707`/`a6de2799`/`a060a217`/`ab23d96f` (开发), Agent `a9235baf`/`a754fd1a`/`a7f79eda` (独立测试)  
**完成时间**: 2026-06-14 (第三阶段)

---

### Fix 3.3: D1-D4 — 设计文档更新

**D1+D2: `docs/design.md`**
- Section 5.1: 组件目录从旧 `attention/mlp/norm/position/fusion` 更新为实际 `common/text/vision`
- Section 8: 目录树完全重写匹配实际结构（adapters 移到 src/、移除 src/preprocess/、添加 src/families/runners/utils、tests 从 unit/benchmark 改为 level0-4、include 数从 4 更新为 10）
- 项目名 `atb_cpp_llm_engine` → `atb_cpp_llm`
- 修复 stale 路径 `components/mlp/moe_mlp_graph.h` → `components/common/moe_mlp_graph.h`

**D3: `docs/refactoring-plan.md`**
- Section 1.2: 移除 `PrepareInputs`/`RunTextDecoder` 自矛盾（标注已删除）
- Section 1.3: 待解决问题表添加状态列（InjectFeatures=已解决，Debug冲突=仍存在）

**D4: `docs/platform-310p.md`**
- 相关文件索引表添加 Python 侧 3 个入口（utils.py, engine.py, float_utils.h）
- 新增 "Python 侧 NZ mask 生成" 子章节（含 `make_causal_mask_nz_npu()` 代码示例）
- 更新状态标记："方案 6" 待实施→已实施，GQA→MHA 已确认

**测试验证**: 
- ✅ Test: 交叉验证无 stale 引用残留
- ✅ Test: `make_causal_mask_nz_npu()` 代码示例补全（import torch_npu + shape 定义）

**验证代理**: Agent `a6bdadff` (开发), Agent `a754fd1a` (独立测试), Agent `acb7187e` (小 bug 修复)  
**完成时间**: 2026-06-14 (第三阶段)

---

### Fix 3.4: L07+L11+L03 — 潜在 bug 修复

**L07: `set_value_op.cpp` — starts/ends 长度验证**
- 在 `Create()` 顶部添加 `if (starts.size() != ends.size())` 检查
- 不匹配时返回空 `OperationHandle` 并记录 `LOG_ERROR`

**L11: `vis_rope_npu_graph.cpp` — MaxGridHW 边界检查**
- 添加 `kMaxImages = 256` 上限和 `num_images <= 0` 检查
- 无效输入返回 0 并记录 `LOG_ERROR`，防止 OOB 读

**L03: `npu_tensor.h` — Release() 标记 deprecated**
- 添加 `[[deprecated("Use Get() for read access...")]]` 属性
- 警告调用者 Release() 转移裸 NPU 指针所有权，需手动 `aclrtFree()`

**测试验证**: 
- ✅ Test: 全量编译 100% 成功
- ✅ Test: 全量 CTest 51/51 PASS
- ✅ Test: 验证代码确实存在于文件中

**验证代理**: Agent `aa19a4cc` (开发), Agent `a7f79eda` (独立测试)  
**完成时间**: 2026-06-14 (第三阶段)

---

### Fix 3.5: L02+L18+L23 — 死代码清理 + 风格统一

**L02: 删除死代码 `nd_to_nz_fp16()`**
- `atb_python_qwen3vl_embedding/utils.py`: 删除 `nd_to_nz_fp16()` 函数（零调用者，活跃路径使用 `make_causal_mask_nz_npu()`）

**L04: 保持 `new` + `unique_ptr`（未引入 make_unique）**
- 审计分类为 "truly LOW (cosmetic)"
- 第一轮尝试 `make_unique` 导致构造函数必须 public（CRITICAL bug）
- WP6-FIX 还原：构造函数保持 private，使用原始 `new` + `std::unique_ptr`

**L18: 清理 stale CopyToNPU 注释**
- `test_io_adapters.cpp:465-476`: 测试名和注释更新，标注 CopyToNPU 已在 Phase 1 删除

**L23: 更新 cpp11_compat.h 注释**
- 过期 "将来升级到 C++14" 注释改为 "Provides C++14/17 backports for C++11 compatibility"

**测试验证**: 
- ✅ Test: 全量编译 100% 成功
- ✅ Test: 构造函数为 private（工厂模式安全）
- ✅ Test: `nd_to_nz_fp16` 零引用残留

**验证代理**: Agent `a3845fe3` (开发), Agent `a7edb364` (L04 修复), Agent `a9235baf` (独立测试)  
**完成时间**: 2026-06-14 (第三阶段)

---

## 第三阶段全量测试

**全量测试代理**: Agent `a27a846e`

| 指标 | 结果 |
|------|------|
| 编译警告 | **0** |
| 测试总数 | **51** |
| 通过 | **51** (100%) |
| 失败 | **0** |
| 跳过 | **0** |

**Phase 3 修改验证（12/12 确认）**:

| 工作包 | 检查项 | 结果 |
|--------|--------|------|
| WP1 | `cpp11_compat.h` 从 2 文件移除, `acl.h` 从 logger.h 移除 | ✅ |
| WP2 | 默认 WARN, `SetLogLevel()` API 存在 | ✅ |
| WP3 | 3 文件 24 TEST_CASE doctest 化 | ✅ |
| WP4 | 3 设计文档更新, 无 stale 引用 | ✅ |
| WP5 | `set_value_op` 验证, `MaxGridHW` 边界, `Release()` deprecated | ✅ |
| WP6 | 构造函数 private, `nd_to_nz_fp16` 删除, 注释更新 | ✅ |
| WP7 | `timing.h` 存在, `StageTimings` 提取 | ✅ |
| WP8 | 7 辅助函数收拢 13 处 debug 调用 | ✅ |

---

## 修复完成状态总览

| 编号 | 描述 | 严重级别 | 状态 | 完成时间 | 测试结果 |
|------|------|----------|------|----------|----------|
| 1.1 | get_platform() 读 .env | HIGH | ✅ 已完成 | 2026-06-14 | 5/5 PASS |
| 1.2 | AllocNpuFloat16 null 检查 | HIGH | ✅ 已完成 | 2026-06-14 | 5/5 PASS (含补充) |
| 1.3 | 删除 CopyToNPU 死代码 | HIGH | ✅ 已完成 | 2026-06-14 | 7/7 PASS |
| 1.4 | build_result.h 私有头文件泄露 | HIGH | ✅ 已完成 | 2026-06-14 | 7/7 PASS |
| 1.5 | test_io_adapters needs_refdata | CRITICAL | ✅ 已完成 | 2026-06-14 | 6/6 PASS |
| 1.6 | 4 文件 NZ mask 适配 | HIGH | ✅ 已完成 | 2026-06-14 | 24/24 PASS |
| 2.1 | SelfAttention 旧路径委托到 Builder | HIGH | ✅ 已完成 | 2026-06-14 | 4/4 PASS |
| 2.2 | SwiGLU 旧路径委托到 Builder | HIGH | ✅ 已完成 | 2026-06-14 | 2/2 PASS |
| 2.3 | 提取共享 add_op lambda | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 2.4 | 补充 model_registry 测试 | HIGH | ✅ 已完成 | 2026-06-14 | 10/10 TEST_CASE |
| 2.5 | 补充 pos_embed_interp 测试 + 除零修复 | HIGH | ✅ 已完成 | 2026-06-14 | 9/9 TEST_CASE |
| 2.6 | bf16 测试加 310P 守卫 | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 2.7 | 统一 src/util/ 和 src/utils/ 目录 | MEDIUM | ✅ 已完成 | 2026-06-14 | 4/4 PASS |
| 2.8 | DeepstackFusion 复用 ExecuteOperation | MEDIUM | ✅ 已完成 | 2026-06-14 | 4/4 PASS |
| 2.9 | embed_vocab_size 整除检查 | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 2.10 | Reshape lambda 整除检查 | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 2.11 | 模型缓存字段线程安全文档 | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 2.12 | buffer_size 可配置化 | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 2.13 | 删除孤儿生成器 gen_python_reference.py | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 3.1 | StageTimings 提取到 timing.h (M5) | MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 3.2 | M6/M11/M12/M15 批量修复 | MEDIUM | ✅ 已完成 | 2026-06-14 | 51/51 PASS |
| 3.3 | D1-D4 设计文档更新 | MEDIUM | ✅ 已完成 | 2026-06-14 | 4/4 PASS |
| 3.4 | L07+L11+L03 潜在 bug 修复 | LOW→MEDIUM | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| 3.5 | L02+L18+L23 死代码+风格统一 | LOW | ✅ 已完成 | 2026-06-14 | 3/3 PASS |
| — | **全量回归测试** | — | ✅ 已完成 | 2026-06-14 | **51/51 PASS** |
