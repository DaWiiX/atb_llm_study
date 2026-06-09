# 测试加固记录 — 修复"宽松度"问题

> **状态**: P0 (4/4) ✅ · P1 (11/11) ✅ · P2 (0/18) — 低优先级，未排期  
> **完成日期**: 2026-06-08  
> **P2 简要记录**: [refactoring-plan.md §1.3](../refactoring-plan.md) 待解决问题

> **审计起源**：测试工程师反馈测试存在 4 类问题：
> 1. 过于宽松（只检查整除性/非零/shape）
> 2. 硬编码浮点（magic number 无来源）
> 3. 测试数据过于简单（uniform 输入）
> 4. 缺少边界条件

---

## 🎯 架构师职责声明（每次操作前重读）

**身份**: Architect。**禁止**：
- ❌ 亲自写测试代码
- ❌ 亲自跑测试 / 改 CMakeLists
- ❌ 亲自 build / 提交 git

**职责**：
1. 把每个测试加固任务拆成**尽可能小**的原子任务（一个文件一个 case，或同一组紧密相关的几个 case）
2. 用 `Agent` 工具派发给 **coder** subagent（subagent_type=`general-purpose`），prompt 中明确「你是 coder，只做这一件事，不要修改无关代码，不要扩大范围」
3. coder 完成后，把同一个改动派发给独立的 **reviewer** subagent：
   - 编译并跑该测试，确认 PASS
   - 检查是否引入回归（其他测试是否仍 PASS）
   - 验证修改是否真正解决审计指出的问题（不是表面 PASS）
4. 如果 reviewer 报告问题，**派发给 coder 修复，不要自己改**
5. 每完成 / 失败一个任务，更新本表格状态

**测试质量准则**（写进每个 coder + reviewer prompt）：
- 阈值必须有**理论依据**（如 fp16 round-to-nearest 误差 ≤ 2^-10）
- 数据从简单到复杂（small / medium / real-scale）
- **绝不**通过降低阈值"通过"测试（CLAUDE.md 明文规定）
- magic number 必须**注释出处**（"由 Python `xxx.py` 生成"）

**派发流程改进**（2026-06-08 H4 教训 + 用户新规则）：
- ⚠️ **subagent 绝对禁止做任何 git 操作**（add / commit / checkout / stash / push） — 全部由用户做
- ⚠️ **architect 自己也不做 git 操作** — 由用户统一处理（用户明确要求）
- ⚠️ **绝不**让 reviewer 用 `git checkout <file>` 做反向验证清理 — 会清掉 coder 未提交的修改
- ✅ coder 完成后**只**编辑文件，不 commit；architect 只更新 TODO 文档，等用户 commit
- ✅ reviewer 反向验证：先 `cp <file> /tmp/<id>_bak.cpp` 备份，sed 改注入，跑测试，**用 `cp` 还原**不碰 git
- ✅ reviewer 结束时 `git diff <file>` 必须为空（与 HEAD 对比可有合法 coder 修改），作为输出的一部分

---

## ⚠️ 协作禁区（P5 同事正在改的文件）

**避开**以下文件相关的测试，不要在本 TODO 期内动：
- `src/components/common/cross_modal_fusion.h` / `deepstack_fusion.{h,cpp}`
- `src/adapters/qwen3vl_embedding/qwen3vl_model.{h,cpp}`

**禁止动**的测试：
- `tests/level3_integration/test_deepstack.cpp`
- `tests/level3_integration/test_vision_runner_full.cpp`（依赖 qwen3vl_model）
- `tests/level3_integration/test_text_runner_full.cpp`（同上）
- `tests/level4_e2e/*`（端到端，涉及 fusion）
- `tests/level2_op_precision/test_vision_merger_precision.cpp` 中 deepstack 路径

**可以动**的范围：
- ✅ Level 0 / Level 1 / Level 2 中**不**涉及 deepstack 的所有文件
- ✅ `tests/python_reference/` 下的 Python ref 生成脚本

---

## 📊 任务清单

### 🔴 P0 — 高危：测试形同虚设（必须立刻修）

| ID | 文件 / 函数 | 问题 | Coder | Reviewer | Status |
|----|------------|------|-------|----------|--------|
| H1 | `test_preprocess_cpu.cpp::TestSmartResizeBoundary` | 函数末尾固定 `return true`，所有断言失效 | coder x3 (a5348570 → a90fe519 重做) | reviewer (a80b2328) ✅ 4/4 PASS + 反向 ✅ | **done** (commit cf72b86) |
| H2 | `test_vision_rope_cpu.cpp::TestVisionRoPEEdgeCases` | 同上，全部 LOG_INFO 末尾 `return true` | coder (af21493c) | reviewer (a1947971) ✅ 真实有效 | **done** |
| H3 | `test_vision_rope_cpu.cpp::TestVisionRoPEMultiImage` | 只检 sin/cos∈[-1,1]（trivial），无数值对照 | coder (af21493c) | reviewer (a1947971) ✅ cosine=1.0 | **done** |
| H13↑ | `test_mrope_cpu.cpp::TestGetRopeIndexEdgeCases` | Case A 不聚合，Case B 无断言 | coder (a5bfcd50) | reviewer (a2b0a129) ✅ | **done** |

### 🟠 P1 — 中危：覆盖严重不足

| ID | 文件 / 函数 | 问题 | Coder | Reviewer | Status |
|----|------------|------|-------|----------|--------|
| H4 | `test_preprocess_cpu.cpp` | 完全缺 BicubicResize / PreprocessImage 的 Python ref 对比 | coder (a93fb6f0 → a90fe519 重做 + 自 commit) | reviewer (a80b2328) ✅ 5/5 PASS + 反向 ✅ | **done** (commit cf72b86) |
| H5 | `test_io_adapters.cpp::PreprocessImage` | uniform pixel 无法发现 channel-mean 顺序 bug | coder (a11d10de) | reviewer (a91ac674) ✅ 37/37 PASS + 反向 ✅ | **done** |
| H6 | `test_io_adapters.cpp::BicubicResize` | magic numbers 无注释来源 | coder (a93fb6f0) | reviewer (a2b0a129) ✅ 36/36 PASS | **done** |
| H7 | `test_vision_attention_precision.cpp` | cos=1,sin=0 identity RoPE，集成层未测真实 RoPE | coder (adb2c1dc) | reviewer (a0f75dc3) ✅ identity + real RoPE 均 cosine=1.0 | **done** |
| H8 | `test_vision_block_precision.cpp` | 同 H7 + 只 1 个 case，缺真实规模 | coder (adb2c1dc) | reviewer (a0f75dc3) ✅ 3 cases (identity/real_small/real_typical) cosine≥0.999996 | **done** |
| H9 | `test_rms_norm_precision.cpp` | PRENORM / POSTNORM 只测 Create 成功，零精度覆盖 | coder (a6aa8bbb) | reviewer (a91ac674) ✅ 7/7 PASS (cosine=1.0) + 反向 ✅ | **done** |
| H10 | `test_vision_mlp_precision.cpp` | 只 1 个 case，缺真实规模 (H=1024, I=4096) | coder (a4dbdd1d) | reviewer (a0f75dc3) ✅ 3 cases cosine=1.0 | **done** |
| H11 | `test_patch_embed_precision.cpp` | 只 1 个 case，缺真实 Qwen3VL embed=1024 | coder (a4dbdd1d) | reviewer (a0f75dc3) ✅ 3 cases cosine=1.0 | **done** |

### 🟡 P2 — 低危：质量改进

| ID | 文件 / 函数 | 问题 | Coder | Reviewer | Status |
|----|------------|------|-------|----------|--------|
| H12 | `test_gather_reduce_precision.cpp::Gather` | 注释说 bit-identical 但阈值 0.99 | - | - | pending |
| H13 | (已升级至 P0，见上方) | — | — | — | — |
| H14 | `test_softmax_precision.cpp` | sum-to-1 阈值 1e-2 无 H 相关公式注释 | - | - | pending |
| H15 | `test_core.cpp::TensorAllocator` | 只检查 host_dst[0]/[3]，应全 buffer 比对 | - | - | pending |
| H16 | `test_layer_norm_precision.cpp` | 缺 3D 输入 + axis≠last 的 case | - | - | pending |
| H17 | `test_rope_precision.cpp` | 缺 rotaryCoeff=4 路径 | - | - | pending |
| H18 | `test_split_concat_precision.cpp` | 缺 unequal split 和 dim=0 split | - | - | pending |
| H19 | `test_elewise_precision.cpp` | 缺 Add 溢出 + Muls(0/-1/极小) | - | - | pending |
| H20 | `test_transpose_set_value_precision.cpp` | 缺 5D transpose + SetValue 写满 dst | - | - | pending |
| H21 | `test_self_attention_precision.cpp` | 缺 S=1 decode + 超长 seq + padding mask | - | - | pending |
| H22 | `test_swiglu_mlp_precision.cpp` | 缺瓶颈型 I=H/2 + 稀疏权重 case | - | - | pending |
| H23 | `test_text_decoder_layer_precision.cpp` | 缺 use_qk_norm=false + S=1 case | - | - | pending |
| H24 | `test_linear_precision.cpp` | 缺 transpose_a=True + 非方形 K!=N | - | - | pending |
| H25 | `test_activation_precision.cpp` | 缺极端值 ±10 + 1D 输入 | - | - | pending |
| H26 | `test_pos_embed_cpu.cpp` | 缺 merge_size≠2 + MaxAbsDiff 上界断言 | - | - | pending |
| H27 | `test_base_model_utils.cpp` | 缺 RunPooling::LAST_TOKEN CPU 单测 | - | - | pending |
| H28 | `test_float_utils.cpp` | 缺 count=0/1 极小输入 + 多线程 | - | - | pending |
| H29 | `test_mrope_cpu.cpp` | 缺 multi-batch + multi-image + t>1（视频） | - | - | pending |

---

## 📋 派发策略

**串行小批次原则**（防 CMakeLists / Python ref 冲突）：
- 每批 1~3 个无文件冲突的任务
- 每批：派发 coder → 等结果 → 派发 reviewer → 等结果 → 更新表

**派发顺序**（按优先级 + 风险）：
1. **Wave A**: P0 全部（H1, H2, H3）— 这些是"虚假通过"，最危险
2. **Wave B**: H4, H6（同一文件 test_preprocess_cpu，文件改一次）
3. **Wave C**: H5（test_io_adapters PreprocessImage）
4. **Wave D**: H9（test_rms_norm PRENORM/POSTNORM）— 独立
5. **Wave E**: H7, H8（同一域 vision_attention + vision_block，但不同文件，可并行）
6. **Wave F**: H10, H11（单独文件加 case，可并行）
7. **Wave G**: P2 批次（每批 3 个无关文件并行）

---

## 📜 派发历史

- 2026-06-08: 架构师创建本 TODO，审计员（agent a69a8c45）报告完成
- 2026-06-08: **Wave A** 派发 H1/H2/H3 — coder (a58b9c6a, af21493c) 完成，reviewer (a1947971) 发现 H1 Case D 期望值 bug
- 2026-06-08: H1-fix 派发 coder (a5348570)，reviewer (afe0f7d2) 验证通过，并扫描出 H13 是同类"假 PASS"已升级到 P0
- 2026-06-08: **Wave A 完成**（H1 ✅ H2 ✅ H3 ✅），准备派发 Wave A+ (H13) 和 Wave B (H4/H6 同一文件)
- 2026-06-08: **Wave A+ / Wave B** 派发 H13 (coder a5bfcd50)、H4+H6 (coder a93fb6f0)。H4 coder 发现 C++ Catmull-Rom 实际**不**匹配 PyTorch bicubic（注释撒谎），改用 NumPy 复现 C++ 算法做 ref；同时发现 H6 注释加得正确
- 2026-06-08: reviewer (a2b0a129) 验证：H13 ✅、H6 ✅、H4 ⚠️ random_8x8 案例真实超阈值（0.62 > 1e-3）由 Python ref 用 fp 输入 vs C++ 用 uint8 量化误差导致；**且 reviewer 反向验证 `git checkout` 误删了 H4 的 310 行 C++ 修改（python ref 和 bin 都保留）**
- 2026-06-08: 架构师补充流程规则（绝不允许 reviewer git checkout 未提交修改），派发 H4 重做
- 2026-06-08: H1+H4 重做 by coder (a90fe519) 完成并 commit (cf72b86)；reviewer (a80b2328) 用 cp 备份做反向验证，全部 ✅ 5/5 PASS，git 干净，回归全绿
- 2026-06-08: **P0 全部完成**（H1 ✅ H2 ✅ H3 ✅ H13 ✅）+ H6 ✅。下一步派发 Wave C (H5)，Wave D (H9), Wave E (H7/H8)，Wave F (H10/H11)
- 2026-06-08: **Batch 1**: H5 (coder a11d10de) + H9 (coder a6aa8bbb) 编码完成，reviewer (a91ac674) ✅ 
  - H5: test_io_adapters 新增 gradient image vs Python ref 用例 + uniform case max-min < 0.01
  - H9: test_rms_norm 新增 PRENORM/POSTNORM × typical/small 共 4 个精度 case，cosine=1.0
- 2026-06-08: **Batch 2 + 3 reviewer (a0f75dc3) 验证全部 ✅**
  - H7: identity + real RoPE 均 cosine=1.000000
  - H8: 3 cases (identity 0.999999 / real_small 1.000000 / real_typical Qwen3VL 0.999996)
  - H10: 3 cases (原 + typical Qwen3VL + nobias) 全 cosine=1.000000
  - H11: 3 cases (原 + typical Qwen3VL + tiny minimal) 全 cosine=1.000000
  - 反向验证 4 个 ID 全部成功触发 FAIL；回归 test_rms_norm/io_adapters/preprocess 全绿
- 2026-06-08: **P1 全部完成（11/11）**：H4 ✅ H5 ✅ H6 ✅ H7 ✅ H8 ✅ H9 ✅ H10 ✅ H11 ✅，等待用户统一 commit
