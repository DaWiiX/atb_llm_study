# 测试修复计划

基于 2026-06-15 对 Python（22 文件）和 C++（79 文件）测试套件的全面审计，整理出以下问题及修复方案。

---

## 问题索引

| # | 严重度 | 区域 | 问题摘要 |
|---|--------|------|----------|
| [P1](#p1) | 🔴 高 | C++ | IMAGE_ONLY 余弦阈值 0.90 可能隐藏精度 bug |
| [P2](#p2) | 🔴→🟢 已修复 | C++ | Stage L2a 静默忽略低于阈值的精度结果 | ✅ 2026-06-15 |
| [P3](#p3) | 🔴→🟢 已修复 | C++ | DecoderLayer MHA+causal mask 精度测试缺失 | ✅ 2026-06-15 |
| [P4](#p4) | 🔴 高 | C++ | DeepSeek/Mixtral/Qwen3(non-VL) 零测试覆盖 |
| [P5](#p5) | 🔴 高 | Python | Deepstack 跨模态融合无自动化测试 |
| [P6](#p6) | 🔴→🟢 已修复 | Python | test_text_model.py 只用 2 层测试（真实模型 28 层） | ✅ 2026-06-15 |
| [P7](#p7) | 🔴 高 | C++ | BicubicResize 参考实现自证（非独立参考） |
| [P8](#p8) | 🟡 中 | Python | test_text_attention.py 与 test_310p_diag.py 冗余 |
| [P9](#p9) | 🟡 中 | Python | test_nz_quick_verify.py 与 test_nz_format_verify.py 冗余 |
| [P10](#p10) | 🟡 中 | Python | test_embedder_e2e.py 与 test_e2e.py + benchmark.py 重叠 |
| [P11](#p11) | 🟡 中 | C++ | test_consistency.cpp 是 test_accuracy.cpp 的子集 |
| [P12](#p12) | 🟡 中 | C++/Python | 预处理阶段在多个文件中重复测试 |
| [P13](#p13) | 🟡 中 | Python | set_atb_buffer_size 调用位置不一致（7+9 个文件） |
| [P14](#p14) | 🟡 中 | Python | 模型加载代码在 6 个文件中重复 |
| [P15](#p15) | 🟡 中 | Python | 单元测试使用非真实模型维度 |
| [P16](#p16) | 🟡 中 | Python | 无自动化测试运行器 / 汇总脚本 |
| [P17](#p17) | 🟡 中 | C++ | test_config_wiring.cpp 硬编码 epsilon 阈值 |
| [P18](#p18) | 🟢 低 | Python | IMAGE_TOKEN_ID 硬编码（应从 config 读取） |
| [P19](#p19) | 🟢 低 | Python | 死代码：VISION_START_TOKEN_ID / run_quick_tf 未使用参数 |
| [P20](#p20) | 🟢 低 | Python | 测试间余弦阈值不一致且缺乏文档说明 |
| [P21](#p21) | 🟢 低 | Python | benchmark.py 手动复制 engine._run_vision 逻辑 |

---

## 🔴 高优先级

### P1: IMAGE_ONLY 余弦阈值 0.90 可能隐藏精度 bug {#p1}

**问题是什么**

`atb_cpp_llm/tests/level4_e2e/test_stage_precision.cpp:40` 中 `COSINE_THRESHOLD_IMG_ONLY = 0.90f`。同一模型、同一输入，C++ 和 Python 的 IMAGE_ONLY 推理结果余弦相似度仅 0.90。代码注释称"vision-only tokens 无 text anchor，fp16 精度差异导致更大发散"，但 0.90 远低于 fp16 正常预期（通常 ≥0.98），可能是真实的精度 bug（如权重加载路径、position embedding 计算、或 RoPE 实现的差异）。

**怎么修复**

1. **逐阶段分解对比**：将 IMAGE_ONLY 推理拆分为预处理 → PatchEmbed → PosEmbed → VisionBlock×24 → Merger 各阶段，逐阶段对比 C++ 和 Python 的中间输出余弦相似度，定位精度下降发生在哪个阶段。
2. **交叉验证**：用 Python 的 TF reference 生成各阶段中间输出，让 C++ 加载并对比。
3. **权重逐元素对比**：验证 C++ 和 Python 加载的 vision 权重逐元素一致（不仅是 cosine，还要看 max_diff）。
4. 定位根因后修复，将阈值提升至 ≥0.98。

**怎么验收**

- IMAGE_ONLY 余弦相似度 ≥ 0.98（或定位到不可消除的 fp16 累积误差后，将阈值设为合理值并附带逐阶段精度分析文档）。
- 在 `test_stage_precision.cpp` 中为降低后的阈值添加详细注释，说明每一阶段的精度贡献。

**如何避免再犯**

- 任何精度阈值 < 0.98 必须有逐阶段精度分解文档支撑。
- 新增 E2E 测试用例时，必须同时记录各阶段的余弦相似度，形成精度基线。
- CI 中对比历史精度基线，阈值下降 > 0.01 时报警。

---

### P2: Stage L2a 静默忽略低于阈值的精度结果 {#p2}

**问题是什么**

`atb_cpp_llm/tests/level3_integration/test_vision_stages.cpp:534-540`，Stage L2a（Vision RoPE + PosEmbed）精度检查：当 `cos < 0.99` 时只打印 `"Expected below 0.99 due to CPU vs NPU precision differences"`，不计数为失败。这意味着精度回归会被静默忽略，测试永远绿色。

**怎么修复**

1. 确认 L2a 当前实际余弦值。如果实际值 ≥0.99，移除该静默逻辑，改为正常断言。
2. 如果实际值确实 <0.99，定位根因（是 CPU/NPU 差异还是代码 bug），修复后改为正常断言。
3. 如果差异确实不可消除（需提供证据），将阈值设为可达到的值，并在低于该值时明确报 `FAIL`。

**怎么验收**

- L2a 精度检查在低于阈值时输出 `FAIL` 而非 `INFO`。
- `ctest` 在精度不达标时返回非零退出码。

**如何避免再犯**

- 代码审查规范：任何 `cos < threshold` 分支如果不报 FAIL，必须有逐阶段精度分析文档链接和 TODO 追踪编号。
- CI 中禁止测试输出包含 `Expected below` 模式的日志行。

---

### P3: DecoderLayer MHA+causal mask 精度测试缺失 {#p3}

**问题是什么**

`atb_cpp_llm/docs/testing-guide.md:152-153` 明确记录："Currently no MHA + causal mask DecoderLayer precision test." 现有 `test_text_decoder_layer_precision.cpp` 只覆盖 GQA+mask 和 MHA+no-mask 两种组合，缺少 MHA+causal mask 场景。MHA（kv_head_num == head_num）是 910B 和部分模型配置的标准路径，缺少 mask 场景的测试意味着 causal mask 在 MHA decoder layer 中的正确性未被验证。

**怎么修复**

1. 在 `gen_cpu_reference.py` 的 `gen_op_decoder_layer` 中添加 `mha_causal` 测试用例（nh=32, kv_nh=32, hd=128, S=16, use_mask=true）。
2. 在 `test_text_decoder_layer_precision.cpp` 中添加对应的 C++ 测试用例，加载参考数据并验证 `cosine >= 0.99`。
3. 更新 `testing-guide.md` 移除已知缺口记录。

**怎么验收**

- `gen_all.py` 生成 `mha_causal` 参考数据成功。
- `test_text_decoder_layer_precision` 的 `mha_causal` 用例通过（cos ≥ 0.99）。

**如何避免再犯**

- `testing-guide.md` 中的测试覆盖矩阵作为 CI check 的一部分：每个"应覆盖"的组合必须在对应 test 文件中找到对应的 `TEST_CASE`。
- 新增模型配置时，检查覆盖矩阵是否需要扩展。

---

### P4: DeepSeek/Mixtral/Qwen3(non-VL) 零测试覆盖 {#p4}

**问题是什么**

C++ 测试套件全部针对 Qwen3VL-Embedding-2B。`atb_cpp_llm/src/` 中存在 `moe_mlp_graph.h`（MoE 架构）和多种模型适配器，但 DeepSeek-V2/V3、Mixtral、Qwen3（纯文本）没有任何测试。这些模型的 MoE 路由、expert 负载均衡、不同的 attention 变体均未验证。

**怎么修复**

分阶段推进：

1. **Phase 1（最小可行）**：为 Qwen3（纯文本，非 VL）添加 Level 2 算子精度测试和 Level 3 集成测试。Qwen3 与 Qwen3VL 共享 text decoder layer 结构，复用现有测试基础设施成本最低。
2. **Phase 2**：为 Mixtral 添加 MoE MLP 精度测试（验证 expert 路由和合并逻辑）。
3. **Phase 3**：为 DeepSeek-V2/V3 添加完整测试（MLA attention + MoE + 多 token 预测）。

**怎么验收**

- Phase 1：Qwen3 text model 完整推理与 Python 参考余弦 ≥ 0.99。
- Phase 2：Mixtral MoE layer 输出与 Python 参考余弦 ≥ 0.99。
- Phase 3：DeepSeek E2E 推理与 Python 参考余弦 ≥ 0.98。

**如何避免再犯**

- 新增模型适配器时，必须同步提交至少 Level 2 算子精度测试。
- 在 `design.md` 的 IModel 接口文档中标注"新增实现必须包含的测试清单"。
- 代码审查规范：`src/adapters/` 下新增目录时，检查 `tests/` 下是否有对应覆盖。

---

### P5: Deepstack 跨模态融合无自动化测试 {#p5}

**问题是什么**

Deepstack 是 Qwen3VL 的核心跨模态融合机制：vision block [5, 11, 17] 的输出通过 deepstack merger MLP 处理后注入到对应的早期 text decoder layer。CLAUDE.md 明确描述这是关键架构特性，但：

- `test_vision_model.py` 设置 `deepstack_visual_indexes=[]`（显式禁用）
- `test_text_model.py` 只用 2 层，不涉及 deepstack
- `test_pipeline_trace.py` 有手动 trace 但不做断言
- 无任何测试验证 `build_deepstack_merger` 图或 `run_merger_npu` 的 deepstack 路径

**怎么修复**

1. **Level 2 算子测试**：新增 `test_deepstack_merger_precision.py`，验证单个 deepstack merger MLP 的 ATB 输出与 TF 参考一致（cos ≥ 0.999）。
2. **Level 3 集成测试**：新增 deepstack 注入测试 — 用真实模型配置（depth=24, deepstack_indexes=[5,11,17]），验证 vision block 在 index 5/11/17 处的输出经过 merger 后与 Python 参考一致。
3. **Level 4 E2E 测试**：在 `test_e2e.py` 中增加 deepstack 专项验证 — 对比 ATB 和 TF 在 deepstack 注入前后的 text layer hidden states。

**怎么验收**

- `test_deepstack_merger_precision.py`：cos ≥ 0.999。
- Deepstack 注入集成测试：各 deepstack index 的输出 cos ≥ 0.99。
- E2E deepstack 专项：ATB vs TF text layer hidden states（deepstack 注入前后）cos ≥ 0.99。

**如何避免再犯**

- CLAUDE.md 中标注的"关键架构特性"必须在测试覆盖矩阵中有对应条目。
- 代码审查规范：新增架构特性时，检查是否同步提交了单元测试 + 集成测试 + E2E 覆盖。

---

### P6: test_text_model.py 只用 2 层测试 {#p6}

**问题是什么**

`atb_python_qwen3vl_embedding/tests/test_text_model.py:10` 中 `num_layers=2`。真实 Qwen3VL-Embedding-2B 有 28 层。2 层无法暴露 fp16 累积误差在深层模型中的放大效应。代码注释（line 63-64）承认"float16 accumulation through multi-layer"会降低余弦相似度，但阈值 0.99 仅针对 2 层——28 层时可能远低于此。

**怎么修复**

1. 新增 `test_text_model_deep` 测试用例，使用 `num_layers=28` 和真实模型维度（hidden=2048, nh=32, nkv=4, hd=128, intermediate=6144）。
2. 记录 28 层的实际余弦值，设定合理阈值（预计 0.95-0.98，取决于 fp16 累积）。
3. 保留 2 层测试作为快速 smoke test，新增 28 层测试作为精度验证。

**怎么验收**

- 28 层 text model ATB 输出与 TF 参考余弦 ≥ 0.95。
- 如果实际值 < 0.95，逐层分解定位精度损失最大的层，确认是 fp16 累积而非代码 bug。

**如何避免再犯**

- 单元测试中凡是涉及"多层循环"的测试，必须至少有一个用例使用真实层数。
- 测试文件顶部注释应说明测试维度与真实维度的对应关系。

---

### P7: BicubicResize 参考实现自证 {#p7}

**问题是什么**

`atb_cpp_llm/tests/python_reference/gen_cpu_reference.py:394-418` 中 `_cpp_bicubic_resize` 是手工编写的 Catmull-Rom edge-clamp 实现，用于验证 C++ BicubicResize 算子。代码注释明确承认这不是独立参考——它是 C++ 算法的 Python 翻版。如果 C++ 算法有 bug，Python"参考"会复现同样的 bug，测试永远通过。

**怎么修复**

1. 使用独立参考实现（如 PIL.Image.resize with BICUBIC、OpenCV resize with INTER_CUBIC）作为 ground truth。
2. 同时保留 `_cpp_bicubic_resize` 作为一致性检查（验证 C++ 和 Python 翻版行为一致），但将独立参考作为精度断言的主要依据。
3. 如果独立参考与 C++ 结果有差异（边界处理方式不同），在文档中明确记录差异点和可接受的误差范围。

**怎么验收**

- C++ BicubicResize vs PIL/OpenCV 参考：pixel-level max_diff ≤ 1（边界 ±1 像素偏差可接受）。
- C++ vs `_cpp_bicubic_resize`：bit-exact 或 near-bit-exact（验证 C++ 和 Python 翻版行为一致）。

**如何避免再犯**

- 测试编写规范：参考实现必须是独立来源（不同库、不同语言、或手工计算的理论值），不能是被测代码的翻译。
- 代码审查时，对任何"参考实现"追溯其来源，确认其独立性。

---

## 🟡 中优先级

### P8: test_text_attention.py 与 test_310p_diag.py 冗余 {#p8}

**问题是什么**

`test_text_attention.py` 只测试 `B=1, S=16, MHA (nh=4, kv_nh=4)`。`test_310p_diag.py` 的 T1（MHA+nomask）和 T2（MHA+mask）覆盖了相同的场景，且额外测试了 GQA、hd=128、S=4/16/880 等配置。`test_text_attention.py` 是 `test_310p_diag.py` 的严格子集。

**怎么修复**

1. 将 `test_text_attention.py` 中独有的功能合并到 `test_310p_diag.py`：使用 `compare_tensors`（打印 MSE 和 max_diff）替代 `test_310p_diag.py` 中的裸余弦比较。
2. 删除 `test_text_attention.py`。
3. 或者保留 `test_text_attention.py` 作为最简示例/教程，但添加注释说明完整测试在 `test_310p_diag.py`。

**怎么验收**

- 删除后所有 attention 相关测试场景仍被覆盖。
- `test_310p_diag.py` 输出包含 MSE 和 max_diff 信息。

**如何避免再犯**

- 新增测试文件时，先搜索现有测试是否已覆盖相同或更广的场景。
- 测试文件头部注释应列出"相关测试文件"和"本文件与它们的区别"。

---

### P9: test_nz_quick_verify.py 与 test_nz_format_verify.py 冗余 {#p9}

**问题是什么**

`test_nz_quick_verify.py`（创建 + 格式检查）的所有检查都被 `test_nz_format_verify.py`（创建 + 格式检查 + data copy + value verification + format cast + ATB acceptance）覆盖。前者是后者的严格子集。

**怎么修复**

删除 `test_nz_quick_verify.py`。

**怎么验收**

- `test_nz_format_verify.py` 全部 6 个 test function 通过。

**如何避免再犯**

- 同 P8。

---

### P10: test_embedder_e2e.py 与 test_e2e.py + benchmark.py 重叠 {#p10}

**问题是什么**

`test_embedder_e2e.py` 的 quick test（3 个 case：text/image/mixed）重复了 `test_e2e.py` 的测试矩阵。benchmark mode 重复了 `benchmark.py` 的分辨率/seqlen 列表和模型加载逻辑。`ATBRunner`/`TFRunner` 类与 `benchmark.py` 中的 benchmark 代码高度重复。

**怎么修复**

1. 将 quick test 的 3 个 case 合并到 `test_e2e.py`（或以 `test_e2e.py` 为准，删除 `test_embedder_e2e.py` 的 quick test）。
2. 将 benchmark mode 合并到 `benchmark.py`。
3. 抽取共享的 `ATBRunner`/`TFRunner` 到 `data_utils.py` 或新建 `test_utils.py`。
4. `test_embedder_e2e.py` 如保留，应仅保留其独特功能：pooled+normalized embedding 的端到端对比。

**怎么验收**

- 合并后所有原有测试场景仍被覆盖。
- `test_embedder_e2e.py` 代码量减少 ≥50%。

**如何避免再犯**

- 同 P8。此外，E2E 测试文件必须明确声明"本文件与其他 E2E 测试的区别"。

---

### P11: test_consistency.cpp 是 test_accuracy.cpp 的子集 {#p11}

**问题是什么**

`test_consistency.cpp` 的 TEXT_ONLY 推理 + 保存 pooler 输出的逻辑是 `test_accuracy.cpp` TEXT_ONLY 路径的子集。前者是后者的早期简化版本。

**怎么修复**

1. 确认 `test_consistency.cpp` 无独有功能后删除。
2. 如果有独有功能（如多次推理一致性检查），提取到 `test_accuracy.cpp`。

**怎么验收**

- 删除后 `test_accuracy.cpp` 覆盖所有 consistency 场景。

**如何避免再犯**

- 新增 C++ 测试文件时，检查是否已有文件覆盖相同路径。在文件头部注释中声明与现有测试的关系。

---

### P12: 预处理阶段在多个文件中重复测试 {#p12}

**问题是什么**

预处理（preprocess_image）的验证同时出现在：
- `test_stage_precision.cpp`（C++）
- `test_vision_stages.cpp`（C++）
- `test_preprocess.py`（Python）
- `test_vision_diagnostics.py::test_preprocess_match`（Python）

四个文件测试同一个功能，代码和参考数据重复。

**怎么修复**

1. **C++ 侧**：将预处理精度验证集中到 `test_vision_stages.cpp`（已有 Stage L0），`test_stage_precision.cpp` 引用该 stage 的结果或直接移除重复检查。
2. **Python 侧**：将 `test_preprocess.py` 扩展为覆盖多个分辨率和边界情况，`test_vision_diagnostics.py::test_preprocess_match` 改为调用 `test_preprocess.py` 的共享函数。
3. 抽取共享的预处理测试数据生成到 `gen_cpu_reference.py`。

**怎么验收**

- 预处理测试逻辑不重复（每处检查不同的维度或使用共享的验证函数）。
- 测试覆盖的预处理场景不减少。

**如何避免再犯**

- 测试代码也遵循 DRY 原则。发现重复时立即重构为共享函数。

---

### P13: set_atb_buffer_size 调用位置不一致 {#p13}

**问题是什么**

7 个文件在模块级调用 `set_atb_buffer_size`，9 个文件在测试函数内调用。CLAUDE.md 警告"must be called exactly once before any ATB graph build"，不一致的调用位置可能导致：
- 模块级调用在 import 时触发，可能在其他模块的 graph build 之前或之后
- 测试函数内调用如果与其他测试组合运行，可能重复调用

**怎么修复**

1. 统一为**模块级调用**（在 `import` 之后、任何函数定义之前），这是最安全的模式——确保在任何 ATB graph build 之前调用且只调用一次。
2. 添加防护：在 `set_atb_buffer_size` 内部增加 `_buffer_size_set = True` 标记，重复调用时打印警告（不崩溃，因为某些场景可能需要调整）。
3. 在所有测试文件顶部添加注释说明 buffer size 的选择依据。

**怎么验收**

- 所有测试文件使用统一的 `set_atb_buffer_size` 调用模式。
- 连续运行多个测试文件不会因 buffer size 问题崩溃。

**如何避免再犯**

- 创建测试文件模板（`tests/template.py`），包含正确的 import 顺序和 `set_atb_buffer_size` 调用位置。
- CLAUDE.md 中补充测试文件的标准结构说明。

---

### P14: 模型加载代码在 6 个文件中重复 {#p14}

**问题是什么**

`load_tf_ref`（加载 Qwen3VLModel from safetensors）模式出现在：
- `test_e2e.py`
- `benchmark.py`
- `test_embedder_e2e.py`
- `test_pipeline_trace.py`
- `test_text_diagnostics.py`
- `test_vision_diagnostics.py`

每份实现有细微差异：有的 check missing/unexpected keys，有的不 check；有的用 float32，有的用 half。

**怎么修复**

1. 在 `data_utils.py` 中添加统一的 `load_tf_reference(model_dir, precision="float32")` 函数。
2. 所有文件改为调用该共享函数。
3. 统一行为：默认 check missing/unexpected keys，默认 float32。

**怎么验收**

- `grep -r "load_tf_ref\|from_pretrained\|load_state_dict" tests/` 只在 `data_utils.py` 中有实现。
- 所有测试文件的 TF 加载行为一致。

**如何避免再犯**

- 代码审查规范：同一模式出现 ≥3 次时，必须抽取为共享函数。

---

### P15: 单元测试使用非真实模型维度 {#p15}

**问题是什么**

| 测试 | 测试维度 | 真实维度 | 比例 |
|------|----------|----------|------|
| test_text_attention | nh=4, hd=64 | nh=32, hd=128 | 1/8 |
| test_text_mlp | intermediate=512 | intermediate=6144 | 1/12 |
| test_vision_attention | hs=128, nh=4 | hs=1280, nh=16 | 1/10 |
| test_vision_mlp | hs=128, interm=256 | hs=1280, interm=5120 | 1/10~1/20 |
| test_vision_model | patch_size=4 | patch_size=16 | 1/4 |

小维度适合快速单元测试，但无法暴露内存对齐、matmul tiling、大矩阵 fp16 精度等问题。

**怎么修复**

1. 保留现有小维度测试作为快速 smoke test（标记为 `_small`）。
2. 为每个单元测试新增一个 `_full_dim` 用例，使用真实模型维度。
3. 真实维度测试可标记为慢速测试（如 `pytest.mark.slow`），日常开发跑小维度，CI/PR 跑全维度。

**怎么验收**

- 每个测试文件至少有 1 个真实维度用例。
- 真实维度用例的余弦阈值不低于小维度用例。

**如何避免再犯**

- 测试编写规范：每个测试文件必须包含至少一个真实维度用例。
- 新增测试 review 时检查维度是否与真实模型匹配。

---

### P16: 无自动化测试运行器 / 汇总脚本 {#p16}

**问题是什么**

所有 Python 测试是独立脚本（`if __name__ == "__main__"`），没有 `conftest.py`、`pytest.ini` 或汇总脚本。无法一键运行全部测试并获取汇总报告。

**怎么修复**

1. 创建 `tests/run_all.py` 汇总脚本，依次运行所有测试文件，收集退出码和输出，打印汇总表格。
2. （可选）添加 `conftest.py` 使用 pytest，将每个测试文件的 `__main__` 逻辑包装为 `test_*` 函数。
3. C++ 侧已有 `build_and_test.sh` 和 CTest，确保 Python 侧也有等价物。

**怎么验收**

- `python tests/run_all.py` 运行所有 Python 测试并输出通过/失败汇总。
- 退出码反映整体结果（全部通过 = 0，任一失败 = 非 0）。

**如何避免再犯**

- CI 中调用 `run_all.py`（Python）和 `build_and_test.sh`（C++），确保每次提交都跑全量测试。

---

### P17: test_config_wiring.cpp 硬编码 epsilon 阈值 {#p17}

**问题是什么**

`test_config_wiring.cpp:119` 检查 `cfg.vis_epsilon < 1e-4`、`cfg.text_rms_norm_eps < 1e-4`、`cfg.normalize == true` 等。这些值来自模型 config.json，如果模型配置变更（如不同的 epsilon），测试会误报失败。

**怎么修复**

1. 改为从 Python 参考数据中读取期望值（在 `gen_cpu_reference.py` 中导出 config 字段到 meta `.bin` 文件）。
2. C++ 测试加载期望值并与解析结果对比，而非硬编码。

**怎么验收**

- 更换模型配置后，`test_config_wiring` 不会误报失败（只要 C++ 解析正确）。

**如何避免再犯**

- 任何与外部数据源（config.json, safetensors）相关的断言，期望值必须来自参考数据生成器，不得硬编码。

---

## 🟢 低优先级

### P18: IMAGE_TOKEN_ID 硬编码 {#p18}

**问题是什么**

`test_embedder_e2e.py:68` 硬编码 `IMAGE_TOKEN_ID = 151655`，而 engine 从 config 读取 `cfg["image_token_id"]`。模型 token ID 变更时测试会静默产生错误结果。

**怎么修复**

从 engine 或 config 中读取 `image_token_id`，删除硬编码常量。

**怎么验收**

- `grep -r "151655\|151652" tests/` 无结果（或仅在参考数据/注释中）。
- 测试使用与 engine 相同的 token ID 来源。

**如何避免再犯**

- 编码规范：token ID、模型维度、epsilon 等模型相关常量必须从 config 读取，不得硬编码。

---

### P19: 死代码清理 {#p19}

**问题是什么**

- `test_embedder_e2e.py:64`：`VISION_START_TOKEN_ID = 151652` 定义但从未使用。
- `test_embedder_e2e.py:378-425`：`run_quick_tf` 接受 `n_warmup`/`n_iter` 参数但从未使用。

**怎么修复**

删除未使用的常量和无用参数。

**怎么验收**

- Python linter（如 ruff/flake8）不再报告未使用的变量。

**如何避免再犯**

- CI 中集成 Python linter，检测未使用变量/导入。

---

### P20: 测试间余弦阈值不一致且缺乏文档 {#p20}

**问题是什么**

| 测试类型 | 阈值 | 依据 |
|----------|------|------|
| 单层单元测试 | 0.999 | fp16 单次运算 |
| 2 层 text model | 0.99 | fp16 多层累积 |
| 28 层 E2E | 0.99 | 可配置 |
| IMAGE_ONLY | 0.90 | 可疑 |

不同测试的阈值选择缺乏统一标准和文档说明。

**怎么修复**

1. 在 `data_utils.py` 或测试文档中定义阈值标准：
   - 单层算子（fp16）：0.999
   - 2-5 层组合：0.99
   - 6+ 层或完整模型：0.98
   - 跨框架（ATB vs TF）：0.97（考虑实现差异）
2. 每个测试文件的阈值引用该标准，偏差需注释说明原因。
3. 在测试输出中打印阈值来源（如 `threshold=0.99 (standard: multi-layer fp16)`）。

**怎么验收**

- 所有测试文件的阈值可追溯到统一标准文档。
- 无注释的阈值偏差 ≤ 0.01。

**如何避免再犯**

- 新增测试 review 时检查阈值是否符合标准，偏差是否已文档化。

---

### P21: benchmark.py 手动复制 engine._run_vision 逻辑 {#p21}

**问题是什么**

`benchmark.py:220-252` 中 `_run_vision_stages` 手动复制了 `engine._run_vision` 的内部逻辑以实现分阶段计时。注释说 "mirrors engine.py" 但无自动检查确保同步。engine 内部变更时 benchmark 会悄然漂移。

**怎么修复**

1. 在 `engine.py` 中暴露分阶段接口（如 `_run_vision_stage1()`, `_run_vision_stage2()` 等），让 benchmark 调用公开 API 而非复制内部逻辑。
2. 或为 `_run_vision` 添加 `return_intermediates=True` 参数，返回各阶段输出和时间戳。

**怎么验收**

- `benchmark.py` 不再包含 `engine._run_vision` 的复制代码。
- benchmark 的分阶段计时结果与原来一致（± 测量误差）。

**如何避免再犯**

- 编码规范：benchmark 代码应调用公开 API，不得复制被测代码的内部实现。

---

## 执行顺序建议

| 阶段 | 问题 | 预计工作量 | 理由 |
|------|------|-----------|------|
| **第 1 批** | P1, P2, P3, P6 | 3-5 天 | 可能隐藏精度 bug，必须优先修复 |
| **第 2 批** | P5, P7, P8, P9, P11, P19 | 3-4 天 | 补关键覆盖 + 清理冗余 |
| **第 3 批** | P13, P14, P16, P18 | 2-3 天 | 统一基础设施 |
| **第 4 批** | P4, P15, P17 | 5-10 天 | 扩展覆盖，工作量较大 |
| **第 5 批** | P10, P12, P20, P21 | 3-5 天 | 合并重叠 + 标准化 |

---

## 附录：审计方法

- **Python 审计**：逐文件阅读 `atb_python_qwen3vl_embedding/tests/` 下全部 22 个文件 + `data_utils.py` + `transformers_runner.py`
- **C++ 审计**：逐文件阅读 `atb_cpp_llm/tests/` 下全部 79 个文件 + `CMakeLists.txt` + `build_and_test.sh` + Python 参考生成器
- **评估维度**：正确性（阈值是否合理）、覆盖完整性（缺失场景）、冗余性（重复测试）、可信度（参考实现是否独立）、架构对齐（是否匹配设计文档）
- **审计日期**：2026-06-15
