# Python → C++ 迁移项目：增量测试策略指南

## 核心原则

**永不通过降低阈值来“通过”测试。** 如果 C++ 和 Python 在相同输入下余弦相似度低于 0.99，说明存在 bug，必须定位并修复根因。

参考 [refactoring-plan.md](./refactoring-plan.md) 中已修复的 Bug —— 每一个 Bug 都对应着一个本应更早被发现的测试缺口。

---

## 一、测试层次设计

理想的测试金字塔应如下（下层越厚越好）：

```
          ┌──────────────┐
          │   E2E        │  3 个模式 × 2 套验证 = 6 个 case
          │   (Level 4)  │  ATB vs Python 最终输出
          ├──────────────┤
          │   集成       │  每个子模型 (Vision/Text) 1-2 个 case
          │   (Level 3)  │  多组件组合，验证数据流正确
          ├──────────────┤
          │   算子精度    │  每个 Graph 组件 2-4 个 case
          │   (Level 2)  │  相同 input+weights → ATB vs NumPy/PyTorch
          ├──────────────┤
          │  CPU 纯函数  │  每个纯函数 ≥ 3 个 case
          │   (Level 1)  │  相同输入 → 逐元素精确匹配/byte-exact
          ├──────────────┤
          │   基础框架   │  框架正确性，不依赖模型
          │   (Level 0)  │  内存分配、JSON 解析、RAII 安全
          └──────────────┘
```

### Level 0: 基础框架测试 (CPU, 无 NPU)

- **目的**: 确保基础设施可用
- **覆盖**: ContextManager 创建, TensorAllocator 分配/释放, GraphBuilder 构建, BufferPool 扩容, JsonConfig 解析, NpuTensor RAII 安全
- **验证**: Doctest 断言, 不需要精度对比
- **当前状态**: ✅ `test_core.cpp`, `test_io_adapters.cpp` 已有

### Level 1: CPU 纯函数测试 (最重要的缺失层)

- **目的**: 验证所有不依赖 NPU 的计算逻辑，在 CI 上即可运行
- **覆盖**: 位置编码、RoPE 频率计算、图像尺寸 resize、mask 生成、token 索引计算
- **验证**: 相同输入 → 逐元素精确匹配 (Python `get_rope_index` ↔ C++ `GetRopeIndex`)
- **当前状态**: ❌ 几乎完全缺失
- **速度**: 毫秒级，可在 CI 上运行

#### Level 1 必须测试的函数清单

| 函数 | C++ 位置 | Python 参考 | 优先级 |
|---|---|---|---|
| `GetRopeIndex` | `components/common/mrope.cpp` | `engine_utils.py:get_rope_index` | 🔴 最高 |
| `MRoPE::Compute` | `components/common/mrope.cpp` | `rotary_emb(pid)` | 🔴 最高 |
| `MRoPE::ApplyInterleaved` | `components/common/mrope.cpp` | MRoPE 内部逻辑 | 🟡 |
| `VisionRotaryEmbedding::ComputeFreqTable` | `components/common/mrope.cpp` | `Qwen3VLVisionRotaryEmbedding(mh)` | 🔴 最高 |
| `VisionRotaryEmbedding::ComputeRoPE` | `components/common/mrope.cpp` | `vm.rot_pos_emb(gth)` | 🔴 最高 |
| `ComputeVisionRotPosEmb` | `components/common/mrope.cpp` | 视觉 RoPE 计算 | 🔴 最高 |
| `ComputePosEmbedInterp` | `components/vision/pos_embed_interp.cpp` | `fast_pos_embed_interpolate()` | 🔴 最高 |
| `MakeCausalMask` | `runners/text_runner.cpp` | `text_model.py:make_causal_mask` | 🟡 |
| `SmartResize` | `adapters/.../qwen3vl_preprocess.cpp` | `preprocess.py` | 🔴 最高 |

#### Level 1 测试的侵入式策略

测试纯 CPU 函数时，不要只对比最终模型的输出。要**在相同的已知输入下，分别调用 Python 和 C++ 的对应函数，逐元素比较输出**。

**示例: GetRopeIndex 测试**

```cpp
// test_mrope_cpu.cpp — 不依赖 NPU
TEST_CASE("GetRopeIndex against Python reference") {
    // 1. 准备输入 (与 Python 完全一致)
    int64_t input_ids[] = {151652, 151655, 151655, 151655, 151655};  // vision_start + 4 image tokens
    int64_t grid_thw[] = {1, 2, 2};                                   // 1×2×2 grid
    int64_t position_ids[3 * 1 * 5];                                  // (3, B=1, S=5)

    // 2. 调用 C++ 函数
    GetRopeIndex(input_ids, 1, 5, grid_thw, 1,
                 151655, 151652, 2, position_ids);

    // 3. 与 Python 参考值比较 (预计算的已知正确值)
    // Python: get_rope_index(...) → 3D position_ids
    int64_t expected[15] = {
        /* T: */ 0, 0,1,2, 0,1,2,3, 3,
        /* H: */ 0, 0,0,0, 1,1,1,1, 3,
        /* W: */ 0, 0,1,0, 01,0,1,0,1,2,3, 3
    };
    // 逐元素比较
    for (int i = 0; i < 15; i++) {
        CHECK(position_ids[i] == expected[i]);
    }
}
```

关键: 不要用模型推理的最终结果来验证。用**独立计算的小输入**来验证。

### Level 2: 算子精度测试 (NPU 依赖)

- **目的**: 验证每个 ATB graph 组件的数值精度
- **覆盖**: RmsNorm, Linear, SwiGLU, SelfAttention, VisionAttention, VisionMLP, LayerNorm, PatchEmbed
- **验证**: 相同随机输入 + 相同权重 → ATB vs PyTorch/NumPy cosine ≥ 0.999
- **当前状态**: ✅ Python 侧很完善, ❌ C++ 侧完全缺失

#### C++ 侧缺失的算子精度测试

每个算子需要 2-4 个 case:
- **正常尺寸**: 模拟实际模型的维度
- **边界尺寸**: seq_len=1, 最小 hidden_size
- **不同精度组合**: fp16 vs fp32

```cpp
// test_rms_norm_precision.cpp 示例
TEST_CASE("RmsNorm precision vs PyTorch") {
    // 1. 随机种子 42, 生成 hidden_states [4, 64] fp32
    // 2. 权重全 1.0 (或随机但 C++/Python 一致)
    // 3. C++ RmsNormGraph → 输出 fp16 → 转 fp32
    // 4. Python F.rms_norm(x, [64], weight, eps=1e-6) → fp32
    // 5. cosine ≥ 0.999
}
```

### Level 3: 集成测试

- **目的**: 验证多个组件组合后的数据流正确
- **覆盖**: VisionModel (first_layer → blocks → merger), TextModel (多层 loop + norm), Deepstack 注入
- **验证**: 固定输入 → ATB vs TF cosine ≥ 0.99
- **当前状态**: ✅ Python 侧有, 🟡 C++ 侧只有 build 验证

### Level 4: E2E 测试

- **目的**: 验证完整推理管线的最终输出
- **覆盖**: TEXT_ONLY, IMAGE_ONLY, IMAGE_AND_TEXT 三种模式
- **验证**: ATB vs Python/TF 最终 embedding cosine ≥ 0.99
- **当前状态**: ✅ 两侧都有

---

## 一·五、本地开发循环：只跑你关心的测试

跑全量 `build_and_test.sh` 一次大概 15–20 分钟，迭代单个失败用例时太慢。脚本提供了三类"快速复跑"开关，**建议日常 debug 全部走这条路径**，只在合并前跑一次全量。

### 1. `--test-only`：跳过 build，复用现有产物

```bash
bash atb_cpp_llm/build_and_test.sh --test-only
```

前提：`build/` 目录已经存在且至少全量构建过一次（否则脚本会清晰报错让你先 `bash build_and_test.sh --no-test`）。开启 `--test-only` 后跳过 CMake configure 和 `cmake --build`，只跑 ctest，省下绝大部分时间。

### 2. 按 level 过滤（位置参数）

把要跑的 level 目录名当位置参数传进去，可以一次给多个：

```bash
# 只跑 Level 1（CPU 纯函数，秒级反馈）
bash atb_cpp_llm/build_and_test.sh --test-only level1_cpu_pure

# 跑 Level 1 + Level 3
bash atb_cpp_llm/build_and_test.sh --test-only level1_cpu_pure level3_integration
```

可用 level 名（来自 `tests/` 子目录）：

- `level0_framework`
- `level1_cpu_pure`
- `level2_op_precision`
- `level3_integration`
- `level4_e2e`

### 3. 按测试名过滤（位置参数）

直接给单/多个测试可执行文件名，**精确匹配**（已自动加 `^...$` 锚定）：

```bash
# 只跑一个测试，反复改 → 反复跑，最快的 debug 循环
bash atb_cpp_llm/build_and_test.sh --test-only test_vision_stages

# 跑两个完全不相关的测试
bash atb_cpp_llm/build_and_test.sh --test-only test_bin_format test_text_model
```

### 4. 同时给 level 和测试名（交集）

ctest 的 `-L` 和 `-R` 默认是 **AND** 关系——测试必须既属于指定 level，又匹配指定名字才会跑。脚本会在执行前提示这一点：

```bash
bash atb_cpp_llm/build_and_test.sh --test-only level3_integration test_text_model
# → 只有同时是 level3 且名字叫 test_text_model 的才会跑（通常就是 1 个）
```

如果你想要"跑 level3 + 也跑 test_xxx"这种并集，分两次运行即可。

### 5. 怎么自动识别 level 还是测试名？

脚本里有一个固定的 `KNOWN_LEVELS` 数组（5 个 level 名）。位置参数命中数组 → 当 level（`ctest -L`）；否则当测试名（`ctest -R`）。**结论**：永远不要把测试取成跟 level 同名，否则会被脚本当过滤器误判。加新 `tests/levelN_xxx/` 目录时记得同步更新这个数组（详见 [§ 八 #1](#八硬编码契约清单加东西前先扫一眼)）。

### 6. `--list`：看有哪些测试 + 它们的 label

```bash
bash atb_cpp_llm/build_and_test.sh --list
```

先列出所有 label，再列出全部测试名，不执行任何用例。

### 7. `-v` / `--verbose`：失败时看完整 stdout

```bash
bash atb_cpp_llm/build_and_test.sh --test-only -v test_vision_stages
# 等价于 ctest -V，把测试可执行文件的全部输出打到终端
```

### 8. 参考数据 (refdata) 的三种模式

`/tmp/{cpu_,stage_,posembed_npu_,visrope_npu_}*.bin` 这些参考数据是 27 个 Level 1/2/3/4 精度测试比较 C++ 输出和 Python 参考实现的基础。脚本对它有三种处理方式：

| 模式 | 行为 | 用法 |
|---|---|---|
| 默认（refresh） | 每次都重新生成全部 refdata (~90 秒) | `bash build_and_test.sh` |
| reuse | 复用 /tmp/ 已有数据；缺失则自动 fallback | `bash build_and_test.sh --no-refresh-refdata` |
| none | 不生成 + 主动排除 27 个依赖测试 | `bash build_and_test.sh --no-refdata` |

**为什么默认每次刷新？** 改了 Python 参考代码、换了模型权重、清过 /tmp 都会让数据失效。"默认刷新" 保证你看到的测试结果反映当前代码状态，而不是历史快照。代价是 ~90 秒生成时间——`--no-refresh-refdata` 给你一个明确的 opt-out。

**为什么 `--no-refdata` 要主动排除测试？** 否则那 27 个测试会因为读不到 `.bin` 走 `LOG_ERROR("SKIP...") + return` 静默"通过"（exit code 0），从外面看是 PASS，实际啥都没测——比明确报错还隐蔽。脚本通过 `ctest -LE "needs_refdata"` 主动把它们排除掉，并打印被排除的测试清单，让"跳过"这件事可见。

**自动 fallback 触发条件**：`--no-refresh-refdata` 模式下脚本会扫所有哨兵文件（共 8 个，覆盖 5 个生成器；部分生成器登记多个代表性输出），任一缺失或为空就 fallback 到 `--no-refdata` 行为（排除 27 测试 + WARN）。同事的机器、新克隆的仓库、清过 /tmp 的环境都会自动走这条路径，不会"假装在测试"。

**边界用例**：

- `--no-refdata test_vision_stages` → 脚本 WARN："test_vision_stages requires reference data and will be excluded"，然后 ctest 跑 0 测试。要测这个就别带 `--no-refdata`。
- `--no-refdata level1_cpu_pure` → level1 5 个测试全部 needs_refdata，ctest 也跑 0 测试，输出会清楚。
- 同时给 `--refresh-refdata` 和 `--no-refdata` → exit 2，refdata flag 冲突。

**维护契约**：新加的测试如果读 `/tmp/` 里的 `.bin`，**必须**在 `CMakeLists.txt` 的 `REFDATA_DEPENDENT_TESTS` 集合里登记。否则 `--no-refdata` 模式不会排除它，依然会走 SKIP 路径骗 PASS。发现方式：

```bash
grep -lE '/tmp/(cpu_|stage_|posembed_npu_|visrope_npu_)' tests/level*/*.cpp
```

把命中结果跟 `REFDATA_DEPENDENT_TESTS` 列表对一下，缺谁补谁。同类需要手动维护的契约还有 4 处，统一在 [§ 八](#八硬编码契约清单加东西前先扫一眼) 列出。

### 命令速查表

| 场景 | 命令 |
|---|---|
| 全量 build + 全量 test（CI 标准） | `bash build_and_test.sh` |
| 只 build，不 test | `bash build_and_test.sh --no-test` |
| 只 test，不 build | `bash build_and_test.sh --test-only` |
| 只 test，复用 /tmp 现有 refdata | `bash build_and_test.sh --test-only --no-refresh-refdata` |
| 跑不依赖 refdata 的测试（~23 个） | `bash build_and_test.sh --test-only --no-refdata` |
| 只跑 Level 1 | `bash build_and_test.sh --test-only --no-refresh-refdata level1_cpu_pure` |
| 只跑 1 个测试 | `bash build_and_test.sh --test-only --no-refresh-refdata test_vision_stages` |
| 只跑 1 个测试 + 看完整输出 | `bash build_and_test.sh --test-only --no-refresh-refdata -v test_vision_stages` |
| 列出所有测试 | `bash build_and_test.sh --list` |

---

## 二、增量测试策略：跟着代码走

以下是在迁移 Python → C++ 的每个阶段应该同步编写的测试：

### 阶段 1: 基础框架 (Phase 0-2)

**写什么代码**: ContextManager, TensorAllocator, GraphBuilder, JsonConfig
**同步写什么测试**:
- `test_core.cpp` — 每个类的创建、基本操作、错误路径
- 纯 CPU, 不依赖模型

### 阶段 2: 配置和数据加载 (Phase 3-4)

**写什么代码**: Qwen3VLConfig, SafetensorsReader, WeightLoader
**同步写什么测试**:
- `test_io_adapters.cpp` — JSON 解析正确性、safetensors 读取、weight 形状验证
- 先创建最小的 safetensors 文件作为 test fixture

### 阶段 3: CPU 纯计算函数 (关键阶段！)

**写什么代码**: GetRopeIndex, MRoPE, VisionRotaryEmbedding, ComputePosEmbedInterp, SmartResize
**同步写什么测试**:
- **每个函数至少 3 个 case**: 正常尺寸、边界条件、多图像/多 batch
- 用 Python 生成 reference 数据（byte-exact 的 int64 数组或 float 数组）
- 这些测试**不依赖 NPU**，应该在 CI 上毫秒级完成

**为什么这一层至关重要**:
- GetRopeIndex 是一个 ~100 行的纯逻辑函数，但包含了 image_token 识别、2D grid 位置生成、连续位置计数等复杂逻辑
- 如果在这个阶段就写好了 Python ↔ C++ 逐元素对比测试
- GetRopeIndex 的 bug（是我们项目中最严重的精度问题，IMAGE_ONLY cosine 仅 0.844）会在**写好函数后 5 分钟内**就被发现和修复
- 而不是在整个模型 E2E 跑通后才从 0.844 的 cosine 反向定位

### 阶段 4: 基础算子 (Phase 5-7)

**写什么代码**: RmsNormGraph, LinearOp, ElewiseOp, SplitOp
**同步写什么测试**:
- `test_rms_norm_precision.cpp` — 与 PyTorch F.rms_norm 对比
- `test_linear_precision.cpp` — 与 PyTorch F.linear 对比
- **每个算子 2-4 个 case**: 不同维度、不同精度

### 阶段 5: 子图组件 (Phase 8-11)

**写什么代码**: SwiGluMlpGraph, SelfAttentionGraph, VisionAttentionGraph, VisionMLPGraph, VisionBlockGraph
**同步写什么测试**:
- 每个 component graph 至少 1 个精度测试: 随机输入 + 随机权重 → ATB vs TF/numpy
- VisionBlock 包含 RoPE → 验证 RoPE 集成正确

### 阶段 6: 子模型 Runner (Phase 12-14)

**写什么代码**: VisionRunner, TextRunner, DeepstackFusion
**同步写什么测试**:
- VisionRunner: 相同 pixel_values + weights → ATB vs TF cosine ≥ 0.99
- TextRunner: 相同 inputs_embeds + position_ids → ATB vs TF cosine ≥ 0.99
- Deepstack: 验证 feature 注入正确性（位置、值）

### 阶段 7: 适配器和引擎 (Phase 15-16)

**写什么代码**: Qwen3VLModel, LLMEngine, Preprocess
**同步写什么测试**:
- 三模式 E2E 测试
- 错误路径测试 (null input, batch_size≠1, seq_len=0)

---

## 三、测试辅助工具建议

### 1. Python Reference Generator（应该统一为一个）

当前 C++ 项目有 5+ 个不同的 Python reference 生成脚本 (`gen_python_reference.py`, `gen_stage_reference.py`, `test_first_layer_ref.py`, `test_vision_block_ref.py`, `test_stage_reference.py`)，应该统一为一个 `gen_reference.py`，支持：

```bash
python tests/gen_reference.py --stage get_rope_index   # 生成 position_ids
python tests/gen_reference.py --stage mrope            # 生成 cos/sin
python tests/gen_reference.py --stage vision_rope      # 生成 vision cos/sin
python tests/gen_reference.py --stage all              # 生成所有 stage
```

### 2. 统一测试输入

建议建立一个 `test_inputs/` 目录：

```
test_inputs/
  images/
    gradient_720x1280.bin       # 梯度图 raw bytes
    solid_red_64x64.bin          # 纯色图
    solid_blue_120x200.bin       # 纯色图
  tokens/
    text_desc.json               # "Describe the image." token IDs
    image_only_880tokens.json    # IMAGE_ONLY token IDs
  references/
    position_ids_1x2x2.bin       # GetRopeIndex 期望输出
    mrope_cos_sin_3tokens.bin    # MRoPE 期望输出
    ...
```

### 3. CI 流水线分层

```
Layer 0 (CPU only, 秒级):  test_core, test_io_adapters, test_*_cpu (Level 1 pure functions)
Layer 1 (NPU, 分钟级):     test_*_precision (Level 2 op precision + Level 3 integration)
Layer 2 (NPU, 10 分钟级):  test_accuracy, test_e2e (Level 4 E2E)
```

---

## 四、从已有 Bug 反推缺失的测试

### Bug #1: GetRopeIndex 位置编码错误

- **影响**: IMAGE_ONLY cosine 0.844, IMAGE_AND_TEXT cosine 0.987
- **根因**: C++ 直接识别 image_token_id 生成 2D grid 位置，绕过了 vision_start_token_id 检查
- **缺失的测试**: Level 1 — `test_mrope_cpu.cpp` 中缺少 `GetRopeIndex` 的纯 CPU 测试
- **应该如何测**: 构造 `[vision_start_token_id, image_token_id, ...]` 输入，与 Python `get_rope_index` 逐元素对比

### Bug #2: LayerNorm epsilon 和 axis 参数

- **影响**: 视觉模型精度偏差
- **根因**: `beginNormAxis` 和 `beginParamsAxis` 默认为 `-1` 而 Python 用的 `1`
- **缺失的测试**: Level 2 — `test_layer_norm_precision.cpp`，相同输入 + 权重 vs PyTorch `F.layer_norm`
- **应该如何测**: 构造 `[seq_len, hidden_size]` 输入，epsilon=1e-6，与 PyTorch 对比

### Bug #3: SmartResize banker's rounding

- **影响**: 预处理输出不一致
- **根因**: C++ 用标准四舍五入，Python 用 banker's rounding
- **缺失的测试**: Level 1 — `test_smart_resize.cpp` 中对特定尺寸（如 720→704）的精确验证
- **应该如何测**: 输入 (720, 1280)，验证 resize 结果与 Python 的 `smart_resize` 完全一致

---

## 五、快速检查清单

在新模块迁移到 C++ 时，问自己：

- [ ] 这个模块有没有纯 CPU 的计算逻辑？→ 写 Level 1 测试，与 Python 逐元素对比
- [ ] 这个 graph/算子可以用随机输入+随机权重独立跑吗？→ 写 Level 2 精度测试
- [ ] 这个模块的边界条件是什么？（seq_len=1, batch_size>1, 空输入）→ 每种边界一个 case
- [ ] 这个模块的输入输出是否与 Python 参考实现有 1:1 的对应关系？→ 用相同的 numpy seed 生成输入
- [ ] 之前修复过的类似 bug 在这里可能重复出现吗？→ 参考 Bug Fixes 列表

---

## 六、与 Python 项目自身测试的关系

Python 项目的测试（`atb_python_qwen3vl_embedding/tests/`）提供了**参考实现**：
- Python 测试的 `data_utils.generate_base_data()` 生成随机输入 → 可用于 C++ 测试
- Python 测试的 `transformers_runner.py` 提供了 TF reference → 可用于 C++ 测试的参考值
- Python 诊断测试 (`test_*_diagnostics.py`) 的模式可以直接搬用到 C++

**建议**: 对于 C++ 项目的每个新模块，在 Python 项目中先写一个对应的 reference 生成函数，输出 binary 文件。C++ 测试读取这个 binary 文件进行精度对比。这样两端使用**完全相同的输入和参考值**。

---

## 七、给加测试的人：label 是自动推出来的

新加测试时**不需要手动配 LABELS**——只要把 `.cpp` 放到对应的 `tests/levelN_xxx/` 目录下，然后在 `CMakeLists.txt` 里照样调一次 `add_atb_test(test_xxx tests/levelN_xxx/test_xxx.cpp)`，CMake 的 `add_atb_test()` 函数会从源文件路径里抓最后一段目录名当作 label（前提是匹配 `^level[0-9]` 前缀）。这样 `build_and_test.sh levelN_xxx` 就能自动把新测试包含进来。

**反例 / 易踩坑**：

- 把 `.cpp` 放到 `tests/` 根目录或某个非 `levelN_*` 子目录 → 不会有 label，`-L` 过滤选不上。
- 测试名跟某个 level 名同名（理论上不可能，但提醒一下）→ `build_and_test.sh` 会优先当 label 处理。
- 不通过 `add_atb_test()` 而是手写 `add_executable + add_test` → 需要手动跟一行 `set_tests_properties(<name> PROPERTIES LABELS "levelN_xxx")`，参考 `test_io_adapters` 的写法。

还有几种"加测试时必须同步更新的硬编码列表"（refdata 依赖、E2E 的 RESOURCE_LOCK 等），统一在 [§ 八](#八硬编码契约清单加东西前先扫一眼) 列出。

---

## 八、硬编码契约清单（加东西前先扫一眼）

为了让 `build_and_test.sh` 的各模式真正可信、ctest 并行不踩坑，仓库里有 **5 处需要手动维护同步的硬编码列表**。改这些功能时一并更新对应列表，否则会出现"看起来在工作但实际漏了"——而且失败方式都是**静默退化**而不是显式报错。

| # | 列表 / 约束 | 位置 | 什么时候改 | 验证方式 |
|---|---|---|---|---|
| 1 | `KNOWN_LEVELS` 数组 | `atb_cpp_llm/build_and_test.sh` 顶部 | 新增 `tests/levelN_xxx/` 目录时 | `bash build_and_test.sh --test-only levelN_xxx` 应识别成 level（看 `[build_and_test] Label filter:` 那一行），而不是当测试名走 `-R` |
| 2 | `REFDATA_SENTINELS` 数组 | `atb_cpp_llm/build_and_test.sh` 顶部 | `tests/python_reference/gen_all.py` 的 `GENERATORS` 表新增/删除生成器时 | 每个生成器至少有一个代表性输出文件登记；删/改一个哨兵后 `--no-refresh-refdata` 应正确触发 fallback（看 `WARN: ... N/M sentinel files are missing`） |
| 3 | `set_tests_properties(... RESOURCE_LOCK "npu_e2e")` | `atb_cpp_llm/CMakeLists.txt` 文末 | 新增 E2E 或会一次性吃 >10 GB HBM 的测试时 | 不加会被 `ctest -j4` 并行触发，立刻 `aclrtMalloc error=207001`；目前 7 个测试在锁里 |
| 4 | `REFDATA_DEPENDENT_TESTS` 集合 | `atb_cpp_llm/CMakeLists.txt` 文末 | 新增测试读 `/tmp/{cpu_,stage_,posembed_npu_,visrope_npu_}*.bin` 时 | 见下面"两条 review 命令"#1 |
| 5 | `tests/levelN_*/` 目录命名 + `add_atb_test()` 第 193 行的正则 `MATCHES "^level[0-9]"` | 物理目录约定 + `CMakeLists.txt` 的正则 | 新增 level 目录时 | 目录名必须满足 `^level[0-9]`；如果你想用 `level_perf` 这种命名，要同时改正则，否则 add_atb_test() 不会给新 level 的测试打 LABEL，过滤/排除全部失效 |

**根本原则**：上面任意一条契约失效，对应的 CLI 选项或 ctest 过滤都会**静默退化**而不是报错。新人最容易踩 #2 和 #4 —— 加生成器或加测试时往往只想自己的事，不会想到要登记。如果发现 `--no-refdata` 行为不对劲、或 `--no-refresh-refdata` 该 fallback 时没 fallback，第一件事是回这个表对一遍。

### 两条 review 命令（review PR 时跑一下）

```bash
# 验证 #4: 实际读 /tmp/*.bin 的测试是否全部登记在 REFDATA_DEPENDENT_TESTS
diff \
    <(grep -lE '/tmp/(cpu_|stage_|posembed_npu_|visrope_npu_)' \
        atb_cpp_llm/tests/level*/*.cpp | xargs -n1 basename -s .cpp | sort) \
    <(awk '/^set\(REFDATA_DEPENDENT_TESTS/,/^\)/' atb_cpp_llm/CMakeLists.txt \
        | grep -oE 'test_[a-z_0-9]+' | sort -u)
# 期望: 无输出 (两边完全一致)

# 验证 #2: gen_all.py 每个生成器的哨兵都被 REFDATA_SENTINELS 覆盖
for f in $(grep -oE '/tmp/[A-Za-z_0-9]+\.bin' atb_cpp_llm/tests/python_reference/gen_all.py | sort -u); do
    grep -qF "$f" atb_cpp_llm/build_and_test.sh && echo "OK  $f" || echo "MISS $f"
done
# 期望: 全部 OK
```

### 不重构成"单一来源"的原因

理论上可以让 CMake 在 configure 阶段调 Python 解析 gen_all.py，或者反过来生成 build_and_test.sh —— 但这会引入构建期依赖（CMake → Python → C++ 测试），收益不值。5 个列表加起来不到 50 项，更新频率每月不到 1 次，人工同步 + review 命令兜底足够。
