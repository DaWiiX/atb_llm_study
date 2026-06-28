# Testing Architecture

> 面向 Architect 的测试体系设计文档。目标是统一测试语义：每个测试属于哪一层、使用哪个参考真相源、覆盖哪个 `(平台 × 分辨率 × 路径 × 参考)` 四元组。

---

## 1. 测试分层

| 层级 | 名称 | 目标 | 示例 |
|------|------|------|------|
| L0 | framework / config / refdata | 验证测试框架、配置读取、参考数据存在性和格式契约 | `test_config_wiring`、refdata sentinel、generator import 检查 |
| L1 | CPU pure | 验证纯 CPU 逻辑、几何、序列、bin 格式，不依赖 NPU 图 | CPU preprocess、token/bin loader、shape/grid 计算 |
| L2 | op precision | 单算子/小算子链精度 gate，定位局部数值差异 | `test_aclnn_bicubic_spike`、patch embed、level2 vision op tests |
| L3 | integration / stage | 多组件阶段输出，验证 graph/stage 对齐 | `test_vision_stages`、`test_pipeline_trace` |
| L4 | e2e / full embedding | 端到端 embedding 级别 gate，最终验收模型行为 | `test_engine_vs_official`、`test_embedder_e2e` |
| manual | benchmark / performance | 手动性能、mode 冒烟、跨语言 compare | `benchmark --mode compare`、`compare_py_cpp.py`、throughput/cold/all |

分层原则：

- L2/L3 负责定位问题；L4 负责最终端到端正确性。
- manual benchmark 不因“手动”而低价值，但它不是 CTest 自动 gate，提交前必须人工跑。
- 同一个测试必须清楚标注自己是 official gate、self-consistency，还是 diagnostic。

---

## 2. 参考真相源原则

### 2.1 黄金标准

**official `Qwen3VLEmbedder` runtime 是黄金标准。** 以下都不是最终黄金标准：

- `preprocessor_config.json` 默认值
- `AutoProcessor` 默认链
- Python ATB 自家实现
- CPU `PreprocessImage`
- C++/Python benchmark compare 输出

这些实现可以用于定位、回归或跨语言一致性，但不能单独证明“对齐官方”。

### 2.2 Qwen3VL-Embedding-2B official runtime 关键常量

当前 official embedding gate 必须遵守：

- `max_pixels = 1843200`，来自 Qwen3VLEmbedder embedder 常量 `1800 * 32 * 32`。
- `do_resize = False`，resize 信息由 `process_vision_info` 产出。
- `process_vision_info(image_patch_size=16)`。
- 默认 prompt：`Represent the user's input.`。

注意：`preprocessor_config.json` 中的 `max_pixels=1310720` 是 image processor 通用默认值；官方 embedder runtime 会覆盖它。对标官方时看 runtime，不看 config 默认值。

### 2.3 Full embedding gate 的 token 同源原则

full embedding vs official gate 中，C++ 使用的 token 必须从**同一次 official preprocess** 捕获，不能手写或复用旧 token bin。

原因：embedding 不只由图像决定，还由 chat template、默认 prompt、image token 个数、special tokens 共同决定。只要 token 不同，就是“同图不同问题”。

当前 `gen_official_embedding.py` 的正确模式：

1. 调用真实 `Qwen3VLEmbedder.process(inputs, normalize=True)` 得到 official pooled embedding。
2. wrapper 捕获同一次 `_preprocess_inputs` 的 `input_ids` 写入 token bin。
3. 校验 public embedding vs captured-chain embedding `max_diff=0`。
4. guard 默认 prompt、`max_pixels=1843200`、image token 数。

---

## 3. 覆盖矩阵：平台 × 分辨率 × 路径 × 参考

### 3.1 当前已覆盖

| 平台 | 分辨率 | 路径 | 参考 | 覆盖状态 |
|------|--------|------|------|----------|
| 910B | 416×672、720×1280、1080×1920、1440×2560 | path C raw image full engine | official `Qwen3VLEmbedder` pooled embedding | `test_engine_vs_official`，cos=0.999882/0.999235/0.999469/0.999690 |
| 910B | 同上 | NPU preprocess pixel_values | official pixel_values | `test_aclnn_bicubic_spike` TC4，cos=1.0/0.999924/0.999878/0.999951 |
| 910B / CPU | 同上 | CPU/Python self path | CPU/Python self-consistency | 阶段0已实测；证明自家链一致，不证明 official |
| 910B | 96×96 toy | path C raw image vs PREPROCESSED | self-consistency | `test_path_c_raw_image`，验证 dispatch/path C，不是 official gate |

### 3.2 当前未覆盖或弱覆盖

| 缺口 | 状态 | 风险 |
|------|------|------|
| 310P vs official full embedding | 未覆盖；official full gate skip | full-engine attention 路径需 310P 硬件验证（CI 在 910B）；bicubic 降采样已由 small-op AA 对齐 PIL，非 skip 理由 |
| benchmark compare vs official | 未覆盖；compare 非 official gate | compare PASS 只能说明 C++/Python 自家链一致，可能共同偏离 official |
| Python e2e production resolution vs official | 弱覆盖/未系统覆盖 | Python toy/e2e sanity 不能替代 4 个生产分辨率 official gate |
| natural image official gate | 当前 official gate 使用 noise prod inputs；natural 图已有阶段0 preprocess 观测 | noise 是 worst-case 高频输入；natural coverage 可补充代表性，但不能替代 worst-case |

阶段0证明的盲区：CPU non-AA vs official AA 在 noise 1440 上 cosine 只有 `0.902562175`，所以 CPU/Python 自家链互比不能作为 official 精度证据。

---

## 4. 各测试语义分类

| 测试 / 脚本 | 分类 | 语义 |
|-------------|------|------|
| `test_engine_vs_official` | 核心 official e2e gate | 910B path C raw image full engine embedding vs 官方 `Qwen3VLEmbedder` pooled embedding；4 个生产分辨率；`CHECK cos>=0.99` |
| `test_aclnn_bicubic_spike` TC4 | preprocess official pv gate | NPU preprocess pixel_values vs official pixel_values；证明 path C 入口前的图像预处理对齐 official |
| `benchmark --mode compare` | 跨语言一致性 / 性能 compare | C++ vs Python 自家链对比；用于性能和二进制输出冒烟；非 official gate |
| `test_accuracy` | 历史跨语言一致性 / 弱信号 | 验证 C++/Python 一致；不能证明 official |
| `test_path_c_raw_image` | path C self-consistency | raw_image path C vs PREPROCESSED 自比对；验证 dispatch、device tensor 路径和 path C 不破；非 official gate |
| Python `test_e2e` / `test_embedder_e2e` | toy-level Transformers sanity | 用 toy 或较小输入做端到端 sanity；不是 4 个生产分辨率 official gate |
| `test_vision_stages` | L3 stage gate | vision stage 输出分段对齐；必须退码；定位 vision graph 差异 |
| `test_pipeline_trace` | L3 text/deepstack trace + diagnostic | Step 1-7 gated；Step 8+ alternate transformers path 是 diagnostic/non-gated |

---

## 5. 新增测试决策树

### 5.1 何时新增 official gate

新增 official gate，当改动满足任一条件：

- 影响最终 embedding：vision graph、text graph、deepstack、pooling、token/chat template、path C dispatch。
- 影响 official runtime 参数：`max_pixels`、`min_pixels`、`image_patch_size`、prompt、`do_resize`、chat template。
- 影响 preprocess 数值：AA、resize kernel、normalize、patch extraction、grid。
- 当前测试只能证明 self-consistency，而需求声称“对齐官方”。

official gate 要求：

- 用真实 official runtime 或经 bit-exact probe 证明等价的链。
- 覆盖生产分辨率，除非明确是 toy gate。
- `CHECK cos>=0.99`，不降阈值。
- refdata 生成、CTest、sentinel、needs_refdata 全登记。

### 5.2 何时使用自比对 / self-consistency

使用自比对，当目标是验证：

- 新路径与旧路径是否等价，如 path C raw image vs PREPROCESSED。
- 跨语言接口是否一致，如 C++/Python bin 消费格式。
- 性能改造是否保持旧行为。

限制：自比对不能证明 official。如果旧路径本身偏离 official，自比对只会证明“两个路径同错”。

### 5.3 何时使用 benchmark

使用 benchmark，当目标是：

- 验证对外二进制 mode/flag 可运行。
- 获取性能、cold start、throughput、stage timing。
- 验证 C++/Python compare 文件输出和人工可读表。

限制：benchmark compare 不在 CTest；必须手动跑。benchmark compare 非 official gate。

### 5.4 何时使用 diagnostic

使用 diagnostic，当目标是：

- 观察中间差异趋势。
- 打印辅助定位信息。
- 对已知不等价链路做探索性比较。

要求：diagnostic 必须显式写 `diagnostic` / `not gated`，不能让读者误以为 PASS 代表精度达标。

---

## 6. Refdata discipline

### 6.1 `gen_all.py` 顺序

所有 C++ 测试用 `.bin` 参考数据统一由 `atb_cpp_llm/tests/python_reference/gen_all.py` 编排。新增 generator 时：

1. 使用全限定 import，例如 `from atb_python_qwen3vl_embedding.preprocess import smart_resize`。
2. 保证 generator 可从 repo root 子进程运行。
3. 将 generator 加入 `gen_all.py`。
4. 若生成 official refdata，打印关键 guard：prompt、max_pixels、grid、token/image token 数、public-vs-captured diff。

### 6.2 `needs_refdata` / CTest 登记

任何读取 `/tmp/*.bin` refdata 的 C++ 测试都必须登记为 refdata-dependent。否则 `--no-refdata` 下可能静默跳过核心验证并假 PASS。

固定检查：

```bash
grep -RIn "REFDATA_DEPENDENT_TESTS\|needs_refdata\|test_engine_vs_official\|test_aclnn_bicubic_spike" atb_cpp_llm
```

### 6.3 `build_and_test.sh` sentinel

新增 refdata 文件后，要让 `build_and_test.sh` sentinel 能发现缺失文件。目标是缺 refdata 时 fail fast，而不是进入 CTest 后 skip 或读空数据。

### 6.4 bit-exact probe

重实现 official 参考链时，仅 cosine 高不够。应优先做 bit-exact probe：

- 用真实 official 类的 unbound 方法 + stand-in，或真实 public API wrapper。
- 对同一输入比较 generator 输出与 official 内部输出。
- 目标：`max_diff=0` 或解释清楚为何不可能 bit-exact。

阶段2 `gen_official_embedding.py` 的 public-vs-captured `max_diff=0` 是 full embedding refdata 的关键证据。

### 6.5 token 同源

full embedding refdata 必须保证：

- embedding bin 和 token bin 来自同一次 official preprocess。
- token bin 不手写、不从历史文件复用。
- C++ 测试校验 image token 数与 SmartResize/grid 预期一致。

---

## 7. 310P 架构定位

310P 当前可作为：

- platform correctness gate：NZ mask、GQA、平台检测、known unsupported configs。
- self-consistency gate：C++/Python 自家链 compare、benchmark mode smoke。
- regression gate：build_and_test、run_all、compare_py_cpp。

310P 当前不能作为：

- 910B AA official full embedding gate 的替代。
- “已对齐 official Qwen3VLEmbedder”的证明。

原因：`test_engine_vs_official` 跑完整 engine（含 SelfAttention），310P 上需 NZ mask 路径且需 310P 硬件实测，当前 CI 在 910B 硬件，故 310P skip 是明确 limitation。**注意**：bicubic 降采样本身在 310P 已由 small-op AA 拼装（`NpuBicubicResizeAASmallOp`，端到端与 910B aclnn AA 数值等价，见 STATUS §2.9 / roadmap P10-C）对齐 PIL，**不再是 skip 的理由**——skip 仅因 full-engine attention 路径需 310P 硬件验证。

---

## 8. 设计规则摘要

1. 先问测试语义：official、self-consistency、diagnostic？
2. 声称 official 就必须走 official runtime 或 bit-exact 证明等价。
3. full embedding gate 的 token 必须同源。
4. 生产精度必须覆盖生产分辨率。
5. gate 必须退码。
6. benchmark compare 不能升级成 official gate。
7. 310P skip 必须写 limitation，不写成功。
8. 新增 refdata 必须同时更新 generator、CTest、needs_refdata、sentinel。
