# atb_llm

基于华为昇腾 NPU 和 ATB（**Ascend Transformer Boost**，开源地址 <https://gitcode.com/cann/ascend-transformer-boost>）图编译器的 LLM 推理实现集合。

仓库包含两个相互独立的子项目：

| 子项目 | 语言 | 定位 |
|---|---|---|
| [`atb_python_qwen3vl_embedding/`](./atb_python_qwen3vl_embedding/) | Python | Qwen3-VL-Embedding-2B 的纯 ATB 图实现，**推理热路径零 transformers 依赖**。直接从 safetensors 加载权重并构建 ATB 计算图。 |
| [`atb_cpp_llm/`](./atb_cpp_llm/) | C++17 | 多模型 LLM 推理引擎，已支持 Qwen3、Qwen3VL、DeepSeek-V2/V3、Mixtral 等架构。 |

两个子项目共享同一份 `.env`（位于仓库根目录），其他无耦合，可独立构建运行。

---

## 适用场景

- 想在 Ascend 910B 上跑 Qwen3-VL-Embedding-2B 推理 → 用 Python 子项目。
- 想在 Ascend NPU 上跑多种主流 LLM 推理（C++ 性能优先） → 用 C++ 子项目。
- 想学习如何用 torch_atb / atb C++ API 手动搭计算图、做权重加载、做 split-graph 优化 → 两个都看。

---

## 硬件与系统要求

| 类别 | 要求 |
|---|---|
| 硬件 | 华为昇腾 NPU（在 910B 上验证过） |
| 操作系统 | Linux aarch64（其他架构未验证） |
| 驱动 | 昇腾 NPU 驱动 + `npu-smi` 可用 |
| 软件栈 | CANN 9.0.0、NNAL/ATB（带 `cxx_abi=1`） |
| Python 运行时 | `torch`、`torch_npu`、`torch_atb`、`safetensors` |
| C++ 构建 | CMake ≥ 3.16、GCC（支持 C++17）、Ascend toolkit + ATB |

**没有 NPU 的开发机**：C++ 项目可以配置 + 尝试编译（链接阶段会失败，符合预期），用于代码静态检查；测试运行必须在 NPU 主机上完成。Python 项目导入即触发 NPU 初始化，没有 NPU 跑不了。

---

## 从 0 开始跑起来

### Step 1 — 克隆仓库

```bash
git clone <repo-url> atb_llm
cd atb_llm
```

### Step 2 — 准备模型权重

下载 `Qwen3-VL-Embedding-2B` 权重到本地任意目录，目录里必须有：

```
<your-model-dir>/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

> C++ 项目还需要其他模型的权重（Qwen3、DeepSeek-V2 等），按需准备即可，路径在测试文件里配置。

### Step 3 — 配置 `.env`

复制模板并改成自己的路径：

```bash
cp .env.example .env
$EDITOR .env
```

`.env` 内容示例：

```bash
QWEN3VL_EMB_MODEL_DIR=/abs/path/to/Qwen3-VL-Embedding-2B
```

> `.env` 已被 `.gitignore` 忽略，不会泄漏机器路径。Python 和 C++ 都会从仓库根目录自动加载它。

### Step 4 — Source 昇腾环境（每个新 shell 都要执行）

CANN 和 ATB 的安装路径取决于安装方式，下面两种是常见位置：

```bash
# 方式 A：非 root 用户装在 ~/Ascend/
source ~/Ascend/ascend-toolkit/set_env.sh
source ~/Ascend/nnal/atb/set_env.sh --cxx_abi=1

# 方式 B：root 用户用默认安装器装在 /usr/local/Ascend/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
```

C++ 项目的 `build_and_test.sh` 会**自动探测**这两个根目录并 source 对应脚本，所以走脚本入口可以跳过这一步；如果你直接用 `cmake` 手工构建，或者跑 Python 测试，需要自己 source。

### Step 5 — 选其中一个子项目跑

#### A. Python：Qwen3-VL-Embedding-2B

```bash
cd atb_python_qwen3vl_embedding

# 单组件单元测试
python tests/test_text_attention.py
python tests/test_vision_model.py

# 端到端推理引擎测试（text-only / image-only / text+image）
python tests/test_engine.py

# 完整 pipeline vs transformers 参考实现对比
python tests/test_e2e.py

# 性能基准
python tests/benchmark.py --iter 50 --warmup 10
```

所有单元测试都会与 transformers 参考实现做余弦相似度对比，阈值通常为 `0.99`。

更多细节见 [`atb_python_qwen3vl_embedding/__init__.py`](./atb_python_qwen3vl_embedding/__init__.py) 中的模块清单，以及项目顶层的 [`CLAUDE.md`](./CLAUDE.md)（包含架构图、split-graph 策略、deepstack 融合机制说明）。

#### B. C++：多模型 LLM 推理引擎

一键构建 + 测试：

```bash
cd atb_cpp_llm
bash build_and_test.sh                      # Release + 默认每次刷新参考数据 + 跑所有 CTest
bash build_and_test.sh --debug              # Debug 构建
bash build_and_test.sh --clean              # 清掉 build/ 重建
bash build_and_test.sh --no-test            # 只构建不跑测试
bash build_and_test.sh --no-refresh-refdata # 复用 /tmp 已有参考数据（缺则自动 fallback）
bash build_and_test.sh --no-refdata         # 跳过生成且主动排除 27 个依赖参考数据的测试
bash build_and_test.sh --refresh-refdata    # 显式刷新（等价于默认）
```

**快速迭代（不重 build，只跑你关心的测试）**：

```bash
bash build_and_test.sh --test-only --no-refresh-refdata                       # 复用 refdata，跑全部
bash build_and_test.sh --test-only --no-refresh-refdata level1_cpu_pure       # 只跑 Level 1
bash build_and_test.sh --test-only --no-refresh-refdata test_vision_stages    # 只跑 1 个测试
bash build_and_test.sh --test-only --no-refresh-refdata test_bin_format test_text_model  # 跑多个
bash build_and_test.sh --test-only --no-refresh-refdata -v test_vision_stages # 加 -v 看完整输出
bash build_and_test.sh --list                                                 # 列出所有测试 + label
```

- **位置参数自动识别**：命中 `level0_framework` / `level1_cpu_pure` / `level2_op_precision` / `level3_integration` / `level4_e2e` 当作 level 过滤，否则当作测试名过滤。
- **参考数据三态**：默认每次刷新（~90s），`--no-refresh-refdata` 复用已有（缺失自动 fallback 到 `--no-refdata`），`--no-refdata` 主动排除 27 个依赖参考数据的测试并打印清单。

详细规则、维护契约、边界用例见 [`atb_cpp_llm/docs/testing-guide.md` § 一·五](atb_cpp_llm/docs/testing-guide.md)。

脚本会：

1. 加载仓库根目录的 `.env`；
2. Source `~/Ascend/` 或 `/usr/local/Ascend/` 下的 CANN/ATB 环境脚本；
3. 用 CMake 配置 + 并行构建；
4. **默认每次运行**都重新生成全部参考数据到 `/tmp/`（约 ~90 秒，详见下文"参考数据三种模式"）；
5. 检测到 `npu-smi` 才会跑 CTest（避免在无 NPU 主机上失败）。

**为什么需要参考数据？** C++ 端 Level1/Level2/Level3/Level4 的 27 个精度测试都是和 Python 参考实现做对比，参考值由 `gen_*_reference.py` 在 NPU 上跑 Python 模型后写到 `/tmp/cpu_*.bin`、`/tmp/stage_*.bin`、`/tmp/posembed_npu_*.bin`、`/tmp/visrope_npu_*.bin`。**注意**：这 27 个测试在数据缺失时会走 `LOG_ERROR(SKIP) + return` 静默"通过"——所以脚本默认每次刷新，加 `--no-refresh-refdata` 可以复用已有数据（缺则自动 fallback 到 `--no-refdata` 模式主动排除它们）。

手动单独生成参考数据：

```bash
# 一次生成全部（5 个生成器串行；每个独立进程避免 set_atb_buffer_size 冲突）
python atb_cpp_llm/tests/python_reference/gen_all.py

# gen_all.py 自身的 --skip-fresh: 跳过哨兵文件已存在的生成器（脚本默认 mode 不传这个 flag，
# 用户在脚本层用 --no-refresh-refdata 控制，详见 atb_cpp_llm/docs/testing-guide.md § 一·五 #8）
python atb_cpp_llm/tests/python_reference/gen_all.py --skip-fresh
```

手动构建（不想用脚本）：

```bash
cd atb_cpp_llm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
# ⚠️ 跑测试前先生成参考数据
python tests/python_reference/gen_all.py
ctest --test-dir build --output-on-failure -j4
```

测试按金字塔分层（详见 [`atb_cpp_llm/tests/README.md`](./atb_cpp_llm/tests/README.md)）：

- `level0_framework/` — 基础框架（ContextManager / TensorAllocator / JsonConfig）
- `level1_cpu_pure/` — CPU 纯函数精度
- `level2_op_precision/` — ATB 算子精度
- `level3_integration/` — 多组件集成
- `level4_e2e/` — 端到端推理 + 与 Python 参考实现对比

C++ 项目的架构、重构计划、性能优化路线见 [`atb_cpp_llm/docs/README.md`](./atb_cpp_llm/docs/README.md) 索引。

---

## 仓库结构

```
atb_llm/
├── .env.example                       # 环境变量模板（拷贝为 .env 使用）
├── CLAUDE.md                          # 项目说明（架构、约束、调试方法）
├── README.md                          # 本文件
│
├── atb_python_qwen3vl_embedding/      # Python 子项目
│   ├── engine.py                      # Qwen3VLEngine 顶层推理引擎
│   ├── engine_utils.py                # 权重加载、MRoPE、RoPE 索引
│   ├── text_*.py / vision_*.py        # 各组件的 ATB 图实现
│   ├── preprocess.py                  # CPU 端图像预处理（纯 Python）
│   └── tests/                         # 单元测试 + E2E + benchmark
│
└── atb_cpp_llm/                       # C++ 子项目
    ├── CMakeLists.txt
    ├── build_and_test.sh              # 一键构建 + 测试入口
    ├── include/atb_llm/               # 公共头文件（engine.h, model.h, ...）
    ├── src/
    │   ├── core/                      # ContextManager / Allocator / GraphBuilder
    │   ├── engine/                    # LlmEngine / Embedder / Runtime
    │   ├── families/                  # 多模型适配层（Qwen3, Qwen3VL, DeepSeek, ...）
    │   ├── components/                # text / vision / common 组件
    │   ├── ops/                       # ATB 算子薄包装
    │   ├── runners/                   # text_runner / vision_runner
    │   ├── io/                        # safetensors_reader / weight_loader
    │   └── adapters/ utils/ log/
    ├── tests/                         # level0~4 分层测试
    └── docs/                          # 设计文档 / 重构计划 / 优化路线
```

---

## 排查问题

### Python

- **测试报 `RuntimeError: Required environment variable QWEN3VL_EMB_MODEL_DIR is not set`**
  → 仓库根目录还没有 `.env`，或里面没填模型路径。回到 Step 3。
- **`torch_npu` / `torch_atb` 导入失败**
  → 没 source 昇腾环境脚本，或安装的 CANN/ATB 版本与 torch_npu 不匹配。
- **ATB 图输出全 0 或精度异常**
  → 多半是 `set_atb_buffer_size()` 被调了两次。本仓库约束是**整个进程只能调用一次**。
- **调试时看 ATB 日志**：

  ```bash
  cat /root/ascend/log/atb/$(ls -rt /root/ascend/log/atb/ | tail -n 1)
  ```

### C++

- **CMake 找不到 `acl/acl.h` 或链接 `libatb.so` 失败**
  → 没 source 环境脚本，或 `ATB_DIR` / `ASCEND_TOOLKIT_DIR` 指错路径。可手动传：

  ```bash
  cmake -S . -B build -DATB_DIR=/path/to/atb/cxx_abi_1 -DASCEND_TOOLKIT_DIR=/path/to/cann
  ```
- **`build_and_test.sh` 提示 `npu-smi not found`**
  → 这是当前主机无 NPU 时的正常行为，构建完成但跳过测试。把构建产物拷到 NPU 主机后执行：

  ```bash
  ctest --test-dir build --output-on-failure -j4
  ```
- **调试时看 ATB 日志**：

  ```bash
  cat /root/atb/log/$(ls -rt /root/atb/log/ | tail -n 1)
  ```

---

## 测试精度原则

**绝不通过降低验收标准来"通过"测试。** 如果 C++ 和 Python 在相同输入下余弦相似度低于 `0.99`，说明存在 bug，必须定位并修复根因，而不是放宽阈值。在有 Python 参考实现的情况下，C++ 应严格对齐 Python 的计算逻辑，逐阶段排查差异直到余弦相似度 ≥ `0.99`。

---

## 进一步阅读

- [`CLAUDE.md`](./CLAUDE.md) — 项目总览（Python 子项目架构、split-graph 策略、deepstack 融合、关键约束）
- [`atb_python_qwen3vl_embedding/__init__.py`](./atb_python_qwen3vl_embedding/__init__.py) — Python 模块清单
- [`atb_cpp_llm/docs/README.md`](./atb_cpp_llm/docs/README.md) — C++ 设计文档索引（架构、重构计划、优化路线、测试指南）
- [`atb_cpp_llm/tests/README.md`](./atb_cpp_llm/tests/README.md) — C++ 测试金字塔说明
