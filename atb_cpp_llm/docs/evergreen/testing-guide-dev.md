# Testing Guide for Developers

> 面向 Developer 的实操手册。目标是让每次改动都能明确回答：这次验证的是 **official gate**、跨语言 self-consistency，还是 diagnostic？

---

## 1. 快速结论

1. **精度测试主线 = official gate 优先**。
   - 端到端精度以官方 `Qwen3VLEmbedder` runtime 输出为黄金标准。
   - 当前核心 gate：`test_engine_vs_official`，验证 910B path C raw image full engine embedding vs 官方 pooled embedding。
   - preprocess 官方 gate：`test_aclnn_bicubic_spike` TC4，验证 NPU pixel_values vs official pixel_values。

2. **`benchmark --mode compare` 只是跨语言一致性/性能 compare，不是 vs official**。
   - compare 路径保留 CPU/Python 自家链，用于 C++/Python 输出一致、性能对比、二进制冒烟。
   - 它不能证明 C++ 与官方 `Qwen3VLEmbedder` 对齐。

3. **所有精度 gate 必须硬性退码**。
   - gate 计算 cosine 后必须 `CHECK` / 非零退出。
   - 只打印 `LOG_ERROR` / `LOG_INFO` 不算 gate。
   - diagnostic 可以不退码，但必须显式标注 `not gated`。

4. **310P limitation 必须诚实标注**。
   - 910B AA 路径可对齐 official full gate。
   - 310P 上 AA 不可用，official full gate 当前 skip；这是平台精度限制，不得伪装成已覆盖。

---

## 2. 修改类型 → 必跑测试矩阵

| 修改类型 | 必跑测试 | 说明 |
|----------|----------|------|
| preprocess / path C / `max_pixels` / AA | `python atb_cpp_llm/tests/python_reference/gen_all.py`；`ctest -R test_aclnn_bicubic_spike`；`ctest -R test_engine_vs_official`；`ctest -R test_path_c_raw_image`；benchmark smoke + compare | official gate 优先。`test_path_c_raw_image` 是 path C vs PREPROCESSED 自比对，不是 official gate。benchmark compare 只证明跨语言一致性。 |
| vision graph | `ctest -R test_vision_stages`；`ctest -R test_engine_vs_official`；相关 level2 vision tests | stage gate 要退码；最终仍以 official full embedding gate 确认端到端影响。 |
| text graph / deepstack | `python atb_python_qwen3vl_embedding/tests/test_pipeline_trace.py`；`python atb_python_qwen3vl_embedding/tests/test_embedder_e2e.py`；`ctest -R test_engine_vs_official` | pipeline trace 中 diagnostic 段不能当 gate；最终以 official full embedding gate 兜底。 |
| benchmark | 所有 mode 冒烟；`./atb_cpp_llm/build/benchmark --mode compare`；`python atb_python_qwen3vl_embedding/tests/benchmark.py --mode all --load-pixel-values`；`python atb_cpp_llm/tests/compare_py_cpp.py` | benchmark 不在 CTest；必须手动跑。compare 不证明 official，只证明 C++/Python 自家链一致和输出可用。 |
| 310P / platform | 在 310P 上 `bash atb_cpp_llm/build_and_test.sh`；Python `run_all.py`；benchmark compare；`compare_py_cpp.py`；核对 known skips | 310P AA 不可用，`test_engine_vs_official` / official full gate 当前应 skip。known skip 必须写清原因，不能静默 PASS。 |

---

## 3. 910B 推荐命令顺序

> 原则：先生成同源 refdata，再跑 official gates，再跑全量，再跑手动 benchmark。不要把 benchmark compare 当 official gate。

### 3.1 加载环境

Root 用户：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
```

普通用户：

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
source ~/Ascend/cann/set_env.sh
source ~/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source ~/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=~/Ascend/nnal/atb/latest/atb/cxx_abi_1
```

### 3.2 生成参考数据

```bash
cd /mnt/workspace/gitCode/atb_llm_study
python atb_cpp_llm/tests/python_reference/gen_all.py
```

验收：所有 generator OK，尤其 official pixel_values / official embedding 生成成功。

### 3.3 先跑 official gates

```bash
cd /mnt/workspace/gitCode/atb_llm_study
cmake -S atb_cpp_llm -B atb_cpp_llm/build -DCMAKE_BUILD_TYPE=Release
cmake --build atb_cpp_llm/build --target test_aclnn_bicubic_spike test_engine_vs_official -j8
ctest --test-dir atb_cpp_llm/build -R 'test_aclnn_bicubic_spike|test_engine_vs_official' --output-on-failure
```

当前 910B 已知结果：

| Gate | 分辨率 | cosine |
|------|--------|--------|
| `test_aclnn_bicubic_spike` TC4 NPU pv vs official pv | 416 / 720 / 1080 / 1440 | 1.0 / 0.999924 / 0.999878 / 0.999951 |
| `test_engine_vs_official` full embedding | 416 / 720 / 1080 / 1440 | 0.999882 / 0.999235 / 0.999469 / 0.999690 |

### 3.4 C++ 全量

```bash
cd /mnt/workspace/gitCode/atb_llm_study
bash atb_cpp_llm/build_and_test.sh
```

如只复用已有 refdata：

```bash
bash atb_cpp_llm/build_and_test.sh --no-refresh-refdata
```

### 3.5 Python 全量

```bash
cd /mnt/workspace/gitCode/atb_llm_study
python atb_python_qwen3vl_embedding/tests/run_all.py
```

### 3.6 Benchmark smoke / compare

```bash
cd /mnt/workspace/gitCode/atb_llm_study
python atb_cpp_llm/scripts/gen_compare_tokens.py
./atb_cpp_llm/build/benchmark --mode text
./atb_cpp_llm/build/benchmark --mode io
./atb_cpp_llm/build/benchmark --mode mm
./atb_cpp_llm/build/benchmark --mode compare
python atb_python_qwen3vl_embedding/tests/benchmark.py --mode all --load-pixel-values
python atb_cpp_llm/tests/compare_py_cpp.py
./atb_cpp_llm/build/benchmark --mode all
./atb_cpp_llm/build/benchmark --mode cold
./atb_cpp_llm/build/benchmark --mode throughput
```

验收：每个 mode 退出码 0、human-readable 表正常、`BENCH_RESULT` 行正常、compare cosine ≥ 0.99。

注意：`benchmark --mode compare` 只生成 C++ 侧 `/tmp/cpp_*.bin` 和中间 pixel_values；必须随后运行 Python benchmark 的 `--load-pixel-values` 生成 `/tmp/py_*.bin`，最后 `compare_py_cpp.py` 才能完成 13 case cosine 对比。

---

## 4. Reviewer checklist

Reviewer 破坏者审查测试相关改动时，逐项回答：

1. **复现**
   - 是否先运行了本次修改应影响的最小命令？
   - 若是 bugfix，是否复现原问题再确认修复？

2. **参考真相源**
   - 这是 vs official、self-consistency，还是 diagnostic？文档/日志是否说清？
   - 若声称 official，是否真的使用官方 `Qwen3VLEmbedder` runtime 链，而不是 processor config 默认链、AutoProcessor 默认链、Python ATB、CPU `PreprocessImage`？

3. **生产分辨率**
   - 是否覆盖 416×672、720×1280、1080×1920、1440×2560 等生产分辨率？
   - 若只用 toy input，是否明确只代表 toy sanity？

4. **退码**
   - 所有 gate 是否有 `CHECK` / 非零退出？
   - 是否存在只打 `LOG_ERROR` / `LOG_INFO` 但 CTest 仍 PASS 的假阳性？

5. **Refdata 登记**
   - 新增读取 `/tmp/*.bin` 的测试是否登记到 `REFDATA_DEPENDENT_TESTS` / `needs_refdata`？
   - `gen_all.py` 是否包含新生成器？
   - `build_and_test.sh` sentinel 是否能检测缺失 refdata？

6. **Token 同源**
   - full embedding vs official gate 的 `input_ids` 是否来自同一次 official preprocess 捕获？
   - 是否 guard 默认 prompt `Represent the user's input.`、`max_pixels=1843200`、image token 数？

7. **310P skip 诚实**
   - 310P skip 是否有硬件/算子限制理由？
   - 是否把 310P 未覆盖误写成已通过？

---

## 5. 假阳性自检

### 5.1 grep cosine 日志

```bash
cd /mnt/workspace/gitCode/atb_llm_study
grep -RIn "LOG_ERROR.*cos\|LOG_INFO.*cos" atb_cpp_llm/tests atb_python_qwen3vl_embedding/tests
```

检查原则：

- gate：日志附近必须有 `CHECK`、`REQUIRE`、`return 1`、`sys.exit(1)` 或等价退码。
- diagnostic：必须在日志或注释中显式标注 `diagnostic` / `not gated`。
- 不能出现“cos 低但只打印，测试仍 PASS”的路径。

### 5.2 grep official gate 入口

```bash
grep -RIn "test_engine_vs_official\|gen_official_embedding\|gen_official_pixel_values\|test_aclnn_bicubic_spike" atb_cpp_llm/tests atb_cpp_llm/CMakeLists.txt atb_cpp_llm/build_and_test.sh
```

检查原则：

- CTest target 已注册。
- refdata generator 已注册。
- 缺 refdata 时不会静默跳过关键 gate。

---

## 6. 310P 注意事项

1. **AA 不可用**
   - `aclnnUpsampleBicubic2dAA` 当前不可用于 310P。
   - 因官方链降采样带 AA，而 310P 没有对应 AA 路径，full official gate 当前 skip。

2. **official full gate skip 是 limitation，不是成功**
   - `test_engine_vs_official` 在非 910B 上 skip，应在状态/报告中写明“未覆盖 official full embedding”。

3. **310P precision limitation 要文档化**
   - 310P 可以跑平台适配、NZ mask、GQA、self-consistency、benchmark compare。
   - 但不能把这些结果等同于 910B official full embedding gate。

4. **已知 skip 要诚实**
   - 平台不支持的组合用 expected skip，并写清 CANN/ATB 限制。
   - 禁止靠跳过把未知失败伪装成通过。

---

## 7. 当前事实速查

阶段0盲区数据：

| 对比 | 输入 | 416 | 720 | 1080 | 1440 |
|------|------|-----|-----|------|------|
| CPU non-AA vs official AA pixel_values | noise | 1.0 | 0.999102734 | 0.995964392 | 0.902562175 |
| CPU non-AA vs official AA pixel_values | natural | 1.0 | 0.999994070 | 0.999991300 | 0.999992786 |
| NPU pv vs official pv | noise | 1.0 | 0.999924 | 0.999878 | 0.999951 |
| NPU pv vs CPU pv | noise | 1.0 | 0.999786 | 0.994999 | 0.899187 |

阶段2 full gate：

| Test | 416 | 720 | 1080 | 1440 |
|------|-----|-----|------|------|
| `test_engine_vs_official` | 0.999882 | 0.999235 | 0.999469 | 0.999690 |

结论：CPU/Python 自家链 vs C++ compare 是弱信号；official gate 才能证明对齐官方。
