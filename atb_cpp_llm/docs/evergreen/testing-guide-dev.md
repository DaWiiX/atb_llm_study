# Testing Guide for Developers

> 面向 C++ Developer 的实操手册。目标是让每次 C++ 改动都能明确回答：这次验证的是 **official gate**、C++ 自一致性，还是 diagnostic？

---

## 1. 快速结论

1. **精度测试主线 = official gate 优先**。
   - 端到端精度以官方 `Qwen3VLEmbedder` runtime 输出为黄金标准。
   - 当前核心 gate：`test_engine_vs_official`，验证 910B path C raw image full engine embedding vs 官方 pooled embedding。
   - preprocess 官方 gate：`test_aclnn_bicubic_spike` TC4，验证 NPU pixel_values vs official pixel_values。

2. **`benchmark --mode compare` 只是 C++ benchmark 输出链冒烟，不是 vs official**。
   - compare mode 可用于确认 C++ 侧 bin/pixel_values 输出路径可用。
   - 它不能证明 C++ 与官方 `Qwen3VLEmbedder` 对齐。

3. **所有精度 gate 必须硬性退码**。
   - gate 计算 cosine 后必须 `CHECK` / 非零退出。
   - 只打印 `LOG_ERROR` / `LOG_INFO` 不算 gate。
   - diagnostic 可以不退码，但必须显式标注 `not gated`。

4. **official reference 是全平台真相源，不是 910B-only**。
   - `gen_official_embedding.py` 应能在 310P 上生成官方 refdata；NPU 官方链失败时应 fallback 到 CPU 官方链。
   - CPU fallback 也按 OFFICIAL_EMBED_CASES 生成；310P 用户应自行在 .env 中只保留 416x672,720x1280 以缩短耗时。
   - 当前 `test_engine_vs_official` 在 310P skip 是待修平台 gap，不是最终目标；最终仍要求全平台 vs official cosine ≥ 0.99。

5. **`build_and_test.sh` 是 C++ 全量回归主入口**。
   - 默认执行：加载环境 → CMake configure/build → 生成 C++ 测试所需 refdata → 跑全部 C++ CTest。
   - 它定义的是 **C++ 全量**，不包含 Python package `run_all.py`，也不包含跨语言 compare 的 Python 侧命令。
   - 如果日志出现 `Falling back to --no-refdata` 或 `Excluding ... needs_refdata`，不能算完整 C++ 全量通过。

---

## 2. 修改类型 → 必跑测试矩阵

| 修改类型 | 必跑测试 | 说明 |
|----------|----------|------|
| preprocess / path C / `max_pixels` / AA | `python atb_cpp_llm/tests/python_reference/gen_all.py`；`ctest -R test_aclnn_bicubic_spike`；`ctest -R test_engine_vs_official`；`ctest -R test_path_c_raw_image`；`bash atb_cpp_llm/build_and_test.sh` | official gate 优先。`test_path_c_raw_image` 是 path C vs PREPROCESSED 自比对，不是 official gate。最后用 `build_and_test.sh` 做 C++ 全量回归。 |
| vision graph | `ctest -R test_vision_stages`；`ctest -R test_engine_vs_official`；相关 level2 vision tests；`bash atb_cpp_llm/build_and_test.sh` | stage gate 要退码；最终仍以 official full embedding gate 确认端到端影响；C++ 全量回归用 `build_and_test.sh`。 |
| text graph / deepstack | `ctest` 相关 text/deepstack tests；`ctest -R test_engine_vs_official`；`bash atb_cpp_llm/build_and_test.sh` | C++ text/deepstack 改动以 C++ CTest 和 official full embedding gate 兜底。Python package 测试不属于本 C++ 全量定义。 |
| benchmark | 所有 C++ benchmark mode 冒烟；`./atb_cpp_llm/build/benchmark --mode compare` | benchmark 不在 CTest；必须手动跑。compare 不证明 official，只证明 C++ benchmark 输出路径可用。 |
| 310P / platform | 在 310P 上 `python atb_cpp_llm/tests/python_reference/gen_all.py`；`bash atb_cpp_llm/build_and_test.sh`；C++ benchmark smoke；核对 known skips 和 official gap | official reference generator 不能因 NPU 算子不支持而失败，应 fallback CPU 生成 golden refdata。`test_engine_vs_official` 当前 skip 是待修平台 gap，不能写成最终覆盖。 |

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

### 3.4 C++ 全量：`build_and_test.sh`

`build_and_test.sh` 是 C++ 侧全量回归的标准入口。默认行为是：

1. 加载 repo root 或 `atb_cpp_llm/` 下的 `.env`。
2. source Ascend CANN / ATB 环境脚本。
3. CMake configure + build。
4. 运行 `tests/python_reference/gen_all.py` 生成 C++ CTest 读取的 `/tmp/*.bin` refdata。
5. 检测到 NPU 后运行全部 C++ CTest。

```bash
cd /mnt/workspace/gitCode/atb_llm_study
bash atb_cpp_llm/build_and_test.sh
```

常用入口：

```bash
# 最完整 C++ 全量：构建 + 重生 refdata + 全部 C++ CTest
bash atb_cpp_llm/build_and_test.sh

# 复用已有 /tmp refdata，加快回归；只在确认 refdata 属于当前代码时使用
bash atb_cpp_llm/build_and_test.sh --no-refresh-refdata

# 只构建，不跑测试
bash atb_cpp_llm/build_and_test.sh --no-test

# 已有 build 后，只重跑全部 C++ CTest
bash atb_cpp_llm/build_and_test.sh --test-only

# 只跑某个 level label
bash atb_cpp_llm/build_and_test.sh --test-only level4_e2e

# 只跑某个测试名
bash atb_cpp_llm/build_and_test.sh --test-only test_engine_vs_official

# 查看已注册测试和 label
bash atb_cpp_llm/build_and_test.sh --list
```

验收注意：

- 默认模式必须成功生成 refdata；如果日志出现 `Falling back to --no-refdata`，不能算完整 C++ 全量通过。
- 如果日志出现 `Excluding ... tests labelled needs_refdata`，说明依赖 `/tmp/*.bin` 的测试被排除了，也不能算完整 C++ 全量通过。
- `--no-refdata` 只用于快速验证不依赖 refdata 的测试，不能用于 release / push 前的完整验收。
- `--no-refresh-refdata` 是加速入口，不是精度闸口；当改过 preprocess、token、shape、官方参考生成器、bin 格式或 CMake refdata 登记时，必须回到默认模式重生 refdata。
- official gate 失败不能靠 `--no-refdata` 或过滤测试绕过。
- official reference generator 是全平台真相源；如果官方 NPU reference 在 310P 上遇到不支持算子，应 fallback 到 CPU 官方链生成 refdata，而不是把 reference 标成 910B-only。
- CPU fallback 也按 OFFICIAL_EMBED_CASES 生成；310P 用户应自行在 .env 中只保留 416x672,720x1280 以缩短耗时。
- `test_engine_vs_official` 在 310P 的 skip 是当前 C++ 实现缺口，报告时必须写成 tracked platform gap。
- 本节定义的“C++ 全量”只包含 C++ build + C++ CTest；Python package 自身测试不属于这个入口。

### 3.5 Benchmark smoke / compare

```bash
cd /mnt/workspace/gitCode/atb_llm_study
python atb_cpp_llm/scripts/gen_compare_tokens.py
./atb_cpp_llm/build/benchmark --mode text
./atb_cpp_llm/build/benchmark --mode io
./atb_cpp_llm/build/benchmark --mode mm
./atb_cpp_llm/build/benchmark --mode compare
./atb_cpp_llm/build/benchmark --mode all
./atb_cpp_llm/build/benchmark --mode cold
./atb_cpp_llm/build/benchmark --mode throughput
```

验收：每个 mode 退出码 0、human-readable 表正常、`BENCH_RESULT` 行正常；`compare` mode 至少确认 C++ 侧 `/tmp/cpp_*.bin` 和中间 pixel_values 能正常生成。

注意：`benchmark --mode compare` 是 C++ benchmark 的对外二进制冒烟入口，不是 official gate。它不在 CTest，也不属于 `build_and_test.sh` 的覆盖范围；提交 benchmark 相关改动前必须手动跑。

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
   - 如果声称跑了 C++ 全量，是否确认 `build_and_test.sh` 没有 fallback 到 `--no-refdata`、没有排除 `needs_refdata`？

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

1. **AA：310P 用 small-op AA 拼装替代 aclnn AA**
   - `aclnnUpsampleBicubic2dAA` 在 310P 上不支持（aclnnStatus=561103）。
   - 因官方链降采样带 AA，310P 改用 ATB small-op 拼装的 separable AA bicubic
     （`NpuBicubicResizeAASmallOp`，Linear×2+Transpose×2，见 STATUS §2.9 /
     optimization-roadmap P10-C），端到端 cos 与 910B aclnn AA 数值等价。
   - 分发（`qwen3vl_preprocess.cpp:438`）：降采样时 910B→aclnn AA、310P→small-op AA；
     非降采样→非 AA aclnn（跨平台）。310P 降采样不再降级 CPU / skip。

2. **official full gate skip 是 limitation，不是成功**
   - `test_engine_vs_official` 在非 910B 上 skip，应在状态/报告中写明“310P official full embedding gate 是待修平台 gap”。
   - official reference 生成器仍应全平台可运行；310P 上 NPU 官方链不支持时 fallback CPU 官方链。

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
