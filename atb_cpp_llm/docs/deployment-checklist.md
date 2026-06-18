# ATB C++ LLM Engine — 部署检查清单

本文档用于生产部署 Qwen3VL-Embedding-2B 推理引擎的逐项核查。每项 `[ ]` 需由操作员在变更窗口内确认并签署。

---

## 1. 部署前验证

### 1.1 硬件要求

| 项目 | 最低要求 | 验证方式 |
|------|---------|---------|
| NPU 型号 | Ascend 910B 或 Ascend 310P | `npu-smi info -t board` |
| NPU 显存 | >= 8 GiB (fp16 模型约 5 GiB) | `npu-smi info -t memory` |
| NPU 可用数量 | >= 1 | `npu-smi info -t board` (统计 "NPU ID" 行数) |
| 系统内存 | >= 32 GiB (含 safetensors 加载) | `free -h` |
| ECC 状态 | 正常 (无 uncorrected errors) | `npu-smi info -t health` |

- [ ] NPU 型号确认 (`910B` 或 `310P`)
- [ ] NPU 显存 >= 8 GiB 且当前可用 >= 6 GiB
- [ ] NPU 健康状态正常
- [ ] 系统内存 >= 32 GiB 可用

### 1.2 软件依赖

| 组件 | 位置 | 验证方式 |
|------|------|---------|
| CANN Toolkit | `~/Ascend/ascend-toolkit/` or `/usr/local/Ascend/ascend-toolkit/` | `source set_env.sh && npu-smi info` |
| ATB Runtime | `~/Ascend/nnal/atb/latest/atb/cxx_abi_1/` | `ls $ATB_BUILD_DEPENDENCY_PATH/lib/libatb.so` |
| ACL (AscendCL) | `~/Ascend/cann/` | `ls $ASCEND_TOOLKIT_DIR/lib64/libascendcl.so` |
| CMake | >= 3.16 | `cmake --version` |
| Python (仅参考数据/诊断) | >= 3.8 | `python3 --version` |

<!-- TODO: 确认最低 CANN/ATB 版本要求。CMakeLists.txt 中默认路径引用 9.0.0，但未明确声明为最低版本。
     确认方法：在多个 CANN/ATB 版本上运行完整测试套件，记录最早能通过的版本号。 -->

| 组件 | 已验证版本 | 说明 |
|------|-----------|------|
| CANN | <!-- TODO: 填写已验证版本 --> | `cat /usr/local/Ascend/ascend-toolkit/version.cfg` 或 `npu-smi info \| head -5` |
| ATB | <!-- TODO: 填写已验证版本 --> | `strings $ATB_BUILD_DEPENDENCY_PATH/lib/libatb.so \| grep -i "version\|build" \| head -3` |

**加载环境变量**（构建和执行都必须执行）：

```bash
source ~/Ascend/ascend-toolkit/set_env.sh
source ~/Ascend/cann/set_env.sh
source ~/Ascend/nnal/atb/latest/atb/set_env.sh --cxx_abi=1
source ~/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ATB_BUILD_DEPENDENCY_PATH=~/Ascend/nnal/atb/latest/atb/cxx_abi_1
```

> **非 root 用户**：使用 `~/Ascend/...` 路径。**root 用户**：使用 `/usr/local/Ascend/...` 路径。
>
> **重要**：`set_env.sh` 不仅是构建时需要 — 运行时也必须 source，因为它设置了 `LD_LIBRARY_PATH` 指向 ATB/ACL 动态库。如果未 source 即运行 benchmark，会得到 `libatb.so: cannot open shared object file` 错误。

- [ ] CANN Toolkit 已加载且 `npu-smi info` 正常
- [ ] ATB 动态库 (`libatb.so`) 存在
- [ ] ACL 动态库 (`libascendcl.so`) 存在
- [ ] 运行时 `LD_LIBRARY_PATH` 已确认包含 ATB 和 ACL 库路径（`source set_env.sh` 后生效）

### 1.3 模型检查点完整性

```bash
MODEL_DIR=/path/to/Qwen3-VL-Embedding-2B

# 验证必需文件存在
ls -lh "$MODEL_DIR/config.json" \
      "$MODEL_DIR/preprocessor_config.json" \
      "$MODEL_DIR/model.safetensors"

# 验证 config.json 模型类型
python3 -c "
import json
with open('$MODEL_DIR/config.json') as f:
    cfg = json.load(f)
assert cfg['model_type'] == 'qwen3vl_embedding', \
    f\"Expected qwen3vl_embedding, got {cfg.get('model_type')}\"
print('model_type:', cfg['model_type'])
print('hidden_size:', cfg['hidden_size'])
print('num_hidden_layers:', cfg['num_hidden_layers'])
print('vision_config present:', 'vision_config' in cfg)
"

# 验证 safetensors 文件可读且大小合理 (> 4 GiB)
STAT_SIZE=$(stat -c%s "$MODEL_DIR/model.safetensors" 2>/dev/null)
if [ "$STAT_SIZE" -lt 4000000000 ]; then
    echo "WARNING: model.safetensors appears too small ($STAT_SIZE bytes)"
else
    echo "model.safetensors size: $STAT_SIZE bytes (OK)"
fi
```

- [ ] `config.json` 存在且 `model_type == "qwen3vl_embedding"`
- [ ] `preprocessor_config.json` 存在
- [ ] `model.safetensors` 存在且 >= 4 GiB
- [ ] 文件 MD5 校验值与发布版本一致（如有发布记录）

### 1.4 构建验证

```bash
cd atb_cpp_llm
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target benchmark -j$(nproc)

# 验证二进制可以加载模型并执行单次推理
QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./benchmark --mode all --iter 1 --warmup 0
# 预期: 6 阶段输出正常，exit code 0
```

- [ ] `cmake --build` 无错误退出
- [ ] `benchmark` 二进制成功生成 (`build/benchmark` 存在且可执行)
- [ ] `QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./benchmark --mode all --iter 1` 推理成功
- [ ] 验证生成的 `.so` 链接到正确的 ATB 版本：
  ```bash
  ldd build/libatb_llm_engine.so | grep atb
  ```

---

## 2. 运行时配置

### 2.1 环境变量

创建或编辑 `.env` 文件（仓库根目录）：

```bash
# .env — 部署环境配置
QWEN3VL_EMB_MODEL_DIR=/path/to/Qwen3-VL-Embedding-2B
ASCEND_PLATFORM=910B          # 910B 或 310P
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN3VL_EMB_MODEL_DIR` | （必填） | 模型检查点目录路径 |
| `ASCEND_PLATFORM` | `910B` | `910B` (Atlas A2) 或 `310P` (Atlas 推理) |
| `ASCEND_DEVICE_ID` | `0` | 多卡时指定 NPU 设备 ID |
| `LOG_LEVEL` | `2` (WARN) | C++ 日志级别：0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, >=4 (含未定义值) 全部映射为 NONE (静默) |

> **310P 注意**：`ASCEND_PLATFORM=310P` 时，mask 格式自动切换为 FRACTAL_NZ。这是平台差异的**唯一**运行时可配置差异。GQA 在 310P 上原生支持，无需额外配置。

- [ ] `.env` 文件已创建，`QWEN3VL_EMB_MODEL_DIR` 指向正确的模型路径
- [ ] `ASCEND_PLATFORM` 与硬件一致（`npu-smi info` 确认）
- [ ] 多卡环境 `ASCEND_DEVICE_ID` 正确配置
- [ ] `LOG_LEVEL` 已审查（生产建议 `WARN` 或 `ERROR`）
- [ ] `.env` 已备份：`cp .env .env.bak.$(date +%Y%m%d)`（任何修改前执行）

### 2.2 ATB Buffer Pool 大小

Buffer pool 通过 `EngineConfig::buffer_size` 控制（单位：字节）。0 表示自动。

| 场景 | 推荐 buffer_size (字节) | 说明 |
|------|-----------------|------|
| 单实例，短文本 (S <= 512) | `2 GiB` (`2147483648`) | 低负载 |
| 单实例，长文本 (S <= 4096) | `5 GiB` (`5368709120`) | 正常负载 |
| 高并发 or 视觉+文本混用 | `10 GiB` (`10737418240`) | 重负载 |

**Python 侧**：

```python
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
set_atb_buffer_size(5 * 1024 * 1024 * 1024)  # 5 GiB (字节精确值)
```

**C++ 侧**：

```cpp
atb_llm::EngineConfig config;
config.buffer_size = 5LL * 1024 * 1024 * 1024;  // 5 GiB (字节精确值)
```

> 设置过小会导致 `ERROR_NPU_MEMORY` 或 Setup 阶段失败。设置过大浪费系统内存，但不会导致错误。buffer pool 按需扩容，`buffer_size` 只是上限。

- [ ] `buffer_size` 已根据预期负载设置
- [ ] NPU 显存总用量确认：模型权重 (~4 GiB) + buffer pool 上限 + 预留

### 2.3 日志配置

**C++ 引擎日志**（stderr 输出）：

```bash
export LOG_LEVEL=2   # WARN 级别（生产建议）
```

日志格式：`[YYYY-MM-DD HH:MM:SS][LEVEL][filename:line] message`

**ATB 内部日志**（文件输出，ATB 框架自动管理）：

| 平台 | 日志路径 |
|------|---------|
| root 用户 | `/root/atb/log/` |
| 普通用户 | `~/ascend/log/atb/` (C++) 或 `/root/ascend/log/atb/` (Python) |

> ATB 日志按时间命名（如 `atb_20260616_143052.log`），不会自动轮转。建议配置 cron job 定期清理（见 5.1 节）。

- [ ] `LOG_LEVEL` 环境变量已设置（生产建议 2）
- [ ] ATB 日志目录存在且可写
- [ ] 日志轮转策略已配置（见 5.1）

### 2.4 内存限制与 OOM 防护

- **NPU 显存**：模型权重约 4 GiB，buffer pool 按需分配。单次推理峰值约为权重 + workspace（通常 < 2 GiB）。
- **系统内存**：safetensors 加载需要将整个模型文件（~4.5 GiB）映射到内存，加上 NPU 侧拷贝。
- **OOM 信号**：
  - C++：`ERROR_NPU_MEMORY`（错误码 -5）或 `ERROR_ATB_BASE + aclError`（错误码 < -1000）
  - Python：`torch_npu` 抛 `TORCH_NPU_ERROR`

**OOM 排查**：

```bash
# 实时监控 NPU 内存
watch -n 1 npu-smi info -t memory

# 查看 ATB 日志中的错误
cat ~/ascend/log/atb/$(ls -rt ~/ascend/log/atb/ | tail -n 1)
```

- [ ] NPU 内存监控已就绪（`npu-smi info -t memory`，无需 `-i`，默认显示所有卡）
- [ ] 系统 OOM killer 配置已审查（避免 kill 关键进程）
- [ ] 告警：NPU 内存使用 > 80% 时出发通知

---

## 3. 健康检查

### 3.1 启动健康检查

部署后首次运行前，执行完整健康检查：

```bash
cd atb_cpp_llm

# Step 1: 运行完整基线验证（13 种输入组合）
bash scripts/verify_baseline.sh

# 预期: 所有 13 个组合的 cosine >= 0.99
# 如果任何一项 FAIL，禁止继续部署 — 优先排查
```

`verify_baseline.sh` 覆盖 13 种组合：

| 模式 | 组合 | 说明 |
|------|------|------|
| TEXT_ONLY | S=100, 512, 1024, 2048, 4096 | 纯文本，不同序列长度 |
| IMAGE_ONLY | 416x672, 720x1280, 1080x1920, 1440x2560 | 纯图片，不同分辨率 |
| MULTIMODAL | 416x672, 720x1280, 1080x1920, 1440x2560 | 图文混合 |

**验收标准**：所有 13 个组合 cosine >= 0.99，`BASELINE PASS`。

如果使用独立 C++ 二进制（非 Python 封装），手动执行最小健康检查：

```bash
# 单次推理精度验证 (compare 模式覆盖 13 种输入组合)
QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./benchmark --mode compare --iter 1 --warmup 0
# 检查输出中 BENCH_RESULT 行、exit code 0
```

- [ ] `verify_baseline.sh` 全部 `PASS`
- [ ] 所有 cosine >= 0.99，无 FAIL
- [ ] 最低 cosine 值记录在案：`_______`

### 3.2 运行时健康检查

**定期验证**（建议每小时一次 cron job）：

```bash
#!/bin/bash
# /etc/cron.hourly/atb_health_check.sh

MODEL_DIR="${QWEN3VL_EMB_MODEL_DIR:-/path/to/model}"

# 运行单次全模式推理，检查 exit code
cd /path/to/atb_cpp_llm/build
QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./benchmark --mode all --iter 1 --warmup 0 > /tmp/health_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "$(date): HEALTH_OK"
else
    echo "$(date): HEALTH_FAIL"
fi
exit $?
```

**监控指标**：

| 指标 | 告警阈值 | 检查方式 |
|------|---------|---------|
| 推理延迟 P99 | > 2x baseline | 应用层记录 |
| NPU 温度 | > 85°C | `npu-smi info -t temp` |
| NPU 内存可用 | < 2 GiB | `npu-smi info -t memory` |
| ECC 错误 | 任何 uncorrectable | `npu-smi info -t health` |

- [ ] 定期健康检查 cron job 已配置
- [ ] 延迟、温度、内存告警已接入监控系统
- [ ] 健康检查失败时的告警通知渠道已确认

### 3.3 延迟基准

使用 `benchmark` 获取服务基线延迟：

```bash
cd atb_cpp_llm/build

# 全面基准测试（text + image-only + multimodal）
QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./benchmark --mode all --iter 10 --warmup 3

# 或完整对比矩阵（13 种输入组合，含 5 种 S 长度的 TEXT_ONLY + 4 种分辨率 IO/MM）
QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./benchmark --mode compare --iter 5 --warmup 2
```

典型延迟参考值（Qwen3VL-Embedding-2B, fp16, Ascend 910B, 单卡）：

| 模式 | 输入规格 | 预期延迟 | 说明 |
|------|---------|---------|------|
| TEXT_ONLY | S=512 | ~15-30 ms | 纯文本编码 |
| TEXT_ONLY | S=2048 | ~60-120 ms | 长文本编码 |
| TEXT_ONLY | S=4096 | ~150-300 ms | 超长文本编码 |
| IMAGE_ONLY | 416x672 | ~20-50 ms | 小图视觉编码 |
| IMAGE_ONLY | 1440x2560 | ~80-200 ms | 大图视觉编码 |
| MULTIMODAL | 720x1280 + 文本 | ~100-250 ms | 图文联合编码 |

> 延迟受 sequence length、图像分辨率、NPU 型号 (910B vs 310P) 和 ATB 版本影响。以实际 `verify_baseline.sh` 输出为基准。

- [ ] 基准延迟已记录（attach 本次 `benchmark --mode all` 输出）
- [ ] P50 / P95 / P99 延迟已写入 SLA 文档

---

## 4. 优雅关闭

### 4.1 关闭流程

```
1. 停止接受新请求
2. 等待所有正在处理的请求完成（drain timeout）
3. 销毁 LLMEngine 实例（触发 RAII 析构链）
4. 释放 NPU 资源
5. 退出进程
```

### 4.2 信号处理

建议在应用层注册以下信号处理器：

| 信号 | 行为 |
|------|------|
| `SIGTERM` | 优雅关闭：停止 accept，等待 drain，析构 engine |
| `SIGINT` | 同 SIGTERM |
| `SIGQUIT` | 优雅关闭 + 打印 backtrace（调试用） |
| `SIGKILL` | **无法捕获** — 内核直接终止进程，跳过所有清理。仅作为 final fallback，不应主动使用 |

### 4.3 Drain 超时

```cpp
// 伪代码：应用层关闭逻辑
void GracefulShutdown(int signum) {
    // 1. 设置 draining 标志 — 拒绝新请求
    SetDraining();
    
    // 2. 等待 in-flight 请求完成（超时 30s）
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (InFlightCount() > 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            LOG_WARN("Drain timeout — %d requests still in-flight, forcing shutdown",
                     InFlightCount());
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // 3. 销毁 engine（RAII 释放 NPU 资源）
    engine_.reset();  // 析构顺序：
                      //   LLMEngine::Impl → IModel → OperationHandle* → 
                      //   TensorAllocator::FreeAll() → BufferPool::Free() →
                      //   ContextManager (DestroyContext → DestroyStream → aclFinalize)
    
    LOG_INFO("Engine destroyed, NPU resources released.");
    exit(0);
}
```

### 4.4 RAII 析构链

C++ engine 通过 RAII 自动释放资源，无需手动逐项释放：

| 析构顺序 | 组件 | 释放的资源 |
|---------|------|-----------|
| 1 | `OperationHandle` (各图算子) | `atb::DestroyOperation()` |
| 2 | `TensorAllocator::FreeAll()` | 所有 NPU tensor (`aclrtFree`) |
| 3 | `BufferPool::Free()` | workspace 内存 |
| 4 | `ContextManager` 析构 | `atb::DestroyContext()` + `aclrtDestroyStream()` + `aclFinalize()` |

**Python 侧**等价于 `del engine` 或退出 `with` 块。

- [ ] 信号处理器已注册（SIGTERM / SIGINT）
- [ ] Drain 超时已配置（建议 30s）
- [ ] 关闭日志已确认 engine 析构成功（无 NPU 资源泄漏）

---

## 5. 操作流程

### 5.1 日志轮转

ATB 内部日志不自动轮转。配置定期清理：

```bash
# /etc/cron.daily/atb_log_rotate
#!/bin/bash
# 保留最近 7 天的 ATB 日志
ATB_LOG_DIR="${HOME}/ascend/log/atb"
if [ -d "$ATB_LOG_DIR" ]; then
    find "$ATB_LOG_DIR" -name "atb_*.log" -mtime +7 -delete
fi
# Python 侧日志目录同理
ATB_PY_LOG_DIR="/root/ascend/log/atb"
[ -d "$ATB_PY_LOG_DIR" ] && find "$ATB_PY_LOG_DIR" -name "*.log" -mtime +7 -delete
```

- [ ] ATB 日志清理 cron job 已配置（建议保留 7 天）
- [ ] 日志目录磁盘使用监控已接入报警（> 10 GiB 时通知）
- [ ] `/tmp` 磁盘空间监控已就绪：benchmark `compare` 模式将中间产物写入 `/tmp/`（token 文件、pixel_values、pooler 输出），累计可达数百 MB。建议 `df -h /tmp` 纳入定期检查。

### 5.2 模型权重更新

```
1. 准备新权重目录 /path/to/Qwen3-VL-Embedding-2B-v2/
2. 验证文件完整性（1.3 节步骤）
3. 运行 verify_baseline.sh 验证新权重精度
4. 信号旧进程 SIGTERM，等待优雅关闭
5. 更新 .env 的 QWEN3VL_EMB_MODEL_DIR 指向新目录
6. 启动新进程
7. 启动健康检查确认正常
```

> 权重更新不要求重新编译 C++ 代码。模型结构不变时，二进制兼容。

- [ ] 新权重已通过 verify_baseline.sh
- [ ] 旧进程已优雅关闭
- [ ] 新进程健康检查通过

### 5.3 回滚到先前版本

```
1. 将 .env 的 QWEN3VL_EMB_MODEL_DIR 指回旧权重目录
2. 或从备份恢复 model.safetensors（已有校验和）
3. 重启进程
4. 健康检查确认正常
```

- [ ] 旧权重备份已保留（至少保留最近 2 个版本的 model.safetensors 校验和）
- [ ] .env 已回滚，健康检查通过

### 5.4 代码更新（C++ engine 升级）

```
1. git pull（拉取新代码）
2. cd atb_cpp_llm/build && cmake --build . -j$(nproc)
3. 运行完整测试套件：bash build_and_test.sh
4. 运行 verify_baseline.sh 确认精度无回归
5. 信号旧进程 SIGTERM，等待优雅关闭
6. 启动新进程
7. 健康检查确认正常
```

- [ ] 新代码 `build_and_test.sh` 全部通过
- [ ] `verify_baseline.sh` 全部 PASS
- [ ] 旧进程已优雅关闭，新进程正常

### 5.5 事件响应

| 症状 | 第一步诊断 | 参考章节 |
|------|-----------|---------|
| 推理延迟突然升高 | `npu-smi info -t temp` | 3.2 — 检查是否限频 |
| 推理延迟持续升高 | `npu-smi info -t memory` | 2.4 — 检查内存是否接近上限 |
| Cosine 精度下降 | `verify_baseline.sh` | 3.1 — 对比基准精度 |
| `ERROR_NPU_MEMORY` | `npu-smi info -t memory` | 2.4 — NPU OOM |
| `ERROR_GRAPH_BUILD` | 检查 S 值是否在缓存范围内 | 2.2 — buffer pool 大小不足 |
| ATB 错误 (code < -1000) | 查看 ATB 日志：`cat ~/ascend/log/atb/$(ls -rt ~/ascend/log/atb/ \| tail -n 1)` | 2.3 — ATB 内部错误 |
| `TransdataOperation ... infer shape fail` | 仅在 310P 上出现；检查 mask 是否为 NZ format | platform-310p.md — mask format 错误 |
| Setup 失败 (error 4) | tensor shape/format 不匹配；检查 ATB 日志 | design.md §4.5.6 — VariantPack 约定 |

**升级路径**：

1. 应用层告警触发
2. 检查症状表，执行对应第一步诊断
3. 如果 5 分钟内无法定位，回滚到上一个已知良好版本（5.3）
4. 收集诊断信息（ATB 日志 + `npu-smi info` 完整输出 + 项目日志）
5. 向开发团队反馈

- [ ] 值班人员知道如何查看 ATB 日志
- [ ] 回滚 SOP 已打印/书签化
- [ ] 升级联系人已明确

---

## 6. 容量规划

### 6.1 单卡吞吐量

| 指标 | 910B 参考值 | 310P 参考值 | 说明 |
|------|-----------|-----------|------|
| 最大输入序列长度 | 4096 | 4096 | 受 Attention 图缓存限制 |
| 最大图像分辨率 | 2560x1440 (approx) | 2560x1440 | 受 preprocessor `max_pixels` 限制 |
| 模型权重显存占用 | ~4 GiB (fp16) | ~4 GiB (fp16) | safetensors 加载后 |
| buffer pool 峰值 | 2-10 GiB (可配置) | 2-10 GiB (可配置) | 取决于 S 和视觉输入大小 |
| 单次推理延迟 (TEXT, S=512) | ~15-30 ms | ~20-40 ms | 参考值，以实际基准为准 |
| 单次推理延迟 (MULTIMODAL, 720p) | ~100-200 ms | ~120-250 ms | 参考值，以实际基准为准 |

> 吞吐量受 sequence length 和图像分辨率影响显著。以 `benchmark --mode all` 输出的均值延迟为规划基准。

### 6.2 并发考虑

当前 C++ engine **不支持并发 Forward 调用**（单 stream，无请求队列）。并发方案：

| 方案 | 适用场景 | 实现方式 |
|------|---------|---------|
| 单实例串行 | QPS < 10 | 应用层串行调用 `Encode()` |
| 多实例（每卡一个） | QPS 10-50 | 每 NPU 卡一个进程，应用层负载均衡 |
| 多卡并行 | QPS > 50 | 每卡一个实例 + 应用层分发 |

**多实例配置示例**（2 卡）：

项目仅提供 `libatb_llm_engine.so` 共享库和 `benchmark` 命令行工具。需要将 engine 库嵌入到带有网络接口的自定义服务进程中。多卡部署时每个进程绑定不同 `ASCEND_DEVICE_ID`：

```bash
# Instance 0 (自定义 server，使用 EngineConfig::device_id = 0)
ASCEND_DEVICE_ID=0 QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./my_server --port 8080 &
# Instance 1
ASCEND_DEVICE_ID=1 QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" ./my_server --port 8081 &
```

> 不同 `ASCEND_DEVICE_ID` 的实例完全独立 — 无显存共享，无 mutual exclusion。

- [ ] 预期 QPS 已评估
- [ ] 并发方案已选定（单实例 / 多实例 / 多卡）
- [ ] 负载均衡配置已就绪

### 6.3 扩展指引

| 当前 QPS | 目标 QPS | 扩展方案 |
|---------|---------|---------|
| < 10 | < 10 | 单实例串行 |
| < 10 | 10-50 | 增加 NPU 卡，每卡一实例 |
| 10-50 | > 50 | 增加 NPU 卡，每卡一实例 + 应用层分发 |
| > 50 | > 100 | 考虑多机部署 + 共享模型存储 |

---

## 附录 A：部署检查清单汇总

以下为部署窗口期间需逐项确认的全部项目（与正文各节一一对应）。

### A.1 部署前验证

- [ ] 1.1 硬件：NPU 型号、显存、健康状态均正常
- [ ] 1.2 软件：CANN / ATB / ACL 版本兼容，环境脚本可正常 source
- [ ] 1.3 模型：`config.json` + `model.safetensors` 完整性通过
- [ ] 1.4 构建：`cmake --build` 成功，`QWEN3VL_EMB_MODEL_DIR=... ./benchmark --mode all --iter 1` 通过

### A.2 运行时配置

- [ ] 2.1 环境变量：`.env` 文件正确配置 `QWEN3VL_EMB_MODEL_DIR` 和 `ASCEND_PLATFORM`
- [ ] 2.2 Buffer Pool：`buffer_size` 已根据预期负载设置（建议 5-10 GiB）
- [ ] 2.3 日志：`LOG_LEVEL` 设为 WARN (2)，ATB 日志轮转已配置
- [ ] 2.4 OOM 防护：NPU 内存监控已就绪，告警阈值已配置

### A.3 健康检查

- [ ] 3.1 启动检查：`verify_baseline.sh` 全部 PASS (cosine >= 0.99)
- [ ] 3.2 运行时检查：定期健康检查 cron job 已配置
- [ ] 3.3 延迟基准：已记录并写入 SLA 文档

### A.4 优雅关闭

- [ ] 4.2 信号：SIGTERM / SIGINT 处理器已注册
- [ ] 4.3 Drain：超时已配置（建议 30s）
- [ ] 4.4 析构：RAII 链确认，关闭日志无 NPU 资源泄漏

### A.5 操作流程

- [ ] 5.1 日志：ATB 日志清理 cron job 已配置（保留 7 天）
- [ ] 5.2 权重更新：SOP 已确认
- [ ] 5.3 回滚：旧权重备份已保留
- [ ] 5.5 事件响应：值班 SOP 已分发，ATB 日志路径已知

### A.6 容量规划

- [ ] 6.2 并发：方案已选定，QPS 预期已评估

---

## 附录 B：快速诊断脚本

部署到 `/usr/local/bin/atb_diag.sh`：

```bash
#!/bin/bash
# atb_diag.sh — 一键诊断 ATB 推理引擎环境
set -euo pipefail

echo "=== NPU Info ==="
npu-smi info -t board 2>&1 | head -15

echo ""
echo "=== NPU Memory ==="
npu-smi info -t memory 2>&1 | head -10

echo ""
echo "=== NPU Temperature ==="
npu-smi info -t temp 2>&1

echo ""
echo "=== NPU Health ==="
npu-smi info -t health 2>&1

echo ""
echo "=== ATB Libraries ==="
for lib in ~/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so \
           /usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so; do
    if [ -f "$lib" ]; then
        echo "ATB: $lib"
        strings "$lib" 2>/dev/null | grep -iE "version|build" | head -3
        break
    fi
done

echo ""
echo "=== Latest ATB Log ==="
# Note: ATB log directory differs by user — see section 2.3
for d in ~/ascend/log/atb /root/atb/log; do
    if [ -d "$d" ]; then
        latest=$(ls -rt "$d" 2>/dev/null | tail -n 1)
        if [ -n "$latest" ]; then
            echo "Latest ATB log: $d/$latest"
            echo "Last 10 lines:"
            tail -10 "$d/$latest" 2>/dev/null
        fi
        break
    fi
done

echo ""
echo "=== Disk Usage ==="
for d in ~/ascend/log/atb /root/atb/log /tmp; do
    if [ -d "$d" ]; then
        du -sh "$d" 2>/dev/null
    fi
done

echo ""
echo "=== Environment ==="
echo "ASCEND_PLATFORM=${ASCEND_PLATFORM:-unset}"
echo "QWEN3VL_EMB_MODEL_DIR=${QWEN3VL_EMB_MODEL_DIR:-unset}"
echo "LOG_LEVEL=${LOG_LEVEL:-unset}"
echo "ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-unset}"
echo "ATB_BUILD_DEPENDENCY_PATH=${ATB_BUILD_DEPENDENCY_PATH:-unset}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

echo ""
echo "=== Model Files ==="
if [ -n "${QWEN3VL_EMB_MODEL_DIR:-}" ] && [ -d "$QWEN3VL_EMB_MODEL_DIR" ]; then
    ls -lh "$QWEN3VL_EMB_MODEL_DIR/config.json" \
          "$QWEN3VL_EMB_MODEL_DIR/preprocessor_config.json" \
          "$QWEN3VL_EMB_MODEL_DIR/model.safetensors" 2>&1
fi
```

---

## 附录 C：ATB 错误码速查

| 错误码 | 宏名 | 含义 |
|--------|------|------|
| 0 | `STATUS_OK` | 成功 |
| -1 | `ERROR_INVALID_PARAM` | 无效参数（model_dir 不存在或为空） |
| -2 | `ERROR_FILE_NOT_FOUND` | 文件未找到（config.json / model.safetensors 缺失） |
| -3 | `ERROR_WEIGHT_LOAD` | 权重加载失败（safetensors 损坏或 key 不匹配） |
| -4 | `ERROR_GRAPH_BUILD` | 图构建失败（ATB CreateOperation / Build 错误） |
| -5 | `ERROR_NPU_MEMORY` | NPU 内存不足（aclrtMalloc 失败） |
| -6 | `ERROR_INFERENCE` | 推理执行失败（Setup / Execute 错误） |
| -7 | `ERROR_UNSUPPORTED` | 不支持的操作（平台/模型不兼容） |
| < -1000 | `ERROR_ATB_BASE + atb::Status` | ATB 内部错误（具体错误码见 ATB 文档） |

**常见 ATB 内部错误**：

| atb::Status | 含义 | 常见原因 |
|-------------|------|---------|
| 4 | ERROR_RT_FAIL | 运行时失败。通常是 tensor shape/format 不匹配或硬件限制 |
| 8 | ERROR_INVALID_TENSOR_DIM | 无效 tensor 维度。检查 VariantPack shape 是否与 Graph 定义一致 |
| 9 | ERROR_GRAPH_INFERSHAPE_FUNC_FAIL | 图形状推导失败。检查中间 tensor 连接是否正确 |
| 13 | ERROR_INVALID_TENSOR_INI_MATCH | tensor 初始化不匹配。检查 input tensor 数量是否与 Graph 一致 |
