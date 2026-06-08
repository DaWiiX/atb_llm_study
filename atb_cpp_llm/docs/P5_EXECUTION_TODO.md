# P5 Execution TODO — Deepstack 端到端 NPU 化

> **架构师工作台**——记录每个原子任务的状态、责任人 agent、验证记录。

---

## 🎯 架构师职责（不可遗忘）

**身份**: Architect。**禁止**：亲自写代码、亲自跑测试、亲自做 git 操作。
**职责**：
1. 把 P5 拆成尽可能小的、可独立验证的原子任务（每次一个接口 / 一个函数）。
2. 把每个原子任务派发给 `coder` subagent（用 Agent 工具，subagent_type=`general-purpose`，prompt 中明确「你只是 coder，只做这一件事」）。
3. coder 完成后，把同一个改动派发给 `reviewer` subagent（独立的 agent 实例），让它：
   - 编写/补充对应级别的测试（L1/L2/L3，遵循 `docs/TEST_STRATEGY_GUIDE.md`）
   - 跑测试
   - 检查改动是否真正符合需求 + 是否引入回归
4. 如果 reviewer 报告问题，把问题派发给 coder 修复，**不要**自己改。
5. 每完成/失败一个任务，更新本 TODO 表格（status / 经办 agent / 备注）。
6. **测试要求**（写进每个 reviewer prompt）：
   - 不可过于宽松；阈值要有理论依据（fp16 round-to-nearest-even 等），不可硬编码神奇数字
   - 数据从简单到复杂（small / medium / large 三档）
   - 覆盖边界条件（空输入、最小尺寸、最大尺寸、单 token、多 token 重复位置等）
7. **不可**通过降低精度阈值"通过"测试。任何 cosine < 0.99 都要让 coder 修根因。

---

## 📋 任务清单

| ID | 描述 | Status | 经办 | 备注 |
|----|------|--------|------|------|
| T1 | 修改 `src/components/common/cross_modal_fusion.h` 接口签名：`std::vector<std::vector<uint16_t>>&` → `std::vector<NpuTensor>&`；`const std::vector<uint16_t>&` → `const NpuTensor&` | done | coder:a3fd042987676b457 | - |
| T1-R | Reviewer 验证 T1：编译通过（只编 header dependent 部分），interface 语义自洽 | done | reviewer:a71baf4ab78c92312 | ✅ NpuTensor noexcept move 语义保证 vector 扩容安全 |
| T2 | 修改 `src/components/common/deepstack_fusion.h` 匹配新接口签名 | done | coder:ab9f1e831e96928d7 | - |
| T2-R | Reviewer 验证 T2 | done | reviewer:ae5750644d3771eba | ✅ PASS |
| T3 | 修改 `deepstack_fusion.cpp::ExtractFeatures`：移除 `Synchronize` + `CopyToHost`，直接保留 `ds_out` (NpuTensor) move 到 ds_features[fusion_idx] | done | coder:af376bade7399bd54 | sync/CopyToHost 已删 |
| T3-R | Reviewer 验证 ExtractFeatures：dangling reference + move 释放安全 | done | reviewer:ada834dc8685c1510 | ✅ PASS（VariantPack 析构无 RAII、NpuTensor move-assign 正确释放旧资源） |
| T4 | 修改 `deepstack_fusion.cpp::InjectFeatures`：删除 `CopyToDevice(*upd_npu.Get(), ds_feat.data(), ...)`，改为直接接收 NpuTensor 输入 | done | coder:ad42c4a51b421f2e1 | 用 *ds_feat.Get() 直接传入 IndexAdd |
| T4-R | Reviewer 深度验证 InjectFeatures + stream 同一性分析 | done | reviewer:a668d0d29436a7fdb | ✅ PASS — vision+text 共享同一 stream FIFO |
| T5 | 修改 `src/adapters/qwen3vl_embedding/qwen3vl_model.h`：`ds_features_` 类型 `std::vector<std::vector<uint16_t>>` → `std::vector<NpuTensor>` | done (no-op) | coder:a4d66fbe441d6282d | 发现 ds_features 不是成员变量而是函数参数；3 个函数签名 (RunVision/PrepareInputs/RunTextDecoder) 留给 T5b |
| T5b | 修改 qwen3vl_model.h 中 3 个函数签名 (RunVision/PrepareInputs/RunTextDecoder) 的 ds_features 参数类型 | done | coder:af5c61e4c180e1e99 | - |
| T5b-R | Reviewer 验证 3 个签名匹配新接口 | done | reviewer:a03c3829200134b10 | ✅ PASS |
| T5-R | Reviewer 检查头文件依赖正确 | done (skipped, T5 is no-op) | - | - |
| T6 | 修改 `qwen3vl_model.cpp` 5 个调用点（Forward / ForwardWithTiming / RunVision / RunTextDecoder / PrepareInputs） | done | coder:ad8c68921b2dff9b5 | 7 个 diff 块改完 5 个函数 |
| T6-R | 编译全工程并定位剩余编译错误 | done | reviewer:ab8f06ae1b7f19d62 | 2 个错误：test_deepstack.cpp 类型不匹配 (P5 引入)，test_accuracy.cpp 链接 (pre-existing) |
| T7 | Coder 修复 test_deepstack.cpp:190 调用签名（旧 host vector → NpuTensor） | done | coder:aa371e8b1e1cc860e | ds_feat → AllocNpuFloat16 + CopyToDevice |
| T7-R | 编译 + 运行 test_deepstack，验证通过 | done | reviewer:a24c91a99e44d3357 | ✅ 4/4 test cases PASS |
| T8 | 全量编译 + 运行 test_accuracy + Python 验证三模式 cosine ≥ 0.99 | done | reviewer:a7d61191f86fc6607 + 用户 | ✅ 1.000000/0.999489/0.999857 = P3 基线，无精度回归 |
| T8b | 调查 benchmark 两个问题：Preprocess=0.00ms、只跑了一个分辨率 | done | reviewer:a6340efbe9b390dd8 | preprocess_ms 从未被赋值；benchmark 只接受单个 --width/--height |
| T8c | 修复 benchmark：1) Preprocess 时间计入 StageTimings 2) 新增 `mode=bench` 自动跑 4 分辨率 | done | coder:afb7cd5bba5bc4ce0 | pre_ms 在 RunImageOnlyBenchmark + RunMultimodalBenchmark 中赋值 |
| T9 | Reviewer 编译 + 跑 `--mode bench` 4 分辨率 benchmark，对比 P3 基线 | done | reviewer:ae032871561544405 | 全部超越预期！896×896 -12.2% |
| T10 | 架构师更新 `docs/NPU_PIPELINE_OPTIMIZATION.md` P5 status=done + 实测收益表 | done | 架构师 | 全部文档已更新 |

---

## 📜 派发历史 / 决议记录

（架构师每次派发 / 收到结果后追加一行）

- 2026-06-08: 架构师创建 TODO，退出 plan mode，开始派发 T1
- 2026-06-08: T1 done + T1-R done（reviewer 确认 noexcept move 安全），派发 T2

