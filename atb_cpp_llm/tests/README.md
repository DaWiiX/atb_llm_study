# Tests

按测试金字塔分层组织：

- `level0_framework/` — 基础框架测试（ContextManager, TensorAllocator, JsonConfig 等）
- `level1_cpu_pure/` — CPU 纯函数精度测试（不依赖 NPU）
- `level2_op_precision/` — ATB 算子/组件精度测试（NPU 必需）
- `level3_integration/` — 集成测试（多组件、Runner 完整流程）
- `level4_e2e/` — 端到端测试（完整推理 + Python 对比）
- `python_reference/` — Python reference 数据生成脚本
- `diagnostics/` — 调试和对比脚本

辅助文件（根目录）：

- `doctest.h` — doctest 头文件
- `test_env.h` — 共用的环境变量帮助函数（`GetModelDir()`）
- `benchmark.cpp` — 性能基准测试
