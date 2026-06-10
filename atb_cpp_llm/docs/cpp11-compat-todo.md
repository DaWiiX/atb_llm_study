# C++ 标准降级 — 完成报告

**目标**：把 `atb_cpp_llm/` 从 C++17 降到尽可能低的版本，以扩大可部署的工具链范围。

**最终结果**（截至 2026-06-10）：**C++14**（无法再低）。

ATB SDK 自带头文件 `atb/svector.h` 使用了 C++14 才允许的"扩展 constexpr 构造函数"
（构造函数体内含循环），这是不可控的第三方约束。**C++11 不可行**，除非更换 SDK。

构建与测试通过：`bash atb_cpp_llm/build_and_test.sh` → Build succeeded，
**49/50 ctest 通过**，与降级前基线一致（唯一失败的 `test_vision_stages`
是 `/tmp/stage_L2_pos_embed_cpu.bin` 参考数据缺失导致的 SKIP 误计入失败，
非语法/语义退步；数值检查 cosine ≥ 0.999997）。

---

## 已完成改动一览

### 1. CMakeLists.txt
- `set(CMAKE_CXX_STANDARD 17)` → `set(CMAKE_CXX_STANDARD 14)`
- 保留 `set(CMAKE_CXX_STANDARD_REQUIRED ON)` —— 编译器会拒绝任何 C++15+ 语法，
  这是验收的硬约束。

### 2. 新增 shim 头文件：`src/util/cpp11_compat.h`

提供 C++14+ 才有的若干工具的 shim（命名空间 `atb_llm::`），让代码即使将来
再降到 C++11 也能用。实际目前在 C++14 模式下编译，shim 与标准库等价但不污染 std。

| Shim | 替代 | 处数 |
|---|---|---|
| `atb_llm::make_unique<T>(args...)` | `std::make_unique` (C++14) | 19 |
| `atb_llm::exchange(obj, new_value)` | `std::exchange` (C++14) | 6 |
| `atb_llm::clamp(v, lo, hi)` | `std::clamp` (C++17) | 2 |

### 3. C++17 结构化绑定全部展开（57 处）

原模式：`auto [a, b] = expr;` （全部从 `std::pair` 解构）

改写为：
```cpp
auto __atb_pair_a = expr;
auto& a = __atb_pair_a.first;
auto& b = __atb_pair_a.second;
```

55 处用 `perl -i -pe` 一行命令批量替换，2 处多行调用手工改写
（`tests/level2_op_precision/test_rms_norm_precision.cpp`）。

涉及 30 个文件，主要在 `tests/level*/` 下；`src/` 命中 2 个文件
（`families/base_model.cpp`、`components/common/deepstack_fusion.cpp`）。

### 4. C++14 泛型 lambda 替换（10 处）

`tests/benchmark.cpp` 内：
```cpp
[](auto& t) { return t.preprocess_ms; }
        ↓
[](const atb_llm::StageTimings& t) { return t.preprocess_ms; }
```

（`MeanStage` 接受 `std::function<double(const atb_llm::StageTimings&)>`，
泛型 lambda 实际推导的就是这个类型。）

### 5. 杂项

- `src/core/tensor_allocator.h`：给嵌套结构 `Allocation` 加显式构造函数
  `Allocation(void*, size_t)`，让 `unordered_map::emplace(key, Allocation{a, b})` 这种
  含 brace-init-list 转发的调用在 C++14 之前也能编（虽然 C++14 模式不强制
  需要，但显式构造函数语义更清晰，也兼容更低标准）。

---

## TODO 文档（这份的前身）漏掉的项

原 TODO 声明 "0 处" 的几项实际不为 0，是降级编译过程中被编译器主动暴露的：

| 项目 | 原 TODO 声明 | 实际 |
|---|---|---|
| `std::exchange` (C++14) | 未提及 | 6 处（`src/core/raii.h`） |
| `std::clamp` (C++17) | 未提及 | 2 处（`qwen3vl_preprocess.cpp`） |
| 结构化绑定 | 0 处 | **57 处** |
| `auto [...]` 形式聚合初始化 + emplace 兼容 | 未提及 | 1 处（`tensor_allocator`） |

这印证了 TODO 文档结尾的话："把 `CMAKE_CXX_STANDARD` 降下来之后，
编译器会主动拒绝任何不合规语法 —— 这就是验收手段。"

## 如果将来再升级到 C++17/20

shim 仍然可用、零成本，无需回退。如果想拥抱原生标准库：

1. `s/atb_llm::make_unique/std::make_unique/g` over `src/ tests/`
2. `s/atb_llm::exchange/std::exchange/g`（仅 `src/core/raii.h`）
3. `s/atb_llm::clamp/std::clamp/g`（仅 `qwen3vl_preprocess.cpp`）
4. 结构化绑定可选择性还原（`__atb_pair_x` 模式 → `auto [a, b] = ...`），
   但功能等价，没必要批量改回。
5. 删 `src/util/cpp11_compat.h` 并清理 `#include "util/cpp11_compat.h"`。

## 参考

- ATB SDK 强制 C++14 的来源：`atb/svector.h:48-53`（扩展 constexpr 构造）
- shim 实现参考 N3656（make_unique 提案）和 cppreference 标准实现
