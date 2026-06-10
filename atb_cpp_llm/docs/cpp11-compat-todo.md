# C++11 兼容降级 TODO

**目标**：把 `atb_cpp_llm/` 从当前的 C++17（实际只用到 C++14 子集）降到 C++11。

**当前状态**（截至 2026-06-10）：
- ✅ 0 个 C++17-only 语法（`[[maybe_unused]]` 已全部改为 `(void)x;`）
- ❌ 仍依赖 C++14 的 `std::make_unique`（19 处）
- ❌ 仍依赖 C++14 的泛型 lambda `[](auto& t) {...}`（10 处，全在 `tests/benchmark.cpp`）
- ❌ `CMakeLists.txt` 里 `CMAKE_CXX_STANDARD` 仍是 `17`

## 改动清单

### 1. 把 19 处 `std::make_unique<X>(args...)` 改成 `std::unique_ptr<X>(new X(args...))`

C++11 没有 `std::make_unique`，必须手写 `unique_ptr` 的初始化。

**文件清单**（每处都是单行机械替换）：

| 文件 | 行号 |
|---|---|
| `src/io/safetensors_reader.cpp` | 8, 23 |
| `src/engine/llm_engine.cpp` | 72 |
| `src/engine/runtime_impl.cpp` | 21, 23, 24 |
| `src/adapters/qwen3vl_embedding/qwen3vl_model.cpp` | 113, 116, 132, 146, 180 |
| `src/adapters/qwen3vl_embedding/register.cpp` | 15 |
| `src/components/common/mlp_builder.cpp` | 70, 71, 72, 73 |
| `src/components/common/gqa_attention_builder.cpp` | 196, 197, 198 |
| `src/core/graph_builder.cpp` | 14 |

**模板替换**：

```cpp
// before (C++14)
auto x = std::make_unique<Foo>(a, b);
// after (C++11)
std::unique_ptr<Foo> x(new Foo(a, b));

// before (C++14, in return / assignment)
return std::make_unique<Foo>(a, b);
// after (C++11)
return std::unique_ptr<Foo>(new Foo(a, b));
```

也可以在工程内自定义一个 `make_unique` shim（很多 C++11 代码库这样做），但只为 19 处用不划算。

**注意**：如果原代码用了 `auto x = std::make_unique<Foo>(...)`，类型推导那一侧也得想清楚。`std::unique_ptr<Foo> x(new Foo(...))` 是最直白的；用 `auto x = std::unique_ptr<Foo>(new Foo(...))` 也行，看口味。

### 2. 改 `tests/benchmark.cpp` 的 10 处泛型 lambda

**行号**：228-234, 263-269

**改法**：把 `[](auto& t) { ... }` 改成具体类型，或者把 lambda 改成显式模板的 functor。最简洁做法是**用具体类型替代 `auto`**：

```cpp
// before (C++14)
double preprocess = MeanStage(results, [](auto& t) { return t.preprocess_ms; });

// after (C++11)
double preprocess = MeanStage(results, [](const StageTimings& t) { return t.preprocess_ms; });
```

`StageTimings` 是 `results` 元素的真实类型，要在 benchmark.cpp 里看一下 `results` 的声明对应替换。

### 3. CMakeLists.txt

```cmake
# 在 atb_cpp_llm/CMakeLists.txt:3-4 把
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 改成
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

### 4. 验证

```bash
cd atb_cpp_llm
bash build_and_test.sh --clean --no-test    # 确认能编译通过
bash build_and_test.sh                       # 跑完整 ctest，确认 49/50 通过率没退步
```

## 不需要动的部分（已确认 C++11 兼容）

| 检查项 | 结果 |
|---|---|
| `[[maybe_unused]]` / `[[nodiscard]]` 等 C++17 attribute | 已清零 |
| `if constexpr` | 0 处 |
| `<filesystem>` / `std::filesystem` | 0 处 |
| `<string_view>` / `std::string_view` | 0 处 |
| `<optional>` / `std::optional` | 0 处 |
| `<variant>` / `std::variant` | 0 处 |
| Structured bindings `auto [a, b] = ...` | 0 处 |
| Binary literals `0b101` / 数字分隔符 `1'000` | 0 处 |
| Lambda init capture `[x = expr]` | 0 处 |
| Variable templates / `template<auto>` | 0 处 |
| 第三方头文件（`utils/safetensors.hh`, `utils/cJSON.c`, ATB SDK headers） | 不受影响，已通过 SYSTEM include 屏蔽其内部要求 |

## 注意事项

- **`std::unique_ptr<T>` 本身是 C++11**，所以 `make_unique` 改成构造 + `new` 没风险。
- 改完之后**编译 warning 数量不会变**（warning 修复跟标准无关）。
- `gen_all.py` / `build_and_test.sh` 等脚本和 Python 端都跟 C++ 标准无关。
- 把 `CMAKE_CXX_STANDARD` 降到 11 之后，编译器会**主动拒绝**任何 C++14+ 语法 —— 这就是验收手段，构建过了就说明改干净了。

## 参考

- 项目目前在 `CMAKE_CXX_STANDARD 17` + `STANDARD_REQUIRED ON`（`atb_cpp_llm/CMakeLists.txt:3-4`）
- `gen_cpu_reference.py` 等 Python 生成器不受影响
- `[[maybe_unused]]` 降级到 `(void)x;` 已经在 commit 中完成
