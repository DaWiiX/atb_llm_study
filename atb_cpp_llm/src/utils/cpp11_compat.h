// SPDX-License-Identifier: Apache-2.0
//
// C++11 compatibility shims.
//
// 本工程编译标准为 C++11。本头文件提供 C++14+ 才有的若干轻量工具，
// 命名与 std:: 对齐但放在 atb_llm 命名空间下，避免污染 std。
//
// 使用方式：
//   #include "utils/cpp11_compat.h"
//   auto p = atb_llm::make_unique<Foo>(a, b);
//
// Provides C++14/17 backports for C++11 compatibility. Used project-wide
// instead of std::make_unique for consistency.

#pragma once

#include "atb_llm/operation_handle.h"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace atb_llm {

// 单对象版本：make_unique<Foo>(args...)
template <typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// 未知长度数组版本：make_unique<Foo[]>(n)
template <typename T>
typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0,
                        std::unique_ptr<T>>::type
make_unique(std::size_t n) {
    using U = typename std::remove_extent<T>::type;
    return std::unique_ptr<T>(new U[n]());
}

// 已知长度数组（make_unique<Foo[N]>()）：跟标准库行为一致，显式禁用。
template <typename T, typename... Args>
typename std::enable_if<std::extent<T>::value != 0, void>::type
make_unique(Args&&...) = delete;

// ─────────────────────────────────────────────────────────────
// std::clamp (C++17) 等价实现。
//
// 返回 v 被夹在 [lo, hi] 区间内的值。lo > hi 时行为未定义，
// 与标准库一致。
// ─────────────────────────────────────────────────────────────
template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
    return v < lo ? lo : (hi < v ? hi : v);
}

}  // namespace atb_llm

// ─────────────────────────────────────────────────────────────
// Platform detection helpers (inline).
//
// Resolve ASCEND_PLATFORM via GetEnv() (utils/dotenv.h), which applies the
// same three-tier precedence as GetModelDir(): shell getenv > .env file >
// "910B" default. A bare `./benchmark` invocation — where the shell has not
// sourced .env — now honours ASCEND_PLATFORM=310P written in .env. Previously
// Is310P() read getenv only and silently fell back to 910B on 310P, which
// selected the 910B ND mask path and crashed SelfAttention's ND->NZ transdata
// ("TransdataOperation mki node infer shape fail"). ctest avoided this because
// build_and_test.sh / CMakeLists ENVIRONMENT bake ASCEND_PLATFORM into the
// process environment.
// Valid values: "910B" (Atlas A2), "310P" (Atlas推理系列).
// ─────────────────────────────────────────────────────────────
#include "utils/dotenv.h"

namespace atb_llm {

inline bool Is310P() {
    return GetEnv("ASCEND_PLATFORM", "910B") == "310P";
}

inline bool Is910B() {
    return GetEnv("ASCEND_PLATFORM", "910B") == "910B";
}

}  // namespace atb_llm
