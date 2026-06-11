// SPDX-License-Identifier: Apache-2.0
//
// C++11 compatibility shims.
//
// 本工程编译标准为 C++11。本头文件提供 C++14+ 才有的若干轻量工具，
// 命名与 std:: 对齐但放在 atb_llm 命名空间下，避免污染 std。
//
// 使用方式：
//   #include "util/cpp11_compat.h"
//   auto p = atb_llm::make_unique<Foo>(a, b);
//
// 将来若升级到 C++14+，可以全局把 atb_llm::make_unique 替换回
// std::make_unique，无其他副作用。

#pragma once

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
// std::exchange (C++14) 等价实现。
//
// 用 obj 当前值返回，并把 obj 设为 new_value。常用于 move 构造/
// move 赋值里把源对象置空。
// ─────────────────────────────────────────────────────────────
template <typename T, typename U = T>
T exchange(T& obj, U&& new_value) {
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}

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
// Platform detection helpers (inline, no dependency).
//
// Reads ASCEND_PLATFORM from environment once per call.
// Valid values: "910B" (Atlas A2), "310P" (Atlas推理系列).
// ─────────────────────────────────────────────────────────────
#include <cstdlib>
#include <cstring>

namespace atb_llm {

inline bool Is310P() {
    const char* p = std::getenv("ASCEND_PLATFORM");
    return p != nullptr && std::strcmp(p, "310P") == 0;
}

inline bool Is910B() {
    const char* p = std::getenv("ASCEND_PLATFORM");
    return p == nullptr || std::strcmp(p, "910B") == 0;
}

}  // namespace atb_llm
