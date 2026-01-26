#pragma once
#include <complex>
#include <concepts>
#include <type_traits>

namespace polygnition {
template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_complex_v = is_complex<std::remove_cvref_t<T>>::value;
template <typename T>
concept complex = is_complex_v<T>;

template <typename T>
concept closed_ring = std::regular<T> && requires(T a, T b) {
  T{};
  T{0};
  T{1};
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { -a } -> std::same_as<T>;
  { a * b } -> std::same_as<T>;
};
} // namespace polygnition
