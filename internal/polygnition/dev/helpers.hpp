#pragma once
#include <array>
#include <cstddef>
#include <polygnition/poly.hpp>
#include <utility>

namespace polygnition::dev {

namespace poly = polygnition::polynomial;

// constexpr iota array: {start + step*0, start + step*1, ...}
template <typename T, std::size_t K, std::size_t... I>
constexpr auto iota_array_impl(T start, T step,
                               std::index_sequence<I...>) noexcept {
  return std::array<T, K>{static_cast<T>(start + step * static_cast<T>(I))...};
}

template <typename T, std::size_t K>
constexpr auto iota_array(T start = T{1}, T step = T{1}) noexcept {
  return iota_array_impl<T, K>(start, step, std::make_index_sequence<K>{});
}

// constexpr polynomial with iota coefficients {c_n = 1, …, c_0 = N+1}
template <typename T = double, int N>
constexpr auto iota_poly(T start = T{1}, T step = T{1}) noexcept {
  auto const a = iota_array<T, static_cast<std::size_t>(N + 1)>(start, step);
  return poly::polynomial_t<T, N>{a};
}

} // namespace polygnition::dev
