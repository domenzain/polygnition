#pragma once
#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <utility>
#include <tuple>
#include <type_traits>

namespace polygnition::polynomial {
// Dense, fixed-degree runtime polynomial: cₙxⁿ + … + c₀.
template <typename T, int N>
  requires(N >= 0)
struct polynomial_t : std::array<T, static_cast<std::size_t>(N + 1)> {
  using value_type = T;
  static constexpr int degree = N;

  template <int Degree>
  [[nodiscard]] constexpr auto coeff() noexcept -> T & {
    static_assert(Degree >= 0 && Degree <= N);
    return (*this)[static_cast<std::size_t>(N - Degree)];
  }

  template <int Degree>
  [[nodiscard]] constexpr auto coeff() const noexcept -> T {
    static_assert(Degree >= 0 && Degree <= N);
    return (*this)[static_cast<std::size_t>(N - Degree)];
  }

  [[nodiscard]] constexpr auto coeff(int degree_index) const noexcept -> T {
    return degree_index < 0 || degree_index > N
               ? T{}
               : (*this)[static_cast<std::size_t>(N - degree_index)];
  }

  template <typename X>
  [[nodiscard]] constexpr auto operator()(X const &x) const;
};

// Horner starts from the leading coefficient, so the return type need only model
// the coefficient-variable product rather than construction from an integer zero.
template <typename T, int N, typename X>
[[nodiscard]] constexpr auto evaluate_horner(polynomial_t<T, N> const &p,
                                             X const &x) {
  using result_t = decltype(std::declval<T>() * std::declval<X>());
  return std::accumulate(std::next(p.begin()), p.end(),
                         static_cast<result_t>(p[0]),
                         [&](auto const &accum, auto const &coefficient) {
                           return (accum * x) + coefficient;
                         });
}

template <typename T, int N>
  requires(N >= 0)
template <typename X>
[[nodiscard]] constexpr auto polynomial_t<T, N>::operator()(X const &x) const {
  return evaluate_horner(*this, x);
}

template <typename T, int N>
[[nodiscard]] constexpr auto degree(polynomial_t<T, N> const &) noexcept -> int {
  return N;
}

template <class T, class... U>
polynomial_t(T, U...) -> polynomial_t<std::common_type_t<T, U...>, sizeof...(U)>;
} // namespace polygnition::polynomial

namespace std {
template <typename T, int N>
struct tuple_size<::polygnition::polynomial::polynomial_t<T, N>>
    : integral_constant<size_t, N + 1> {};

template <size_t I, typename T, int N>
struct tuple_element<I, ::polygnition::polynomial::polynomial_t<T, N>> {
  using type = T;
};
} // namespace std
