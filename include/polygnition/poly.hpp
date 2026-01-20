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

  template <typename... Vars>
    requires(sizeof...(Vars) > 0)
  [[nodiscard]] constexpr auto operator()(Vars const &...vars) const;
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

template <typename T, int N, typename X, typename... Remaining>
  requires(sizeof...(Remaining) > 0)
[[nodiscard]] constexpr auto evaluate_multivariate(
    polynomial_t<T, N> const &p, X const &x, Remaining const &...remaining) {
  using coefficient_result_t = decltype(std::declval<T>()(remaining...));
  using result_t = std::common_type_t<
      decltype(std::declval<X>() * std::declval<coefficient_result_t>()),
      coefficient_result_t>;
  return std::accumulate(
      p.begin(), p.end(), result_t{},
      [&](auto const &accum, auto const &coefficient) {
        return (accum * x) + coefficient(remaining...);
      });
}

template <typename T, int N>
  requires(N >= 0)
template <typename... Vars>
  requires(sizeof...(Vars) > 0)
[[nodiscard]] constexpr auto polynomial_t<T, N>::operator()(
    Vars const &...vars) const {
  if constexpr (sizeof...(Vars) == 1) {
    return evaluate_horner(*this, vars...);
  } else {
    return evaluate_multivariate(*this, vars...);
  }
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
