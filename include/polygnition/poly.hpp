#pragma once
#include "polygnition/util.hpp"
#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <utility>
#include <tuple>
#include <type_traits>

namespace polygnition::polynomial {
namespace detail {
template <typename...> struct first_type;
template <typename T, typename... Ts> struct first_type<T, Ts...> {
  using type = T;
};
template <typename... Ts> using first_type_t = typename first_type<Ts...>::type;
} // namespace detail
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

// Knuth's real-coefficient recurrence avoids generic complex multiplies.
template <typename T, int N, typename U>
  requires arithmetic<T> && std::floating_point<U>
[[nodiscard]] constexpr auto evaluate_complex_knuth(
    polynomial_t<T, N> const &p, std::complex<U> const &z) {
  struct state_t {
    U a{};
    U b{};
  };

  auto const two_x = z.real() + z.real();
  auto const minus_norm_sqr = -std::norm(z);
  auto const state = std::accumulate(
      p.begin(), p.end(), state_t{},
      [two_x, minus_norm_sqr](state_t const &previous, T const coefficient) {
        return state_t{.a = (two_x * previous.a) + previous.b,
                       .b = (minus_norm_sqr * previous.a) +
                            static_cast<U>(coefficient)};
      });
  return std::complex<U>{(z.real() * state.a) + state.b,
                         z.imag() * state.a};
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
  if constexpr (sizeof...(Vars) > 1) {
    return evaluate_multivariate(*this, vars...);
  } else {
    using variable_t = std::remove_cvref_t<detail::first_type_t<Vars...>>;
    if constexpr (arithmetic<T> && complex<variable_t>) {
      if constexpr (std::floating_point<typename variable_t::value_type>) {
        return evaluate_complex_knuth(*this, vars...);
      } else {
        return evaluate_horner(*this, vars...);
      }
    } else {
      return evaluate_horner(*this, vars...);
    }
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
