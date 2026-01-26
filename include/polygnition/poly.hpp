#pragma once
#include "polygnition/detail/static_pack.hpp"
#include "polygnition/util.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace polygnition::polynomial {
template <auto V> using cval = detail::cval_t<V>;
template <auto V> inline constexpr cval<V> constant{};

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

// Literal coefficients are part of the type. The object itself stores nothing.
template <polynomial_t P> struct static_poly {
  using polynomial_type = std::remove_cvref_t<decltype(P)>;
  using value_type = typename polynomial_type::value_type;
  static constexpr int degree = polynomial_type::degree;
  static inline constexpr polynomial_type value = P;

  template <typename... Vars>
    requires(sizeof...(Vars) > 0)
  [[nodiscard]] constexpr auto operator()(Vars const &...vars) const;

  template <int Degree>
  [[nodiscard]] constexpr auto coeff() const noexcept -> value_type {
    static_assert(Degree >= 0 && Degree <= degree);
    return value[static_cast<std::size_t>(degree - Degree)];
  }

  [[nodiscard]] constexpr auto coeff(int degree_index) const noexcept
      -> value_type {
    return degree_index < 0 || degree_index > degree
               ? value_type{}
               : value[static_cast<std::size_t>(degree - degree_index)];
  }

  [[nodiscard]] constexpr auto operator[](std::size_t index) const noexcept
      -> value_type {
    return value[index];
  }

  [[nodiscard]] static constexpr auto size() noexcept -> std::size_t {
    return static_cast<std::size_t>(degree + 1);
  }
  [[nodiscard]] static constexpr auto empty() noexcept -> bool { return false; }

  [[nodiscard]] constexpr auto begin() const noexcept { return value.begin(); }
  [[nodiscard]] constexpr auto end() const noexcept { return value.end(); }
  [[nodiscard]] constexpr auto cbegin() const noexcept { return value.cbegin(); }
  [[nodiscard]] constexpr auto cend() const noexcept { return value.cend(); }
};

template <typename T, int N>
[[nodiscard]] constexpr auto degree(polynomial_t<T, N> const &) noexcept -> int {
  return N;
}

template <polynomial_t P>
[[nodiscard]] constexpr auto degree(static_poly<P> const &) noexcept -> int {
  return static_poly<P>::degree;
}

namespace detail {
template <typename...> struct first_type;
template <typename T, typename... Ts> struct first_type<T, Ts...> {
  using type = T;
};
template <typename... Ts> using first_type_t = typename first_type<Ts...>::type;

template <typename> struct is_polynomial : std::false_type {};
template <typename T, int N>
struct is_polynomial<polynomial_t<T, N>> : std::true_type {};
template <polynomial_t P> struct is_polynomial<static_poly<P>> : std::true_type {};
template <typename T>
inline constexpr bool is_polynomial_v =
    is_polynomial<std::remove_cvref_t<T>>::value;

template <typename> struct is_static_polynomial : std::false_type {};
template <polynomial_t P>
struct is_static_polynomial<static_poly<P>> : std::true_type {};
template <typename T>
inline constexpr bool is_static_polynomial_v =
    is_static_polynomial<std::remove_cvref_t<T>>::value;

template <typename P> struct stored_polynomial_type {
  using type = std::remove_cvref_t<P>;
};
template <polynomial_t P> struct stored_polynomial_type<static_poly<P>> {
  using type = typename static_poly<P>::polynomial_type;
};
template <typename P>
using stored_polynomial_type_t =
    typename stored_polynomial_type<std::remove_cvref_t<P>>::type;

template <std::size_t I, typename P>
  requires is_polynomial_v<P>
[[nodiscard]] constexpr auto coefficient_at(P const &p) noexcept
    -> typename stored_polynomial_type_t<P>::value_type {
  static_assert(I < static_cast<std::size_t>(stored_polynomial_type_t<P>::degree + 1));
  return p[I];
}

template <int Degree, typename P>
  requires is_polynomial_v<P>
[[nodiscard]] constexpr auto coefficient_by_degree(P const &p) noexcept
    -> typename stored_polynomial_type_t<P>::value_type {
  static_assert(Degree >= 0 && Degree <= stored_polynomial_type_t<P>::degree);
  return coefficient_at<static_cast<std::size_t>(
      stored_polynomial_type_t<P>::degree - Degree)>(p);
}

template <typename R, typename P>
  requires is_polynomial_v<P>
[[nodiscard]] constexpr auto coefficient_by_degree_as(P const &p,
                                                      int degree_index) -> R {
  constexpr auto source_degree = stored_polynomial_type_t<P>::degree;
  return degree_index < 0 || degree_index > source_degree
             ? R{}
             : static_cast<R>(p[static_cast<std::size_t>(source_degree -
                                                         degree_index)]);
}
} // namespace detail

// Horner starts from the leading coefficient, avoiding a synthetic zero.
template <typename P, typename X>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto evaluate_horner(P const &p, X const &x) {
  using coefficient_t = typename detail::stored_polynomial_type_t<P>::value_type;
  using result_t = decltype(std::declval<coefficient_t>() * std::declval<X>());
  return std::accumulate(std::next(p.begin()), p.end(),
                         static_cast<result_t>(p[0]),
                         [&](auto const &accum, auto const &coefficient) {
                           return (accum * x) + coefficient;
                         });
}

template <typename P, typename X, typename... Remaining>
  requires detail::is_polynomial_v<P> && (sizeof...(Remaining) > 0)
[[nodiscard]] constexpr auto evaluate_multivariate(
    P const &p, X const &x, Remaining const &...remaining) {
  using coefficient_t = typename detail::stored_polynomial_type_t<P>::value_type;
  using coefficient_result_t = decltype(std::declval<coefficient_t>()(remaining...));
  using result_t = std::common_type_t<
      decltype(std::declval<X>() * std::declval<coefficient_result_t>()),
      coefficient_result_t>;
  return std::accumulate(
      p.begin(), p.end(), result_t{},
      [&](auto const &accum, auto const &coefficient) {
        return (accum * x) + coefficient(remaining...);
      });
}

// Knuth's real-coefficient recurrence avoids generic complex multiplies.
template <typename P, typename U>
  requires detail::is_polynomial_v<P> &&
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> &&
           std::floating_point<U>
[[nodiscard]] constexpr auto evaluate_complex_knuth(
    P const &p, std::complex<U> const &z) {
  using coefficient_t = typename detail::stored_polynomial_type_t<P>::value_type;
  struct state_t {
    U a{};
    U b{};
  };

  auto const two_x = z.real() + z.real();
  auto const minus_norm_sqr = -std::norm(z);
  auto const state = std::accumulate(
      p.begin(), p.end(), state_t{},
      [two_x, minus_norm_sqr](state_t const &previous,
                              coefficient_t const coefficient) {
        return state_t{.a = (two_x * previous.a) + previous.b,
                       .b = (minus_norm_sqr * previous.a) +
                            static_cast<U>(coefficient)};
      });
  return std::complex<U>{(z.real() * state.a) + state.b,
                         z.imag() * state.a};
}

namespace detail {
template <typename P, typename... Vars>
  requires is_polynomial_v<P> && (sizeof...(Vars) > 0)
[[nodiscard]] constexpr auto evaluate_automatic(P const &p,
                                                Vars const &...vars) {
  if constexpr (sizeof...(Vars) > 1) {
    return evaluate_multivariate(p, vars...);
  } else {
    using coefficient_t = typename stored_polynomial_type_t<P>::value_type;
    using variable_t = std::remove_cvref_t<first_type_t<Vars...>>;
    if constexpr (arithmetic<coefficient_t> && complex<variable_t>) {
      if constexpr (std::floating_point<typename variable_t::value_type>) {
        return evaluate_complex_knuth(p, vars...);
      } else {
        return evaluate_horner(p, vars...);
      }
    } else {
      return evaluate_horner(p, vars...);
    }
  }
}
} // namespace detail

template <typename T, int N>
  requires(N >= 0)
template <typename... Vars>
  requires(sizeof...(Vars) > 0)
[[nodiscard]] constexpr auto polynomial_t<T, N>::operator()(
    Vars const &...vars) const {
  return detail::evaluate_automatic(*this, vars...);
}

template <polynomial_t P>
template <typename... Vars>
  requires(sizeof...(Vars) > 0)
[[nodiscard]] constexpr auto static_poly<P>::operator()(
    Vars const &...vars) const {
  return detail::evaluate_automatic(*this, vars...);
}

template <std::size_t I, typename T, int N>
[[nodiscard]] constexpr decltype(auto) get(polynomial_t<T, N> &p) noexcept {
  static_assert(I < static_cast<std::size_t>(N + 1));
  return (p[I]);
}

template <std::size_t I, typename T, int N>
[[nodiscard]] constexpr decltype(auto) get(polynomial_t<T, N> const &p) noexcept {
  static_assert(I < static_cast<std::size_t>(N + 1));
  return (p[I]);
}

template <std::size_t I, typename T, int N>
[[nodiscard]] constexpr decltype(auto) get(polynomial_t<T, N> &&p) noexcept {
  static_assert(I < static_cast<std::size_t>(N + 1));
  return std::move(p[I]);
}

template <std::size_t I, typename T, int N>
[[nodiscard]] constexpr decltype(auto) get(polynomial_t<T, N> const &&p) noexcept {
  static_assert(I < static_cast<std::size_t>(N + 1));
  return std::move(p[I]);
}

template <std::size_t I, polynomial_t P>
[[nodiscard]] constexpr auto get(static_poly<P>) noexcept
    -> typename static_poly<P>::value_type {
  static_assert(I < static_poly<P>::size());
  return static_poly<P>::value[I];
}

template <typename A, typename B>
  requires detail::is_polynomial_v<A> && detail::is_polynomial_v<B>
[[nodiscard]] constexpr auto operator==(A const &a, B const &b) -> bool {
  if constexpr (detail::stored_polynomial_type_t<A>::degree !=
                detail::stored_polynomial_type_t<B>::degree) {
    return false;
  } else {
    return std::equal(a.begin(), a.end(), b.begin());
  }
}

template <class T, class... U>
  requires(!detail::is_cval_v<std::remove_cvref_t<T>>)
polynomial_t(T, U...) -> polynomial_t<std::common_type_t<T, U...>, sizeof...(U)>;

template <auto... Cs>
  requires(sizeof...(Cs) > 0)
[[nodiscard]] consteval auto literal() {
  using value_type = std::common_type_t<decltype(Cs)...>;
  return static_poly<polynomial_t<value_type, sizeof...(Cs) - 1>{
      static_cast<value_type>(Cs)...}>{};
}
} // namespace polygnition::polynomial

namespace std {
template <typename T, int N>
struct tuple_size<::polygnition::polynomial::polynomial_t<T, N>>
    : integral_constant<size_t, N + 1> {};

template <size_t I, typename T, int N>
struct tuple_element<I, ::polygnition::polynomial::polynomial_t<T, N>> {
  using type = T;
};

template <::polygnition::polynomial::polynomial_t P>
struct tuple_size<::polygnition::polynomial::static_poly<P>>
    : integral_constant<size_t, static_cast<size_t>(
                                        ::polygnition::polynomial::static_poly<P>::degree + 1)> {};

template <size_t I, ::polygnition::polynomial::polynomial_t P>
struct tuple_element<I, ::polygnition::polynomial::static_poly<P>> {
  using type = typename ::polygnition::polynomial::static_poly<P>::value_type;
};
} // namespace std
