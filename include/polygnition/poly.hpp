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

template <typename Return, typename Coefficient>
[[nodiscard]] constexpr auto lift_coefficient(Coefficient const &coefficient)
    -> Return {
  if constexpr (std::convertible_to<Coefficient, Return>) {
    return static_cast<Return>(coefficient);
  } else {
    return Return{} + coefficient;
  }
}

template <std::size_t... I, typename A, typename B>
[[nodiscard]] constexpr auto equal_coefficients(
    A const &a, B const &b, std::index_sequence<I...>) -> bool {
  return ((coefficient_at<I>(a) == coefficient_at<I>(b)) && ...);
}

template <int Degree, typename P>
[[nodiscard]] constexpr auto coefficient_by_degree_or_zero(P const &p) noexcept
    -> typename stored_polynomial_type_t<P>::value_type {
  using value_type = typename stored_polynomial_type_t<P>::value_type;
  if constexpr (Degree > stored_polynomial_type_t<P>::degree) {
    return value_type{};
  } else {
    return coefficient_by_degree<Degree>(p);
  }
}

template <int... Degree, typename A, typename B>
[[nodiscard]] constexpr auto equal_coefficients_by_degree(
    A const &a, B const &b, std::integer_sequence<int, Degree...>) -> bool {
  return ((coefficient_by_degree_or_zero<Degree>(a) ==
           coefficient_by_degree_or_zero<Degree>(b)) && ...);
}

struct is_zero_t {
  template <typename T>
  [[nodiscard]] constexpr auto operator()(T const &value) const
      noexcept(noexcept(value == T{})) -> bool {
    return value == T{};
  }
};
inline constexpr is_zero_t is_zero{};

template <typename Coefficients>
[[nodiscard]] constexpr auto effective_degree_from_coefficients(
    Coefficients const &coefficients, int static_degree) noexcept -> int {
  auto const first_nonzero = std::ranges::find_if_not(coefficients, is_zero);
  return static_degree -
         static_cast<int>(first_nonzero - std::ranges::begin(coefficients));
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

// Motzkin rewrites a quartic into two dependent quadratic stages. Runtime
// coefficients can be preprocessed once; literal quartics can preprocess at
// compile time.
template <typename T>
  requires std::floating_point<T>
struct motzkin_preprocessed_t {
  T leading;
  T beta0;
  T beta1;
  T beta2;
  T beta3;

  template <typename X>
  [[nodiscard]] constexpr auto operator()(X const &x) const;
};

namespace detail {
template <typename T>
[[nodiscard]] constexpr auto preprocess_motzkin_coefficients(
    T leading, T c3, T c2, T c1, T c0) {
  auto const inverse_leading = T{1} / leading;
  auto const a3 = c3 * inverse_leading;
  auto const a2 = c2 * inverse_leading;
  auto const a1 = c1 * inverse_leading;
  auto const a0 = c0 * inverse_leading;
  auto const beta0 = T{0.5} * (a3 - T{1});
  auto const z = a2 - (beta0 * (beta0 + T{1}));
  auto const beta1 = a1 - (beta0 * z);
  auto const beta2 = z - (T{2} * beta1);
  auto const beta3 = a0 - (beta1 * (beta1 + beta2));
  return motzkin_preprocessed_t<T>{leading, beta0, beta1, beta2, beta3};
}
} // namespace detail

template <typename P>
  requires detail::is_polynomial_v<P> &&
           (detail::stored_polynomial_type_t<P>::degree == 4) &&
           std::floating_point<
               typename detail::stored_polynomial_type_t<P>::value_type>
[[nodiscard]] constexpr auto preprocess_motzkin(P const &p) {
  using value_type = typename detail::stored_polynomial_type_t<P>::value_type;
  return detail::preprocess_motzkin_coefficients(
      static_cast<value_type>(p[0]), static_cast<value_type>(p[1]),
      static_cast<value_type>(p[2]), static_cast<value_type>(p[3]),
      static_cast<value_type>(p[4]));
}

template <typename T, typename X>
  requires std::floating_point<T> && arithmetic<std::remove_cvref_t<X>>
[[nodiscard]] constexpr auto evaluate_motzkin(
    motzkin_preprocessed_t<T> const &preprocessed, X const &x) {
  using result_t = std::common_type_t<T, std::remove_cvref_t<X>>;
  auto const xc = static_cast<result_t>(x);
  auto const b0 = static_cast<result_t>(preprocessed.beta0);
  auto const b1 = static_cast<result_t>(preprocessed.beta1);
  auto const b2 = static_cast<result_t>(preprocessed.beta2);
  auto const b3 = static_cast<result_t>(preprocessed.beta3);
  auto const leading = static_cast<result_t>(preprocessed.leading);
  auto const y = ((xc + b0) * xc) + b1;
  return ((((y + xc) + b2) * y) + b3) * leading;
}

template <typename P, typename X>
  requires detail::is_polynomial_v<P> &&
           (detail::stored_polynomial_type_t<P>::degree == 4) &&
           std::floating_point<
               typename detail::stored_polynomial_type_t<P>::value_type> &&
           arithmetic<std::remove_cvref_t<X>>
[[nodiscard]] constexpr auto evaluate_motzkin(P const &p, X const &x) {
  return evaluate_motzkin(preprocess_motzkin(p), x);
}

template <typename T>
  requires std::floating_point<T>
template <typename X>
[[nodiscard]] constexpr auto motzkin_preprocessed_t<T>::operator()(
    X const &x) const {
  return evaluate_motzkin(*this, x);
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
    if constexpr (complex<variable_t>) {
      if constexpr (arithmetic<coefficient_t> &&
                    std::floating_point<typename variable_t::value_type>) {
        return evaluate_complex_knuth(p, vars...);
      } else {
        return evaluate_horner(p, vars...);
      }
    } else if constexpr (is_static_polynomial_v<P> &&
                         stored_polynomial_type_t<P>::degree == 4 &&
                         std::floating_point<coefficient_t> &&
                         arithmetic<variable_t>) {
      return evaluate_motzkin(p, vars...);
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

// Equality is structural; same_polynomial ignores leading-zero storage.
template <typename A, typename B>
  requires detail::is_polynomial_v<A> && detail::is_polynomial_v<B>
[[nodiscard]] constexpr auto operator==(A const &a, B const &b) -> bool {
  constexpr auto a_degree = detail::stored_polynomial_type_t<A>::degree;
  constexpr auto b_degree = detail::stored_polynomial_type_t<B>::degree;
  if constexpr (a_degree != b_degree) {
    return false;
  } else {
    return detail::equal_coefficients(
        a, b,
        std::make_index_sequence<static_cast<std::size_t>(a_degree + 1)>{});
  }
}

template <typename A, typename B>
  requires detail::is_polynomial_v<A> && detail::is_polynomial_v<B>
[[nodiscard]] constexpr auto operator!=(A const &a, B const &b) -> bool {
  return !(a == b);
}

template <typename A, typename B>
  requires detail::is_polynomial_v<A> && detail::is_polynomial_v<B>
[[nodiscard]] constexpr auto same_polynomial(A const &a, B const &b) -> bool {
  constexpr auto a_degree = detail::stored_polynomial_type_t<A>::degree;
  constexpr auto b_degree = detail::stored_polynomial_type_t<B>::degree;
  constexpr auto max_degree = a_degree > b_degree ? a_degree : b_degree;
  return detail::equal_coefficients_by_degree(
      a, b, std::make_integer_sequence<int, max_degree + 1>{});
}

template <typename P>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto effective_degree(P const &p) noexcept -> int {
  return detail::effective_degree_from_coefficients(p, degree(p));
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
