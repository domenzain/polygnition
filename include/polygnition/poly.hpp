#pragma once
#include "polygnition/detail/static_pack.hpp"
#include "polygnition/util.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace polygnition::polynomial {
template <auto V> using cval = detail::cval_t<V>;
template <auto V> inline constexpr cval<V> constant{};

namespace algorithm {
struct horner_t {};
struct knuth_t {};
struct motzkin_t {};
struct multivariate_horner_t {};
struct estrin_t {};
struct compensated_t {};
template <int K> struct dorn_t {
  static_assert(K > 1, "Dorn decomposition needs at least two residue classes");
  static constexpr int stride = K;
};
} // namespace algorithm

inline constexpr algorithm::horner_t horner{};
inline constexpr algorithm::knuth_t knuth{};
inline constexpr algorithm::motzkin_t motzkin{};
inline constexpr algorithm::multivariate_horner_t multivariate_horner{};
inline constexpr algorithm::estrin_t estrin{};
inline constexpr algorithm::compensated_t compensated{};
template <int K> inline constexpr algorithm::dorn_t<K> dorn{};

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

struct evaluate_t {
  template <typename P, typename... Vars>
    requires detail::is_polynomial_v<P> && (sizeof...(Vars) > 0)
  [[nodiscard]] constexpr auto operator()(P const &p, Vars &&...vars) const;

  template <typename Algorithm, typename P, typename... Vars>
    requires requires(evaluate_t const &self, Algorithm algorithm, P const &p,
                      Vars &&...vars) {
      polygnition::tag_invoke(self, algorithm, p,
                              std::forward<Vars>(vars)...);
    }
  [[nodiscard]] constexpr auto operator()(Algorithm algorithm, P const &p,
                                          Vars &&...vars) const {
    return polygnition::tag_invoke(*this, algorithm, p,
                                   std::forward<Vars>(vars)...);
  }
};

inline constexpr evaluate_t evaluate{};

// Horner starts from the leading coefficient, avoiding a synthetic zero.
template <typename P, typename X>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto evaluate_horner(P const &p, X const &x) {
  using coefficient_t = typename detail::stored_polynomial_type_t<P>::value_type;
  using result_t = decltype(std::declval<coefficient_t>() * std::declval<X>());
  return std::accumulate(std::next(p.begin()), p.end(),
                         detail::lift_coefficient<result_t>(p[0]),
                         [&](auto const &accum, auto const &coefficient) {
                           return (accum * x) + coefficient;
                         });
}

template <typename P, typename X>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::horner_t,
                                        P const &p, X &&x) {
  return evaluate_horner(p, std::forward<X>(x));
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

template <typename P, typename... Vars>
  requires detail::is_polynomial_v<P> && (sizeof...(Vars) > 1)
[[nodiscard]] constexpr auto tag_invoke(evaluate_t,
                                        algorithm::multivariate_horner_t,
                                        P const &p, Vars &&...vars) {
  return evaluate_multivariate(p, std::forward<Vars>(vars)...);
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

template <typename P, typename Z>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::knuth_t,
                                        P const &p, Z &&z) {
  return evaluate_complex_knuth(p, std::forward<Z>(z));
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
  return evaluate(motzkin, *this, x);
}

template <typename T, typename X>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::motzkin_t,
                                        motzkin_preprocessed_t<T> const &p,
                                        X &&x) {
  return evaluate_motzkin(p, std::forward<X>(x));
}

template <typename P, typename X>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::motzkin_t,
                                        P const &p, X &&x) {
  return evaluate_motzkin(p, std::forward<X>(x));
}

namespace detail {
template <int Exponent, typename Return, typename X>
[[nodiscard]] constexpr auto power(X const &x) -> Return {
  if constexpr (Exponent == 0) {
    return lift_coefficient<Return>(1);
  } else if constexpr (Exponent == 1) {
    return lift_coefficient<Return>(x);
  } else if constexpr (Exponent % 2 == 0) {
    auto const half = power<Exponent / 2, Return>(x);
    return half * half;
  } else {
    return power<Exponent - 1, Return>(x) * lift_coefficient<Return>(x);
  }
}

template <int J, int I, int K, typename P, typename Y, typename Return>
[[nodiscard]] constexpr auto dorn_fold(P const &p, Y const &y,
                                       Return const &accum) -> Return {
  if constexpr (J < I) {
    return accum;
  } else {
    return dorn_fold<J - K, I, K>(
        p, y, (accum * y) +
                  lift_coefficient<Return>(coefficient_by_degree<J>(p)));
  }
}

template <int I, int K, typename P, typename X, typename Y, typename Return>
[[nodiscard]] constexpr auto dorn_partial(P const &p, X const &x,
                                          Y const &y) -> Return {
  constexpr int degree = stored_polynomial_type_t<P>::degree;
  if constexpr (I > degree) {
    return Return{};
  } else {
    constexpr int highest = degree - ((degree - I) % K);
    return power<I, Return>(x) *
           dorn_fold<highest - K, I, K>(
               p, y,
               lift_coefficient<Return>(coefficient_by_degree<highest>(p)));
  }
}

template <int I, int K, typename P, typename X, typename Y, typename Return>
[[nodiscard]] constexpr auto dorn_sum(P const &p, X const &x, Y const &y)
    -> Return {
  if constexpr (I == K) {
    return Return{};
  } else {
    return dorn_partial<I, K, P, X, Y, Return>(p, x, y) +
           dorn_sum<I + 1, K, P, X, Y, Return>(p, x, y);
  }
}

template <std::size_t I, typename Return, typename P, typename X>
[[nodiscard]] constexpr auto estrin_pair(P const &p, X const &x) {
  return (lift_coefficient<Return>(coefficient_at<I>(p)) * x) +
         lift_coefficient<Return>(coefficient_at<I + 1>(p));
}

template <typename P, typename X>
[[nodiscard]] constexpr auto estrin_eval(P const &p, X x) {
  constexpr int degree = stored_polynomial_type_t<P>::degree;
  using coefficient_t = typename stored_polynomial_type_t<P>::value_type;
  using return_t = decltype(std::declval<coefficient_t>() * std::declval<X>());
  if constexpr (degree == 0) {
    return lift_coefficient<return_t>(coefficient_at<0>(p));
  } else if constexpr (degree % 2 == 0) {
    constexpr int half_degree = degree / 2;
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto const reduced = polynomial_t<return_t, half_degree>{
          lift_coefficient<return_t>(coefficient_at<0>(p)),
          estrin_pair<2 * (I + 1) - 1, return_t>(p, x)...};
      return estrin_eval(reduced, x * x);
    }(std::make_index_sequence<static_cast<std::size_t>(half_degree)>{});
  } else {
    constexpr int half_degree = degree / 2;
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto const reduced = polynomial_t<return_t, half_degree>{
          estrin_pair<2 * I, return_t>(p, x)...};
      return estrin_eval(reduced, x * x);
    }(std::make_index_sequence<static_cast<std::size_t>(half_degree + 1)>{});
  }
}
} // namespace detail

template <int K, typename P, typename X>
  requires(K > 1) && detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto evaluate_dorn(P const &p, X const &x) {
  using coefficient_t = typename detail::stored_polynomial_type_t<P>::value_type;
  using return_t = decltype(std::declval<coefficient_t>() * std::declval<X>());
  auto const y = detail::power<K, return_t>(x);
  return detail::dorn_sum<0, K, P, X, return_t, return_t>(p, x, y);
}

template <int K, typename P, typename X>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::dorn_t<K>,
                                        P const &p, X &&x) {
  return evaluate_dorn<K>(p, std::forward<X>(x));
}

template <typename P, typename X>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto evaluate_estrin(P const &p, X &&x) {
  return detail::estrin_eval(p, std::forward<X>(x));
}

template <typename P, typename X>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::estrin_t,
                                        P const &p, X &&x) {
  return evaluate_estrin(p, std::forward<X>(x));
}

namespace detail {
template <typename T> [[nodiscard]] constexpr auto abs_value(T value) {
  return value < T{} ? -value : value;
}

template <int Exponent, typename T>
[[nodiscard]] constexpr auto pow_unsigned(T value) -> T {
  if constexpr (Exponent == 0) {
    return T{1};
  } else {
    return value * pow_unsigned<Exponent - 1>(value);
  }
}

template <std::size_t I, typename Bound, typename P>
[[nodiscard]] constexpr auto abs_weighted_coefficient(P const &p,
                                                      Bound xmax) -> Bound {
  constexpr int degree = stored_polynomial_type_t<P>::degree;
  constexpr int exponent = degree - static_cast<int>(I);
  return abs_value(static_cast<Bound>(coefficient_at<I>(p))) *
         pow_unsigned<exponent>(abs_value(xmax));
}

template <typename Bound, typename P, std::size_t... I>
[[nodiscard]] constexpr auto weighted_coefficient_sum(
    P const &p, Bound xmax, std::index_sequence<I...>) -> Bound {
  return (Bound{} + ... + abs_weighted_coefficient<I, Bound>(p, xmax));
}

template <typename T> struct eft_pair {
  T value;
  T error;
};

template <typename T> [[nodiscard]] constexpr auto splitter() -> T {
  if constexpr (std::numeric_limits<T>::digits <= 24) {
    return T{4097};
  } else {
    return T{134217729};
  }
}

template <typename T> [[nodiscard]] constexpr auto split(T a) -> eft_pair<T> {
  auto const c = splitter<T>() * a;
  auto const abig = c - a;
  auto const high = c - abig;
  return {.value = high, .error = a - high};
}

template <typename T>
[[nodiscard]] constexpr auto two_sum(T a, T b) -> eft_pair<T> {
  auto const sum = a + b;
  auto const bb = sum - a;
  return {.value = sum, .error = (a - (sum - bb)) + (b - bb)};
}

template <typename T>
[[nodiscard]] constexpr auto two_prod(T a, T b) -> eft_pair<T> {
  auto const product = a * b;
  auto const as = split(a);
  auto const bs = split(b);
  auto const error = ((as.value * bs.value - product) +
                      as.value * bs.error + as.error * bs.value) +
                     as.error * bs.error;
  return {.value = product, .error = error};
}

template <typename P, typename X, typename Return, std::size_t... I>
[[nodiscard]] constexpr auto compensated_horner_indexed(
    P const &p, X const &x, std::index_sequence<I...>) -> Return {
  auto result = lift_coefficient<Return>(coefficient_at<0>(p));
  auto correction = Return{};
  auto const xc = static_cast<Return>(x);
  ([&] {
    auto const product = two_prod(result, xc);
    auto const sum =
        two_sum(product.value, static_cast<Return>(coefficient_at<I + 1>(p)));
    correction = (correction * xc) + (product.error + sum.error);
    result = sum.value;
  }(),
   ...);
  return result + correction;
}
} // namespace detail

template <typename P, typename X>
  requires detail::is_polynomial_v<P> &&
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> &&
           arithmetic<std::remove_cvref_t<X>>
[[nodiscard]] constexpr auto error_bound(P const &p, X const &xmax) {
  using coefficient_t = typename detail::stored_polynomial_type_t<P>::value_type;
  constexpr auto degree = detail::stored_polynomial_type_t<P>::degree;
  using bound_t = std::common_type_t<double, coefficient_t,
                                     std::remove_cvref_t<X>>;
  constexpr auto operations = 2 * degree;
  constexpr auto epsilon = std::numeric_limits<bound_t>::epsilon() / bound_t{2};
  constexpr auto gamma =
      (operations * epsilon) / (bound_t{1} - (operations * epsilon));
  return gamma * detail::weighted_coefficient_sum<bound_t>(
                     p, static_cast<bound_t>(xmax),
                     std::make_index_sequence<
                         static_cast<std::size_t>(degree + 1)>{});
}

template <typename P, typename X>
  requires detail::is_polynomial_v<P> &&
           std::floating_point<std::remove_cvref_t<decltype(
               std::declval<typename detail::stored_polynomial_type_t<P>::value_type>() *
               std::declval<X>())>>
[[nodiscard]] constexpr auto evaluate_compensated(P const &p, X const &x) {
  using coefficient_t = typename detail::stored_polynomial_type_t<P>::value_type;
  using return_t = std::remove_cvref_t<
      decltype(std::declval<coefficient_t>() * std::declval<X>())>;
  constexpr auto degree = detail::stored_polynomial_type_t<P>::degree;
  return detail::compensated_horner_indexed<P, X, return_t>(
      p, x, std::make_index_sequence<static_cast<std::size_t>(degree)>{});
}

template <typename P, typename X>
  requires detail::is_polynomial_v<P> &&
           requires(P const &p, X const &x) { evaluate_compensated(p, x); }
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::compensated_t,
                                        P const &p, X &&x) {
  return evaluate_compensated(p, std::forward<X>(x));
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

template <typename P, typename... Vars>
  requires detail::is_polynomial_v<P> && (sizeof...(Vars) > 0)
[[nodiscard]] constexpr auto evaluate_t::operator()(P const &p,
                                                    Vars &&...vars) const {
  return detail::evaluate_automatic(p, std::forward<Vars>(vars)...);
}

template <typename T, int N>
  requires(N >= 0)
template <typename... Vars>
  requires(sizeof...(Vars) > 0)
[[nodiscard]] constexpr auto polynomial_t<T, N>::operator()(
    Vars const &...vars) const {
  return evaluate(*this, vars...);
}

template <polynomial_t P>
template <typename... Vars>
  requires(sizeof...(Vars) > 0)
[[nodiscard]] constexpr auto static_poly<P>::operator()(
    Vars const &...vars) const {
  return evaluate(*this, vars...);
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
