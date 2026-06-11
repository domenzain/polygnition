#pragma once
#include "polygnition/detail/static_pack.hpp"
#include "polygnition/target.hpp"
#include "polygnition/util.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
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

template <target::profile Profile> struct tuned_t {
  static constexpr auto profile = Profile;
};

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
template <target::profile Profile> inline constexpr algorithm::tuned_t<Profile> tuned{};
static constexpr auto automatic = tuned<target::configured>;

struct evaluate_t {
  template <target::profile Profile, typename P, typename... Vars>
    requires requires(evaluate_t const &self, P const &p, Vars &&...vars) {
      polygnition::tag_invoke(self, tuned<Profile>, p,
                              std::forward<Vars>(vars)...);
    }
  [[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE static constexpr auto
  call_tuned(P const &p, Vars &&...vars) {
    return polygnition::tag_invoke(evaluate_t{}, tuned<Profile>, p,
                                   std::forward<Vars>(vars)...);
  }

  template <typename P, typename... Vars>
    requires requires(P const &p, Vars &&...vars) {
      evaluate_t::template call_tuned<target::configured>(
          p, std::forward<Vars>(vars)...);
    }
  [[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
  operator()(P const &p, Vars &&...vars) const {
    return evaluate_t::template call_tuned<target::configured>(
        p, std::forward<Vars>(vars)...);
  }

  template <typename Algorithm, typename P, typename... Vars>
    requires requires(evaluate_t const &self, Algorithm algorithm, P const &p,
                      Vars &&...vars) {
      polygnition::tag_invoke(self, algorithm, p, std::forward<Vars>(vars)...);
    }
  [[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
  operator()(Algorithm algorithm, P const &p, Vars &&...vars) const {
    return polygnition::tag_invoke(*this, algorithm, p,
                                   std::forward<Vars>(vars)...);
  }
};

inline constexpr evaluate_t evaluate{};

// lightweight runtime polynomial of degree n: cₙxⁿ + cₙ₋₁xⁿ⁻¹ + … + c₀x⁰
template <typename T, int N>
  requires(N >= 0)
struct polynomial_t : std::array<T, static_cast<std::size_t>(N + 1)> {
  using value_type = T;
  static constexpr int degree = N;

  template <int Degree>
  [[nodiscard]] constexpr auto coeff() noexcept -> T & {
    static_assert(Degree >= 0);
    static_assert(Degree <= N);
    return (*this)[static_cast<std::size_t>(N - Degree)];
  }

  template <target::profile Profile = target::configured, typename... Vars>
  [[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
  operator()(Vars &&...x) const {
    return evaluate(tuned<Profile>, *this, std::forward<Vars>(x)...);
  }

  template <int Degree>
  [[nodiscard]] constexpr auto coeff() const noexcept -> T {
    static_assert(Degree >= 0);
    static_assert(Degree <= N);
    return (*this)[static_cast<std::size_t>(N - Degree)];
  }

  [[nodiscard]] constexpr auto coeff(int const degree_index) const noexcept -> T {
    if (degree_index < 0 || degree_index > N) {
      return T{};
    }
    return (*this)[static_cast<std::size_t>(N - degree_index)];
  }
};

template <polynomial_t P> struct static_poly {
  using polynomial_type = std::remove_cvref_t<decltype(P)>;
  using value_type = typename polynomial_type::value_type;
  static constexpr int degree = polynomial_type::degree;
  static inline constexpr polynomial_type value = P;

  template <target::profile Profile = target::configured, typename... Vars>
  [[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
  operator()(Vars &&...x) const {
    return evaluate(tuned<Profile>, *this, std::forward<Vars>(x)...);
  }

  template <int Degree>
  [[nodiscard]] constexpr auto coeff() const noexcept -> value_type {
    static_assert(Degree >= 0);
    static_assert(Degree <= degree);
    return value[static_cast<std::size_t>(degree - Degree)];
  }

  [[nodiscard]] constexpr auto coeff(int const degree_index) const noexcept
      -> value_type {
    if (degree_index < 0 || degree_index > degree) {
      return value_type{};
    }
    return value[static_cast<std::size_t>(degree - degree_index)];
  }

  [[nodiscard]] constexpr auto operator[](std::size_t i) const noexcept
      -> value_type {
    return value[i];
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

template <target::profile Profile, typename P> struct profile_view_t {
  P const *polynomial{};

  template <typename... Vars>
  [[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
  operator()(Vars &&...vars) const {
    return evaluate(tuned<Profile>, *polynomial, std::forward<Vars>(vars)...);
  }
};

template <target::profile Profile, typename P>
using profiled_ref = profile_view_t<Profile, P>;

template <target::profile Profile, typename P>
[[nodiscard]] constexpr auto on(P const &p) noexcept
    -> profile_view_t<Profile, std::remove_cvref_t<P>> {
  return {&p};
}

template <target::profile Profile, typename Key> struct tuning;

namespace detail {
template <typename> struct is_polynomial : std::false_type {};
template <typename T, int N>
struct is_polynomial<polynomial_t<T, N>> : std::true_type {};
template <polynomial_t P> struct is_polynomial<static_poly<P>> : std::true_type {};
template <typename T>
inline constexpr bool is_polynomial_v =
    is_polynomial<std::remove_cvref_t<T>>::value;

template <typename> struct is_static_polynomial : std::false_type {};
template <polynomial_t P> struct is_static_polynomial<static_poly<P>> : std::true_type {};
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
using stored_polynomial_type_t = typename stored_polynomial_type<std::remove_cvref_t<P>>::type;

template <typename T> struct is_tuned : std::false_type {};
template <target::profile Profile>
struct is_tuned<algorithm::tuned_t<Profile>> : std::true_type {};
template <typename T>
inline constexpr bool is_tuned_v =
    is_tuned<std::remove_cvref_t<T>>::value;

template <typename T> struct is_std_span : std::false_type {};
template <typename T, std::size_t Extent>
struct is_std_span<std::span<T, Extent>> : std::true_type {};
template <typename T>
inline constexpr bool is_std_span_v =
    is_std_span<std::remove_cvref_t<T>>::value;

template <typename...> struct first_type;
template <typename T, typename... Ts> struct first_type<T, Ts...> {
  using type = T;
};
template <typename... Ts> using first_type_t = typename first_type<Ts...>::type;

template <typename T> struct complex_value_type {
  using type = void;
};
template <typename T> struct complex_value_type<std::complex<T>> {
  using type = T;
};
template <typename T>
using complex_value_type_t = typename complex_value_type<T>::type;

template <typename T> struct complex_or_real_scalar {
  using type = std::remove_cvref_t<T>;
};
template <typename T> struct complex_or_real_scalar<std::complex<T>> {
  using type = T;
};
template <typename T>
using complex_or_real_scalar_t = typename complex_or_real_scalar<std::remove_cvref_t<T>>::type;

template <typename Like, typename Scalar> struct rebind_lane_like {
  using type = Scalar;
};

template <typename Old, std::size_t W, typename Scalar>
struct rebind_lane_like<lanes<Old, W>, Scalar> {
  using type = lanes<Scalar, W>;
};

template <typename Like, typename Scalar>
using rebind_lane_like_t =
    typename rebind_lane_like<std::remove_cvref_t<Like>, Scalar>::type;

template <typename T> struct split_complex_scalar {
  using component = typename std::remove_cvref_t<T>::value_type;
  using type = scalar_value_type_t<component>;
};

template <typename T>
using split_complex_scalar_t = typename split_complex_scalar<T>::type;

template <typename Coeff, typename Var, typename = void>
struct eval_result_from_coeff_and_var {
  using type = decltype(std::declval<Coeff>() * std::declval<Var>());
};

template <typename Coeff, simd_like Var>
struct eval_result_from_coeff_and_var<Coeff, Var, void> {
  using type = std::remove_cvref_t<Var>;
};

template <typename Coeff, split_complex_like Var>
struct eval_result_from_coeff_and_var<Coeff, Var, void> {
  using var_component = typename std::remove_cvref_t<Var>::value_type;
  using scalar = std::common_type_t<complex_or_real_scalar_t<Coeff>,
                                    split_complex_scalar_t<Var>>;
  using component = rebind_lane_like_t<var_component, scalar>;
  using type = split_complex<component>;
};

template <typename Coeff, typename Var>
  requires(!simd_like<std::remove_cvref_t<Var>>) &&
          (!split_complex_like<std::remove_cvref_t<Var>>) &&
          (complex<std::remove_cvref_t<Coeff>> || complex<std::remove_cvref_t<Var>>)
struct eval_result_from_coeff_and_var<Coeff, Var, void> {
  using scalar = std::common_type_t<complex_or_real_scalar_t<Coeff>,
                                    complex_or_real_scalar_t<Var>>;
  using type = std::complex<scalar>;
};

template <typename Coeff, typename Var>
  requires(!simd_like<std::remove_cvref_t<Var>>) &&
          (!split_complex_like<std::remove_cvref_t<Var>>) &&
          (!complex<std::remove_cvref_t<Coeff>>) &&
          (!complex<std::remove_cvref_t<Var>>) && arithmetic<std::remove_cvref_t<Coeff>> &&
          arithmetic<std::remove_cvref_t<Var>>
struct eval_result_from_coeff_and_var<Coeff, Var, void> {
  using type = std::common_type_t<std::remove_cvref_t<Coeff>, std::remove_cvref_t<Var>>;
};

template <typename Coeff, typename Var>
using eval_result_from_coeff_and_var_t = typename eval_result_from_coeff_and_var<
    std::remove_cvref_t<Coeff>, std::remove_cvref_t<Var>>::type;

template <typename P, typename Var>
using eval_result_t = eval_result_from_coeff_and_var_t<
    typename stored_polynomial_type_t<P>::value_type, std::remove_cvref_t<Var>>;

} // namespace detail
} // namespace polygnition::polynomial

#include "polygnition/detail/selection.hpp"

namespace polygnition::polynomial {
namespace detail {

template <typename P>
[[nodiscard]] constexpr decltype(auto) runtime_value(P const &p) noexcept {
  if constexpr (is_static_polynomial_v<P>) {
    return std::remove_cvref_t<P>::value;
  } else {
    return (p);
  }
}

template <std::size_t I, typename T, int N>
[[nodiscard]] constexpr auto coefficient_at(polynomial_t<T, N> const &p) noexcept
    -> T {
  static_assert(I < static_cast<std::size_t>(N + 1));
  return p[I];
}

template <std::size_t I, polynomial_t P>
[[nodiscard]] constexpr auto coefficient_at(static_poly<P> const &) noexcept
    -> typename static_poly<P>::value_type {
  using polynomial_type = typename static_poly<P>::polynomial_type;
  static_assert(I < static_cast<std::size_t>(polynomial_type::degree + 1));
  return static_poly<P>::value[I];
}

template <int Degree, typename P>
  requires is_polynomial_v<P>
[[nodiscard]] constexpr auto coefficient_by_degree(P const &p) noexcept
    -> typename stored_polynomial_type_t<P>::value_type {
  constexpr int degree = stored_polynomial_type_t<P>::degree;
  static_assert(Degree >= 0);
  static_assert(Degree <= degree);
  return coefficient_at<static_cast<std::size_t>(degree - Degree)>(p);
}

template <typename R, typename P>
  requires is_polynomial_v<P>
[[nodiscard]] constexpr auto coefficient_by_degree_as(P const &p,
                                                      int degree) noexcept -> R {
  constexpr int static_degree = stored_polynomial_type_t<P>::degree;
  if (degree < 0 || degree > static_degree) {
    return R{};
  }
  return static_cast<R>(p.coeff(degree));
}

template <typename Return, typename Coeff>
[[nodiscard]] constexpr auto lift_coefficient(Coeff const &coeff) -> Return {
  using return_type = std::remove_cvref_t<Return>;
  using coeff_type = std::remove_cvref_t<Coeff>;
  if constexpr (split_complex_like<return_type>) {
    using component_t = typename return_type::value_type;
    if constexpr (split_complex_like<coeff_type>) {
      return return_type{lift_coefficient<component_t>(coeff.real),
                         lift_coefficient<component_t>(coeff.imag)};
    } else if constexpr (complex<coeff_type>) {
      return return_type{lift_coefficient<component_t>(coeff.real()),
                         lift_coefficient<component_t>(coeff.imag())};
    } else {
      return return_type{lift_coefficient<component_t>(coeff), component_t{}};
    }
  } else if constexpr (simd_like<return_type> && !simd_like<coeff_type> &&
                std::convertible_to<Coeff, scalar_value_type_t<return_type>>) {
    return return_type{static_cast<scalar_value_type_t<return_type>>(coeff)};
  } else if constexpr (complex<return_type>) {
    using scalar = typename return_type::value_type;
    if constexpr (complex<coeff_type>) {
      return return_type{static_cast<scalar>(coeff.real()),
                         static_cast<scalar>(coeff.imag())};
    } else {
      return return_type{static_cast<scalar>(coeff), scalar{}};
    }
  } else if constexpr (std::convertible_to<Coeff, Return>) {
    return static_cast<Return>(coeff);
  } else {
    return Return{} + coeff;
  }
}

template <typename P, typename Var, typename Return, std::size_t... I>
[[nodiscard]] constexpr auto horner_indexed(P const &p, Var const &x,
                                            std::index_sequence<I...>)
    -> Return {
  auto accum = lift_coefficient<Return>(coefficient_at<0>(p));
  if constexpr (requires(Return const &a, Var const &raw) {
                  { a * raw } -> std::convertible_to<Return>;
                }) {
    ((accum = static_cast<Return>(accum * x) +
               lift_coefficient<Return>(coefficient_at<I + 1>(p))),
     ...);
  } else {
    auto const lifted_x = lift_coefficient<Return>(x);
    ((accum = (accum * lifted_x) +
               lift_coefficient<Return>(coefficient_at<I + 1>(p))),
     ...);
  }
  return accum;
}

template <int Exponent, typename Return, typename Var>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
power(Var const &x) -> Return {
  if constexpr (Exponent == 0) {
    return lift_coefficient<Return>(1);
  } else if constexpr (Exponent == 1) {
    return static_cast<Return>(x);
  } else if constexpr (Exponent % 2 == 0) {
    auto const half = power<Exponent / 2, Return>(x);
    return half * half;
  } else {
    return power<Exponent - 1, Return>(x) * static_cast<Return>(x);
  }
}

template <int J, int I, int K, typename P, typename Y, typename Return>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
dorn_fold(P const &p, Y const &y, Return const &accum) -> Return {
  if constexpr (J < I) {
    return accum;
  } else {
    return dorn_fold<J - K, I, K>(
        p, y,
        (accum * y) + lift_coefficient<Return>(coefficient_by_degree<J>(p)));
  }
}

template <int I, int K, typename P, typename Var, typename Y, typename Return>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
dorn_partial(P const &p, Var const &x, Y const &y) -> Return {
  if constexpr (I > std::remove_cvref_t<P>::degree) {
    return Return{};
  } else {
    constexpr int degree = std::remove_cvref_t<P>::degree;
    constexpr int j_max = degree - ((degree - I) % K);
    return power<I, Return>(x) *
           dorn_fold<j_max - K, I, K>(
               p, y, lift_coefficient<Return>(coefficient_by_degree<j_max>(p)));
  }
}

template <int I, int K, typename P, typename Var, typename Y, typename Return>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
dorn_sum(P const &p, Var const &x, Y const &y) -> Return {
  if constexpr (I == K) {
    return Return{};
  } else {
    return dorn_partial<I, K, P, Var, Y, Return>(p, x, y) +
           dorn_sum<I + 1, K, P, Var, Y, Return>(p, x, y);
  }
}

template <typename P, typename State, typename U, std::size_t... I>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
knuth_indexed(P const &p, U const &two_x, U const &minus_norm_sqr,
              std::index_sequence<I...>) -> State {
  auto state = State{};
  ([&] {
    auto const a = state.a;
    auto const b = state.b;
    state = State{.a = (two_x * a) + b,
                  .b = (minus_norm_sqr * a) + static_cast<U>(coefficient_at<I>(p))};
  }(),
   ...);
  return state;
}

template <typename P, typename Component, std::size_t... I>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
knuth_split_components_indexed(P const &p, Component const &x,
                               Component const &y, std::index_sequence<I...>)
    -> split_complex<Component> {
  auto a = Component{};
  auto b = Component{};
  auto const two_x = x + x;
  auto const minus_norm_sqr = Component{} - ((x * x) + (y * y));
  ([&] {
    auto const old_a = a;
    auto const old_b = b;
    a = (two_x * old_a) + old_b;
    b = (minus_norm_sqr * old_a) +
        lift_coefficient<Component>(coefficient_at<I>(p));
  }(),
   ...);
  return {(x * a) + b, y * a};
}

template <typename P, typename Component>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
knuth_split_components(P const &p, Component const &x, Component const &y)
    -> split_complex<Component> {
  constexpr auto degree = stored_polynomial_type_t<P>::degree;
  return knuth_split_components_indexed<P, Component>(
      p, x, y, std::make_index_sequence<static_cast<std::size_t>(degree + 1)>{});
}

template <std::size_t... I, typename A, typename B>
[[nodiscard]] constexpr auto equal_coefficients(A const &a, B const &b,
                                                std::index_sequence<I...>) -> bool {
  return ((coefficient_at<I>(a) == coefficient_at<I>(b)) and ...);
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
[[nodiscard]] constexpr auto
 equal_coefficients_by_degree(A const &a, B const &b,
                              std::integer_sequence<int, Degree...>) -> bool {
  return ((coefficient_by_degree_or_zero<Degree>(a) ==
           coefficient_by_degree_or_zero<Degree>(b)) and
          ...);
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
[[nodiscard]] constexpr auto
effective_degree_from_coefficients(Coefficients const &coefficients,
                                   int static_degree) noexcept -> int {
  auto const first_nonzero = std::ranges::find_if_not(coefficients, is_zero);
  return static_degree -
         static_cast<int>(first_nonzero - std::ranges::begin(coefficients));
}
} // namespace detail

template <target::profile Profile, typename Key> struct tuning {
  [[nodiscard]] static consteval auto try_select_choice() {
    return detail::profile_tuning<Profile>::table::
        template try_select_valid_choice<Key, typename Key::polynomial_type,
                                         typename Key::var_type>();
  }

  [[nodiscard]] static consteval auto select() {
    return detail::profile_tuning<Profile>::table::template select_valid<
        Key, typename Key::polynomial_type, typename Key::var_type>();
  }
};

using select_t = detail::select_t;
inline constexpr select_t select_strategy{};

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

// Equality is structural: the static degree and every coefficient must match.
// Use same_polynomial where leading-zero storage should be ignored.
template <typename A, typename B>
  requires detail::is_polynomial_v<A> && detail::is_polynomial_v<B>
[[nodiscard]] constexpr auto operator==(A const &a, B const &b) -> bool {
  constexpr auto a_degree = detail::stored_polynomial_type_t<A>::degree;
  constexpr auto b_degree = detail::stored_polynomial_type_t<B>::degree;
  if constexpr (a_degree != b_degree) {
    return false;
  } else {
    return detail::equal_coefficients(
        a, b, std::make_index_sequence<static_cast<std::size_t>(a_degree + 1)>{});
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
  constexpr auto max_degree = (a_degree > b_degree) ? a_degree : b_degree;
  return detail::equal_coefficients_by_degree(
      a, b, std::make_integer_sequence<int, max_degree + 1>{});
}

template <typename P>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto effective_degree(P const &p) noexcept -> int {
  return detail::effective_degree_from_coefficients(p, degree(p));
}

// preprocessed Motzkin coefficients for degree-4 monic normalization
template <typename T>
  requires std::floating_point<T>
struct motzkin_preprocessed_t {
  T leading;
  T beta0;
  T beta1;
  T beta2;
  T beta3;

  template <typename Var>
  [[nodiscard]] constexpr auto operator()(Var &&x) const {
    return evaluate(motzkin, *this, std::forward<Var>(x));
  }
};

namespace detail {
template <typename T>
[[nodiscard]] constexpr auto
preprocess_motzkin_coeffs(T const &leading, T const &c3, T const &c2,
                          T const &c1, T const &c0) {
  auto const inv_leading = T{1} / leading;
  auto const a3 = c3 * inv_leading;
  auto const a2 = c2 * inv_leading;
  auto const a1 = c1 * inv_leading;
  auto const a0 = c0 * inv_leading;

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
           std::floating_point<typename detail::stored_polynomial_type_t<P>::value_type>
[[nodiscard]] constexpr auto preprocess_motzkin(P const &p) {
  return detail::preprocess_motzkin_coeffs(
      detail::coefficient_at<0>(p), detail::coefficient_at<1>(p),
      detail::coefficient_at<2>(p), detail::coefficient_at<3>(p),
      detail::coefficient_at<4>(p));
}

template <typename T, typename Var>
  requires std::floating_point<T> and
           (arithmetic<std::remove_cvref_t<Var>> or
            simd_like<std::remove_cvref_t<Var>>)
[[nodiscard]] constexpr auto
evaluate_motzkin(motzkin_preprocessed_t<T> const &pre, Var const &x) {
  using compute_t = detail::eval_result_from_coeff_and_var_t<T, Var>;
  auto const xc = detail::lift_coefficient<compute_t>(x);
  auto const b0 = detail::lift_coefficient<compute_t>(pre.beta0);
  auto const b1 = detail::lift_coefficient<compute_t>(pre.beta1);
  auto const b2 = detail::lift_coefficient<compute_t>(pre.beta2);
  auto const b3 = detail::lift_coefficient<compute_t>(pre.beta3);
  auto const lead = detail::lift_coefficient<compute_t>(pre.leading);

  auto const y = (xc + b0) * xc + b1;
  return ((y + xc + b2) * y + b3) * lead;
}

template <typename P, typename Var>
  requires detail::is_polynomial_v<P> &&
           (detail::stored_polynomial_type_t<P>::degree == 4) &&
           std::floating_point<typename detail::stored_polynomial_type_t<P>::value_type> and
           (arithmetic<std::remove_cvref_t<Var>> or
            simd_like<std::remove_cvref_t<Var>>)
[[nodiscard]] constexpr auto evaluate_motzkin(P const &p, Var const &x) {
  if constexpr (detail::is_static_polynomial_v<P>) {
    constexpr auto pre = preprocess_motzkin(std::remove_cvref_t<P>{});
    return evaluate_motzkin(pre, x);
  } else {
    return evaluate_motzkin(preprocess_motzkin(p), x);
  }
}

// evaluate polynomial using Horner's method as left fold over [cₙ, cₙ₋₁, …, c₀]
// P(x) = (…(cₙ * x + cₙ₋₁) * x … ) * x + c₀
template <typename P, typename Var>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto evaluate_horner(P const &p, Var const &x) {
  using return_t = detail::eval_result_t<P, Var>;
  constexpr auto degree = detail::stored_polynomial_type_t<P>::degree;
  return detail::horner_indexed<P, Var, return_t>(
      p, x, std::make_index_sequence<static_cast<std::size_t>(degree)>{});
}

template <typename P, typename V>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::horner_t,
                                        P const &p, V &&v) {
  return evaluate_horner(p, std::forward<V>(v));
}

// Dorn's residue-class decomposition:
// P(x) = Σᵢ xⁱ Pᵢ(xᴷ), where Pᵢ holds coefficients with degree ≡ i mod K.
template <int K, typename P, typename Var>
  requires(K > 1) && detail::is_polynomial_v<P>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
evaluate_dorn(P const &p, Var const &x) {
  using return_t = detail::eval_result_t<P, Var>;
  auto const y = detail::power<K, return_t>(x);
  return detail::dorn_sum<0, K, P, Var, return_t, return_t>(p, x, y);
}

template <int K, typename P, typename V>
  requires detail::is_polynomial_v<P>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::dorn_t<K>, P const &p, V &&v) {
  return evaluate_dorn<K>(p, std::forward<V>(v));
}

namespace detail {
template <std::size_t I, typename Return, typename P, typename Var>
[[nodiscard]] constexpr auto estrin_pair(P const &p, Var const &x) {
  return (lift_coefficient<Return>(coefficient_at<I>(p)) * x) +
         lift_coefficient<Return>(coefficient_at<I + 1>(p));
}

// Estrin folds adjacent coefficients into linear blocks, then recursively
// evaluates those blocks as a polynomial in x². This trades extra independent
// multiplies for shorter dependency chains than Horner.
template <typename P, typename Var>
[[nodiscard]] constexpr auto estrin_eval(P const &p, Var x) {
  constexpr int degree = stored_polynomial_type_t<P>::degree;
  using return_t = eval_result_t<P, Var>;

  if constexpr (degree == 0) {
    return lift_coefficient<return_t>(coefficient_at<0>(p));
  } else if constexpr (degree % 2 == 0) {
    constexpr int half_degree = degree / 2;
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto const q = polynomial_t<return_t, half_degree>{
          lift_coefficient<return_t>(coefficient_at<0>(p)),
          estrin_pair<2 * (I + 1) - 1, return_t>(p, x)...};
      return estrin_eval(q, x * x);
    }(std::make_index_sequence<static_cast<std::size_t>(half_degree)>{});
  } else {
    constexpr int half_degree = degree / 2;
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto const q = polynomial_t<return_t, half_degree>{
          estrin_pair<2 * I, return_t>(p, x)...};
      return estrin_eval(q, x * x);
    }(std::make_index_sequence<static_cast<std::size_t>(half_degree + 1)>{});
  }
}
} // namespace detail

template <typename P, typename Var>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto evaluate_estrin(P const &p, Var &&x) {
  return detail::estrin_eval(p, std::forward<Var>(x));
}

template <typename P, typename V>
  requires detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::estrin_t,
                                        P const &p, V &&v) {
  return evaluate_estrin(p, std::forward<V>(v));
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
  constexpr int power_index = degree - static_cast<int>(I);
  return abs_value(static_cast<Bound>(coefficient_at<I>(p))) *
         pow_unsigned<power_index>(abs_value(xmax));
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
  auto const low = a - high;
  return {.value = high, .error = low};
}

template <typename T>
[[nodiscard]] constexpr auto two_sum(T a, T b) -> eft_pair<T> {
  auto const s = a + b;
  auto const bb = s - a;
  auto const err = (a - (s - bb)) + (b - bb);
  return {.value = s, .error = err};
}

template <typename T>
[[nodiscard]] constexpr auto two_prod(T a, T b) -> eft_pair<T> {
  auto const p = a * b;
  auto const as = split(a);
  auto const bs = split(b);
  auto const err = ((as.value * bs.value - p) + as.value * bs.error +
                    as.error * bs.value) +
                   as.error * bs.error;
  return {.value = p, .error = err};
}

template <typename P, typename Var, typename Return, std::size_t... I>
[[nodiscard]] constexpr auto compensated_horner_indexed(
    P const &p, Var const &x, std::index_sequence<I...>) -> Return {
  auto r = lift_coefficient<Return>(coefficient_at<0>(p));
  auto e = Return{};
  auto const xc = static_cast<Return>(x);
  ([&] {
    auto const product = two_prod(r, xc);
    auto const sum =
        two_sum(product.value,
                static_cast<Return>(coefficient_at<I + 1>(p)));
    e = (e * xc) + (product.error + sum.error);
    r = sum.value;
  }(),
   ...);
  return r + e;
}
} // namespace detail

template <typename P, typename X>
  requires detail::is_polynomial_v<P> &&
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> and
           arithmetic<std::remove_cvref_t<X>>
[[nodiscard]] constexpr auto error_bound(P const &p, X const &xmax) {
  using coeff_t = typename detail::stored_polynomial_type_t<P>::value_type;
  constexpr auto degree = detail::stored_polynomial_type_t<P>::degree;
  using bound_t = std::common_type_t<double, coeff_t, std::remove_cvref_t<X>>;
  auto constexpr ops = 2 * degree;
  auto constexpr eps = std::numeric_limits<bound_t>::epsilon() / bound_t{2};
  auto constexpr gamma = (ops * eps) / (bound_t{1} - (ops * eps));
  return gamma * detail::weighted_coefficient_sum<bound_t>(
                     p, static_cast<bound_t>(xmax),
                     std::make_index_sequence<static_cast<std::size_t>(degree + 1)>{});
}

template <typename P, typename Var>
  requires detail::is_polynomial_v<P> && arithmetic<std::remove_cvref_t<Var>> and
           std::floating_point<detail::eval_result_t<P, Var>>
[[nodiscard]] constexpr auto evaluate_compensated(P const &p, Var const &x) {
  using return_t = detail::eval_result_t<P, Var>;
  constexpr auto degree = detail::stored_polynomial_type_t<P>::degree;
  return detail::compensated_horner_indexed<P, Var, return_t>(
      p, x, std::make_index_sequence<static_cast<std::size_t>(degree)>{});
}

template <typename P, typename V>
  requires detail::is_polynomial_v<P> and
           requires(P const &p, V const &v) { evaluate_compensated(p, v); }
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::compensated_t,
                                        P const &p, V &&v) {
  return evaluate_compensated(p, std::forward<V>(v));
}

// evaluate multivariate polynomial as left fold over [cₙ, cₙ₋₁, …, c₀]
// P(x, xs…) = (…(0 * x + cₙ(xs…)) * x + cₙ₋₁(xs…) … ) * x + c₀(xs…)
// NOTE: multivariate polynomials have callable coefficients (like polynomials)
template <typename P, typename Var, typename... Remaining>
  requires detail::is_polynomial_v<P> && (sizeof...(Remaining) > 0)
[[nodiscard]] constexpr auto evaluate_multivariate(P const &p, Var const &x,
                                                   Remaining const &...xs) {
  using coeff_t = typename detail::stored_polynomial_type_t<P>::value_type;
  using inner_return_t = decltype(std::declval<coeff_t>()(xs...));
  using return_t = std::common_type_t<decltype(std::declval<Var>() *
                                               std::declval<inner_return_t>()),
                                      inner_return_t>;
  auto accum = return_t{};
  for (auto const &coeff : p) {
    accum = (accum * x) + coeff(xs...);
  }
  return accum;
}

template <typename P, typename... V>
  requires detail::is_polynomial_v<P> and (sizeof...(V) > 1)
[[nodiscard]] constexpr auto
tag_invoke(evaluate_t, algorithm::multivariate_horner_t, P const &p, V &&...v) {
  return evaluate_multivariate(p, std::forward<V>(v)...);
}

// evaluate real polynomial u(z) = uₙ zⁿ + ... + u₀ at complex point z = x + iy
// using Knuth's algorithm (AoCP, Vol. 2, Sec. 4.6.4, Eq. 2).
// computes u(x + iy) = z ⋅ aₙ + bₙ where aⱼ, bⱼ are defined by recurrence:
// a₀ = 0, b₀ = 0
// aⱼ = bⱼ₋₁ + 2x aⱼ₋₁
// bⱼ = uₙ₋ⱼ − (x² + y²) aⱼ₋₁
template <typename P, typename U>
  requires detail::is_polynomial_v<P> &&
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> and
           std::floating_point<U>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
evaluate_complex_knuth(P const &u, std::complex<U> const &z) {
  using result_t = detail::eval_result_t<P, std::complex<U>>;
  using scalar_t = typename result_t::value_type;
  auto const y = detail::knuth_split_components(
      u, static_cast<scalar_t>(z.real()), static_cast<scalar_t>(z.imag()));
  return result_t{y.real, y.imag};
}


template <typename P, typename Z>
  requires detail::is_polynomial_v<P> &&
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> &&
           split_complex_like<std::remove_cvref_t<Z>> &&
           std::floating_point<detail::split_complex_scalar_t<std::remove_cvref_t<Z>>>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
evaluate_complex_knuth(P const &u, Z const &z) {
  using result_t = detail::eval_result_t<P, Z>;
  using component_t = typename result_t::value_type;
  auto const y = detail::knuth_split_components(
      u, detail::lift_coefficient<component_t>(z.real),
      detail::lift_coefficient<component_t>(z.imag));
  return result_t{y.real, y.imag};
}


template <typename P, typename V>
  requires detail::is_polynomial_v<P> and
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> and
           complex<std::remove_cvref_t<V>> and
           std::floating_point<typename std::remove_cvref_t<V>::value_type>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::knuth_t, P const &p, V &&v) {
  return evaluate_complex_knuth(p, std::forward<V>(v));
}

template <typename P, typename V>
  requires detail::is_polynomial_v<P> &&
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> &&
           split_complex_like<std::remove_cvref_t<V>> &&
           std::floating_point<detail::split_complex_scalar_t<std::remove_cvref_t<V>>>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::knuth_t, P const &p, V &&v) {
  return evaluate_complex_knuth(p, std::forward<V>(v));
}

template <typename P, typename V>
  requires detail::is_polynomial_v<P> &&
           (detail::stored_polynomial_type_t<P>::degree == 4) &&
           std::floating_point<typename detail::stored_polynomial_type_t<P>::value_type> and
           (arithmetic<std::remove_cvref_t<V>> or
            simd_like<std::remove_cvref_t<V>>)
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::motzkin_t,
                                        P const &p, V &&v) {
  return evaluate_motzkin(p, std::forward<V>(v));
}

template <typename T, typename V>
  requires std::floating_point<T> and
           (arithmetic<std::remove_cvref_t<V>> or
            simd_like<std::remove_cvref_t<V>>)
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::motzkin_t,
                                        motzkin_preprocessed_t<T> const &p,
                                        V &&v) {
  return evaluate_motzkin(p, std::forward<V>(v));
}

namespace detail {
constexpr void require_equal_batch_sizes(std::size_t xs, std::size_t out) {
  if (xs != out) {
    throw std::invalid_argument{
        "polygnition batch input and output sizes differ"};
  }
}

template <typename Algorithm, typename E, typename X, typename R>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
evaluate_batch_scalar(Algorithm algorithm, E const &e, std::span<X const> xs,
                      std::span<R> out) -> std::span<R> {
  require_equal_batch_sizes(xs.size(), out.size());
  for (auto i = std::size_t{}; i < xs.size(); ++i) {
    out[i] = static_cast<R>(evaluate(algorithm, e, xs[i]));
  }
  return out;
}

template <typename X> [[nodiscard]] consteval auto default_batch_lanes() {
  return static_cast<std::size_t>(default_batch_lane_count<X>());
}


template <std::size_t J, typename Algorithm, typename E, typename X, typename R,
          std::size_t LaneCount>
POLYGNITION_DETAIL_ALWAYS_INLINE constexpr void evaluate_batch_lane(
    Algorithm algorithm, E const &e, std::span<X const> xs, std::span<R> out,
    std::size_t i) {
  using lane_t = lanes<X, LaneCount>;
  auto const x = lane_t::load(xs.data() + i + (J * LaneCount));
  auto const y = evaluate(algorithm, e, x);
  y.store(out.data() + i + (J * LaneCount));
}

template <typename Algorithm, typename E, typename X, typename R,
          std::size_t LaneCount, std::size_t... J>
POLYGNITION_DETAIL_ALWAYS_INLINE constexpr void evaluate_batch_unrolled(
    Algorithm algorithm, E const &e, std::span<X const> xs, std::span<R> out,
    std::size_t i, std::index_sequence<J...>) {
  (evaluate_batch_lane<J, Algorithm, E, X, R, LaneCount>(algorithm, e, xs, out,
                                                        i),
   ...);
}

template <typename Algorithm, typename E, typename X, typename R,
          std::size_t LaneCount, std::size_t Unroll>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
evaluate_batch_blocked_width(Algorithm algorithm, E const &e,
                              std::span<X const> xs, std::span<R> out)
    -> std::span<R> {
  require_equal_batch_sizes(xs.size(), out.size());
  if constexpr (LaneCount <= 1 || !std::same_as<X, R> || !arithmetic<X>) {
    return evaluate_batch_scalar(algorithm, e, xs, out);
  } else {
    using lane_t = lanes<X, LaneCount>;
    if constexpr (requires(lane_t x) {
                    { evaluate(algorithm, e, x) } -> std::same_as<lane_t>;
                  }) {
      if (std::is_constant_evaluated()) {
        return evaluate_batch_scalar(algorithm, e, xs, out);
      }
      auto i = std::size_t{};
      constexpr auto block = LaneCount * Unroll;
      for (; i + block <= xs.size(); i += block) {
        evaluate_batch_unrolled<Algorithm, E, X, R, LaneCount>(
            algorithm, e, xs, out, i, std::make_index_sequence<Unroll>{});
      }
      for (; i + LaneCount <= xs.size(); i += LaneCount) {
        auto const x = lane_t::load(xs.data() + i);
        auto const y = evaluate(algorithm, e, x);
        y.store(out.data() + i);
      }
      for (; i < xs.size(); ++i) {
        out[i] = static_cast<R>(evaluate(algorithm, e, xs[i]));
      }
      return out;
    } else {
      return evaluate_batch_scalar(algorithm, e, xs, out);
    }
  }
}

template <typename Algorithm, typename E, typename X, typename R>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto evaluate_batch_blocked(Algorithm algorithm,
                                                    E const &e,
                                                    std::span<X const> xs,
                                                    std::span<R> out)
    -> std::span<R> {
  return evaluate_batch_blocked_width<Algorithm, E, X, R,
                                      default_batch_lanes<X>(), 4>(algorithm, e,
                                                                   xs, out);
}
} // namespace detail

template <typename Algorithm, typename E, typename X, typename R>
  requires(!detail::is_tuned_v<Algorithm>) &&
          requires(Algorithm algorithm, E const &e, X const &x) {
            { evaluate(algorithm, e, x) } -> std::convertible_to<R>;
          }
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, Algorithm algorithm, E const &e, std::span<X const> xs,
           std::span<R> out) -> std::span<R> {
  return detail::evaluate_batch_blocked(algorithm, e, xs, out);
}

namespace detail {
template <target::profile Cpu, evaluation_intent Intent, typename P,
          typename... Vars>
[[nodiscard]] consteval auto select_algorithm_for() {
  static_assert(sizeof...(Vars) > 0,
                "polynomial may only be evaluated on one point per variable");
  if constexpr (sizeof...(Vars) > 1) {
    return multivariate_horner;
  } else {
    using var_type = std::remove_cvref_t<first_type_t<Vars...>>;
    constexpr auto effective_intent = simd_like<var_type>
                                          ? evaluation_intent::throughput
                                          : Intent;
    using key = selection_key<P, var_type, effective_intent>;
    return select_t{}(tuned<Cpu>, std::type_identity<key>{});
  }
}

template <target::profile Cpu, typename P, typename... Vars>
[[nodiscard]] consteval auto select_algorithm_for() {
  return select_algorithm_for<Cpu, evaluation_intent::latency, P, Vars...>();
}

template <target::profile Cpu, typename P, typename... Vars>
[[nodiscard]] consteval auto select_throughput_algorithm_for() {
  return select_algorithm_for<Cpu, evaluation_intent::throughput, P, Vars...>();
}

template <typename P, typename... Vars>
[[nodiscard]] consteval auto select_algorithm() {
  return select_algorithm_for<target::configured, P, Vars...>();
}

template <target::profile Cpu, typename P, typename... Vars>
using selected_algorithm_t_for =
    decltype(select_algorithm_for<Cpu, P, Vars...>());

template <target::profile Cpu, typename P, typename... Vars>
using selected_throughput_algorithm_t_for =
    decltype(select_throughput_algorithm_for<Cpu, P, Vars...>());

template <typename P, typename... Vars>
using selected_algorithm_t = decltype(select_algorithm<P, Vars...>());
} // namespace detail

template <target::profile Profile, typename P, typename X, typename R>
  requires detail::is_polynomial_v<P>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::tuned_t<Profile>, P const &p,
           std::span<X const> xs, std::span<R> out) -> std::span<R> {
  if constexpr (arithmetic<X> && std::same_as<X, R>) {
    using key = detail::selection_key<P, X, detail::evaluation_intent::throughput>;
    constexpr auto batch =
        detail::profile_tuning<Profile>::table::template select_batch_choice<key, P, X>();
    constexpr auto algorithm = detail::strategy_constant<batch.algorithm>();
    return detail::evaluate_batch_blocked_width<decltype(algorithm), P, X, R,
                                                static_cast<std::size_t>(batch.lanes),
                                                static_cast<std::size_t>(batch.unroll)>(
        algorithm, p, xs, out);
  } else {
    return evaluate(detail::select_throughput_algorithm_for<Profile, P, X>(), p,
                    xs, out);
  }
}

template <target::profile Profile, typename P, std::ranges::contiguous_range Xs,
          std::ranges::contiguous_range Out>
  requires detail::is_polynomial_v<P> && (!detail::is_std_span_v<Xs>) &&
           (!detail::is_std_span_v<Out>) && std::ranges::sized_range<Xs> &&
           std::ranges::sized_range<Out> &&
           (!std::is_const_v<
               std::remove_reference_t<std::ranges::range_reference_t<Out>>>) &&
           requires(P const &p, Xs &&xs, Out &&out) {
             evaluate(tuned<Profile>, p,
                      std::span<std::ranges::range_value_t<Xs> const>{
                          std::ranges::data(xs), std::ranges::size(xs)},
                      std::span<std::ranges::range_value_t<Out>>{
                          std::ranges::data(out), std::ranges::size(out)});
           }
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, algorithm::tuned_t<Profile>,
                                        P const &p, Xs &&xs, Out &&out) {
  using x_value = std::ranges::range_value_t<Xs>;
  using r_value = std::ranges::range_value_t<Out>;
  return evaluate(tuned<Profile>, p,
                  std::span<x_value const>{std::ranges::data(xs),
                                           std::ranges::size(xs)},
                  std::span<r_value>{std::ranges::data(out),
                                     std::ranges::size(out)});
}

template <typename Algorithm, typename E, std::ranges::contiguous_range Xs,
          std::ranges::contiguous_range Out>
  requires(!detail::is_std_span_v<Xs> && !detail::is_std_span_v<Out> &&
           std::ranges::sized_range<Xs> && std::ranges::sized_range<Out> &&
           (!std::is_const_v<std::ranges::range_value_t<Out>>) &&
           requires(Algorithm algorithm, E const &e, Xs &&xs, Out &&out) {
             evaluate(algorithm, e,
                      std::span<std::ranges::range_value_t<Xs> const>{
                          std::ranges::data(xs), std::ranges::size(xs)},
                      std::span<std::ranges::range_value_t<Out>>{
                          std::ranges::data(out), std::ranges::size(out)});
           })
[[nodiscard]] constexpr auto tag_invoke(evaluate_t, Algorithm algorithm,
                                        E const &e, Xs &&xs, Out &&out) {
  using x_value = std::ranges::range_value_t<Xs>;
  using r_value = std::ranges::range_value_t<Out>;
  return evaluate(algorithm, e,
                  std::span<x_value const>{std::ranges::data(xs),
                                           std::ranges::size(xs)},
                  std::span<r_value>{std::ranges::data(out),
                                     std::ranges::size(out)});
}

template <typename R> struct complex_split_span {
  std::span<R> real;
  std::span<R> imag;
};

#if defined(__GNUC__) || defined(__clang__)
#define POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE inline
#endif

namespace detail {
template <typename Algorithm, typename P, typename X, typename R>
[[nodiscard]] POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr auto
evaluate_complex_split_batch_scalar(Algorithm algorithm, P const &p,
                                    std::span<X const> real,
                                    std::span<X const> imag,
                                    std::span<R> out_real,
                                    std::span<R> out_imag)
    -> complex_split_span<R> {
  require_equal_batch_sizes(real.size(), imag.size());
  require_equal_batch_sizes(real.size(), out_real.size());
  require_equal_batch_sizes(real.size(), out_imag.size());
  for (auto i = std::size_t{}; i < real.size(); ++i) {
    auto const y = evaluate(algorithm, p, split_complex<X>{real[i], imag[i]});
    out_real[i] = static_cast<R>(y.real);
    out_imag[i] = static_cast<R>(y.imag);
  }
  return {.real = out_real, .imag = out_imag};
}

template <std::size_t J, typename Algorithm, typename P, typename X, typename R,
          std::size_t LaneCount>
POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr void
evaluate_complex_split_batch_lane(Algorithm algorithm, P const &p,
                                  std::span<X const> real,
                                  std::span<X const> imag,
                                  std::span<R> out_real,
                                  std::span<R> out_imag, std::size_t i) {
  static_assert(std::same_as<X, R>,
                "split-complex vector lanes require matching input/output scalars");
  using lane_t = lanes<X, LaneCount>;
  auto const x = lane_t::load(real.data() + i + (J * LaneCount));
  auto const yi = lane_t::load(imag.data() + i + (J * LaneCount));
  if constexpr (std::same_as<std::remove_cvref_t<Algorithm>, algorithm::knuth_t>) {
    auto const y = knuth_split_components(p, x, yi);
    y.real.store(out_real.data() + i + (J * LaneCount));
    y.imag.store(out_imag.data() + i + (J * LaneCount));
  } else {
    auto const y = evaluate(algorithm, p, split_complex<lane_t>{x, yi});
    static_assert(split_complex_like<decltype(y)>);
    y.real.store(out_real.data() + i + (J * LaneCount));
    y.imag.store(out_imag.data() + i + (J * LaneCount));
  }
}

template <typename Algorithm, typename P, typename X, typename R,
          std::size_t LaneCount, std::size_t... J>
POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr void
evaluate_complex_split_batch_unrolled(Algorithm algorithm, P const &p,
                                      std::span<X const> real,
                                      std::span<X const> imag,
                                      std::span<R> out_real,
                                      std::span<R> out_imag, std::size_t i,
                                      std::index_sequence<J...>) {
  (evaluate_complex_split_batch_lane<J, Algorithm, P, X, R, LaneCount>(
       algorithm, p, real, imag, out_real, out_imag, i),
   ...);
}

template <typename Algorithm, typename P, typename X, typename R,
          std::size_t LaneCount, std::size_t Unroll>
[[nodiscard]] POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr auto
evaluate_complex_split_batch_blocked_width(Algorithm algorithm, P const &p,
                                           std::span<X const> real,
                                           std::span<X const> imag,
                                           std::span<R> out_real,
                                           std::span<R> out_imag)
    -> complex_split_span<R> {
  static_assert(LaneCount > 0);
  static_assert(Unroll > 0);
  require_equal_batch_sizes(real.size(), imag.size());
  require_equal_batch_sizes(real.size(), out_real.size());
  require_equal_batch_sizes(real.size(), out_imag.size());
  if constexpr (std::floating_point<X> && std::same_as<X, R>) {
    if (!std::is_constant_evaluated()) {
      constexpr auto block = LaneCount * Unroll;
      auto i = std::size_t{};
      for (; i + block <= real.size(); i += block) {
        evaluate_complex_split_batch_unrolled<Algorithm, P, X, R, LaneCount>(
            algorithm, p, real, imag, out_real, out_imag, i,
            std::make_index_sequence<Unroll>{});
      }
      for (; i + LaneCount <= real.size(); i += LaneCount) {
        evaluate_complex_split_batch_lane<0, Algorithm, P, X, R, LaneCount>(
            algorithm, p, real, imag, out_real, out_imag, i);
      }
      for (; i < real.size(); ++i) {
        auto const y = evaluate(algorithm, p, split_complex<X>{real[i], imag[i]});
        out_real[i] = static_cast<R>(y.real);
        out_imag[i] = static_cast<R>(y.imag);
      }
      return {.real = out_real, .imag = out_imag};
    }
  }
  return evaluate_complex_split_batch_scalar(algorithm, p, real, imag, out_real,
                                             out_imag);
}

template <typename P, typename X, typename R>
[[nodiscard]] POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr auto
evaluate_complex_knuth_batch_split(
    P const &p, std::span<X const> real, std::span<X const> imag,
    std::span<R> out_real, std::span<R> out_imag) -> complex_split_span<R> {
  using key = selection_key<P, split_complex<X>, evaluation_intent::throughput>;
  constexpr auto selected =
      profile_tuning<target::configured>::table::template select_batch_choice<
          key, P, split_complex<X>>();
  constexpr auto knuth_choice = algorithm_choice{strategy_category::knuth, 0};
  constexpr auto lane_count = selected.algorithm == knuth_choice
                                  ? selected.lanes
                                  : static_cast<int>(default_batch_lanes<X>());
  constexpr auto unroll = selected.algorithm == knuth_choice ? selected.unroll : 4;
  return evaluate_complex_split_batch_blocked_width<
      algorithm::knuth_t, P, X, R, static_cast<std::size_t>(lane_count),
      static_cast<std::size_t>(unroll)>(knuth, p, real, imag, out_real,
                                        out_imag);
}
} // namespace detail

template <typename P, typename X, typename R>
  requires detail::is_polynomial_v<P> and
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> and
           std::floating_point<X> and std::floating_point<R>
[[nodiscard]] POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::knuth_t,
           P const &p, std::span<X const> real, std::span<X const> imag,
           std::span<R> out_real, std::span<R> out_imag)
    -> complex_split_span<R> {
  return detail::evaluate_complex_knuth_batch_split(p, real, imag, out_real,
                                                    out_imag);
}


template <target::profile Profile, typename P, typename X, typename R>
  requires detail::is_polynomial_v<P> &&
           arithmetic<typename detail::stored_polynomial_type_t<P>::value_type> &&
           std::floating_point<X> && std::floating_point<R>
[[nodiscard]] POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::tuned_t<Profile>, P const &p,
           std::span<X const> real, std::span<X const> imag,
           std::span<R> out_real, std::span<R> out_imag)
    -> complex_split_span<R> {
  if constexpr (std::same_as<X, R>) {
    using split_x = split_complex<X>;
    using key = detail::selection_key<P, split_x,
                                      detail::evaluation_intent::throughput>;
    constexpr auto batch = detail::profile_tuning<Profile>::table::
        template select_batch_choice<key, P, split_x>();
    constexpr auto algorithm = detail::strategy_constant<batch.algorithm>();
    return detail::evaluate_complex_split_batch_blocked_width<
        decltype(algorithm), P, X, R, static_cast<std::size_t>(batch.lanes),
        static_cast<std::size_t>(batch.unroll)>(algorithm, p, real, imag,
                                                out_real, out_imag);
  } else {
    constexpr auto algorithm =
        detail::select_throughput_algorithm_for<Profile, P, split_complex<X>>();
    return detail::evaluate_complex_split_batch_scalar(algorithm, p, real, imag,
                                                       out_real, out_imag);
  }
}

template <target::profile Profile, typename P,
          std::ranges::contiguous_range Real,
          std::ranges::contiguous_range Imag,
          std::ranges::contiguous_range OutReal,
          std::ranges::contiguous_range OutImag>
  requires detail::is_polynomial_v<P> &&
           (!detail::is_std_span_v<Real> && !detail::is_std_span_v<Imag> &&
            !detail::is_std_span_v<OutReal> && !detail::is_std_span_v<OutImag>) &&
           std::ranges::sized_range<Real> && std::ranges::sized_range<Imag> &&
           std::ranges::sized_range<OutReal> &&
           std::ranges::sized_range<OutImag> &&
           std::same_as<std::ranges::range_value_t<Real>,
                        std::ranges::range_value_t<Imag>> &&
           std::same_as<std::ranges::range_value_t<OutReal>,
                        std::ranges::range_value_t<OutImag>> &&
           (!std::is_const_v<std::ranges::range_value_t<OutReal>>) &&
           requires(P const &p, Real &&real, Imag &&imag, OutReal &&out_real,
                    OutImag &&out_imag) {
             evaluate(tuned<Profile>, p,
                      std::span<std::ranges::range_value_t<Real> const>{
                          std::ranges::data(real), std::ranges::size(real)},
                      std::span<std::ranges::range_value_t<Imag> const>{
                          std::ranges::data(imag), std::ranges::size(imag)},
                      std::span<std::ranges::range_value_t<OutReal>>{
                          std::ranges::data(out_real),
                          std::ranges::size(out_real)},
                      std::span<std::ranges::range_value_t<OutImag>>{
                          std::ranges::data(out_imag),
                          std::ranges::size(out_imag)});
           }
[[nodiscard]] POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::tuned_t<Profile>, P const &p, Real &&real,
           Imag &&imag, OutReal &&out_real, OutImag &&out_imag) {
  using x_value = std::ranges::range_value_t<Real>;
  using r_value = std::ranges::range_value_t<OutReal>;
  return evaluate(tuned<Profile>, p,
                  std::span<x_value const>{std::ranges::data(real),
                                           std::ranges::size(real)},
                  std::span<x_value const>{std::ranges::data(imag),
                                           std::ranges::size(imag)},
                  std::span<r_value>{std::ranges::data(out_real),
                                     std::ranges::size(out_real)},
                  std::span<r_value>{std::ranges::data(out_imag),
                                     std::ranges::size(out_imag)});
}

template <typename P, std::ranges::contiguous_range Real,
          std::ranges::contiguous_range Imag,
          std::ranges::contiguous_range OutReal,
          std::ranges::contiguous_range OutImag>
  requires detail::is_polynomial_v<P> &&
           (!detail::is_std_span_v<Real> && !detail::is_std_span_v<Imag> &&
            !detail::is_std_span_v<OutReal> && !detail::is_std_span_v<OutImag>) &&
           std::ranges::sized_range<Real> && std::ranges::sized_range<Imag> &&
           std::ranges::sized_range<OutReal> &&
           std::ranges::sized_range<OutImag> &&
           std::same_as<std::ranges::range_value_t<Real>,
                        std::ranges::range_value_t<Imag>> &&
           std::same_as<std::ranges::range_value_t<OutReal>,
                        std::ranges::range_value_t<OutImag>> &&
           (!std::is_const_v<std::ranges::range_value_t<OutReal>>) &&
           requires(P const &p, Real &&real, Imag &&imag, OutReal &&out_real,
                    OutImag &&out_imag) {
             evaluate(knuth, p,
                      std::span<std::ranges::range_value_t<Real> const>{
                          std::ranges::data(real), std::ranges::size(real)},
                      std::span<std::ranges::range_value_t<Imag> const>{
                          std::ranges::data(imag), std::ranges::size(imag)},
                      std::span<std::ranges::range_value_t<OutReal>>{
                          std::ranges::data(out_real),
                          std::ranges::size(out_real)},
                      std::span<std::ranges::range_value_t<OutImag>>{
                          std::ranges::data(out_imag),
                          std::ranges::size(out_imag)});
           }
[[nodiscard]] POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::knuth_t, P const &p, Real &&real,
           Imag &&imag, OutReal &&out_real, OutImag &&out_imag) {
  using x_value = std::ranges::range_value_t<Real>;
  using r_value = std::ranges::range_value_t<OutReal>;
  return evaluate(knuth, p,
                  std::span<x_value const>{std::ranges::data(real),
                                           std::ranges::size(real)},
                  std::span<x_value const>{std::ranges::data(imag),
                                           std::ranges::size(imag)},
                  std::span<r_value>{std::ranges::data(out_real),
                                     std::ranges::size(out_real)},
                  std::span<r_value>{std::ranges::data(out_imag),
                                     std::ranges::size(out_imag)});
}

#undef POLYGNITION_DETAIL_BATCH_ALWAYS_INLINE

template <target::profile Profile, typename P, typename... Vars>
  requires detail::is_polynomial_v<P>
[[nodiscard]] POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::tuned_t<Profile>, P const &p,
           Vars &&...vars) {
  return evaluate(detail::select_algorithm_for<Profile, P, Vars...>(), p,
                  std::forward<Vars>(vars)...);
}

#undef POLYGNITION_DETAIL_ALWAYS_INLINE

// NOTE: polynomial_t{cₙ, …, c₀}; deduction guide
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
    : integral_constant<size_t, static_cast<size_t>(::polygnition::polynomial::static_poly<P>::degree + 1)> {};

template <size_t I, ::polygnition::polynomial::polynomial_t P>
struct tuple_element<I, ::polygnition::polynomial::static_poly<P>> {
  using type = typename ::polygnition::polynomial::static_poly<P>::value_type;
};
} // namespace std
