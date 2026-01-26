#pragma once
#include "polygnition/poly.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <utility>

namespace polygnition::polynomial {
namespace detail {
template <typename T>
struct is_field_coefficient : std::bool_constant<std::floating_point<T>> {};

template <typename T>
struct is_field_coefficient<std::complex<T>>
    : std::bool_constant<std::floating_point<T>> {};

template <auto V>
struct is_field_coefficient<cval<V>>
    : is_field_coefficient<std::remove_cvref_t<decltype(V)>> {};

template <typename T> struct field_scalar {
  using type = std::remove_cvref_t<T>;
};

template <typename T> struct field_scalar<std::complex<T>> {
  using type = T;
};

template <auto V> struct field_scalar<cval<V>> {
  using type = typename field_scalar<std::remove_cvref_t<decltype(V)>>::type;
};

template <typename T>
using field_scalar_t = typename field_scalar<std::remove_cvref_t<T>>::type;

template <typename T>
[[nodiscard]] constexpr auto unwrap_scalar(T const &value) -> T const & {
  return value;
}

template <auto V>
[[nodiscard]] constexpr auto unwrap_scalar(cval<V> const &) -> decltype(V) {
  return V;
}

template <typename R, typename T>
[[nodiscard]] constexpr auto as_field_value(T const &value) {
  auto const &actual = unwrap_scalar(value);
  using actual_type = std::remove_cvref_t<decltype(actual)>;
  if constexpr (complex<actual_type>) {
    return std::complex<R>{static_cast<R>(actual.real()),
                           static_cast<R>(actual.imag())};
  } else {
    return static_cast<R>(actual);
  }
}

template <typename T>
concept scalar_divisor =
    arithmetic<std::remove_cvref_t<T>> ||
    is_field_coefficient<std::remove_cvref_t<T>>::value;

template <typename T>
concept field_coefficient_like =
    is_field_coefficient<std::remove_cvref_t<T>>::value;

template <field_coefficient_like T, scalar_divisor U>
[[nodiscard]] constexpr auto divide_coeff(T const &coeff, U const &scalar) {
  using R = std::common_type_t<field_scalar_t<T>, field_scalar_t<U>>;
  if constexpr (complex<std::remove_cvref_t<T>> ||
                complex<std::remove_cvref_t<U>>) {
    return as_field_value<R>(coeff) / as_field_value<R>(scalar);
  } else {
    return static_cast<R>(unwrap_scalar(coeff)) /
           static_cast<R>(unwrap_scalar(scalar));
  }
}

template <typename T> constexpr auto derivative_scale(int degree) {
  if constexpr (complex<T>) {
    return static_cast<typename T::value_type>(degree);
  } else {
    return degree;
  }
}

template <typename T>
constexpr auto derivative_coeff(T const &coeff, int degree) {
  return coeff * derivative_scale<std::remove_cvref_t<T>>(degree);
}

template <typename R, typename P>
  requires is_polynomial_v<P>
constexpr auto coeff_by_degree(P const &p, int degree) -> R {
  return coefficient_by_degree_as<R>(p, degree);
}

template <int ResultDegree, typename R, typename P, std::size_t... I>
[[nodiscard]] constexpr auto rebase_by_degree(P const &p,
                                               std::index_sequence<I...>) {
  return polynomial_t<R, ResultDegree>{
      coefficient_by_degree_as<R>(
          p, ResultDegree - static_cast<int>(I))...};
}

template <int ResultDegree, typename R, typename P>
[[nodiscard]] constexpr auto rebase_by_degree(P const &p) {
  return rebase_by_degree<ResultDegree, R>(
      p, std::make_index_sequence<static_cast<std::size_t>(ResultDegree + 1)>{});
}

template <polynomial_t P>
[[nodiscard]] consteval auto effective_degree_value() -> int {
  for (int offset = 0; offset <= std::remove_cvref_t<decltype(P)>::degree;
       ++offset) {
    if (P[static_cast<std::size_t>(offset)] !=
        typename std::remove_cvref_t<decltype(P)>::value_type{}) {
      return std::remove_cvref_t<decltype(P)>::degree - offset;
    }
  }
  return 0;
}

template <polynomial_t P, int NewDegree, std::size_t... I>
[[nodiscard]] consteval auto trim_static_value_impl(std::index_sequence<I...>) {
  using source_type = std::remove_cvref_t<decltype(P)>;
  using T = typename source_type::value_type;
  constexpr int source_degree = source_type::degree;
  return polynomial_t<T, NewDegree>{
      P[static_cast<std::size_t>(source_degree - NewDegree) + I]...};
}

template <polynomial_t P>
[[nodiscard]] consteval auto trim_static_value() {
  constexpr int new_degree = effective_degree_value<P>();
  return trim_static_value_impl<P, new_degree>(
      std::make_index_sequence<static_cast<std::size_t>(new_degree + 1)>{});
}

template <polynomial_t P>
using trimmed_static_poly_t = static_poly<trim_static_value<P>()>;

} // namespace detail

template <typename T>
concept field_coefficient =
    detail::is_field_coefficient<std::remove_cvref_t<T>>::value;

template <int Order, typename T>
  requires closed_ring<T>
constexpr auto multiply_truncated(polynomial_t<T, Order> const &a,
                                  polynomial_t<T, Order> const &b) {
  polynomial_t<T, Order> out{};
  for (int d = 0; d <= Order; ++d) {
    T sum{};
    for (int i = 0; i <= d; ++i) {
      sum = sum + (a.coeff(i) * b.coeff(d - i));
    }
    out[static_cast<std::size_t>(Order - d)] = sum;
  }
  return out;
}

template <typename T, int N>
  requires closed_ring<T>
[[nodiscard]] constexpr auto trim(polynomial_t<T, N> const &p) {
  return p;
}

template <polynomial_t P>
  requires closed_ring<typename static_poly<P>::value_type>
[[nodiscard]] constexpr auto trim(static_poly<P>) {
  return detail::trimmed_static_poly_t<P>{};
}

template <typename T, int N>
  requires closed_ring<T>
[[nodiscard]] constexpr auto operator-(polynomial_t<T, N> const &p) {
  polynomial_t<T, N> out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = -p[i];
  }
  return out;
}

template <polynomial_t P>
  requires closed_ring<typename static_poly<P>::value_type>
[[nodiscard]] constexpr auto operator-(static_poly<P>) {
  return static_poly<-P>{};
}

template <typename T, int N, typename U, int M>
  requires closed_ring<std::common_type_t<T, U>> &&
           std::convertible_to<T, std::common_type_t<T, U>> &&
           std::convertible_to<U, std::common_type_t<T, U>>
[[nodiscard]] constexpr auto operator+(polynomial_t<T, N> const &a,
                                       polynomial_t<U, M> const &b) {
  using R = std::common_type_t<T, U>;
  constexpr int result_degree = (N > M) ? N : M;
  polynomial_t<R, result_degree> out{};
  for (int d = 0; d <= result_degree; ++d) {
    out[static_cast<std::size_t>(result_degree - d)] =
        detail::coefficient_by_degree_as<R>(a, d) +
        detail::coefficient_by_degree_as<R>(b, d);
  }
  return out;
}

template <typename T, int N, typename U, int M>
  requires closed_ring<std::common_type_t<T, U>> &&
           std::convertible_to<T, std::common_type_t<T, U>> &&
           std::convertible_to<U, std::common_type_t<T, U>>
[[nodiscard]] constexpr auto operator-(polynomial_t<T, N> const &a,
                                       polynomial_t<U, M> const &b) {
  using R = std::common_type_t<T, U>;
  constexpr int result_degree = (N > M) ? N : M;
  polynomial_t<R, result_degree> out{};
  for (int d = 0; d <= result_degree; ++d) {
    out[static_cast<std::size_t>(result_degree - d)] =
        detail::coefficient_by_degree_as<R>(a, d) -
        detail::coefficient_by_degree_as<R>(b, d);
  }
  return out;
}

template <typename T, int N, typename U, int M>
  requires closed_ring<std::common_type_t<T, U>> &&
           std::convertible_to<T, std::common_type_t<T, U>> &&
           std::convertible_to<U, std::common_type_t<T, U>>
[[nodiscard]] constexpr auto operator*(polynomial_t<T, N> const &a,
                                       polynomial_t<U, M> const &b) {
  using R = std::common_type_t<T, U>;
  constexpr int result_degree = N + M;
  polynomial_t<R, result_degree> out{};
  for (int d = 0; d <= result_degree; ++d) {
    R sum{};
    for (int i = 0; i <= d; ++i) {
      sum += detail::coefficient_by_degree_as<R>(a, i) *
             detail::coefficient_by_degree_as<R>(b, d - i);
    }
    out[static_cast<std::size_t>(result_degree - d)] = sum;
  }
  return out;
}

template <polynomial_t A, polynomial_t B>
  requires closed_ring<std::common_type_t<typename static_poly<A>::value_type,
                                         typename static_poly<B>::value_type>>
[[nodiscard]] constexpr auto operator+(static_poly<A>, static_poly<B>) {
  return static_poly<(A + B)>{};
}

template <polynomial_t A, polynomial_t B>
  requires closed_ring<std::common_type_t<typename static_poly<A>::value_type,
                                         typename static_poly<B>::value_type>>
[[nodiscard]] constexpr auto operator-(static_poly<A>, static_poly<B>) {
  return static_poly<(A - B)>{};
}

template <polynomial_t A, polynomial_t B>
  requires closed_ring<std::common_type_t<typename static_poly<A>::value_type,
                                         typename static_poly<B>::value_type>>
[[nodiscard]] constexpr auto operator*(static_poly<A>, static_poly<B>) {
  return static_poly<(A * B)>{};
}

template <polynomial_t A, typename U, int M>
  requires closed_ring<std::common_type_t<typename static_poly<A>::value_type, U>>
[[nodiscard]] constexpr auto operator+(static_poly<A>,
                                       polynomial_t<U, M> const &b) {
  return A + b;
}

template <typename T, int N, polynomial_t B>
  requires closed_ring<std::common_type_t<T, typename static_poly<B>::value_type>>
[[nodiscard]] constexpr auto operator+(polynomial_t<T, N> const &a,
                                       static_poly<B>) {
  return a + B;
}

template <polynomial_t A, typename U, int M>
  requires closed_ring<std::common_type_t<typename static_poly<A>::value_type, U>>
[[nodiscard]] constexpr auto operator-(static_poly<A>,
                                       polynomial_t<U, M> const &b) {
  return A - b;
}

template <typename T, int N, polynomial_t B>
  requires closed_ring<std::common_type_t<T, typename static_poly<B>::value_type>>
[[nodiscard]] constexpr auto operator-(polynomial_t<T, N> const &a,
                                       static_poly<B>) {
  return a - B;
}

template <polynomial_t A, typename U, int M>
  requires closed_ring<std::common_type_t<typename static_poly<A>::value_type, U>>
[[nodiscard]] constexpr auto operator*(static_poly<A>,
                                       polynomial_t<U, M> const &b) {
  return A * b;
}

template <typename T, int N, polynomial_t B>
  requires closed_ring<std::common_type_t<T, typename static_poly<B>::value_type>>
[[nodiscard]] constexpr auto operator*(polynomial_t<T, N> const &a,
                                       static_poly<B>) {
  return a * B;
}

template <typename T, int N, typename U>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator+(polynomial_t<T, N> const &a,
                                       U const &c) {
  return a + polynomial_t{c};
}

template <typename U, typename T, int N>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator+(U const &c,
                                       polynomial_t<T, N> const &a) {
  return polynomial_t{c} + a;
}

template <typename T, int N, typename U>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator-(polynomial_t<T, N> const &a,
                                       U const &c) {
  return a - polynomial_t{c};
}

template <typename U, typename T, int N>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator-(U const &c,
                                       polynomial_t<T, N> const &a) {
  return polynomial_t{c} - a;
}

template <typename T, int N, typename U>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator*(polynomial_t<T, N> const &a,
                                       U const &c) {
  return a * polynomial_t{c};
}

template <typename U, typename T, int N>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator*(U const &c,
                                       polynomial_t<T, N> const &a) {
  return polynomial_t{c} * a;
}

template <polynomial_t P, typename U>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator+(static_poly<P>, U const &c) {
  return P + c;
}

template <typename U, polynomial_t P>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator+(U const &c, static_poly<P>) {
  return c + P;
}

template <polynomial_t P, typename U>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator-(static_poly<P>, U const &c) {
  return P - c;
}

template <typename U, polynomial_t P>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator-(U const &c, static_poly<P>) {
  return c - P;
}

template <polynomial_t P, typename U>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator*(static_poly<P>, U const &c) {
  return P * c;
}

template <typename U, polynomial_t P>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>)
[[nodiscard]] constexpr auto operator*(U const &c, static_poly<P>) {
  return c * P;
}

template <typename T, int N, auto C>
[[nodiscard]] constexpr auto operator+(polynomial_t<T, N> const &a, detail::cval<C>) {
  return a + polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)};
}

template <auto C, typename T, int N>
[[nodiscard]] constexpr auto operator+(detail::cval<C>, polynomial_t<T, N> const &a) {
  return polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)} + a;
}

template <typename T, int N, auto C>
[[nodiscard]] constexpr auto operator-(polynomial_t<T, N> const &a, detail::cval<C>) {
  return a - polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)};
}

template <auto C, typename T, int N>
[[nodiscard]] constexpr auto operator-(detail::cval<C>, polynomial_t<T, N> const &a) {
  return polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)} - a;
}

template <typename T, int N, auto C>
[[nodiscard]] constexpr auto operator*(polynomial_t<T, N> const &a, detail::cval<C>) {
  return a * polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)};
}

template <auto C, typename T, int N>
[[nodiscard]] constexpr auto operator*(detail::cval<C>, polynomial_t<T, N> const &a) {
  return polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)} * a;
}

template <polynomial_t P, auto C>
[[nodiscard]] constexpr auto operator+(static_poly<P>, detail::cval<C>) {
  return static_poly<(P + polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)})>{};
}

template <auto C, polynomial_t P>
[[nodiscard]] constexpr auto operator+(detail::cval<C>, static_poly<P>) {
  return static_poly<(polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)} + P)>{};
}

template <polynomial_t P, auto C>
[[nodiscard]] constexpr auto operator-(static_poly<P>, detail::cval<C>) {
  return static_poly<(P - polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)})>{};
}

template <auto C, polynomial_t P>
[[nodiscard]] constexpr auto operator-(detail::cval<C>, static_poly<P>) {
  return static_poly<(polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)} - P)>{};
}

template <polynomial_t P, auto C>
[[nodiscard]] constexpr auto operator*(static_poly<P>, detail::cval<C>) {
  return static_poly<(P * polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)})>{};
}

template <auto C, polynomial_t P>
[[nodiscard]] constexpr auto operator*(detail::cval<C>, static_poly<P>) {
  return static_poly<(polynomial_t{static_cast<std::remove_cvref_t<decltype(C)>>(C)} * P)>{};
}

template <typename T, int N, typename U>
  requires(!detail::is_polynomial_v<U>) &&
          requires(T coeff, U scalar) { detail::divide_coeff(coeff, scalar); }
[[nodiscard]] constexpr auto operator/(polynomial_t<T, N> const &p,
                                       U const &scalar) {
  using R = decltype(detail::divide_coeff(std::declval<T const &>(),
                                          std::declval<U const &>()));
  polynomial_t<R, N> out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = detail::divide_coeff(p[i], scalar);
  }
  return out;
}

template <polynomial_t P, typename U>
  requires(!detail::is_polynomial_v<U> &&
           !detail::is_cval_v<std::remove_cvref_t<U>>) &&
          requires(typename static_poly<P>::value_type coeff, U scalar) {
            detail::divide_coeff(coeff, scalar);
          }
[[nodiscard]] constexpr auto operator/(static_poly<P>, U const &scalar) {
  return P / scalar;
}

template <polynomial_t P, auto C>
  requires requires(typename static_poly<P>::value_type coeff) {
    detail::divide_coeff(coeff, detail::cval<C>{});
  }
[[nodiscard]] constexpr auto operator/(static_poly<P>, detail::cval<C>) {
  return static_poly<(P / detail::cval<C>{})>{};
}

template <typename T, int N>
  requires requires(T coeff) { detail::derivative_coeff(coeff, 1); }
[[nodiscard]] constexpr auto derivative(polynomial_t<T, N> const &p) {
  using R = decltype(detail::derivative_coeff(std::declval<T const &>(), 1));
  if constexpr (N == 0) {
    return polynomial_t<R, 0>{R{}};
  } else {
    polynomial_t<R, N - 1> out{};
    for (int d = 0; d < N; ++d) {
      out[static_cast<std::size_t>((N - 1) - d)] =
          detail::derivative_coeff(p.coeff(d + 1), d + 1);
    }
    return out;
  }
}

template <polynomial_t P>
  requires requires(typename static_poly<P>::value_type coeff) {
    detail::derivative_coeff(coeff, 1);
  }
[[nodiscard]] constexpr auto derivative(static_poly<P>) {
  return static_poly<derivative(P)>{};
}

// Long division over a field-like coefficient domain. The divisor is assumed to
// have actual degree M, so its stored leading coefficient is non-zero.
template <typename T, int N, typename U, int M>
  requires field_coefficient<T> && field_coefficient<U> &&
           requires(T lhs, U rhs) { detail::divide_coeff(lhs, rhs); }
[[nodiscard]] constexpr auto divide(polynomial_t<T, N> const &u,
                                    polynomial_t<U, M> const &f) {
  using R = decltype(detail::divide_coeff(std::declval<T const &>(),
                                          std::declval<U const &>()));

  if constexpr (N < M) {
    polynomial_t<R, 0> quotient{R{}};
    polynomial_t<R, N> remainder{};
    for (int d = 0; d <= N; ++d) {
      remainder[static_cast<std::size_t>(N - d)] =
          detail::coeff_by_degree<R>(u, d);
    }
    return std::pair{quotient, remainder};
  } else if constexpr (M == 0) {
    return std::pair{u / f.coeff(0), polynomial_t<R, 0>{R{}}};
  } else {
    constexpr int quotient_degree = N - M;
    std::array<R, static_cast<std::size_t>(N + 1)> scratch{};
    for (int d = 0; d <= N; ++d) {
      scratch[static_cast<std::size_t>(N - d)] =
          detail::coeff_by_degree<R>(u, d);
    }

    auto const leading = detail::coeff_by_degree<R>(f, M);
    for (int k = 0; k <= quotient_degree; ++k) {
      scratch[static_cast<std::size_t>(k)] /= leading;
      auto const quotient_coeff = scratch[static_cast<std::size_t>(k)];
      for (int i = 1; i <= M; ++i) {
        scratch[static_cast<std::size_t>(k + i)] -=
            quotient_coeff * detail::coeff_by_degree<R>(f, M - i);
      }
    }

    polynomial_t<R, quotient_degree> quotient{};
    std::ranges::copy(std::views::take(scratch, quotient_degree + 1),
                      quotient.begin());

    polynomial_t<R, M - 1> remainder{};
    std::ranges::copy(std::views::drop(scratch, quotient_degree + 1) |
                          std::views::take(M),
                      remainder.begin());

    return std::pair{quotient, remainder};
  }
}

template <polynomial_t U, polynomial_t F>
[[nodiscard]] consteval auto divide_static() {
  constexpr auto result = divide(U, F);
  return std::pair{static_poly<result.first>{}, static_poly<result.second>{}};
}

template <polynomial_t U, polynomial_t F>
  requires requires { divide(U, F); }
[[nodiscard]] constexpr auto divide(static_poly<U>, static_poly<F>) {
  return divide_static<U, F>();
}

template <polynomial_t U, typename T, int M>
  requires requires { divide(U, std::declval<polynomial_t<T, M> const &>()); }
[[nodiscard]] constexpr auto divide(static_poly<U>, polynomial_t<T, M> const &f) {
  return divide(U, f);
}

template <typename T, int N, polynomial_t F>
  requires requires { divide(std::declval<polynomial_t<T, N> const &>(), F); }
[[nodiscard]] constexpr auto divide(polynomial_t<T, N> const &u, static_poly<F>) {
  return divide(u, F);
}

static_assert(::polygnition::closed_ring<polynomial_t<double, 0>>);
} // namespace polygnition::polynomial
