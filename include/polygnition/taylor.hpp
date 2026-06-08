#pragma once
#include "polygnition/arithmetic.hpp"
#include "polygnition/math.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace polygnition::polynomial {
template <typename T, int Order> struct taylor_series;

namespace detail {
template <typename> struct is_taylor_polynomial : std::false_type {};
template <typename T, int Order>
struct is_taylor_polynomial<taylor_series<T, Order>> : std::true_type {};

template <typename R, int Order, typename Poly>
[[nodiscard]] constexpr auto taylor_from(Poly const &p)
    -> taylor_series<R, Order>;
} // namespace detail

// Represents a truncated Taylor series in ε:
// value + a₁ε + a₂ε² + ... + aₙεⁿ, where aₖ = f⁽ᵏ⁾(x) / k!.
// Layout follows polynomial_t's high-to-low order: [aₙ, ..., a₁, value].
template <typename T, int Order>
struct taylor_series : polynomial_t<T, Order> {
  template <typename U>
  static constexpr bool scalar_like =
      !detail::is_polynomial_v<U> &&
      !detail::is_taylor_polynomial<std::remove_cvref_t<U>>::value;

  friend constexpr auto operator*(taylor_series const &a,
                                  taylor_series const &b) {
    return detail::taylor_from<T, Order>(multiply_truncated<Order>(a, b));
  }

  template <typename U>
    requires scalar_like<U> &&
             requires(polynomial_t<T, Order> const &lhs, U const &rhs) {
               lhs * rhs;
             }
  friend constexpr auto operator*(taylor_series const &lhs, U const &rhs) {
    auto const product = static_cast<polynomial_t<T, Order> const &>(lhs) * rhs;
    using R = typename std::remove_cvref_t<decltype(product)>::value_type;
    return detail::taylor_from<R, Order>(product);
  }

  template <typename U>
    requires scalar_like<U> &&
             requires(U const &lhs, polynomial_t<T, Order> const &rhs) {
               lhs * rhs;
             }
  friend constexpr auto operator*(U const &lhs, taylor_series const &rhs) {
    auto const product = lhs * static_cast<polynomial_t<T, Order> const &>(rhs);
    using R = typename std::remove_cvref_t<decltype(product)>::value_type;
    return detail::taylor_from<R, Order>(product);
  }

  template <typename U>
    requires scalar_like<U> &&
             requires(polynomial_t<T, Order> const &lhs, U const &rhs) {
               lhs + rhs;
             }
  friend constexpr auto operator+(taylor_series const &lhs, U const &rhs) {
    auto const sum = static_cast<polynomial_t<T, Order> const &>(lhs) + rhs;
    using R = typename std::remove_cvref_t<decltype(sum)>::value_type;
    return detail::taylor_from<R, Order>(sum);
  }

  template <typename U>
    requires scalar_like<U> &&
             requires(U const &lhs, polynomial_t<T, Order> const &rhs) {
               lhs + rhs;
             }
  friend constexpr auto operator+(U const &lhs, taylor_series const &rhs) {
    auto const sum = lhs + static_cast<polynomial_t<T, Order> const &>(rhs);
    using R = typename std::remove_cvref_t<decltype(sum)>::value_type;
    return detail::taylor_from<R, Order>(sum);
  }

  template <typename U>
    requires scalar_like<U> &&
             requires(polynomial_t<T, Order> const &lhs, U const &rhs) {
               lhs - rhs;
             }
  friend constexpr auto operator-(taylor_series const &lhs, U const &rhs) {
    auto const difference =
        static_cast<polynomial_t<T, Order> const &>(lhs) - rhs;
    using R = typename std::remove_cvref_t<decltype(difference)>::value_type;
    return detail::taylor_from<R, Order>(difference);
  }

  template <typename U>
    requires scalar_like<U> &&
             requires(U const &lhs, polynomial_t<T, Order> const &rhs) {
               lhs - rhs;
             }
  friend constexpr auto operator-(U const &lhs, taylor_series const &rhs) {
    auto const difference =
        lhs - static_cast<polynomial_t<T, Order> const &>(rhs);
    using R = typename std::remove_cvref_t<decltype(difference)>::value_type;
    return detail::taylor_from<R, Order>(difference);
  }

  friend constexpr auto operator-(taylor_series const &value) {
    return detail::taylor_from<T, Order>(
        -static_cast<polynomial_t<T, Order> const &>(value));
  }

  friend constexpr auto operator+(taylor_series const &lhs,
                                  taylor_series const &rhs) {
    return detail::taylor_from<T, Order>(
        static_cast<polynomial_t<T, Order> const &>(lhs) +
        static_cast<polynomial_t<T, Order> const &>(rhs));
  }

  friend constexpr auto operator-(taylor_series const &lhs,
                                  taylor_series const &rhs) {
    return detail::taylor_from<T, Order>(
        static_cast<polynomial_t<T, Order> const &>(lhs) -
        static_cast<polynomial_t<T, Order> const &>(rhs));
  }

  // K-th derivative: f⁽ᴷ⁾ = cₖ * K!
  template <int K> [[nodiscard]] constexpr T derivative() const {
    static_assert(K >= 0, "Derivative order must be non-negative");
    if constexpr (K > Order)
      return T{0};
    return (*this)[Order - K] * static_cast<T>(math::factorial(K));
  }

  // { f(x), f'(x), f''(x), ... }
  [[nodiscard]] constexpr auto derivatives() const {
    return make_derivatives(std::make_integer_sequence<int, Order + 1>{});
  }

private:
  // unpack sequence 0..N into derivative calls
  template <int... K>
  constexpr auto make_derivatives(std::integer_sequence<int, K...>) const {
    return std::tuple{derivative<K>()...};
  }
};

namespace detail {
template <typename R, int Order, typename Poly, std::size_t... I>
[[nodiscard]] constexpr auto taylor_from(Poly const &p,
                                         std::index_sequence<I...>)
    -> taylor_series<R, Order> {
  return taylor_series<R, Order>{static_cast<R>(p[I])...};
}

template <typename R, int Order, typename Poly>
[[nodiscard]] constexpr auto taylor_from(Poly const &p)
    -> taylor_series<R, Order> {
  return taylor_from<R, Order>(
      p, std::make_index_sequence<static_cast<std::size_t>(Order + 1)>{});
}
} // namespace detail

static_assert(closed_ring<taylor_series<double, 2>>);
static_assert(std::is_aggregate_v<taylor_series<double, 0>>);

// NOTE: taylor_series{cₙ, …, c₀}; deduction guide
template <class T, class... U>
taylor_series(T, U...) -> taylor_series<T, sizeof...(U)>;

template <typename T> using dual = taylor_series<T, 1>;

template <typename T>
inline constexpr bool is_taylor_polynomial_v =
    detail::is_taylor_polynomial<std::remove_cvref_t<T>>::value;

template <int Order, typename T, int N, typename X>
[[nodiscard]] constexpr auto series_expansion(
    polynomial_t<T, N> const &p, X const &x) {
  static_assert(Order >= 0, "series expansion order must be non-negative");
  using value_t = std::common_type_t<T, std::remove_cvref_t<X>>;
  taylor_series<value_t, Order> seed{};
  seed[Order] = static_cast<value_t>(x);
  if constexpr (Order > 0) {
    seed[Order - 1] = value_t{1};
  }
  return detail::taylor_from<value_t, Order>(evaluate(horner, p, seed));
}

template <int Order, polynomial_t P, typename X>
[[nodiscard]] constexpr auto series_expansion(static_poly<P> p, X const &x) {
  return series_expansion<Order>(static_poly<P>::value, x);
}

} // namespace polygnition::polynomial
