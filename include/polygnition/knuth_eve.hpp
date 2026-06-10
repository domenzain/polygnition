#pragma once
#include "polygnition/poly.hpp"
#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace polygnition::polynomial {
namespace algorithm {
struct knuth_eve_t {};
} // namespace algorithm

inline constexpr algorithm::knuth_eve_t knuth_eve{};

#if defined(__GNUC__) || defined(__clang__)
#define POLYGNITION_KNUTH_EVE_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define POLYGNITION_KNUTH_EVE_ALWAYS_INLINE inline
#endif

namespace detail {
template <int N>
inline constexpr int knuth_eve_quadratic_count_v = (N - 1) / 2;

template <int N>
inline constexpr int knuth_eve_base_degree_v =
    N - (2 * knuth_eve_quadratic_count_v<N>);

template <typename R, typename T, int N>
[[nodiscard]] constexpr auto
knuth_eve_shifted_coefficients(polynomial_t<T, N> const &p,
                               R const &shift)
    -> std::array<R, static_cast<std::size_t>(N + 1)> {
  std::array<R, static_cast<std::size_t>(N + 1)> low_to_high{};
  auto active_degree = 0;

  // Horner composition of p(x + shift), with the accumulator stored low-to-high.
  for (auto const &coeff : p) {
    std::array<R, static_cast<std::size_t>(N + 1)> next{};
    for (auto d = 0; d <= active_degree; ++d) {
      auto const i = static_cast<std::size_t>(d);
      auto const c = low_to_high[i];
      next[i] += c * shift;
      if (d < N) {
        next[static_cast<std::size_t>(d + 1)] += c;
      }
    }
    next[0] += static_cast<R>(coeff);
    low_to_high = next;
    active_degree = active_degree < N ? active_degree + 1 : N;
  }

  std::array<R, static_cast<std::size_t>(N + 1)> high_to_low{};
  for (auto d = 0; d <= N; ++d) {
    high_to_low[static_cast<std::size_t>(N - d)] =
        low_to_high[static_cast<std::size_t>(d)];
  }
  return high_to_low;
}

template <std::size_t I, typename Preprocessed, typename S, typename Y>
[[nodiscard]] POLYGNITION_KNUTH_EVE_ALWAYS_INLINE constexpr auto
knuth_eve_fold(Preprocessed const &pre, S const &s, Y const &y) -> Y {
  auto out = y;
  ::polygnition::detail::static_for<I>([&]<std::size_t Step> {
    constexpr auto index = I - 1U - Step;
    out = (out * (s - static_cast<Y>(pre.alpha[index]))) +
          static_cast<Y>(pre.gamma[index]);
  });
  return out;
}
} // namespace detail

// Preprocessed Knuth-Eve form for a source degree N polynomial:
// p(x) = (...((base(u) · (u² - αₘ) + γₘ)...) · (u² - α₁) + γ₁),
// where u = x - shift. The supplied α values must be roots of the odd part of
// p(x + shift), in the same order used during preprocessing.
template <typename T, int N>
  requires std::floating_point<T> && (N >= 0)
struct knuth_eve_preprocessed_t {
  using value_type = T;
  static constexpr int source_degree = N;
  static constexpr int quadratic_count = detail::knuth_eve_quadratic_count_v<N>;
  static constexpr int base_degree = detail::knuth_eve_base_degree_v<N>;

  T shift{};
  std::array<T, static_cast<std::size_t>(quadratic_count)> alpha{};
  std::array<T, static_cast<std::size_t>(quadratic_count)> gamma{};
  polynomial_t<T, base_degree> base{};

  template <typename Var>
  [[nodiscard]] POLYGNITION_KNUTH_EVE_ALWAYS_INLINE constexpr auto
  operator()(Var &&x) const {
    return evaluate(knuth_eve, *this, std::forward<Var>(x));
  }
};

template <typename T, int N, typename Shift>
  requires std::floating_point<
      std::common_type_t<T, std::remove_cvref_t<Shift>>>
[[nodiscard]] constexpr auto
knuth_eve_odd_part(polynomial_t<T, N> const &p, Shift const &shift) {
  using R = std::common_type_t<T, std::remove_cvref_t<Shift>>;
  constexpr int M = detail::knuth_eve_quadratic_count_v<N>;

  auto const shifted = detail::knuth_eve_shifted_coefficients<R>(
      p, static_cast<R>(shift));
  polynomial_t<R, M> odd{};
  for (auto j = 0; j <= M; ++j) {
    auto const degree = (2 * j) + 1;
    odd[static_cast<std::size_t>(M - j)] =
        degree <= N ? shifted[static_cast<std::size_t>(N - degree)] : R{};
  }
  return odd;
}

template <typename T, int N>
  requires std::floating_point<T>
[[nodiscard]] constexpr auto
knuth_eve_odd_part(polynomial_t<T, N> const &p) {
  return knuth_eve_odd_part(p, T{});
}

template <polynomial_t P, typename Shift>
  requires std::floating_point<std::common_type_t<
      typename static_poly<P>::value_type, std::remove_cvref_t<Shift>>>
[[nodiscard]] constexpr auto knuth_eve_odd_part(static_poly<P>,
                                                Shift const &shift) {
  return knuth_eve_odd_part(P, shift);
}

template <polynomial_t P>
  requires std::floating_point<typename static_poly<P>::value_type>
[[nodiscard]] constexpr auto knuth_eve_odd_part(static_poly<P>) {
  return knuth_eve_odd_part(P);
}

template <typename T, int N, typename Shift, typename Alpha,
          std::size_t M>
  requires(M == static_cast<std::size_t>(
                    detail::knuth_eve_quadratic_count_v<N>)) &&
          std::floating_point<std::common_type_t<
              T, std::remove_cvref_t<Shift>, std::remove_cvref_t<Alpha>>>
[[nodiscard]] constexpr auto
preprocess_knuth_eve(polynomial_t<T, N> const &p, Shift const &shift,
                     std::array<Alpha, M> const &alpha) {
  using R = std::common_type_t<T, std::remove_cvref_t<Shift>,
                               std::remove_cvref_t<Alpha>>;
  constexpr int B = detail::knuth_eve_base_degree_v<N>;

  auto current = detail::knuth_eve_shifted_coefficients<R>(
      p, static_cast<R>(shift));
  auto current_degree = N;
  knuth_eve_preprocessed_t<R, N> out{};
  out.shift = static_cast<R>(shift);

  for (auto stage = 0; stage < static_cast<int>(M); ++stage) {
    auto work = current;
    std::array<R, static_cast<std::size_t>(N + 1)> next{};
    auto const a = static_cast<R>(alpha[static_cast<std::size_t>(stage)]);
    for (auto i = 0; i <= current_degree - 2; ++i) {
      auto const q = work[static_cast<std::size_t>(i)];
      next[static_cast<std::size_t>(i)] = q;
      work[static_cast<std::size_t>(i + 2)] += a * q;
    }
    out.alpha[static_cast<std::size_t>(stage)] = a;
    out.gamma[static_cast<std::size_t>(stage)] =
        work[static_cast<std::size_t>(current_degree)];
    current = next;
    current_degree -= 2;
  }

  for (auto i = 0; i <= B; ++i) {
    out.base[static_cast<std::size_t>(i)] = current[static_cast<std::size_t>(i)];
  }
  return out;
}

template <typename T, int N, typename Alpha, std::size_t M>
  requires(M == static_cast<std::size_t>(
                    detail::knuth_eve_quadratic_count_v<N>)) &&
          std::floating_point<
              std::common_type_t<T, std::remove_cvref_t<Alpha>>>
[[nodiscard]] constexpr auto
preprocess_knuth_eve(polynomial_t<T, N> const &p,
                     std::array<Alpha, M> const &alpha) {
  return preprocess_knuth_eve(p, T{}, alpha);
}

template <polynomial_t P, typename Shift, typename Alpha, std::size_t M>
  requires(M == static_cast<std::size_t>(
                    detail::knuth_eve_quadratic_count_v<static_poly<P>::degree>)) &&
          std::floating_point<std::common_type_t<
              typename static_poly<P>::value_type, std::remove_cvref_t<Shift>,
              std::remove_cvref_t<Alpha>>>
[[nodiscard]] constexpr auto
preprocess_knuth_eve(static_poly<P>, Shift const &shift,
                     std::array<Alpha, M> const &alpha) {
  return preprocess_knuth_eve(P, shift, alpha);
}

template <polynomial_t P, typename Alpha, std::size_t M>
  requires(M == static_cast<std::size_t>(
                    detail::knuth_eve_quadratic_count_v<static_poly<P>::degree>)) &&
          std::floating_point<std::common_type_t<
              typename static_poly<P>::value_type, std::remove_cvref_t<Alpha>>>
[[nodiscard]] constexpr auto preprocess_knuth_eve(static_poly<P>,
                                                  std::array<Alpha, M> const &alpha) {
  return preprocess_knuth_eve(P, alpha);
}

template <typename T, int N, typename Var>
  requires std::floating_point<T> && arithmetic<std::remove_cvref_t<Var>>
[[nodiscard]] POLYGNITION_KNUTH_EVE_ALWAYS_INLINE constexpr auto
evaluate_knuth_eve(knuth_eve_preprocessed_t<T, N> const &pre, Var const &x) {
  using compute_t = std::common_type_t<T, std::remove_cvref_t<Var>>;
  auto const u = static_cast<compute_t>(x) - static_cast<compute_t>(pre.shift);
  auto const s = u * u;
  auto const base = evaluate_horner(pre.base, u);
  return detail::knuth_eve_fold<static_cast<std::size_t>(
      knuth_eve_preprocessed_t<T, N>::quadratic_count)>(pre, s, base);
}

template <typename T, int N, typename V>
  requires std::floating_point<T> && arithmetic<std::remove_cvref_t<V>>
[[nodiscard]] POLYGNITION_KNUTH_EVE_ALWAYS_INLINE constexpr auto
tag_invoke(evaluate_t, algorithm::knuth_eve_t,
           knuth_eve_preprocessed_t<T, N> const &p, V &&v) {
  return evaluate_knuth_eve(p, std::forward<V>(v));
}

#undef POLYGNITION_KNUTH_EVE_ALWAYS_INLINE
} // namespace polygnition::polynomial
