#pragma once
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

namespace polygnition {
namespace detail {
#if defined(__GNUC__) || defined(__clang__)
#define POLYGNITION_DETAIL_ALWAYS_INLINE [[gnu::always_inline]] inline
#else
#define POLYGNITION_DETAIL_ALWAYS_INLINE inline
#endif

template <typename... Fs> struct overload : Fs... {
  using Fs::operator()...;
};
template <typename... Fs> overload(Fs...) -> overload<Fs...>;

template <typename...> inline constexpr bool dependent_false_v = false;

template <typename... Ts> struct type_pack {};

template <typename...> struct first_type;
template <typename T, typename... Ts> struct first_type<T, Ts...> {
  using type = T;
};
template <typename... Ts> using first_type_t = typename first_type<Ts...>::type;

template <std::size_t N, typename Fn>
POLYGNITION_DETAIL_ALWAYS_INLINE constexpr void static_for(Fn &&fn) {
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (std::forward<Fn>(fn).template operator()<I>(), ...);
  }(std::make_index_sequence<N>{});
}

template <std::size_t N, typename Fn>
POLYGNITION_DETAIL_ALWAYS_INLINE constexpr void for_each_index(Fn &&fn) {
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (std::forward<Fn>(fn)(std::integral_constant<std::size_t, I>{}), ...);
  }(std::make_index_sequence<N>{});
}

struct tag_invoke_t {
  template <typename Tag, typename... Args>
    requires requires(Tag &&tag, Args &&...args) {
      tag_invoke(std::forward<Tag>(tag), std::forward<Args>(args)...);
    }
  // trampoline
  POLYGNITION_DETAIL_ALWAYS_INLINE constexpr auto
  operator()(Tag &&tag, Args &&...args) const
      noexcept(noexcept(tag_invoke(std::forward<Tag>(tag),
                                   std::forward<Args>(args)...))) {
    return tag_invoke(std::forward<Tag>(tag), std::forward<Args>(args)...);
  }
};

// NOTE: must follow tag_invoke_t definition due to GCC bug affecting
// requires-clause name lookup.
void tag_invoke() = delete; // poison-pill

} // namespace detail

inline namespace customization { // dispatch to user-provided overload
                                 // tag_invoke(tag, args...)
inline constexpr detail::tag_invoke_t tag_invoke{};
}

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

namespace detail {
template <std::size_t N>
inline constexpr auto is_power_of_two_v = N != 0 && ((N & (N - 1)) == 0);

#if defined(__AVX512F__)
inline constexpr auto native_lane_max_bytes = std::size_t{64};
#elif defined(__AVX__)
inline constexpr auto native_lane_max_bytes = std::size_t{32};
#elif defined(__SSE2__) || defined(__ARM_NEON) || defined(__ARM_NEON__) || \
    defined(__aarch64__)
inline constexpr auto native_lane_max_bytes = std::size_t{16};
#else
inline constexpr auto native_lane_max_bytes = std::size_t{0};
#endif

template <typename T, std::size_t W>
inline constexpr auto use_native_lane_storage_v =
    arithmetic<T> && !std::same_as<T, bool> && W > 1 &&
    is_power_of_two_v<sizeof(T) * W> &&
    sizeof(T) * W <= native_lane_max_bytes;

#if defined(__GNUC__) || defined(__clang__)
template <typename T, std::size_t W>
struct native_lane_storage {
  using type __attribute__((vector_size(sizeof(T) * W))) = T;
};
#endif

template <typename T, std::size_t W,
          bool Native =
#if defined(__GNUC__) || defined(__clang__)
              use_native_lane_storage_v<T, W>
#else
              false
#endif
          >
struct lane_storage {
  using type = std::array<T, W>;
};

#if defined(__GNUC__) || defined(__clang__)
template <typename T, std::size_t W>
struct lane_storage<T, W, true> {
  using type = typename native_lane_storage<T, W>::type;
};
#endif
} // namespace detail

template <typename T, std::size_t W>
  requires(W > 0) && arithmetic<T>
struct lanes {
  using value_type = T;
  static constexpr auto uses_native_storage =
#if defined(__GNUC__) || defined(__clang__)
      detail::use_native_lane_storage_v<T, W>;
#else
      false;
#endif
  using storage_type = typename detail::lane_storage<T, W>::type;
  storage_type v{};

  [[nodiscard]] static constexpr auto size() noexcept -> std::size_t { return W; }

  constexpr lanes() noexcept = default;

  constexpr explicit(false) lanes(T scalar) noexcept {
    if constexpr (uses_native_storage) {
      for (auto i = std::size_t{}; i < W; ++i) {
        v[i] = scalar;
      }
    } else {
      v.fill(scalar);
    }
  }

  constexpr explicit lanes(storage_type raw) noexcept
    requires uses_native_storage
      : v(raw) {}

  [[nodiscard]] static constexpr auto load(T const *ptr) noexcept -> lanes {
    lanes out{};
    for (auto i = std::size_t{}; i < W; ++i) {
      out.v[i] = ptr[i];
    }
    return out;
  }

  constexpr void store(T *ptr) const noexcept {
    for (auto i = std::size_t{}; i < W; ++i) {
      ptr[i] = v[i];
    }
  }

  [[nodiscard]] constexpr auto operator[](std::size_t i) const noexcept -> T {
    return v[i];
  }

  friend constexpr auto operator+(lanes a, lanes b) noexcept -> lanes {
    if constexpr (uses_native_storage) {
      return lanes{a.v + b.v};
    } else {
      lanes out{};
      for (auto i = std::size_t{}; i < W; ++i) out.v[i] = a.v[i] + b.v[i];
      return out;
    }
  }

  friend constexpr auto operator-(lanes a, lanes b) noexcept -> lanes {
    if constexpr (uses_native_storage) {
      return lanes{a.v - b.v};
    } else {
      lanes out{};
      for (auto i = std::size_t{}; i < W; ++i) out.v[i] = a.v[i] - b.v[i];
      return out;
    }
  }

  friend constexpr auto operator-(lanes a) noexcept -> lanes {
    if constexpr (uses_native_storage) {
      return lanes{-a.v};
    } else {
      lanes out{};
      for (auto i = std::size_t{}; i < W; ++i) out.v[i] = -a.v[i];
      return out;
    }
  }

  friend constexpr auto operator*(lanes a, lanes b) noexcept -> lanes {
    if constexpr (uses_native_storage) {
      return lanes{a.v * b.v};
    } else {
      lanes out{};
      for (auto i = std::size_t{}; i < W; ++i) out.v[i] = a.v[i] * b.v[i];
      return out;
    }
  }

  friend constexpr auto operator/(lanes a, lanes b) noexcept -> lanes {
    if constexpr (uses_native_storage) {
      return lanes{a.v / b.v};
    } else {
      lanes out{};
      for (auto i = std::size_t{}; i < W; ++i) out.v[i] = a.v[i] / b.v[i];
      return out;
    }
  }
};


template <typename T>
struct split_complex {
  using value_type = T;

  T real{};
  T imag{};

  constexpr split_complex() noexcept = default;
  constexpr explicit(false) split_complex(T r) noexcept : real(r), imag{} {}
  constexpr split_complex(T r, T i) noexcept : real(r), imag(i) {}

  friend constexpr auto operator+(split_complex a, split_complex b) noexcept
      -> split_complex {
    return {a.real + b.real, a.imag + b.imag};
  }

  friend constexpr auto operator-(split_complex a, split_complex b) noexcept
      -> split_complex {
    return {a.real - b.real, a.imag - b.imag};
  }

  friend constexpr auto operator-(split_complex a) noexcept -> split_complex {
    return {-a.real, -a.imag};
  }

  friend constexpr auto operator*(split_complex a, split_complex b) noexcept
      -> split_complex {
    return {(a.real * b.real) - (a.imag * b.imag),
            (a.real * b.imag) + (a.imag * b.real)};
  }
};

template <typename> struct is_split_complex : std::false_type {};
template <typename T>
struct is_split_complex<split_complex<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_split_complex_v =
    is_split_complex<std::remove_cvref_t<T>>::value;
template <typename T>
concept split_complex_like = is_split_complex_v<T>;


template <typename T>
concept simd_like =
    requires(std::remove_cvref_t<T> value,
             typename std::remove_cvref_t<T>::value_type scalar) {
      typename std::remove_cvref_t<T>::value_type;
      { std::remove_cvref_t<T>::size() } -> std::convertible_to<std::size_t>;
      std::remove_cvref_t<T>{scalar};
      { value + value } -> std::same_as<std::remove_cvref_t<T>>;
      { value - value } -> std::same_as<std::remove_cvref_t<T>>;
      { value * value } -> std::same_as<std::remove_cvref_t<T>>;
    };

template <typename T> struct scalar_value_type {
  using type = std::remove_cvref_t<T>;
};

template <simd_like T> struct scalar_value_type<T> {
  using type = typename std::remove_cvref_t<T>::value_type;
};

template <typename T>
using scalar_value_type_t =
    typename scalar_value_type<std::remove_cvref_t<T>>::type;

template <typename T>
inline constexpr std::size_t simd_size_v = [] {
  if constexpr (simd_like<T>) {
    return std::remove_cvref_t<T>::size();
  } else {
    return std::size_t{1};
  }
}();

template <typename> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};
template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;
template <typename T>
concept complex = is_complex_v<T>;

template <typename T>
concept closed_ring = std::regular<T> && requires(T a, T b) {
  T{};
  T{0};
  T{1};
  { a + b } -> std::same_as<T>;
  { a - b } -> std::same_as<T>;
  { -a } -> std::same_as<T>;
  { a * b } -> std::same_as<T>;
};

namespace detail {
template <typename R> struct is_span : std::false_type {};
template <typename T, std::size_t Extent>
struct is_span<std::span<T, Extent>> : std::true_type {};
template <typename R>
inline constexpr bool is_span_v = is_span<std::remove_cvref_t<R>>::value;

template <typename R>
concept sized_contiguous_range =
    std::ranges::contiguous_range<R> && std::ranges::sized_range<R>;

template <typename R>
concept writable_sized_contiguous_range =
    sized_contiguous_range<R> &&
    (!std::is_const_v<
        std::remove_reference_t<std::ranges::range_reference_t<R>>>);

template <sized_contiguous_range R>
[[nodiscard]] constexpr auto span_from_range(R &&range) {
  using value_type = std::remove_cvref_t<std::ranges::range_value_t<R>>;
  return std::span<value_type const>{std::ranges::data(range),
                                     std::ranges::size(range)};
}

template <writable_sized_contiguous_range R>
[[nodiscard]] constexpr auto writable_span_from_range(R &&range) {
  using value_type = std::remove_cvref_t<std::ranges::range_value_t<R>>;
  return std::span<value_type>{std::ranges::data(range),
                               std::ranges::size(range)};
}
} // namespace detail
} // namespace polygnition
