#pragma once
#include <complex>
#include <cstddef>
#include <concepts>
#include <type_traits>
#include <utility>

namespace polygnition {
namespace detail {
template <std::size_t N, typename Fn>
constexpr void static_for(Fn &&fn) {
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (std::forward<Fn>(fn).template operator()<I>(), ...);
  }(std::make_index_sequence<N>{});
}

struct tag_invoke_t {
  template <typename Tag, typename... Args>
    requires requires(Tag &&tag, Args &&...args) {
      tag_invoke(std::forward<Tag>(tag), std::forward<Args>(args)...);
    }
  constexpr auto operator()(Tag &&tag, Args &&...args) const
      noexcept(noexcept(tag_invoke(std::forward<Tag>(tag),
                                   std::forward<Args>(args)...))) {
    return tag_invoke(std::forward<Tag>(tag), std::forward<Args>(args)...);
  }
};

// Keep accidental fallback overloads out of unqualified lookup.
void tag_invoke() = delete;
} // namespace detail

inline namespace customization {
inline constexpr detail::tag_invoke_t tag_invoke{};
}

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_complex_v = is_complex<std::remove_cvref_t<T>>::value;
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
} // namespace polygnition
