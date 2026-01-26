#pragma once
#include <type_traits>

namespace polygnition::polynomial::detail {
template <auto V> struct cval {
  static constexpr auto value = auto(V);
  constexpr operator decltype(value)() const noexcept { return value; }
};

template <auto V> using cval_t = cval<auto(V)>;

template <typename T> struct is_cval : std::false_type {};
template <auto V> struct is_cval<cval<V>> : std::true_type {};
template <typename T> inline constexpr bool is_cval_v = is_cval<T>::value;
} // namespace polygnition::polynomial::detail
