#pragma once
#include "polygnition/batch.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define POLYGNITION_PREPARED_INLINE [[gnu::always_inline]] inline
#else
#define POLYGNITION_PREPARED_INLINE inline
#endif

namespace polygnition::polynomial {
namespace detail {
template <typename Algorithm, typename P>
[[nodiscard]] constexpr auto prepared_representation(Algorithm, P &&p) {
  return std::forward<P>(p);
}

template <typename P>
  requires is_polynomial_v<P> && (!is_static_polynomial_v<P>) &&
           (stored_polynomial_type_t<P>::degree == 4) &&
           std::floating_point<typename stored_polynomial_type_t<P>::value_type>
[[nodiscard]] constexpr auto prepared_representation(algorithm::motzkin_t,
                                                     P &&p) {
  return preprocess_motzkin(p);
}
} // namespace detail

template <typename Algorithm, typename Representation> struct prepared_t {
  [[no_unique_address]] Algorithm algorithm;
  [[no_unique_address]] Representation representation;

  template <typename... Args>
    requires requires(Algorithm const &selected,
                      Representation const &stored, Args &&...args) {
      evaluate(selected, stored, std::forward<Args>(args)...);
    }
  [[nodiscard]] POLYGNITION_PREPARED_INLINE constexpr decltype(auto)
  operator()(Args &&...args) const {
    return evaluate(algorithm, representation, std::forward<Args>(args)...);
  }
};

template <typename Algorithm, typename Representation>
prepared_t(Algorithm, Representation) -> prepared_t<Algorithm, Representation>;

template <typename Algorithm, typename P>
[[nodiscard]] constexpr auto prepare(Algorithm algorithm, P &&p) {
  return prepared_t{
      algorithm,
      detail::prepared_representation(algorithm, std::forward<P>(p))};
}

template <typename Algorithm, typename P>
  requires std::is_lvalue_reference_v<P &&>
[[nodiscard]] constexpr auto borrow(Algorithm algorithm, P &&p) noexcept {
  using representation_type = std::remove_reference_t<P> const &;
  return prepared_t<Algorithm, representation_type>{algorithm, p};
}
} // namespace polygnition::polynomial

#undef POLYGNITION_PREPARED_INLINE
