#pragma once
#include "polygnition/poly.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define POLYGNITION_BATCH_INLINE [[gnu::always_inline]] inline
#else
#define POLYGNITION_BATCH_INLINE inline
#endif

namespace polygnition::polynomial {
template <typename Algorithm, typename Representation> struct prepared_t;

struct evaluate_into_t {
  template <typename Algorithm, typename Representation, typename... Args>
    requires requires(prepared_t<Algorithm, Representation> const &evaluator,
                      Args &&...args) {
      evaluator(std::forward<Args>(args)...);
    }
  [[nodiscard]] POLYGNITION_BATCH_INLINE constexpr decltype(auto)
  operator()(prepared_t<Algorithm, Representation> const &evaluator,
             Args &&...args) const {
    return evaluator(std::forward<Args>(args)...);
  }

  template <typename... Args>
    requires requires(Args &&...args) {
      evaluate(std::forward<Args>(args)...);
    }
  [[nodiscard]] POLYGNITION_BATCH_INLINE constexpr decltype(auto)
  operator()(Args &&...args) const {
    return evaluate(std::forward<Args>(args)...);
  }
};

inline constexpr evaluate_into_t evaluate_into{};
} // namespace polygnition::polynomial

#undef POLYGNITION_BATCH_INLINE
