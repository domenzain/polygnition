#pragma once
#include "polygnition/target.hpp"
#include "polygnition/util.hpp"
#include <climits>
#include <concepts>
#include <expected>
#include <limits>
#include <type_traits>

namespace polygnition::polynomial::detail {
enum class value_category : unsigned {
  any,
  floating_real,
  integral_real,
  complex_floating,
  other,
};

enum class storage_category : unsigned {
  any,
  runtime,
  static_pack,
};

enum class evaluation_intent : unsigned {
  latency,
  throughput,
};

enum class strategy_category : unsigned {
  horner,
  knuth,
  motzkin,
  multivariate_horner,
  estrin,
  compensated,
  dorn,
};

enum class coefficient_shape : unsigned {
  any,
  zero_leading,
  leading_zero = zero_leading,
  leading_nonzero,
  general = leading_nonzero,
  motzkin_safe,
};

enum class accuracy_class : unsigned {
  any,
  motzkin_cancellation_safe,
};

enum class selection_error : unsigned {
  no_matching_row,
};

struct algorithm_choice {
  strategy_category strategy{};
  int stride{};

  friend constexpr auto operator==(algorithm_choice, algorithm_choice)
      -> bool = default;
};

struct batch_algorithm_choice {
  algorithm_choice algorithm{};
  int lanes = 1;
  int unroll = 4;

  friend constexpr auto operator==(batch_algorithm_choice,
                                   batch_algorithm_choice) -> bool = default;
};

struct tuning_row {
  value_category coeff;
  value_category var;
  storage_category storage;
  evaluation_intent intent;
  int first_degree;
  int last_degree;
  strategy_category strategy;
  int stride;
  int coeff_bits = 0;
  int var_bits = 0;
  int lanes = 0;
  coefficient_shape shape = coefficient_shape::any;
  accuracy_class accuracy = accuracy_class::any;
  int unroll = 4;
};

inline constexpr auto any_degree = 1'000'000;
inline constexpr auto any_bits = 0;
inline constexpr auto any_lanes = 0;

template <typename T> [[nodiscard]] constexpr auto abs_constexpr(T value) {
  return value < T{} ? -value : value;
}

template <typename T> [[nodiscard]] constexpr auto max_constexpr(T a, T b) {
  return a < b ? b : a;
}

template <typename T>
[[nodiscard]] constexpr auto cancellation_ratio(T lhs, T rhs, T result) {
  auto const local_scale =
      max_constexpr(T{1}, max_constexpr(abs_constexpr(lhs), abs_constexpr(rhs)));
  auto const floor = std::numeric_limits<T>::epsilon() * local_scale;
  return local_scale / max_constexpr(abs_constexpr(result), floor);
}

template <typename T>
[[nodiscard]] constexpr auto motzkin_cancellation_estimate(
    T leading, T c3, T c2, T c1, T c0) -> T {
  auto const inv_leading = T{1} / leading;
  auto const a3 = c3 * inv_leading;
  auto const a2 = c2 * inv_leading;
  auto const a1 = c1 * inv_leading;
  auto const a0 = c0 * inv_leading;

  auto const beta0_input = a3 - T{1};
  auto const beta0 = T{0.5} * beta0_input;
  auto const beta0_product = beta0 * (beta0 + T{1});
  auto const z = a2 - beta0_product;
  auto const beta0_z = beta0 * z;
  auto const beta1 = a1 - beta0_z;
  auto const two_beta1 = T{2} * beta1;
  auto const beta2 = z - two_beta1;
  auto const beta1_product = beta1 * (beta1 + beta2);
  auto const beta3 = a0 - beta1_product;

  auto const coeff_scale =
      max_constexpr(T{1},
                    max_constexpr(abs_constexpr(a3),
                                  max_constexpr(abs_constexpr(a2),
                                                max_constexpr(abs_constexpr(a1),
                                                              abs_constexpr(a0)))));
  auto const beta_scale = max_constexpr(
      abs_constexpr(beta0),
      max_constexpr(abs_constexpr(z),
                    max_constexpr(abs_constexpr(beta1),
                                  max_constexpr(abs_constexpr(beta2),
                                                abs_constexpr(beta3)))));
  auto const cancellation = max_constexpr(
      cancellation_ratio(a3, T{1}, beta0_input),
      max_constexpr(cancellation_ratio(a2, beta0_product, z),
                    max_constexpr(cancellation_ratio(a1, beta0_z, beta1),
                                  max_constexpr(
                                      cancellation_ratio(z, two_beta1, beta2),
                                      cancellation_ratio(a0, beta1_product,
                                                         beta3)))));
  return max_constexpr(cancellation, beta_scale / coeff_scale);
}

template <typename T>
[[nodiscard]] constexpr auto motzkin_coefficients_are_safe(
    T leading, T c3, T c2, T c1, T c0) -> bool {
  // Roughly six lost decimal digits is where Motzkin's shorter recurrence stops
  // being a good automatic default. Users may still request poly::motzkin
  // explicitly for a preprocessed quartic whose domain is known to be benign.
  return motzkin_cancellation_estimate(leading, c3, c2, c1, c0) <= T{1.0e6};
}

template <typename T> [[nodiscard]] consteval auto value_category_for() {
  using raw_type = std::remove_cvref_t<T>;
  if constexpr (split_complex_like<raw_type>) {
    using scalar = split_complex_scalar_t<raw_type>;
    if constexpr (std::floating_point<scalar>) {
      return value_category::complex_floating;
    } else {
      return value_category::other;
    }
  } else {
    using value_type = scalar_value_type_t<raw_type>;
    if constexpr (complex<value_type>) {
      if constexpr (std::floating_point<typename value_type::value_type>) {
        return value_category::complex_floating;
      } else {
        return value_category::other;
      }
    } else if constexpr (std::floating_point<value_type>) {
      return value_category::floating_real;
    } else if constexpr (arithmetic<value_type>) {
      return value_category::integral_real;
    } else {
      return value_category::other;
    }
  }
}

template <typename P> [[nodiscard]] consteval auto storage_category_for() {
  using polynomial_type = std::remove_cvref_t<P>;
  if constexpr (is_static_polynomial_v<polynomial_type>) {
    return storage_category::static_pack;
  } else {
    return storage_category::runtime;
  }
}

template <typename T> [[nodiscard]] consteval auto element_bits_for() -> int {
  using raw_type = std::remove_cvref_t<T>;
  if constexpr (split_complex_like<raw_type>) {
    return element_bits_for<typename raw_type::value_type>();
  } else {
    using value_type = scalar_value_type_t<raw_type>;
    if constexpr (complex<value_type>) {
      return static_cast<int>(sizeof(typename value_type::value_type) * CHAR_BIT);
    } else if constexpr (arithmetic<value_type>) {
      return static_cast<int>(sizeof(value_type) * CHAR_BIT);
    } else {
      return 0;
    }
  }
}

template <typename T> [[nodiscard]] consteval auto lane_count_for() -> int {
  using raw_type = std::remove_cvref_t<T>;
  if constexpr (split_complex_like<raw_type>) {
    return lane_count_for<typename raw_type::value_type>();
  } else if constexpr (simd_like<raw_type>) {
    return static_cast<int>(raw_type::size());
  } else {
    return 1;
  }
}

template <typename P> [[nodiscard]] consteval auto motzkin_polynomial_is_safe() {
  using polynomial_type = std::remove_cvref_t<P>;
  if constexpr (!is_static_polynomial_v<polynomial_type> ||
                polynomial_type::degree != 4) {
    return false;
  } else {
    using coeff_type = typename polynomial_type::value_type;
    if constexpr (!std::floating_point<coeff_type>) {
      return false;
    } else {
      constexpr auto const &coefficients = polynomial_type::value;
      if constexpr (coefficients[0] == coeff_type{}) {
        return false;
      } else {
        return motzkin_coefficients_are_safe(
          static_cast<long double>(coefficients[0]),
          static_cast<long double>(coefficients[1]),
          static_cast<long double>(coefficients[2]),
          static_cast<long double>(coefficients[3]),
          static_cast<long double>(coefficients[4]));
      }
    }
  }
}

template <typename P> [[nodiscard]] consteval auto coefficient_shape_for() {
  using polynomial_type = std::remove_cvref_t<P>;
  using stored_type = stored_polynomial_type_t<polynomial_type>;
  if constexpr (!is_static_polynomial_v<polynomial_type>) {
    return coefficient_shape::any;
  } else {
    using coeff_type = typename stored_type::value_type;
    constexpr auto leading = static_cast<coeff_type>(polynomial_type::value[0]);
    if constexpr (leading == coeff_type{}) {
      return coefficient_shape::leading_zero;
    } else if constexpr (stored_type::degree == 4 &&
                         std::floating_point<coeff_type> &&
                         motzkin_polynomial_is_safe<polynomial_type>()) {
      return coefficient_shape::motzkin_safe;
    } else {
      return coefficient_shape::leading_nonzero;
    }
  }
}

template <strategy_category Strategy, int Stride>
[[nodiscard]] consteval auto strategy_constant() {
  if constexpr (Strategy == strategy_category::horner) {
    return horner;
  } else if constexpr (Strategy == strategy_category::knuth) {
    return knuth;
  } else if constexpr (Strategy == strategy_category::motzkin) {
    return motzkin;
  } else if constexpr (Strategy == strategy_category::multivariate_horner) {
    return multivariate_horner;
  } else if constexpr (Strategy == strategy_category::estrin) {
    return estrin;
  } else if constexpr (Strategy == strategy_category::compensated) {
    return compensated;
  } else {
    return dorn<Stride>;
  }
}

template <typename P, typename Var, evaluation_intent Intent>
struct selection_key {
  using polynomial_type = std::remove_cvref_t<P>;
  using stored_type = stored_polynomial_type_t<polynomial_type>;
  using coeff_type = typename stored_type::value_type;
  using var_type = std::remove_cvref_t<Var>;

  static constexpr auto degree = stored_type::degree;
  static constexpr auto coeff_kind = value_category_for<coeff_type>();
  static constexpr auto var_kind = value_category_for<var_type>();
  static constexpr auto storage = storage_category_for<polynomial_type>();
  static constexpr auto intent = Intent;
  static constexpr auto coeff_bits = element_bits_for<coeff_type>();
  static constexpr auto var_bits = element_bits_for<var_type>();
  static constexpr auto element_bits = var_bits;
  static constexpr auto lanes = lane_count_for<var_type>();
  static constexpr auto shape = coefficient_shape_for<polynomial_type>();
  static constexpr auto leading = shape;
  static constexpr auto motzkin_safe =
      motzkin_polynomial_is_safe<polynomial_type>();
};

template <tuning_row Row, typename Key>
[[nodiscard]] consteval auto row_accuracy_matches() {
  if constexpr (Row.accuracy == accuracy_class::any) {
    return true;
  } else if constexpr (Row.accuracy == accuracy_class::motzkin_cancellation_safe) {
    return Key::motzkin_safe;
  } else {
    return false;
  }
}

template <tuning_row Row, typename Key>
[[nodiscard]] consteval auto row_matches() {
  auto const coeff_matches =
      Row.coeff == value_category::any || Row.coeff == Key::coeff_kind;
  auto const var_matches =
      Row.var == value_category::any || Row.var == Key::var_kind;
  auto const storage_matches = Row.storage == storage_category::any ||
                               Row.storage == Key::storage;
  auto const coeff_bits_match =
      Row.coeff_bits == 0 || Row.coeff_bits == Key::coeff_bits;
  auto const var_bits_match =
      Row.var_bits == 0 || Row.var_bits == Key::var_bits;
  auto const lanes_match = Row.lanes == 0 || Row.lanes == Key::lanes;
  auto const shape_matches =
      Row.shape == coefficient_shape::any || Row.shape == Key::shape ||
      (Row.shape == coefficient_shape::leading_nonzero &&
       Key::shape == coefficient_shape::motzkin_safe);
  return coeff_matches && var_matches && storage_matches && coeff_bits_match &&
         var_bits_match && lanes_match && shape_matches &&
         row_accuracy_matches<Row, Key>() && Row.intent == Key::intent &&
         Row.first_degree <= Key::degree && Key::degree < Row.last_degree;
}


template <tuning_row Row, typename Key>
[[nodiscard]] consteval auto row_matches_ignoring_lanes() {
  auto const coeff_matches =
      Row.coeff == value_category::any || Row.coeff == Key::coeff_kind;
  auto const var_matches =
      Row.var == value_category::any || Row.var == Key::var_kind;
  auto const storage_matches = Row.storage == storage_category::any ||
                               Row.storage == Key::storage;
  auto const coeff_bits_match =
      Row.coeff_bits == 0 || Row.coeff_bits == Key::coeff_bits;
  auto const var_bits_match =
      Row.var_bits == 0 || Row.var_bits == Key::var_bits;
  auto const shape_matches =
      Row.shape == coefficient_shape::any || Row.shape == Key::shape ||
      (Row.shape == coefficient_shape::leading_nonzero &&
       Key::shape == coefficient_shape::motzkin_safe);
  return coeff_matches && var_matches && storage_matches && coeff_bits_match &&
         var_bits_match && shape_matches && row_accuracy_matches<Row, Key>() &&
         Row.intent == Key::intent && Row.first_degree <= Key::degree &&
         Key::degree < Row.last_degree;
}

template <algorithm_choice Choice>
[[nodiscard]] consteval auto strategy_constant() {
  return strategy_constant<Choice.strategy, Choice.stride>();
}

template <algorithm_choice Choice, typename P, typename Var>
[[nodiscard]] consteval auto strategy_conforms() -> bool {
  if constexpr (requires(P const &p, Var const &x) {
                  { evaluate(strategy_constant<Choice>(), p, x) } ->
                      std::same_as<eval_result_t<P, Var>>;
                }) {
    return true;
  } else {
    return false;
  }
}

template <typename Key, typename P, typename Var, tuning_row Row,
          tuning_row... Rows>
[[nodiscard]] consteval auto try_select_valid_from_rows()
    -> std::expected<algorithm_choice, selection_error> {
  if constexpr (row_matches<Row, Key>()) {
    constexpr auto choice = algorithm_choice{Row.strategy, Row.stride};
    if constexpr (strategy_conforms<choice, P, Var>()) {
      return choice;
    }
  }
  if constexpr (sizeof...(Rows) > 0) {
    return try_select_valid_from_rows<Key, P, Var, Rows...>();
  } else {
    return std::unexpected{selection_error::no_matching_row};
  }
}


template <algorithm_choice Choice, typename P, typename X, int LaneCount>
[[nodiscard]] consteval auto batch_strategy_conforms() -> bool {
  using x_type = std::remove_cvref_t<X>;
  if constexpr (LaneCount > 1 && split_complex_like<x_type>) {
    using component_t = typename x_type::value_type;
    if constexpr (arithmetic<component_t>) {
      using lane_component =
          lanes<component_t, static_cast<std::size_t>(LaneCount)>;
      using lane_t = split_complex<lane_component>;
      return strategy_conforms<Choice, P, lane_t>();
    } else {
      return false;
    }
  } else if constexpr (LaneCount > 1 && arithmetic<x_type>) {
    using lane_t = lanes<x_type, static_cast<std::size_t>(LaneCount)>;
    return strategy_conforms<Choice, P, lane_t>();
  } else {
    return strategy_conforms<Choice, P, x_type>();
  }
}

template <typename X> [[nodiscard]] consteval auto default_batch_lane_count() {
  if constexpr (std::same_as<std::remove_cvref_t<X>, float>) {
    return 8;
  } else if constexpr (std::same_as<std::remove_cvref_t<X>, double>) {
    return 4;
  } else {
    return 1;
  }
}

template <typename Key, typename P, typename X, tuning_row Row,
          tuning_row... Rows>
[[nodiscard]] consteval auto try_select_batch_from_rows()
    -> std::expected<batch_algorithm_choice, selection_error> {
  if constexpr (row_matches_ignoring_lanes<Row, Key>()) {
    constexpr auto choice = algorithm_choice{Row.strategy, Row.stride};
    constexpr auto candidate_lanes = Row.lanes > 0 ? Row.lanes : 1;
    static_assert(candidate_lanes > 0, "batch tuning rows must use a positive lane count");
    if constexpr (batch_strategy_conforms<choice, P, X, candidate_lanes>()) {
      return batch_algorithm_choice{choice, candidate_lanes, Row.unroll};
    }
  }
  if constexpr (sizeof...(Rows) > 0) {
    return try_select_batch_from_rows<Key, P, X, Rows...>();
  } else {
    return std::unexpected{selection_error::no_matching_row};
  }
}

template <tuning_row... Rows> struct tuning_table {
  template <typename Key, typename P, typename Var>
  [[nodiscard]] static consteval auto try_select_valid_choice()
      -> std::expected<algorithm_choice, selection_error> {
    if constexpr (sizeof...(Rows) == 0) {
      return std::unexpected{selection_error::no_matching_row};
    } else {
      return try_select_valid_from_rows<Key, P, Var, Rows...>();
    }
  }

  template <typename Key, typename P, typename Var>
  [[nodiscard]] static consteval auto select_valid_choice() {
    auto choice = try_select_valid_choice<Key, P, Var>();
    if (choice) {
      return *choice;
    }
    constexpr auto fallback = algorithm_choice{strategy_category::horner, 0};
    static_assert(strategy_conforms<fallback, P, Var>(),
                  "no tuned row or Horner fallback can evaluate this polynomial/key with the documented eval_result_t");
    return fallback;
  }

  template <typename Key, typename P, typename Var>
  [[nodiscard]] static consteval auto select_valid() {
    return strategy_constant<select_valid_choice<Key, P, Var>()>();
  }

  template <typename Key, typename P, typename X>
  [[nodiscard]] static consteval auto try_select_batch_choice()
      -> std::expected<batch_algorithm_choice, selection_error> {
    if constexpr (sizeof...(Rows) == 0) {
      return std::unexpected{selection_error::no_matching_row};
    } else {
      return try_select_batch_from_rows<Key, P, X, Rows...>();
    }
  }

  template <typename Key, typename P, typename X>
  [[nodiscard]] static consteval auto select_batch_choice() {
    auto choice = try_select_batch_choice<Key, P, X>();
    if (choice) {
      return *choice;
    }
    constexpr auto fallback = select_valid_choice<Key, P, X>();
    constexpr auto fallback_lanes = default_batch_lane_count<X>();
    if constexpr (batch_strategy_conforms<fallback, P, X, fallback_lanes>()) {
      return batch_algorithm_choice{fallback, fallback_lanes, 4};
    } else {
      return batch_algorithm_choice{fallback, 1, 4};
    }
  }
};

template <tuning_row... ProfileRows>
using profile_table = tuning_table<
    tuning_row{value_category::floating_real, value_category::floating_real,
               storage_category::static_pack, evaluation_intent::latency, 4, 5,
               strategy_category::motzkin, 0, 64, 64, 1,
               coefficient_shape::motzkin_safe,
               accuracy_class::motzkin_cancellation_safe},
    tuning_row{value_category::floating_real, value_category::floating_real,
               storage_category::static_pack, evaluation_intent::throughput, 4,
               5, strategy_category::motzkin, 0, 64, 64, 4,
               coefficient_shape::motzkin_safe,
               accuracy_class::motzkin_cancellation_safe},
    tuning_row{value_category::floating_real, value_category::floating_real,
               storage_category::static_pack, evaluation_intent::throughput, 4,
               5, strategy_category::motzkin, 0, 64, 64, 1,
               coefficient_shape::motzkin_safe,
               accuracy_class::motzkin_cancellation_safe},
    tuning_row{value_category::floating_real, value_category::floating_real,
               storage_category::static_pack, evaluation_intent::latency, 4, 5,
               strategy_category::horner, 0, 64, 64, 1,
               coefficient_shape::leading_zero},
    tuning_row{value_category::floating_real, value_category::floating_real,
               storage_category::static_pack, evaluation_intent::throughput, 4,
               5, strategy_category::horner, 0, 64, 64, 4,
               coefficient_shape::leading_zero},
    tuning_row{value_category::floating_real, value_category::floating_real,
               storage_category::static_pack, evaluation_intent::throughput, 4,
               5, strategy_category::horner, 0, 64, 64, 1,
               coefficient_shape::leading_zero},
    tuning_row{value_category::floating_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::latency, 0, 2,
               strategy_category::horner, 0, 64, 64, 1},
    tuning_row{value_category::floating_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::throughput, 0, 2,
               strategy_category::horner, 0, 64, 64, 4,
               coefficient_shape::any, accuracy_class::any, 4},
    tuning_row{value_category::floating_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::throughput, 0, 2,
               strategy_category::horner, 0, 64, 64, 1},
    tuning_row{value_category::floating_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::latency, 2,
               any_degree, strategy_category::knuth, 0, 64, 64, 1},
    tuning_row{value_category::floating_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::throughput, 2,
               any_degree, strategy_category::knuth, 0, 64, 64, 4,
               coefficient_shape::any, accuracy_class::any, 1},
    tuning_row{value_category::floating_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::throughput, 2,
               any_degree, strategy_category::knuth, 0, 64, 64, 1},
    tuning_row{value_category::integral_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::latency, 0,
               any_degree, strategy_category::knuth, 0},
    tuning_row{value_category::integral_real, value_category::complex_floating,
               storage_category::any, evaluation_intent::throughput, 0,
               any_degree, strategy_category::knuth, 0},
    ProfileRows...>;

template <target::profile Profile> struct profile_tuning;

#include "polygnition/detail/profile_tuning.inc"

struct select_t {
  template <target::profile Profile, typename Key>
    requires requires(select_t self) {
      polygnition::tag_invoke(self, tuned<Profile>, std::type_identity<Key>{});
    }
  [[nodiscard]] consteval auto operator()(algorithm::tuned_t<Profile> profile,
                                          std::type_identity<Key> key) const {
    return polygnition::tag_invoke(*this, profile, key);
  }

  template <target::profile Profile, typename Key>
  [[nodiscard]] consteval auto operator()(algorithm::tuned_t<Profile>,
                                          std::type_identity<Key>) const {
    return ::polygnition::polynomial::tuning<Profile, Key>::select();
  }
};
} // namespace polygnition::polynomial::detail
