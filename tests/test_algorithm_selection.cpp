#include <boost/ut.hpp>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <polygnition/dev/helpers.hpp>
#include <polygnition/poly.hpp>
#include <ranges>
#include <type_traits>
#include <utility>

namespace poly = polygnition::polynomial;
namespace target = polygnition::target;
namespace dev = polygnition::dev;
namespace ut = boost::ut;

namespace polygnition::test {
struct forced_tuning_key {};
struct forced_tag_invoke_key {};
struct forced_tuning_eval_strategy {};
struct forced_tag_invoke_eval_strategy {};
struct forced_tag_invoke_var {
  double value{};
};

using forced_tuning_eval_poly =
    polygnition::polynomial::polynomial_t<float, 3>;
using forced_tuning_eval_key = polygnition::polynomial::detail::selection_key<
    forced_tuning_eval_poly, float,
    polygnition::polynomial::detail::evaluation_intent::latency>;
using forced_tag_invoke_eval_poly =
    polygnition::polynomial::polynomial_t<double, 5>;
using forced_tag_invoke_eval_key =
    polygnition::polynomial::detail::selection_key<
        forced_tag_invoke_eval_poly, forced_tag_invoke_var,
        polygnition::polynomial::detail::evaluation_intent::latency>;

[[nodiscard]] consteval auto tag_invoke(
    polygnition::polynomial::select_t,
    polygnition::polynomial::algorithm::tuned_t<
        polygnition::target::profile::generic>,
    std::type_identity<forced_tag_invoke_key>) {
  return polygnition::polynomial::estrin;
}

[[nodiscard]] consteval auto tag_invoke(
    polygnition::polynomial::select_t,
    polygnition::polynomial::algorithm::tuned_t<
        polygnition::target::profile::generic>,
    std::type_identity<forced_tag_invoke_eval_key>) {
  return forced_tag_invoke_eval_strategy{};
}

[[nodiscard]] constexpr auto
tag_invoke(polygnition::polynomial::evaluate_t, forced_tuning_eval_strategy,
           forced_tuning_eval_poly const &, float) -> double {
  return -123.0;
}

[[nodiscard]] constexpr auto
tag_invoke(polygnition::polynomial::evaluate_t, forced_tag_invoke_eval_strategy,
           forced_tag_invoke_eval_poly const &, forced_tag_invoke_var) -> double {
  return 456.0;
}
} // namespace polygnition::test

namespace polygnition::polynomial {
template <> struct tuning<target::profile::generic,
                           polygnition::test::forced_tuning_key> {
  [[nodiscard]] static consteval auto select() { return dorn<2>; }
};

template <> struct tuning<target::profile::generic,
                           polygnition::test::forced_tuning_eval_key> {
  [[nodiscard]] static consteval auto select() {
    return polygnition::test::forced_tuning_eval_strategy{};
  }
};
} // namespace polygnition::polynomial

namespace {
template <int N, std::size_t... I>
[[nodiscard]] consteval auto static_floating_poly_impl(std::index_sequence<I...>) {
  return poly::literal<static_cast<double>(I + 1)...>();
}

template <int N> [[nodiscard]] consteval auto static_floating_poly() {
  return static_floating_poly_impl<N>(std::make_index_sequence<static_cast<std::size_t>(N + 1)>{});
}

template <typename P, typename V, typename Expected>
inline constexpr bool selects = std::is_same_v<poly::detail::selected_algorithm_t<P, V>, Expected>;

template <target::profile Profile, typename P, typename V, typename Expected>
inline constexpr bool selects_for =
    std::is_same_v<poly::detail::selected_algorithm_t_for<Profile, P, V>,
                   Expected>;

[[nodiscard]] constexpr auto tuning_override_reaches_evaluate() -> bool {
  auto const p =
      polygnition::test::forced_tuning_eval_poly{1.0F, 2.0F, 3.0F, 4.0F};
  return p.template operator()<target::profile::generic>(2.0F) == -123.0;
}

[[nodiscard]] constexpr auto tag_invoke_override_reaches_evaluate() -> bool {
  auto const p = polygnition::test::forced_tag_invoke_eval_poly{
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  return p.template operator()<target::profile::generic>(
             polygnition::test::forced_tag_invoke_var{2.0}) == 456.0;
}

static_assert(tuning_override_reaches_evaluate());
static_assert(tag_invoke_override_reaches_evaluate());

template <int N>
using floating_t = poly::polynomial_t<double, N>;
template <int N>
using complex_t = poly::polynomial_t<std::complex<double>, N>;

struct fake_simd {
  using value_type = double;
  double value{};

  [[nodiscard]] static constexpr auto size() -> std::size_t { return 4; }

  constexpr fake_simd() = default;
  constexpr fake_simd(double v) : value{v} {}
};

[[nodiscard]] constexpr auto operator+(fake_simd a, fake_simd b) -> fake_simd { return {a.value + b.value}; }
[[nodiscard]] constexpr auto operator-(fake_simd a, fake_simd b) -> fake_simd { return {a.value - b.value}; }
[[nodiscard]] constexpr auto operator*(fake_simd a, fake_simd b) -> fake_simd { return {a.value * b.value}; }

static_assert(polygnition::simd_like<fake_simd>);
using fake_simd_key = poly::detail::selection_key<
    floating_t<20>, fake_simd, poly::detail::evaluation_intent::throughput>;
using fake_simd_latency_key = poly::detail::selection_key<
    floating_t<20>, fake_simd, poly::detail::evaluation_intent::latency>;
using float_key = poly::detail::selection_key<
    poly::polynomial_t<float, 20>, float,
    poly::detail::evaluation_intent::latency>;
using float_batch_key = poly::detail::selection_key<
    poly::polynomial_t<float, 20>, float,
    poly::detail::evaluation_intent::throughput>;
static_assert(fake_simd_key::coeff_bits == 64);
static_assert(fake_simd_key::var_bits == 64);
static_assert(fake_simd_key::element_bits == 64);
static_assert(fake_simd_key::lanes == 4);
static_assert(fake_simd_latency_key::intent ==
              poly::detail::evaluation_intent::latency);
static_assert(float_key::coeff_bits == 32);
static_assert(float_key::var_bits == 32);
constexpr auto float_batch_choice =
    poly::detail::profile_tuning<target::profile::generic>::table::
        template select_batch_choice<float_batch_key,
                                     poly::polynomial_t<float, 20>, float>();
static_assert(float_batch_choice.algorithm.strategy ==
              poly::detail::strategy_category::dorn);
static_assert(float_batch_choice.algorithm.stride == 4);
static_assert(float_batch_choice.lanes == 8);
static_assert(float_batch_choice.unroll == 4);

using safe_quartic_t = decltype(poly::literal<1.0, 2.0, 3.0, 4.0, 5.0>());
using safe_quartic_key = poly::detail::selection_key<
    safe_quartic_t, double, poly::detail::evaluation_intent::latency>;
constexpr auto safe_quartic_choice =
    poly::tuning<target::profile::generic, safe_quartic_key>::try_select_choice();
static_assert(safe_quartic_choice.has_value());
static_assert(safe_quartic_choice->strategy ==
              poly::detail::strategy_category::motzkin);
using safe_quartic_batch_key = poly::detail::selection_key<
    safe_quartic_t, double, poly::detail::evaluation_intent::throughput>;
constexpr auto safe_quartic_batch_choice =
    poly::detail::profile_tuning<target::profile::generic>::table::
        template select_batch_choice<safe_quartic_batch_key, safe_quartic_t,
                                     double>();
static_assert(safe_quartic_batch_choice.algorithm.strategy ==
              poly::detail::strategy_category::motzkin);
static_assert(safe_quartic_batch_choice.lanes == 4);
static_assert(safe_quartic_batch_choice.unroll == 4);

using zero_leading_quartic_t =
    decltype(poly::literal<0.0, 1.0, 2.0, 3.0, 4.0>());
using zero_leading_key = poly::detail::selection_key<
    zero_leading_quartic_t, double, poly::detail::evaluation_intent::latency>;
static_assert(zero_leading_key::leading ==
              poly::detail::coefficient_shape::zero_leading);
static_assert(std::is_same_v<
              decltype(poly::select_strategy(
                  poly::tuned<target::profile::generic>,
                  std::type_identity<polygnition::test::forced_tuning_key>{})),
              poly::algorithm::dorn_t<2>>);
static_assert(std::is_same_v<
              decltype(poly::select_strategy(
                  poly::tuned<target::profile::generic>,
                  std::type_identity<polygnition::test::forced_tag_invoke_key>{})),
              poly::algorithm::estrin_t>);
static_assert(std::is_same_v<
              poly::detail::selected_algorithm_t_for<
                  target::profile::generic, floating_t<80>, fake_simd>,
              poly::detail::selected_throughput_algorithm_t_for<
                  target::profile::generic, floating_t<80>, fake_simd>>);

static_assert(selects<decltype(poly::literal<0.0, 1.0, 2.0, 3.0, 4.0>()),
                      double, poly::algorithm::horner_t>);
static_assert(selects<decltype(poly::literal<1.0, 2.0, 3.0, 4.0, 5.0>()),
                      double, poly::algorithm::motzkin_t>);
static_assert(std::is_same_v<poly::detail::selected_throughput_algorithm_t_for<
                                 target::profile::generic,
                                 decltype(poly::literal<1.0, 2.0, 3.0, 4.0,
                                                        5.0>()),
                                 double>,
                             poly::algorithm::motzkin_t>);
static_assert(selects<decltype(poly::literal<1.0, 3.0, 2.0, -100000000.0,
                                             -20000000000000000.0>()),
                      double, poly::algorithm::horner_t>);
static_assert(selects_for<target::profile::generic, floating_t<1>,
                          std::complex<double>, poly::algorithm::horner_t>);
static_assert(selects_for<target::profile::generic, floating_t<2>,
                          std::complex<double>, poly::algorithm::knuth_t>);
static_assert(selects_for<target::profile::generic, floating_t<2>,
                          std::complex<float>, poly::algorithm::horner_t>);
static_assert(std::is_same_v<
              poly::detail::eval_result_t<floating_t<2>, std::complex<float>>,
              std::complex<double>>);
static_assert(requires(floating_t<2> p, std::complex<float> z) {
  { p(z) } -> std::same_as<std::complex<double>>;
});
static_assert(selects_for<target::profile::generic,
                          poly::polynomial_t<std::int64_t, 4>,
                          std::complex<double>, poly::algorithm::knuth_t>);

using batch_real_key = poly::detail::selection_key<
    floating_t<20>, double, poly::detail::evaluation_intent::throughput>;
constexpr auto batch_real_choice =
    poly::detail::profile_tuning<target::profile::generic>::table::
        template select_batch_choice<batch_real_key, floating_t<20>, double>();
static_assert(batch_real_choice.algorithm.strategy ==
              poly::detail::strategy_category::dorn);
static_assert(batch_real_choice.algorithm.stride == 4);
static_assert(batch_real_choice.lanes == 4);
static_assert(batch_real_choice.unroll == 4);
constexpr auto icelake_batch_real_choice =
    poly::detail::profile_tuning<
        target::profile::x86_intel_icelake_avx512>::table::
        template select_batch_choice<batch_real_key, floating_t<20>, double>();
static_assert(icelake_batch_real_choice.algorithm.strategy ==
              poly::detail::strategy_category::dorn);
static_assert(icelake_batch_real_choice.algorithm.stride == 4);
static_assert(icelake_batch_real_choice.lanes == 8);
static_assert(icelake_batch_real_choice.unroll == 4);

using split_complex_double = polygnition::split_complex<double>;
using split_complex_lanes4 =
    polygnition::split_complex<polygnition::lanes<double, 4>>;
using split_complex_key = poly::detail::selection_key<
    floating_t<12>, split_complex_double,
    poly::detail::evaluation_intent::throughput>;
constexpr auto split_complex_batch_choice =
    poly::detail::profile_tuning<target::profile::generic>::table::
        template select_batch_choice<split_complex_key, floating_t<12>,
                                     split_complex_double>();
static_assert(split_complex_batch_choice.algorithm.strategy ==
              poly::detail::strategy_category::knuth);
static_assert(split_complex_batch_choice.lanes == 4);
static_assert(split_complex_batch_choice.unroll == 1);
static_assert(std::is_same_v<
              poly::detail::eval_result_t<floating_t<12>, split_complex_double>,
              split_complex_double>);
static_assert(std::is_same_v<
              poly::detail::eval_result_t<floating_t<12>, split_complex_lanes4>,
              split_complex_lanes4>);
static_assert(poly::detail::strategy_conforms<
              poly::detail::algorithm_choice{poly::detail::strategy_category::knuth, 0},
              floating_t<12>, split_complex_lanes4>());

using split_linear_key = poly::detail::selection_key<
    floating_t<1>, split_complex_double,
    poly::detail::evaluation_intent::throughput>;
constexpr auto split_linear_batch_choice =
    poly::detail::profile_tuning<target::profile::generic>::table::
        template select_batch_choice<split_linear_key, floating_t<1>,
                                     split_complex_double>();
static_assert(split_linear_batch_choice.algorithm.strategy ==
              poly::detail::strategy_category::horner);
static_assert(split_linear_batch_choice.lanes == 4);
static_assert(split_linear_batch_choice.unroll == 4);

template <poly::detail::tuning_row Row,
          poly::detail::evaluation_intent Intent, int Degree>
[[nodiscard]] consteval auto generated_real_row_matches() -> bool {
  return Row.coeff == poly::detail::value_category::floating_real &&
         Row.var == poly::detail::value_category::floating_real &&
         Row.storage == poly::detail::storage_category::any &&
         Row.intent == Intent && Row.first_degree <= Degree &&
         Degree < Row.last_degree && Row.coeff_bits == 64 &&
         Row.var_bits == 64 && Row.shape == poly::detail::coefficient_shape::any &&
         ((Intent == poly::detail::evaluation_intent::throughput &&
           (Row.lanes == 4 || Row.lanes == 8) && Row.unroll == 4) ||
          (Intent == poly::detail::evaluation_intent::latency && Row.lanes == 1));
}

template <typename Table> struct table_traits;
template <poly::detail::tuning_row... Rows>
struct table_traits<poly::detail::tuning_table<Rows...>> {
  template <poly::detail::evaluation_intent Intent, int Degree>
  [[nodiscard]] static consteval auto generated_real_match_count() -> int {
    return (0 + ... + (generated_real_row_matches<Rows, Intent, Degree>() ? 1 : 0));
  }

  template <poly::detail::evaluation_intent Intent, int... Degree>
  [[nodiscard]] static consteval auto generated_real_covers_exactly_once(
      std::integer_sequence<int, Degree...>) -> bool {
    return ((generated_real_match_count<Intent, Degree>() == 1) && ...);
  }
};

template <target::profile Profile>
[[nodiscard]] consteval auto generated_real_rows_are_total() -> bool {
  using table = typename poly::detail::profile_tuning<Profile>::table;
  using traits = table_traits<table>;
  return traits::template generated_real_covers_exactly_once<
             poly::detail::evaluation_intent::latency>(
             std::make_integer_sequence<int, 128>{}) &&
         traits::template generated_real_covers_exactly_once<
             poly::detail::evaluation_intent::throughput>(
             std::make_integer_sequence<int, 128>{});
}

static_assert(generated_real_rows_are_total<target::profile::generic>());
static_assert(generated_real_rows_are_total<target::profile::x86_intel_coffee_lake>());
static_assert(generated_real_rows_are_total<target::profile::x86_intel_icelake_avx512>());
static_assert(generated_real_rows_are_total<target::profile::x86_amd_zen3>());

template <int N> [[nodiscard]] consteval auto check_degree_selection() {
  using static_floating_t = decltype(static_floating_poly<N>());
  using runtime_floating_t = poly::polynomial_t<double, N>;
  using integral_t = poly::polynomial_t<std::int64_t, N>;
  using modular_t = poly::polynomial_t<std::uint32_t, N>;
  using complex_t = poly::polynomial_t<std::complex<double>, N>;
  using z_t = std::complex<double>;

  if constexpr (N == 4) {
    static_assert(selects<static_floating_t, double, poly::algorithm::motzkin_t>);
  }

  static_assert(selects<integral_t, std::int64_t, poly::algorithm::horner_t>);
  static_assert(selects<modular_t, std::uint32_t, poly::algorithm::horner_t>);
  static_assert(selects<complex_t, double, poly::algorithm::horner_t>);

  if constexpr (N < 2) {
    static_assert(selects<runtime_floating_t, z_t, poly::algorithm::horner_t>);
    static_assert(selects<static_floating_t, z_t, poly::algorithm::horner_t>);
  } else {
    static_assert(selects<runtime_floating_t, z_t, poly::algorithm::knuth_t>);
    static_assert(selects<static_floating_t, z_t, poly::algorithm::knuth_t>);
  }
  static_assert(selects<integral_t, z_t, poly::algorithm::knuth_t>);
  static_assert(selects<modular_t, z_t, poly::algorithm::knuth_t>);
  return true;
}

template <int... N>
[[nodiscard]] consteval auto check_selection(std::integer_sequence<int, N...>) {
  return (check_degree_selection<N>() and ...);
}
static_assert(check_selection(std::make_integer_sequence<int, 15>{}));

template <std::size_t I, typename Return, typename P, typename Var>
[[nodiscard]] constexpr auto reference_horner_impl(P const &p, Var const &x,
                                                   Return const &accum)
    -> Return {
  if constexpr (I == static_cast<std::size_t>(P::degree + 1)) {
    return accum;
  } else {
    return reference_horner_impl<I + 1, Return>(
        p, x,
        (accum * x) +
            static_cast<Return>(poly::detail::coefficient_at<I>(p)));
  }
}

template <typename Return, typename P, typename Var>
  requires poly::detail::is_polynomial_v<P>
[[nodiscard]] constexpr auto reference_horner(P const &p, Var const &x)
    -> Return {
  return reference_horner_impl<1, Return>(
      p, x, static_cast<Return>(poly::detail::coefficient_at<0>(p)));
}

template <typename T>
[[nodiscard]] constexpr auto close_enough(T const &a, T const &b) -> bool { return a == b; }
[[nodiscard]] auto close_enough(double a, double b) -> bool { return std::abs(a - b) <= 1e-10; }
[[nodiscard]] auto close_enough(std::complex<double> a, std::complex<double> b) -> bool { return std::abs(a - b) <= 1e-9; }

template <int N> [[nodiscard]] constexpr auto check_degree_values() -> bool {
  auto const static_floating = static_floating_poly<N>();
  auto const runtime_floating = dev::iota_poly<double, N>();
  auto const integral = dev::iota_poly<std::int64_t, N>();
  auto const modular = dev::iota_poly<std::uint32_t, N>();
  auto const complex_coeff = dev::iota_poly<std::complex<double>, N>({1.0, -0.25}, {0.5, 0.125});
  auto const x = 0.875;
  auto const i = std::int64_t{2};
  auto const u = std::uint32_t{7};
  auto const z = std::complex<double>{0.75, -0.3125};
  return close_enough(static_floating(x), reference_horner<double>(static_floating, x)) and
         close_enough(runtime_floating(x), reference_horner<double>(runtime_floating, x)) and
         close_enough(integral(i), reference_horner<std::int64_t>(integral, i)) and
         close_enough(modular(u), reference_horner<std::uint32_t>(modular, u)) and
         close_enough(complex_coeff(x), reference_horner<std::complex<double>>(complex_coeff, x)) and
         close_enough(static_floating(z), reference_horner<std::complex<double>>(static_floating, z)) and
         close_enough(runtime_floating(z), reference_horner<std::complex<double>>(runtime_floating, z)) and
         close_enough(integral(z), reference_horner<std::complex<double>>(integral, z)) and
         close_enough(modular(z), reference_horner<std::complex<double>>(modular, z)) and
         close_enough(complex_coeff(z), reference_horner<std::complex<double>>(complex_coeff, z));
}

template <int... N>
[[nodiscard]] constexpr auto check_values(std::integer_sequence<int, N...>) { return (check_degree_values<N>() and ...); }
} // namespace

int main() {
  ut::test("automatic_low_degree_selection_matrix") = [] {
    ut::expect(check_values(std::make_integer_sequence<int, 15>{}));
  };
}
