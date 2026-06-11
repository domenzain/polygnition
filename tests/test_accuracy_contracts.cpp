#include <boost/ut.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <polygnition/poly.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

namespace {
struct double_double {
  double hi;
  double lo;
};

[[nodiscard]] constexpr auto renormalize(double hi, double lo)
    -> double_double {
  auto const sum = poly::detail::two_sum(hi, lo);
  return {.hi = sum.value, .lo = sum.error};
}

[[nodiscard]] constexpr auto dd_add(double_double a, double b)
    -> double_double {
  auto const sum = poly::detail::two_sum(a.hi, b);
  auto const correction = poly::detail::two_sum(a.lo, sum.error);
  return renormalize(sum.value, correction.value + correction.error);
}

[[nodiscard]] constexpr auto dd_mul(double_double a, double b)
    -> double_double {
  auto const product = poly::detail::two_prod(a.hi, b);
  return renormalize(product.value, (a.lo * b) + product.error);
}

[[nodiscard]] constexpr auto dd_horner(auto const &p, double x) -> double {
  auto accum = double_double{.hi = 0.0, .lo = 0.0};
  for (auto const coeff : p) {
    accum = dd_add(dd_mul(accum, x), static_cast<double>(coeff));
  }
  return accum.hi + accum.lo;
}

[[nodiscard]] auto ordered_bits(double value) -> std::uint64_t {
  auto bits = std::bit_cast<std::uint64_t>(value);
  if ((bits >> 63U) != 0U) {
    bits = ~bits + 1U;
  } else {
    bits |= (std::uint64_t{1} << 63U);
  }
  return bits;
}

[[nodiscard]] auto ulp_distance(double a, double b) -> std::uint64_t {
  auto const ai = ordered_bits(a);
  auto const bi = ordered_bits(b);
  return ai > bi ? ai - bi : bi - ai;
}

template <typename Algorithm, typename P>
void expect_ulp(Algorithm algorithm, P const &p, double x,
                std::uint64_t max_ulp) {
  auto const got = poly::evaluate(algorithm, p, x);
  auto const oracle = dd_horner(p, x);
  ut::expect(ulp_distance(got, oracle) <= max_ulp);
}

template <typename Algorithm, typename P>
void expect_abs(Algorithm algorithm, P const &p, double x, double max_abs) {
  auto const got = poly::evaluate(algorithm, p, x);
  auto const oracle = dd_horner(p, x);
  ut::expect(std::abs(got - oracle) <= max_abs);
}
} // namespace

int main() {
  ut::test("constexpr_horner_error_bound") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0, -4.0, 5.0>();
    constexpr auto bound = poly::error_bound(p, 1.0);
    static_assert(bound > 0.0);
    static_assert(bound < 2.0e-13);
    auto const x = 0.625;
    auto const got = poly::evaluate(poly::horner, p, x);
    auto const oracle = dd_horner(p, x);
    ut::expect(std::abs(got - oracle) <= bound);
  };

  ut::test("motzkin_is_auto_only_when_consteval_gate_is_safe") = [] {
    constexpr auto safe = poly::literal<1.0, 2.0, 3.0, 4.0, 5.0>();
    constexpr auto unsafe = poly::literal<1.0, 3.0, 2.0, -100000000.0,
                                          -20000000000000000.0>();
    static_assert(std::is_same_v<poly::detail::selected_algorithm_t<
                                     decltype(safe), double>,
                                 poly::algorithm::motzkin_t>);
    static_assert(std::is_same_v<poly::detail::selected_algorithm_t<
                                     decltype(unsafe), double>,
                                 poly::algorithm::horner_t>);
    expect_ulp(poly::automatic, safe, 0.25, 4);
    expect_ulp(poly::automatic, unsafe, 0.25, 4);
  };

  ut::test("documented_strategy_ulp_envelopes") = [] {
    constexpr auto randomish = poly::literal<0.125, -1.75, 3.5, -0.0625,
                                             8.0, -11.0, 0.5>();
    constexpr auto wilkinson5 = poly::literal<1.0, -15.0, 85.0, -225.0,
                                              274.0, -120.0>();
    constexpr auto clustered = poly::literal<1.0, -4.00000000000001,
                                             6.00000000000003,
                                             -4.00000000000003,
                                             1.00000000000001>();

    for (auto const x : std::array{-0.875, -0.125, 0.125, 0.875}) {
      expect_ulp(poly::horner, randomish, x, 4);
      expect_ulp(poly::dorn<2>, randomish, x, 8);
      expect_ulp(poly::estrin, randomish, x, 8);
      expect_ulp(poly::compensated, randomish, x, 2);
    }
    for (auto const x : std::array{0.875, 1.125, 1.5, 2.25}) {
      expect_ulp(poly::horner, wilkinson5, x, 64);
      expect_ulp(poly::dorn<2>, wilkinson5, x, 128);
      expect_ulp(poly::estrin, wilkinson5, x, 128);
      expect_ulp(poly::compensated, wilkinson5, x, 16);
    }
    // Around clustered roots the correctly rounded value is close enough to zero
    // that ULP counts mostly measure exponent-bin changes.  Keep those cases in
    // the corpus, but assert absolute forward-error envelopes instead.
    for (auto const x : std::array{0.999, 1.0, 1.001}) {
      expect_abs(poly::horner, clustered, x, 2.0e-16);
      expect_abs(poly::motzkin, clustered, x, 5.0e-16);
      expect_abs(poly::automatic, clustered, x, 5.0e-16);
      expect_abs(poly::compensated, clustered, x, 5.0e-20);
    }
  };
}
