#include <boost/ut.hpp>
#include <array>
#include <cmath>
#include <polygnition/knuth_eve.hpp>
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

int main() {
  ut::test("knuth_eve_odd_part_extracts_roots_polynomial") = [] {
    // p(x)=((2x+3)(x²+7)-13)(x²-5)+11
    // odd part: 2s²+4s-70 = 2(s-5)(s+7)
    constexpr auto p = poly::literal<2.0, 3.0, 4.0, -7.0, -70.0, -29.0>();
    constexpr auto odd = poly::knuth_eve_odd_part(p);

    static_assert(odd == poly::literal<2.0, 4.0, -70.0>());
  };

  ut::test("knuth_eve_preprocessed_degree5_matches_horner") = [] {
    constexpr auto p = poly::literal<2.0, 3.0, 4.0, -7.0, -70.0, -29.0>();
    constexpr auto pre = poly::preprocess_knuth_eve(p, std::array{5.0, -7.0});

    static_assert(pre.base == poly::polynomial_t{2.0, 3.0});
    static_assert(pre.alpha == std::array{5.0, -7.0});
    static_assert(pre.gamma == std::array{11.0, -13.0});
    static_assert(pre(0.25) == poly::evaluate(poly::horner, p, 0.25));

    for (auto const x : std::array{-2.0, -0.5, 0.0, 0.75, 3.0}) {
      auto const got = pre(x);
      auto const expected = poly::evaluate(poly::horner, p, x);
      ut::expect(ut::approx(got, expected, 1e-10));
    }
  };

  ut::test("knuth_eve_preprocessed_even_degree_keeps_quadratic_base") = [] {
    // p(x)=((4x²-2x+1)(x²+5)-11)(x²-3)+7
    constexpr auto p =
        poly::literal<4.0, -2.0, 9.0, -4.0, -69.0, 30.0, 25.0>();
    constexpr auto pre = poly::preprocess_knuth_eve(p, std::array{3.0, -5.0});

    static_assert(pre.base == poly::polynomial_t{4.0, -2.0, 1.0});
    static_assert(pre.gamma == std::array{7.0, -11.0});

    for (auto const x : std::array{-1.5, -0.25, 0.0, 0.5, 2.0}) {
      auto const got = pre(x);
      auto const expected = poly::evaluate(poly::horner, p, x);
      ut::expect(ut::approx(got, expected, 1e-9));
    }
  };

  ut::test("knuth_eve_preprocessed_shifted_polynomial_matches_horner") = [] {
    // p(x) = F(x-1.5), where
    // F(u)=((2u+3)(u²+7)-13)(u²-5)+11.
    constexpr auto p =
        poly::literal<2.0, -12.0, 31.0, -52.0, -11.875, 46.75>();
    constexpr auto pre =
        poly::preprocess_knuth_eve(p, 1.5, std::array{5.0, -7.0});

    static_assert(pre.shift == 1.5);
    static_assert(pre.base == poly::polynomial_t{2.0, 3.0});
    static_assert(pre.gamma == std::array{11.0, -13.0});

    for (auto const x : std::array{-1.0, 0.0, 1.5, 2.25, 4.0}) {
      auto const got = pre(x);
      auto const expected = poly::evaluate(poly::horner, p, x);
      ut::expect(ut::approx(got, expected, 1e-9));
    }
  };
}
