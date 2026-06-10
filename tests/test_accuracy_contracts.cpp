#include <boost/ut.hpp>
#include <array>
#include <cmath>
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

int main() {
  ut::test("Horner exposes a constexpr forward-error bound") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0, -4.0, 5.0>();
    constexpr auto bound = poly::error_bound(p, 1.0);
    static_assert(bound > 0.0);
    static_assert(bound < 2.0e-13);

    auto const x = 0.625;
    auto const value = poly::evaluate(poly::horner, p, x);
    auto const reference = static_cast<long double>(1.0L) * x * x * x * x -
                           static_cast<long double>(2.0L) * x * x * x +
                           static_cast<long double>(3.0L) * x * x -
                           static_cast<long double>(4.0L) * x + 5.0L;
    ut::expect(std::abs(value - static_cast<double>(reference)) <= bound);
  };

  ut::test("compensated Horner improves a cancellation-heavy value") = [] {
    constexpr auto p = poly::literal<1.0, -4.00000000000001,
                                     6.00000000000003, -4.00000000000003,
                                     1.00000000000001>();
    auto const x = 1.0;
    auto const ordinary = poly::evaluate(poly::horner, p, x);
    auto const corrected = poly::evaluate(poly::compensated, p, x);
    auto const reference = 0.0L;
    ut::expect(std::abs(static_cast<long double>(corrected) - reference) <=
               std::abs(static_cast<long double>(ordinary) - reference));
  };
}
