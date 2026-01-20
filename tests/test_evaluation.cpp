#include <boost/ut.hpp>
#include <cstdint>
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

int main() {
  ut::test("Horner evaluates integral coefficients") = [] {
    auto const p = poly::polynomial_t{1, 2, 3, 4};
    auto const x = 5;
    ut::expect(p(x) == 1 * x * x * x + 2 * x * x + 3 * x + 4);
  };

  ut::test("Horner preserves floating-point type") = [] {
    auto const p = poly::polynomial_t{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto const x = 1.25;
    auto const expected =
        1.0 * x * x * x * x * x + 2.0 * x * x * x * x +
        3.0 * x * x * x + 4.0 * x * x + 5.0 * x + 6.0;
    ut::expect(ut::approx(p(x), expected, 1e-12));
  };

  ut::test("callable coefficients compose into multivariate Horner") = [] {
    auto const q = poly::polynomial_t{1.0, 2.0, 3.0};
    auto const p = poly::polynomial_t{q, q, q};
    auto const y = -1.1;
    auto const x = 0.75;
    auto const expected = q(x) * (y * y + y + 1.0);
    ut::expect(ut::approx(p(y, x), expected, 1e-12));
  };

  ut::test("Horner retains modular integer semantics") = [] {
    auto const p = poly::polynomial_t<std::uint32_t, 4>{
        0xFFFFFFF0u, 0xABCDEF01u, 0x12345678u, 0x9ABCDEF0u, 0x1u};
    auto const x = 0xFFFFFFFEu;
    auto const expected = static_cast<std::uint32_t>(
        (((p[0] * x + p[1]) * x + p[2]) * x + p[3]) * x + p[4]);
    ut::expect(p(x) == expected);
  };
}
