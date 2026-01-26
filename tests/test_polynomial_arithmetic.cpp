#include <boost/ut.hpp>
#include <complex>
#include <polygnition/arithmetic.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

int main() {
  ut::test("semantic degree distinguishes stored and actual degree") = [] {
    auto const p = poly::polynomial_t{0.0, 0.0, 2.0, 3.0};
    auto const q = poly::polynomial_t{2.0, 3.0};
    ut::expect(poly::effective_degree(p) == 1);
    ut::expect(poly::same_polynomial(p, q));
    ut::expect(p != q);
    ut::expect(poly::effective_degree(poly::polynomial_t{0.0, 0.0}) == -1);
  };

  ut::test("ring operations align coefficients by degree") = [] {
    auto const f = poly::polynomial_t{1.0, 2.0, 3.0};
    auto const g = poly::polynomial_t{2.0, 4.0};

    ut::expect(f + g == poly::polynomial_t{1.0, 4.0, 7.0});
    ut::expect(f - g == poly::polynomial_t{1.0, 0.0, -1.0});
    ut::expect(f * g == poly::polynomial_t{2.0, 8.0, 14.0, 12.0});
    ut::expect(-g == poly::polynomial_t{-2.0, -4.0});
  };

  ut::test("scalars lift to constant polynomials") = [] {
    auto const f = poly::polynomial_t{1.0, 2.0, 3.0};
    ut::expect(f + 5.0 == poly::polynomial_t{1.0, 2.0, 8.0});
    ut::expect(2.0 * f == poly::polynomial_t{2.0, 4.0, 6.0});
  };

  ut::test("literal arithmetic closes at compile time") = [] {
    constexpr auto f = poly::literal<1.0, 2.0, 3.0>();
    constexpr auto g = poly::literal<2.0, 4.0>();
    constexpr auto sum = f + g;
    constexpr auto product = f * g;

    static_assert(std::is_empty_v<decltype(sum)>);
    ut::expect(sum == poly::literal<1.0, 4.0, 7.0>());
    ut::expect(product == poly::literal<2.0, 8.0, 14.0, 12.0>());
  };

  ut::test("formal differentiation preserves coefficient domains") = [] {
    auto const f = poly::polynomial_t{3.0, 2.0, 1.0};
    ut::expect(poly::derivative(f) == poly::polynomial_t{6.0, 2.0});
    ut::expect(poly::derivative(poly::literal<3.0, 2.0, 1.0>()) ==
               poly::literal<6.0, 2.0>());
  };

  ut::test("field division returns quotient and remainder") = [] {
    auto const dividend = poly::polynomial_t{1.0, 0.0, -1.0};
    auto const divisor = poly::polynomial_t{1.0, -1.0};
    auto const [quotient, remainder] = poly::divide(dividend, divisor);

    ut::expect(quotient == poly::polynomial_t{1.0, 1.0});
    ut::expect(remainder == poly::polynomial_t{0.0});
  };

  ut::test("complex floating coefficients form a field") = [] {
    using z = std::complex<double>;
    auto const p = poly::polynomial_t{z{2.0, 2.0}, z{4.0, -2.0}};
    auto const q = p / z{2.0, 0.0};
    ut::expect(q == poly::polynomial_t{z{1.0, 1.0}, z{2.0, -1.0}});
  };
}
