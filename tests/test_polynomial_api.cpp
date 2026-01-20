#include <boost/ut.hpp>
#include <polygnition/poly.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

static_assert(std::is_aggregate_v<poly::polynomial_t<double, 0>>);

int main() {
  ut::test("construction and coefficient order") = [] {
    auto const p = poly::polynomial_t{1.0, 2.0, 3.0};
    auto const [a, b, c] = p;

    ut::expect(a == 1.0);
    ut::expect(b == 2.0);
    ut::expect(c == 3.0);
    ut::expect(p.coeff<2>() == 1.0);
    ut::expect(p.coeff(1) == 2.0);
    ut::expect(p.coeff(-1) == 0.0);
    ut::expect(poly::degree(p) == 2);
  };

  ut::test("default construction is the zero polynomial") = [] {
    auto const p = poly::polynomial_t<double, 3>{};
    ut::expect(p == poly::polynomial_t{0.0, 0.0, 0.0, 0.0});
  };
}
