#include <boost/ut.hpp>
#include <polygnition/poly.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

static_assert(std::is_aggregate_v<poly::polynomial_t<double, 0>>);
static_assert(std::is_empty_v<decltype(poly::literal<1.0, 2.0, 3.0>())>);
static_assert(sizeof(decltype(poly::literal<1.0, 2.0, 3.0>())) == 1);
static_assert(poly::literal<1.0, 2.0, 3.0>().coeff<2>() == 1.0);
static_assert(poly::literal<1.0, 2.0, 3.0>().coeff(1) == 2.0);

struct sentinel_algorithm {};
constexpr auto tag_invoke(poly::evaluate_t, sentinel_algorithm,
                          poly::polynomial_t<int, 1> const &, int) {
  return 42;
}

int main() {
  ut::test("the CPO admits an external strategy") = [] {
    auto const p = poly::polynomial_t{1, 2};
    ut::expect(poly::evaluate(sentinel_algorithm{}, p, 9) == 42);
  };

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

  ut::test("literal coefficients are structural and storage-free") = [] {
    constexpr auto p = poly::literal<1.0, 2.0, 3.0>();
    auto const [a, b, c] = p;
    ut::expect(p(2.0) == 11.0);
    ut::expect(a == 1.0);
    ut::expect(b == 2.0);
    ut::expect(c == 3.0);
    ut::expect(poly::degree(p) == 2);
  };

  ut::test("mixed literal coefficients share a common value type") = [] {
    constexpr auto p = poly::literal<1, 2.0>();
    ut::expect(p(2.0) == 4.0);
  };

  ut::test("default construction is the zero polynomial") = [] {
    auto const p = poly::polynomial_t<double, 3>{};
    ut::expect(p == poly::polynomial_t{0.0, 0.0, 0.0, 0.0});
  };
}
