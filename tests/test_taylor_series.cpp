#include <boost/ut.hpp>
#include <polygnition/taylor.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

static_assert(std::is_aggregate_v<poly::taylor_series<double, 0>>);

int main() {
  ut::test("derivatives") = [] {
    auto const f = poly::taylor_series<double, 2>{3.0, 2.0, 1.0}; // d2, d1, value
    ut::expect(f.derivative<0>() == 1.0);
    ut::expect(f.derivative<1>() == 2.0);
    ut::expect(f.derivative<2>() == 6.0);

    auto const [v0, v1, v2] = f.derivatives();
    ut::expect(v0 == 1.0);
    ut::expect(v1 == 2.0);
    ut::expect(v2 == 6.0);
  };

  ut::test("multiply_order2") = [] {
    auto const a = poly::taylor_series<double, 2>{3.0, 2.0, 1.0};
    auto const b = poly::taylor_series<double, 2>{5.0, 4.0, 2.0};
    auto const c = a * b;
    ut::expect(c[2] == 2.0);  // value
    ut::expect(c[1] == 8.0);  // d1 = a0*b1 + a1*b0
    ut::expect(c[0] == 19.0); // d2 = a0*b2 + a1*b1 + a2*b0
  };

  ut::test("multiply_order3") = [] {
    auto const a = poly::taylor_series<int, 3>{1, 2, 3, 4}; // e^3..value
    auto const b = poly::taylor_series<int, 3>{2, 1, 0, 1};
    auto const c = a * b;
    ut::expect(c[3] == 4);  // value
    ut::expect(c[2] == 3);  // d1
    ut::expect(c[1] == 6);  // d2
    ut::expect(c[0] == 12); // d3
  };

  ut::test("scalar_operations_preserve_taylor_series") = [] {
    auto const x = poly::dual<double>{1.0, 2.0};
    auto const y = (3 * x) + 4;
    auto const z = 4 - x;

    static_assert(std::is_same_v<decltype(y), poly::dual<double> const>);
    static_assert(std::is_same_v<decltype(z), poly::dual<double> const>);
    ut::expect(y[1] == 10.0);
    ut::expect(y[0] == 3.0);
    ut::expect(z[1] == 2.0);
    ut::expect(z[0] == -1.0);
  };

  ut::test("series_expansion") = [] {
    auto constexpr cubic = poly::literal<1.0, 0.0, 0.0, 0.0>();
    auto constexpr expansion = poly::series_expansion<2>(cubic, 3.0);
    auto constexpr derivatives = expansion.derivatives();

    static_assert(std::is_same_v<decltype(expansion),
                                 poly::taylor_series<double, 2> const>);
    static_assert(std::get<0>(derivatives) == 27.0);
    static_assert(std::get<1>(derivatives) == 27.0);
    static_assert(std::get<2>(derivatives) == 18.0);
  };

  ut::test("series_expansion_lifts_integral_coefficients") = [] {
    auto constexpr p = poly::literal<1, 2, 3>();
    auto constexpr expansion = poly::series_expansion<1>(p, 2.0);
    auto constexpr derivatives = expansion.derivatives();

    static_assert(std::is_same_v<decltype(expansion),
                                 poly::taylor_series<double, 1> const>);
    static_assert(std::get<0>(derivatives) == 11.0);
    static_assert(std::get<1>(derivatives) == 6.0);
  };
}
