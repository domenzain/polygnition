#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <complex>
#include <polygnition/arithmetic.hpp>
#include <polygnition/poly.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

template <typename P>
concept coefficient_assignable = requires(P p) {
  p[std::size_t{}] = typename P::value_type{};
};

// make sure it remains lightweight
static_assert(std::is_aggregate_v<poly::polynomial_t<double, 0>>);
static_assert(coefficient_assignable<poly::polynomial_t<double, 2>>);
static_assert(!coefficient_assignable<decltype(poly::literal<1.0, 2.0>())>);
static_assert(std::is_empty_v<decltype(poly::literal<1.0, 2.0, 3.0>())>);
static_assert(sizeof(decltype(poly::literal<1.0, 2.0, 3.0>())) == 1);
static_assert(poly::literal<1.0, 2.0, 3.0>()[1] == 2.0);
static_assert(poly::literal<1.0, 2.0, 3.0>().coeff<2>() == 1.0);
static_assert(poly::literal<1.0, 2.0, 3.0>().coeff<0>() == 3.0);
static_assert(poly::literal<1.0, 2.0, 3.0>().coeff(1) == 2.0);
static_assert(poly::literal<1.0, 2.0, 3.0>().coeff(3) == 0.0);
static_assert(poly::literal<1.0, 2.0, 3.0>().coeff(99) == 0.0);
static_assert(std::is_same_v<poly::detail::selected_algorithm_t<
                                 poly::polynomial_t<double, 2>, double>,
                             poly::algorithm::horner_t>);
static_assert(std::is_same_v<
              poly::detail::selected_algorithm_t<
                  decltype(poly::literal<1.0, 2.0, 3.0, 4.0, 5.0>()), double>,
              poly::algorithm::motzkin_t>);
static_assert(std::is_same_v<poly::detail::selected_algorithm_t<
                                 poly::polynomial_t<double, 2>, double, double>,
                             poly::algorithm::multivariate_horner_t>);
static_assert(
    std::is_same_v<poly::detail::selected_algorithm_t<
                       poly::polynomial_t<double, 2>, std::complex<double>>,
                   poly::algorithm::knuth_t>);
static_assert(std::is_same_v<poly::detail::selected_algorithm_t<
                                 poly::polynomial_t<double, 12>, double>,
                             poly::algorithm::horner_t>);
static_assert(
    std::is_same_v<
        poly::detail::selected_algorithm_t<poly::polynomial_t<int, 64>, int>,
        poly::algorithm::horner_t>);
static_assert(std::is_same_v<poly::detail::selected_algorithm_t<
                                 poly::polynomial_t<std::complex<double>, 5>,
                                 std::complex<double>>,
                             poly::algorithm::horner_t>);
static_assert(poly::literal<0.0>().degree == 0);
static_assert(poly::degree(poly::literal<1.0, 2.0, 3.0>()) == 2);

struct sentinel_algorithm {};

constexpr auto tag_invoke(poly::evaluate_t, sentinel_algorithm,
                          poly::polynomial_t<int, 1> const &, int) {
  return 42;
}

int main() {
  ut::test("construction") = [] {
    auto const p = poly::polynomial_t{1., 2, 3};
    ut::expect(p[0] == 1.);
  };

  ut::test("bindings") = [] {
    auto const f = poly::polynomial_t{1., 2, 3};
    auto const [a, b, c] = f;
    ut::expect(a == 1.);
    ut::expect(b == 2.);
    ut::expect(c == 3.);
  };

  ut::test("evaluation") = [] {
    auto const f = poly::polynomial_t{3., 0, 0};
    auto const [a, b, c] = f;
    auto const x = 2;
    auto const y = f(x);
    ut::expect(y == a * (x * x) + b * x + c);
    ut::expect(f.coeff<2>() == a);
    ut::expect(f.coeff(1) == b);
    ut::expect(f.coeff(0) == c);
    ut::expect(f.coeff(-1) == 0.0);
  };

  ut::test("named_profile_view") = [] {
    constexpr auto f = poly::literal<1.0, -2.0, 3.0>();
    auto const y = poly::on<polygnition::target::profile::generic>(f)(0.25);
    ut::expect(y == poly::evaluate(
                        poly::tuned<polygnition::target::profile::generic>, f,
                        0.25));
  };

  ut::test("multivariate_evaluation") = [] {
    auto const f = poly::polynomial_t{2., 0, 0};
    using f_t = decltype(f);
    auto const g = poly::polynomial_t{f, f_t{}};
    auto const x = 2.;
    auto const y = 3.;
    auto const v = g(y, x);
    ut::expect(v == y * 2. * (x * x));
  };

  ut::test("default_construction") = [] {
    // given an arbitrary default-constructed degree-N polynomial
    constexpr auto N = 3;
    auto const f = poly::polynomial_t<double, N>{};

    // when it evaluates to 0 at N+1 points, then it is identically zero
    auto const xs = std::array{0.0, 1.0, 2.0, 3.0};
    ut::expect(std::ranges::all_of(xs, [&](auto x) { return f(x) == 0.0; }));
  };

  ut::test("compile_time_coeffs") = [] {
    using expected_t = poly::static_poly<
        poly::polynomial_t<double, 4>{1.0, 2.0, 3.0, 4.0, 5.0}>;

    constexpr auto lit = poly::literal<1.0, 2.0, 3.0, 4.0, 5.0>();
    static_assert(
        std::is_same_v<std::remove_cvref_t<decltype(lit)>, expected_t>);
    static_assert(expected_t::value ==
                  poly::polynomial_t{1.0, 2.0, 3.0, 4.0, 5.0});

    auto const eval_lit = lit(2.0);
    auto const eval_value = poly::polynomial_t{1.0, 2.0, 3.0, 4.0, 5.0}(2.0);
    auto const [a, b, c, d, e] = lit;

    ut::expect(eval_lit == eval_value);
    ut::expect(a == 1.0);
    ut::expect(b == 2.0);
    ut::expect(c == 3.0);
    ut::expect(d == 4.0);
    ut::expect(e == 5.0);
  };

  ut::test("mixed_static_coefficients") = [] {
    constexpr auto p = poly::literal<1, 2.0>();
    ut::expect(p(2.0) == 4.0);
  };

  ut::test("static_polynomial_value_is_semantic") = [] {
    using p_t = decltype(poly::literal<1.0, 2.0, 3.0>());
    auto const p = p_t{};
    auto const shifted = p + poly::literal<1.0>();
    ut::expect(p(2.0) == 11.0);
    ut::expect(p == poly::literal<1.0, 2.0, 3.0>());
    ut::expect(shifted == poly::polynomial_t{1.0, 2.0, 4.0});
  };

  ut::test("degree_zero_zero_polynomial_is_valid_static_pack") = [] {
    constexpr auto zero = poly::literal<0.0>();
    auto const f = poly::polynomial_t{2.0, 3.0, 4.0};
    auto const sum = f + zero;
    ut::expect(sum == f);
    ut::expect(zero(123.0) == 0.0);
  };

  ut::test("explicit_tag_invoke_customization") = [] {
    auto const p = poly::polynomial_t{1, 2};
    ut::expect(poly::evaluate(sentinel_algorithm{}, p, 9) == 42);
  };
}
