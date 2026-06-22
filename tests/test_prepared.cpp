#include <array>
#include <boost/ut.hpp>
#include <polygnition/prepared.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

template <typename P>
concept can_borrow_temporary = requires { poly::borrow(poly::horner, P{}); };

template <typename P>
concept can_profile_temporary = requires {
  poly::on<polygnition::target::profile::generic>(P{});
};

using runtime_quadratic = poly::polynomial_t<double, 2>;
static_assert(!can_borrow_temporary<runtime_quadratic>);
static_assert(!can_profile_temporary<runtime_quadratic>);

constexpr auto static_quadratic = poly::literal<1.0, -2.0, 3.0>();
constexpr auto static_evaluator = poly::prepare(poly::automatic,
                                                 static_quadratic);
static_assert(sizeof(static_evaluator) == 1);
static_assert(static_evaluator(0.25) == static_quadratic(0.25));

int main() {
  ut::test("ownership_is_explicit") = [] {
    auto p = runtime_quadratic{1.0, -2.0, 3.0};
    auto const owned = poly::prepare(poly::horner, p);
    auto const borrowed = poly::borrow(poly::horner, p);
    auto const before = p(0.25);

    static_assert(sizeof(owned) == sizeof(p));
    static_assert(sizeof(borrowed) == sizeof(decltype(&p)));

    p.coeff<0>() = 7.0;
    ut::expect(owned(0.25) == before);
    ut::expect(borrowed(0.25) == p(0.25));
  };

  ut::test("motzkin_is_preprocessed_once") = [] {
    auto p = poly::polynomial_t{2.0, -5.0, 7.0, -11.0, 13.0};
    auto const evaluator = poly::prepare(poly::motzkin, p);
    auto const expected = poly::evaluate(poly::horner, p, 0.375);

    static_assert(std::same_as<
                  decltype(evaluator),
                  poly::prepared_t<poly::algorithm::motzkin_t,
                                   poly::motzkin_preprocessed_t<double>> const>);

    p = {};
    ut::expect(ut::approx(evaluator(0.375), expected, 1e-12));
  };

  ut::test("evaluate_into_is_only_a_name") = [] {
    constexpr auto p = poly::literal<1.0, -3.0, 5.0, -7.0, 11.0>();
    constexpr auto evaluator = poly::prepare(poly::automatic, p);
    auto const xs = std::array{-0.5, 0.0, 0.5, 1.0};
    auto direct = std::array<double, xs.size()>{};
    auto prepared = std::array<double, xs.size()>{};

    (void)poly::evaluate_into(poly::automatic, p, xs, direct);
    (void)poly::evaluate_into(evaluator, xs, prepared);

    ut::expect(direct == prepared);
  };
}
