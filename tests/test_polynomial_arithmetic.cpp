#include <boost/ut.hpp>
#include <complex>
#include <cstdint>
#include <polygnition/arithmetic.hpp>
#include <type_traits>

namespace poly = polygnition::polynomial;
namespace ut = boost::ut;

template <typename P, typename S>
concept scalar_divisible = requires(P p, S s) { p / s; };

template <typename U, typename F>
concept polynomial_divisible = requires(U u, F f) { poly::divide(u, f); };

constexpr auto compose_static_quartic() {
  constexpr auto x = poly::literal<1.0, 0.0>();
  return (((poly::literal<2.0>() * x + poly::literal<-5.0>()) * x +
           poly::literal<7.0>()) *
              x +
          poly::literal<-11.0>()) *
             x +
         poly::literal<13.0>();
}

template <typename T> auto expect_effective_degree_for(T nonzero) -> void {
  auto const middle = poly::polynomial_t<T, 4>{T{}, T{}, nonzero, T{}, T{}};
  auto const leading = poly::polynomial_t<T, 4>{nonzero, T{}, T{}, T{}, T{}};
  auto const constant = poly::polynomial_t<T, 4>{T{}, T{}, T{}, T{}, nonzero};
  auto const zero = poly::polynomial_t<T, 4>{};

  ut::expect(poly::effective_degree(middle) == 2);
  ut::expect(poly::effective_degree(leading) == 4);
  ut::expect(poly::effective_degree(constant) == 0);
  ut::expect(poly::effective_degree(zero) == -1);
}

template <typename T>
auto expect_effective_degree_for_polynomial_coefficients(T nonzero) -> void {
  using coeff_poly_t = poly::polynomial_t<T, 2>;
  auto const zero_coeff = coeff_poly_t{};
  auto const constant_coeff = coeff_poly_t{T{}, T{}, nonzero};
  auto const linear_coeff = coeff_poly_t{T{}, nonzero, T{}};

  auto const middle = poly::polynomial_t<coeff_poly_t, 4>{
      zero_coeff, zero_coeff, constant_coeff, zero_coeff, zero_coeff};
  auto const leading = poly::polynomial_t<coeff_poly_t, 4>{
      constant_coeff, zero_coeff, zero_coeff, zero_coeff, zero_coeff};
  auto const trailing = poly::polynomial_t<coeff_poly_t, 4>{
      zero_coeff, zero_coeff, zero_coeff, zero_coeff, linear_coeff};
  auto const zero = poly::polynomial_t<coeff_poly_t, 4>{};

  ut::expect(poly::effective_degree(middle) == 2);
  ut::expect(poly::effective_degree(leading) == 4);
  ut::expect(poly::effective_degree(trailing) == 0);
  ut::expect(poly::effective_degree(zero) == -1);
}

template <typename T>
auto expect_effective_degree_for_nested_polynomial_coefficients(T nonzero)
    -> void {
  using leaf_poly_t = poly::polynomial_t<T, 2>;
  using coeff_poly_t = poly::polynomial_t<leaf_poly_t, 2>;

  auto const zero_leaf = leaf_poly_t{};
  auto const constant_leaf = leaf_poly_t{T{}, T{}, nonzero};
  auto const linear_leaf = leaf_poly_t{T{}, nonzero, T{}};
  auto const zero_coeff = coeff_poly_t{};
  auto const constant_coeff = coeff_poly_t{zero_leaf, zero_leaf, constant_leaf};
  auto const linear_coeff = coeff_poly_t{zero_leaf, linear_leaf, zero_leaf};

  auto const middle = poly::polynomial_t<coeff_poly_t, 4>{
      zero_coeff, zero_coeff, constant_coeff, zero_coeff, zero_coeff};
  auto const leading = poly::polynomial_t<coeff_poly_t, 4>{
      constant_coeff, zero_coeff, zero_coeff, zero_coeff, zero_coeff};
  auto const trailing = poly::polynomial_t<coeff_poly_t, 4>{
      zero_coeff, zero_coeff, zero_coeff, zero_coeff, linear_coeff};
  auto const zero = poly::polynomial_t<coeff_poly_t, 4>{};

  ut::expect(poly::effective_degree(middle) == 2);
  ut::expect(poly::effective_degree(leading) == 4);
  ut::expect(poly::effective_degree(trailing) == 0);
  ut::expect(poly::effective_degree(zero) == -1);
}

int main() {
  ut::test("addition_same_degree") = [] {
    // given two degree-2 polynomials
    auto const f = poly::polynomial_t{3., 2., 1.};
    auto const g = poly::polynomial_t{1., 0., 4.};

    // when they are added
    auto const h = f + g;

    // then coefficients are summed by degree
    ut::expect(h == poly::polynomial_t{4., 2., 5.});
  };

  ut::test("addition_different_degree") = [] {
    // given a degree-2 polynomial and a constant polynomial
    auto const f = poly::polynomial_t{1., 2., 3.};
    auto const c = poly::polynomial_t{4.};

    // when they are added
    auto const h = f + c;

    // then the result keeps the higher degree and adjusts the constant term
    static_assert(
        std::is_same_v<decltype(h), poly::polynomial_t<double, 2> const>);
    ut::expect(h == poly::polynomial_t{1., 2., 7.});
  };

  ut::test("negation_and_subtraction") = [] {
    // given a degree-2 polynomial
    auto const f = poly::polynomial_t{2., -1., 5.};

    // when it is negated and added
    auto const zero = f + (-f);

    // then the result is the zero polynomial
    ut::expect(zero == poly::polynomial_t{0., 0., 0.});
  };

  ut::test("multiplication_degree_adds") = [] {
    // given degree-1 and degree-2 polynomials
    auto const f = poly::polynomial_t{2., 3.};     // 2x + 3
    auto const g = poly::polynomial_t{1., 0., 4.}; // x^2 + 4

    // when they are multiplied
    auto const h = f * g;

    // then the result has degree 3 with correct coefficients
    static_assert(
        std::is_same_v<decltype(h), poly::polynomial_t<double, 3> const>);
    ut::expect(h == poly::polynomial_t{2., 3., 8., 12.});
  };

  ut::test("static_coefficients_survive_algebraic_composition") = [] {
    using composed_t = decltype(compose_static_quartic());
    using expected_t =
        decltype(poly::literal<2.0, -5.0, 7.0, -11.0, 13.0>());

    static_assert(std::is_same_v<composed_t, expected_t>);
    static_assert(std::is_empty_v<composed_t>);
    static_assert(composed_t::value ==
                  poly::polynomial_t{2.0, -5.0, 7.0, -11.0, 13.0});
    static_assert(std::is_same_v<
                  poly::detail::selected_algorithm_t<composed_t, double>,
                  poly::algorithm::motzkin_t>);
    static_assert(compose_static_quartic() ==
                  poly::literal<2.0, -5.0, 7.0, -11.0, 13.0>());
  };

  ut::test("static_arithmetic_preserves_storage_degree_until_trimmed") = [] {
    auto constexpr sum = poly::literal<1.0, 2.0>() + poly::literal<-1.0, 3.0>();
    auto constexpr zero = poly::literal<1.0, 2.0>() - poly::literal<1.0, 2.0>();

    static_assert(std::is_same_v<decltype(sum),
                                 decltype(poly::literal<0.0, 5.0>()) const>);
    static_assert(std::is_same_v<decltype(zero),
                                 decltype(poly::literal<0.0, 0.0>()) const>);
    static_assert(std::is_same_v<decltype(poly::trim(sum)),
                                 decltype(poly::literal<5.0>())>);
    static_assert(std::is_same_v<decltype(poly::trim(zero)),
                                 decltype(poly::literal<0.0>())>);
    static_assert(poly::trim(sum) == poly::literal<5.0>());
    ut::expect(zero == poly::polynomial_t{0.0, 0.0});
  };

  ut::test("scalar_lifts_to_constant_polynomial") = [] {
    // given a degree-2 polynomial and a scalar
    auto const f = poly::polynomial_t{1., 2., 3.};

    // when adding and multiplying by a scalar
    auto const sum = f + 5.0;
    auto const prod = 2.0 * f;

    // then the scalar acts as a constant polynomial
    ut::expect(sum == poly::polynomial_t{1., 2., 8.});
    ut::expect(prod == poly::polynomial_t{2., 4., 6.});
  };

  ut::test("static_scalar_closure_uses_type_level_constant") = [] {
    auto constexpr sum = poly::literal<1.0, 2.0>() + poly::constant<3.0>;
    auto constexpr product = poly::constant<2.0> * poly::literal<3.0, 4.0>();

    static_assert(std::is_same_v<decltype(sum),
                                 decltype(poly::literal<1.0, 5.0>()) const>);
    static_assert(
        std::is_same_v<decltype(product),
                       decltype(poly::literal<6.0, 8.0>()) const>);
    static_assert(sum == poly::literal<1.0, 5.0>());
    static_assert(product == poly::literal<6.0, 8.0>());
  };

  ut::test("degree_zero_zero_polynomial_adds_without_normalization") = [] {
    auto const zero = poly::polynomial_t{0.0};
    auto const f = poly::polynomial_t{1.0, 2.0, 3.0};
    auto const h = zero + f;
    static_assert(
        std::is_same_v<decltype(h), poly::polynomial_t<double, 2> const>);
    ut::expect(h == f);
  };

  ut::test("effective_degree_ignores_leading_zero_storage") = [] {
    auto const p = poly::polynomial_t<int, 4>{0, 0, 7, 0, -3};
    auto const zero = poly::polynomial_t{0.0};

    ut::expect(poly::effective_degree(p) == 2);
    ut::expect(poly::effective_degree(zero) == -1);
    static_assert(poly::effective_degree(poly::literal<0.0>()) == -1);
    static_assert(poly::effective_degree(poly::literal<2.0, 0.0, 1.0>()) == 2);
    using static_poly_t = decltype(poly::literal<2.0, 0.0, 1.0>());
    static_assert(poly::effective_degree(static_poly_t{}) == 2);
  };

  ut::test("effective_degree_supports_common_coefficient_types") = [] {
    expect_effective_degree_for(1.0F);
    expect_effective_degree_for(1.0);
    expect_effective_degree_for(1);
    expect_effective_degree_for(std::int64_t{1});
    expect_effective_degree_for(1U);
    expect_effective_degree_for(std::uint32_t{1});
    expect_effective_degree_for(std::complex<float>{1.0F, -0.5F});
    expect_effective_degree_for(std::complex<double>{1.0, -0.5});
    expect_effective_degree_for(poly::polynomial_t{0, 0, 1});
    expect_effective_degree_for(poly::polynomial_t{
        std::complex<double>{0.0, 0.0}, std::complex<double>{1.0, -0.5}});

    using coeff_poly_t = poly::polynomial_t<int, 2>;
    static_assert(poly::effective_degree(poly::polynomial_t<coeff_poly_t, 4>{
                      coeff_poly_t{}, coeff_poly_t{}, coeff_poly_t{0, 0, 1},
                      coeff_poly_t{}, coeff_poly_t{}}) == 2);
  };

  ut::test("effective_degree_handles_polynomial_coefficients_with_zero_leading_"
           "terms") = [] {
    expect_effective_degree_for_polynomial_coefficients(1.0F);
    expect_effective_degree_for_polynomial_coefficients(1.0);
    expect_effective_degree_for_polynomial_coefficients(1);
    expect_effective_degree_for_polynomial_coefficients(std::int64_t{1});
    expect_effective_degree_for_polynomial_coefficients(1U);
    expect_effective_degree_for_polynomial_coefficients(std::uint32_t{1});
    expect_effective_degree_for_polynomial_coefficients(
        std::complex<float>{1.0F, -0.5F});
    expect_effective_degree_for_polynomial_coefficients(
        std::complex<double>{1.0, -0.5});

    using coeff_poly_t = poly::polynomial_t<int, 2>;
    static_assert(poly::effective_degree(poly::polynomial_t<coeff_poly_t, 3>{
                      coeff_poly_t{}, coeff_poly_t{0, 0, 2}, coeff_poly_t{},
                      coeff_poly_t{}}) == 2);
  };

  ut::test(
      "effective_degree_handles_polynomial_of_polynomial_coefficients") = [] {
    expect_effective_degree_for_nested_polynomial_coefficients(1.0F);
    expect_effective_degree_for_nested_polynomial_coefficients(1.0);
    expect_effective_degree_for_nested_polynomial_coefficients(1);
    expect_effective_degree_for_nested_polynomial_coefficients(std::int64_t{1});
    expect_effective_degree_for_nested_polynomial_coefficients(1U);
    expect_effective_degree_for_nested_polynomial_coefficients(
        std::uint32_t{1});
    expect_effective_degree_for_nested_polynomial_coefficients(
        std::complex<float>{1.0F, -0.5F});
    expect_effective_degree_for_nested_polynomial_coefficients(
        std::complex<double>{1.0, -0.5});

    using leaf_poly_t = poly::polynomial_t<int, 2>;
    using coeff_poly_t = poly::polynomial_t<leaf_poly_t, 2>;
    static_assert(
        poly::effective_degree(poly::polynomial_t<coeff_poly_t, 3>{
            coeff_poly_t{},
            coeff_poly_t{leaf_poly_t{}, leaf_poly_t{}, leaf_poly_t{0, 0, 2}},
            coeff_poly_t{}, coeff_poly_t{}}) == 2);
  };

  ut::test("equality_preserves_static_degree") = [] {
    auto const zero = poly::polynomial_t{0.0};
    auto const stored_zero = poly::polynomial_t{0.0, 0.0, 0.0};
    auto const p = poly::polynomial_t{0.0, 2.0, -1.0};
    auto const q = poly::polynomial_t{2.0, -1.0};

    ut::expect(stored_zero != zero);
    ut::expect(p != q);
    ut::expect(poly::same_polynomial(stored_zero, zero));
    ut::expect(poly::same_polynomial(p, q));
    ut::expect(!poly::same_polynomial(p, poly::polynomial_t{2.0, 0.0}));
  };

  ut::test("formal_derivative") = [] {
    auto constexpr p = poly::literal<4.0, -3.0, 2.0, -1.0>();
    auto constexpr dp = poly::derivative(p);
    auto constexpr dc = poly::derivative(poly::literal<7.0>());

    static_assert(std::is_same_v<decltype(dp),
                                 decltype(poly::literal<12.0, -6.0, 2.0>())
                                     const>);
    static_assert(dp == poly::polynomial_t{12.0, -6.0, 2.0});
    static_assert(dc == poly::polynomial_t{0.0});
    static_assert(poly::effective_degree(dc) == -1);
    static_assert(std::is_same_v<decltype(poly::derivative(
                                     compose_static_quartic())),
                                 decltype(poly::literal<8.0, -15.0, 14.0,
                                                        -11.0>())>);
  };

  ut::test("scalar_division_preserves_static_degree") = [] {
    auto const p = poly::polynomial_t{6.0, -3.0, 12.0};
    auto const q = p / 3;
    auto constexpr static_q =
        poly::literal<6.0, -3.0, 12.0>() / poly::constant<3.0>;

    static_assert(
        std::is_same_v<decltype(q), poly::polynomial_t<double, 2> const>);
    static_assert(std::is_same_v<decltype(static_q),
                                 decltype(poly::literal<2.0, -1.0, 4.0>())
                                     const>);
    static_assert(!scalar_divisible<poly::polynomial_t<int, 1>, int>);
    ut::expect(q == poly::polynomial_t{2.0, -1.0, 4.0});
    static_assert(static_q == poly::literal<2.0, -1.0, 4.0>());
  };

  ut::test("complex_scalar_division") = [] {
    auto const p = poly::polynomial_t{std::complex<double>{2.0, 4.0},
                                      std::complex<double>{6.0, -2.0}};
    auto const q = p / 2;

    ut::expect(q == poly::polynomial_t{std::complex<double>{1.0, 2.0},
                                       std::complex<double>{3.0, -1.0}});
  };

  ut::test("polynomial_division") = [] {
    auto const u = poly::polynomial_t{2.0, 5.0, 3.0, 4.0};
    auto const f = poly::polynomial_t{1.0, 2.0, 1.0};
    auto const [q, r] = poly::divide(u, f);

    static_assert(poly::field_coefficient<double>);
    static_assert(poly::field_coefficient<std::complex<double>>);
    static_assert(!poly::field_coefficient<int>);
    static_assert(!polynomial_divisible<poly::polynomial_t<int, 1>,
                                        poly::polynomial_t<int, 0>>);
    ut::expect(q == poly::polynomial_t{2.0, 1.0});
    ut::expect(r == poly::polynomial_t{-1.0, 3.0});
    ut::expect(poly::same_polynomial((q * f) + r, u));
    ut::expect(poly::effective_degree(r) < poly::effective_degree(f));
  };

  ut::test("polynomial_division_by_constant") = [] {
    auto constexpr u = poly::polynomial_t{2.0, 5.0, 3.0, 4.0};
    auto constexpr f = poly::polynomial_t{2.0};
    auto constexpr division = poly::divide(u, f);
    auto constexpr expected = std::pair{poly::polynomial_t{1.0, 2.5, 1.5, 2.0},
                                        poly::polynomial_t{0.0}};

    static_assert(division == expected);
  };

  ut::test("polynomial_division_by_larger_degree") = [] {
    auto const u = poly::polynomial_t{2.0, 3.0};
    auto const f = poly::polynomial_t{1.0, -1.0, 4.0};
    auto const [q, r] = poly::divide(u, f);

    ut::expect(q == poly::polynomial_t{0.0});
    ut::expect(r == u);
    ut::expect(poly::same_polynomial((q * f) + r, u));
  };
}
