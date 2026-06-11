#include <boost/ut.hpp>
#include <array>
#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <polygnition/dev/helpers.hpp>
#include <polygnition/poly.hpp>
#include <polygnition/taylor.hpp>

namespace poly = polygnition::polynomial;
namespace dev = polygnition::dev;
namespace ut = boost::ut;

namespace polygnition::test {
struct sentinel_algorithm {};

template <typename P, typename X>
[[nodiscard]] constexpr auto tag_invoke(polygnition::polynomial::evaluate_t,
                                        sentinel_algorithm, P const &p,
                                        X const &x) {
  return polygnition::polynomial::evaluate(polygnition::polynomial::horner, p,
                                           x) + 1.0;
}
} // namespace polygnition::test

int main() {
  ut::test("horner_int") = [] {
    // p(x) = 1·x³ + 2·x² + 3·x + 4
    // coefficients c₃..c₀ = {1,2,3,4}
    constexpr auto N = 3;
    auto const p = dev::iota_poly<int, N>(); // {1,2,3,4} ≡ c₃..c₀
    auto const x = 5;
    auto const x2 = x * x;  // x²
    auto const x3 = x2 * x; // x³
    auto const expected = 1 * x3 + 2 * x2 + 3 * x + 4;
    auto const got = poly::evaluate(poly::horner, p, x);
    ut::expect(got == expected);
  };

  ut::test("horner_double") = [] {
    // p(x) = 1·x⁵ + 2·x⁴ + 3·x³ + 4·x² + 5·x + 6
    // coefficients c₅..c₀ = {1,2,3,4,5,6}
    constexpr auto N = 5;
    auto const p = dev::iota_poly<double, N>(); // {1,2,3,4,5,6} ≡ c₅..c₀
    auto const x = 1.25;
    auto const x2 = x * x;  // x²
    auto const x3 = x2 * x; // x³
    auto const x4 = x3 * x; // x⁴
    auto const x5 = x4 * x; // x⁵
    auto const expected =
        1.0 * x5 + 2.0 * x4 + 3.0 * x3 + 4.0 * x2 + 5.0 * x + 6.0;
    auto const got = poly::evaluate(poly::horner, p, x);
    ut::expect(ut::approx(got, expected, 1e-12));
  };

  ut::test("dorn_int") = [] {
    constexpr auto N = 7;
    auto const p = dev::iota_poly<int, N>();
    auto const x = 3;
    auto const expected = poly::evaluate(poly::horner, p, x);
    auto const got = poly::evaluate(poly::dorn<3>, p, x);
    ut::expect(got == expected);
  };

  ut::test("dorn_double") = [] {
    constexpr auto N = 12;
    auto const p = dev::iota_poly<double, N>();
    auto const x = 0.75;
    auto const expected = poly::evaluate(poly::horner, p, x);
    auto const got = poly::evaluate(poly::dorn<4>, p, x);
    ut::expect(ut::approx(got, expected, 1e-12));
  };

  ut::test("dorn_k_larger_than_degree") = [] {
    constexpr auto N = 3;
    auto const p = dev::iota_poly<double, N>();
    auto const x = 1.125;
    auto const expected = poly::evaluate(poly::horner, p, x);
    auto const got = poly::evaluate(poly::dorn<8>, p, x);
    ut::expect(ut::approx(got, expected, 1e-12));
  };

  ut::test("motzkin_preprocessed_double") = [] {
    // p(x) = 1·x⁴ + 2·x³ + 3·x² + 4·x + 5
    constexpr auto N = 4;
    auto const p = dev::iota_poly<double, N>(); // {1,2,3,4,5} ≡ c₄..c₀
    auto const x = 1.25;
    auto const expected = poly::evaluate(poly::horner, p, x);
    auto const pre = poly::preprocess_motzkin(p);
    auto const got = poly::evaluate(poly::motzkin, pre, x);
    ut::expect(ut::approx(got, expected, 1e-12));
  };

  ut::test("motzkin_static_pack_uses_same_value") = [] {
    constexpr auto p = poly::literal<1.0, 2.0, 3.0, 4.0, 5.0>();
    auto const x = 1.25;
    auto const expected = poly::evaluate(poly::horner, p, x);
    auto const got = p(x);
    ut::expect(ut::approx(got, expected, 1e-12));
  };

  ut::test("motzkin_handles_non_monic_coefficients") = [] {
    constexpr auto p = poly::literal<2.0, -5.0, 7.0, -11.0, 13.0>();
    auto const xs = std::array{-2.0, -0.5, 0.0, 0.75, 3.0};
    for (auto const x : xs) {
      auto const expected = poly::evaluate(poly::horner, p, x);
      auto const pre = poly::preprocess_motzkin(p);
      auto const got = poly::evaluate(poly::motzkin, pre, x);
      ut::expect(ut::approx(got, expected, 1e-10));
    }
  };

  ut::test("dorn_uses_degree_residue_classes") = [] {
    constexpr auto p = poly::literal<2.0, -3.0, 5.0, -7.0, 11.0, -13.0,
                                     17.0, -19.0, 23.0, -29.0, 31.0,
                                     -37.0, 41.0>();
    auto const x = -0.625;
    auto const x2 = x * x;
    auto const x3 = x2 * x;
    auto const y = x2 * x2;
    auto const p0 = ((2.0 * y + 11.0) * y + 23.0) * y + 41.0;
    auto const p1 = ((-7.0 * y - 19.0) * y - 37.0) * x;
    auto const p2 = ((5.0 * y + 17.0) * y + 31.0) * x2;
    auto const p3 = ((-3.0 * y - 13.0) * y - 29.0) * x3;
    auto const got = poly::evaluate(poly::dorn<4>, p, x);
    ut::expect(ut::approx(got, p0 + p1 + p2 + p3, 1e-12));
    ut::expect(ut::approx(got, poly::evaluate(poly::horner, p, x), 1e-12));
  };

  ut::test("dorn_taylor_argument_matches_horner") = [] {
    auto constexpr p = poly::literal<4.0, -3.0, 2.0, -1.0>();
    auto constexpr x = poly::dual<double>{1.0, 2.0};
    auto constexpr got = poly::evaluate(poly::dorn<2>, p, x);
    auto constexpr expected = poly::evaluate(poly::horner, p, x);

    static_assert(got == expected);
    static_assert(got.derivative<0>() == 23.0);
    static_assert(got.derivative<1>() == 38.0);
  };

  ut::test("automatic_mid_degree_matches_horner_value") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0,
                                     -8.0, 9.0, -10.0, 11.0, -12.0, 13.0,
                                     -14.0, 15.0, -16.0, 17.0>();
    auto const x = -0.71875;
    auto const got = p(x);
    ut::expect(ut::approx(got, poly::evaluate(poly::horner, p, x), 1e-12));
  };

  ut::test("automatic_high_degree_matches_horner_value") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0,
                                     -8.0, 9.0, -10.0, 11.0, -12.0, 13.0,
                                     -14.0, 15.0, -16.0, 17.0, -18.0, 19.0>();
    auto const x = 0.8125;
    auto const got = p(x);
    ut::expect(ut::approx(got, poly::evaluate(poly::horner, p, x), 1e-12));
  };

  ut::test("knuth_iota_double_complex") = [] {
    // p(z) = 1·z³ + 2·z² + 3·z + 4
    // coefficients c₃..c₀ = {1,2,3,4}
    constexpr auto N = 3;
    auto const p = dev::iota_poly<double, N>(); // {1,2,3,4} ≡ c₃..c₀
    auto const z = std::complex<double>{1.2, -0.7};
    auto const z2 = z * z;  // z²
    auto const z3 = z2 * z; // z³
    auto const expected = 1.0 * z3 + 2.0 * z2 + 3.0 * z + 4.0;
    auto const got = poly::evaluate(poly::knuth, p, z);
    ut::expect(ut::approx(got.real(), expected.real(), 1e-12));
    ut::expect(ut::approx(got.imag(), expected.imag(), 1e-12));
  };

  ut::test("multivariate") = [] {
    // q(t) = 1·t² + 2·t + 3
    // g(y,t) = q(t)·y² + q(t)·y + q(t) = q(t)·(y² + y + 1)
    constexpr auto M = 2;
    auto const q = dev::iota_poly<double, M>(); // {1,2,3} ≡ c₂..c₀
    auto const g = poly::polynomial_t{q, q, q}; // degree-2 in y with coeffs q(t)
    auto const t = 0.75;
    auto const y = -1.1;
    auto const q_t = (1.0 * (t * t)) + (2.0 * t) + 3.0; // q(t) = 1·t² + 2·t + 3
    auto const expected = q_t * (y * y + y + 1.0);      // q(t)·(y² + y + 1)
    auto const got = poly::evaluate(poly::multivariate_horner, g, y, t);
    ut::expect(ut::approx(got, expected, 1e-12));
  };

  ut::test("estrin_static_pack_matches_horner") = [] {
    constexpr auto p = poly::literal<4.0, -3.0, 2.0, -1.0>();
    constexpr auto x = 2.0;

    static_assert(poly::evaluate(poly::estrin, p, x) ==
                  poly::evaluate(poly::horner, p, x));
  };

  ut::test("estrin_runtime_coefficients_match_horner") = [] {
    auto const p = poly::polynomial_t{1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    auto const xs = std::array{-2.0, -0.5, 0.0, 0.75, 3.0};

    for (auto const x : xs) {
      auto const got = poly::evaluate(poly::estrin, p, x);
      auto const expected = poly::evaluate(poly::horner, p, x);
      ut::expect(ut::approx(got, expected, 1e-12));
    }
  };

  ut::test("estrin_complex_matches_horner") = [] {
    auto const p = poly::polynomial_t{std::complex<double>{1.0, 2.0},
                                      std::complex<double>{-3.0, 0.5},
                                      std::complex<double>{4.0, -1.0},
                                      std::complex<double>{2.0, 0.0}};
    auto const z = std::complex<double>{0.25, -0.75};
    auto const got = poly::evaluate(poly::estrin, p, z);
    auto const expected = poly::evaluate(poly::horner, p, z);

    ut::expect(ut::approx(got.real(), expected.real(), 1e-12));
    ut::expect(ut::approx(got.imag(), expected.imag(), 1e-12));
  };

  ut::test("estrin_taylor_argument_matches_horner") = [] {
    auto constexpr p = poly::literal<4.0, -3.0, 2.0, -1.0>();
    auto constexpr x = poly::dual<double>{1.0, 2.0};
    auto constexpr got = poly::evaluate(poly::estrin, p, x);
    auto constexpr expected = poly::evaluate(poly::horner, p, x);

    static_assert(got == expected);
    static_assert(got.derivative<0>() == 23.0);
    static_assert(got.derivative<1>() == 38.0);
  };

  ut::test("horner_uint32_modular_degree4") = [] {
    constexpr auto N = 4;
    auto const p = poly::polynomial_t<uint32_t, N>{
        0xFFFFFFF0u, 0xABCDEF01u, 0x12345678u, 0x9ABCDEF0u, 0x1u};
    auto const x = 0xFFFFFFFEu;
    constexpr auto mod = uint64_t{1} << 32;
    auto const x64 = static_cast<uint64_t>(x);
    auto const expected = static_cast<uint32_t>(
        (((((uint64_t{0} * x64 + p[0]) * x64 + p[1]) * x64 + p[2]) * x64 +
          p[3]) *
             x64 +
         p[4]) %
        mod);
    auto const got = p(x);
    ut::expect(got == expected);
  };

  ut::test("batch_automatic_matches_scalar_horner") = [] {
    constexpr auto p = poly::literal<1.0, -3.0, 5.0, -7.0, 11.0, -13.0,
                                     17.0, -19.0, 23.0>();
    auto const xs = std::array{-1.0, -0.5, -0.25, 0.0, 0.25, 0.5,
                               0.75, 1.0, 1.25, 1.5, 1.75};
    auto out = std::array<double, xs.size()>{};

    auto const produced = poly::evaluate(poly::automatic, p,
                                         std::span<double const>{xs},
                                         std::span<double>{out});

    ut::expect(produced.size() == xs.size());
    for (auto i = std::size_t{}; i < xs.size(); ++i) {
      ut::expect(ut::approx(out[i], poly::evaluate(poly::horner, p, xs[i]),
                            1e-12));
    }
  };

  ut::test("batch_automatic_knuth_complex_points") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0, -4.0, 5.0>();
    auto const xs = std::array{std::complex<double>{1.0, 0.0},
                               std::complex<double>{0.5, -0.25},
                               std::complex<double>{-0.125, 0.75}};
    auto out = std::array<std::complex<double>, xs.size()>{};

    auto const produced = poly::evaluate(poly::automatic, p, xs, out);

    ut::expect(produced.size() == xs.size());
    for (auto i = std::size_t{}; i < xs.size(); ++i) {
      auto const expected = poly::evaluate(poly::knuth, p, xs[i]);
      ut::expect(ut::approx(out[i].real(), expected.real(), 1e-12));
      ut::expect(ut::approx(out[i].imag(), expected.imag(), 1e-12));
    }
  };

  ut::test("batch_knuth_split_complex_soa") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0, -4.0, 5.0>();
    auto const real = std::array{1.0, 0.5, -0.125, 0.75, -0.5};
    auto const imag = std::array{0.0, -0.25, 0.75, 0.125, -0.875};
    auto out_real = std::array<double, real.size()>{};
    auto out_imag = std::array<double, real.size()>{};

    auto const produced =
        poly::evaluate(poly::knuth, p, real, imag, out_real, out_imag);

    ut::expect(produced.real.size() == real.size());
    ut::expect(produced.imag.size() == imag.size());
    for (auto i = std::size_t{}; i < real.size(); ++i) {
      auto const z = std::complex<double>{real[i], imag[i]};
      auto const expected = poly::evaluate(poly::knuth, p, z);
      ut::expect(ut::approx(out_real[i], expected.real(), 1e-12));
      ut::expect(ut::approx(out_imag[i], expected.imag(), 1e-12));
    }
  };

  ut::test("batch_automatic_split_complex_soa") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0, -4.0, 5.0>();
    auto const real = std::array{1.0, 0.5, -0.125, 0.75, -0.5, 0.25, -0.75};
    auto const imag = std::array{0.0, -0.25, 0.75, 0.125, -0.875, 0.5, 0.375};
    auto out_real = std::array<double, real.size()>{};
    auto out_imag = std::array<double, real.size()>{};

    auto const produced =
        poly::evaluate(poly::automatic, p, real, imag, out_real, out_imag);

    ut::expect(produced.real.size() == real.size());
    ut::expect(produced.imag.size() == imag.size());
    for (auto i = std::size_t{}; i < real.size(); ++i) {
      auto const z = std::complex<double>{real[i], imag[i]};
      auto const expected = poly::evaluate(poly::knuth, p, z);
      ut::expect(ut::approx(out_real[i], expected.real(), 1e-12));
      ut::expect(ut::approx(out_imag[i], expected.imag(), 1e-12));
    }
  };

  ut::test("batch_preprocessed_motzkin_and_user_strategy") = [] {
    auto const p = poly::polynomial_t{2.0, -5.0, 7.0, -11.0, 13.0};
    auto const pre = poly::preprocess_motzkin(p);
    auto const xs = std::array{-1.0, -0.25, 0.0, 0.5, 1.25};
    auto motzkin_out = std::array<double, xs.size()>{};
    auto sentinel_out = std::array<double, xs.size()>{};

    (void)poly::evaluate(poly::motzkin, pre, xs, motzkin_out);
    (void)poly::evaluate(polygnition::test::sentinel_algorithm{}, p, xs,
                         sentinel_out);

    for (auto i = std::size_t{}; i < xs.size(); ++i) {
      ut::expect(ut::approx(motzkin_out[i], poly::evaluate(poly::horner, p, xs[i]),
                            1e-12));
      ut::expect(ut::approx(sentinel_out[i],
                            poly::evaluate(poly::horner, p, xs[i]) + 1.0,
                            1e-12));
    }
  };

  ut::test("batch_size_mismatch_is_an_error") = [] {
    constexpr auto p = poly::literal<1.0, -2.0, 3.0>();
    auto const xs = std::array{0.0, 1.0, 2.0};
    auto out = std::array<double, 2>{};
    auto threw = false;
    try {
      (void)poly::evaluate(poly::horner, p, std::span<double const>{xs},
                           std::span<double>{out});
    } catch (std::invalid_argument const &) {
      threw = true;
    }
    ut::expect(threw);
  };

  ut::test("batch_contiguous_ranges_and_size_precondition") = [] {
    constexpr auto p = poly::literal<1.0, 2.0, 3.0, 4.0, 5.0>();
    auto const xs = std::array{0.0, 0.5, 1.0, 1.5};
    auto out = std::array<double, xs.size()>{};

    auto const produced = poly::evaluate(poly::motzkin, p, xs, out);
    ut::expect(produced.size() == xs.size());
    for (auto i = std::size_t{}; i < xs.size(); ++i) {
      ut::expect(ut::approx(out[i], poly::evaluate(poly::horner, p, xs[i]),
                            1e-12));
    }

    auto too_small = std::array<double, 3>{};
    auto threw = false;
    try {
      (void)poly::evaluate(poly::motzkin, p, xs, too_small);
    } catch (std::invalid_argument const &) {
      threw = true;
    }
    ut::expect(threw);
  };

}
