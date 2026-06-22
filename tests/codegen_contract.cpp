#include "codegen_bridge.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <polygnition/arithmetic.hpp>
#include <polygnition/prepared.hpp>
#include <span>
#include <utility>

namespace poly = polygnition::polynomial;
using polygnition_codegen::make_literal;

static_assert(sizeof(static_degree20_t) == 1);

[[nodiscard]] constexpr auto make_algebra_static_quartic() {
  constexpr auto x = poly::literal<1.0, 0.0>();
  return (((poly::literal<2.0>() * x + poly::literal<-5.0>()) * x +
           poly::literal<7.0>()) *
              x +
          poly::literal<-11.0>()) *
             x +
         poly::literal<13.0>();
}

template <int N, std::size_t... I>
[[nodiscard]] consteval auto make_float_literal_impl(std::index_sequence<I...>) {
  return poly::literal<static_cast<float>(I + 1)...>();
}

template <int N> [[nodiscard]] consteval auto make_float_literal() {
  return make_float_literal_impl<N>(
      std::make_index_sequence<static_cast<std::size_t>(N + 1)>{});
}

[[nodiscard]] auto close_relative(double a, double b, double tolerance)
    -> bool {
  auto const scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
  return std::abs(a - b) <= tolerance * scale;
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_static_motzkin(double x) {
  constexpr auto p = poly::literal<2.0, -5.0, 7.0, -11.0, 13.0>();
  return p(x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_algebra_static_motzkin(double x) {
  constexpr auto p = make_algebra_static_quartic();
  return p(x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_runtime_motzkin(double const *coeffs, double x) {
  auto const p = poly::polynomial_t<double, 4>{coeffs[0], coeffs[1], coeffs[2],
                                               coeffs[3], coeffs[4]};
  return poly::evaluate(poly::motzkin, p, x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_automatic_degree12(double x) {
  constexpr auto p = make_literal<12>();
  return p(x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_automatic_degree20(double x) {
  constexpr auto p = make_literal<20>();
  return p(x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_prepared_degree20(double x) {
  constexpr auto evaluator = poly::prepare(poly::automatic, make_literal<20>());
  return evaluator(x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_automatic_degree48(double x) {
  constexpr auto p = make_literal<48>();
  return p(x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_horner_degree20(double x) {
  constexpr auto p = make_literal<20>();
  return poly::evaluate(poly::horner, p, x);
}

extern "C" [[gnu::noinline, gnu::used]] void
polygnition_codegen_complex_coeff_real_horner(double x, double *out) {
  constexpr auto p = poly::polynomial_t<std::complex<double>, 6>{
      std::complex<double>{1.0, -0.25}, std::complex<double>{-2.0, 0.5},
      std::complex<double>{3.0, -0.75}, std::complex<double>{-4.0, 1.0},
      std::complex<double>{5.0, -1.25}, std::complex<double>{-6.0, 1.5},
      std::complex<double>{7.0, -1.75}};
  auto const y = poly::evaluate(poly::horner, p, x);
  out[0] = y.real();
  out[1] = y.imag();
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_static_bridge_degree20(double x) {
  return polygnition_codegen_static_bridge_target(make_literal<20>(), x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_dorn2_degree20(double x) {
  constexpr auto p = make_literal<20>();
  return poly::evaluate(poly::dorn<2>, p, x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_dorn6_degree48(double x) {
  constexpr auto p = make_literal<48>();
  return poly::evaluate(poly::dorn<6>, p, x);
}

extern "C" [[gnu::noinline, gnu::used]] double
polygnition_codegen_estrin_degree20(double x) {
  constexpr auto p = make_literal<20>();
  return poly::evaluate(poly::estrin, p, x);
}

extern "C" [[gnu::noinline, gnu::used]] void
polygnition_codegen_batch_automatic_degree20(double const *xs, double *out,
                                             std::size_t n) {
  constexpr auto p = make_literal<20>();
  (void)poly::evaluate(poly::automatic, p, std::span<double const>{xs, n},
                       std::span<double>{out, n});
}

extern "C" [[gnu::noinline, gnu::used]] void
polygnition_codegen_batch_prepared_degree20(double const *xs, double *out,
                                            std::size_t n) {
  constexpr auto evaluator = poly::prepare(poly::automatic, make_literal<20>());
  (void)poly::evaluate_into(evaluator, std::span<double const>{xs, n},
                            std::span<double>{out, n});
}

extern "C" [[gnu::noinline, gnu::used]] void
polygnition_codegen_batch_automatic_float_degree20(float const *xs, float *out,
                                                   std::size_t n) {
  constexpr auto p = make_float_literal<20>();
  (void)poly::evaluate(poly::automatic, p, std::span<float const>{xs, n},
                       std::span<float>{out, n});
}

extern "C" [[gnu::noinline, gnu::used]] void
polygnition_codegen_batch_preprocessed_motzkin(double const *xs, double *out,
                                               std::size_t n) {
  constexpr auto p = poly::literal<2.0, -5.0, 7.0, -11.0, 13.0>();
  constexpr auto pre = poly::preprocess_motzkin(p);
  (void)poly::evaluate(poly::motzkin, pre, std::span<double const>{xs, n},
                       std::span<double>{out, n});
}

extern "C" [[gnu::noinline, gnu::used]] void
polygnition_codegen_batch_knuth_split_degree12(double const *real,
                                               double const *imag,
                                               double *out_real,
                                               double *out_imag,
                                               std::size_t n) {
  constexpr auto p = make_literal<12>();
  (void)poly::evaluate(poly::knuth, p, std::span<double const>{real, n},
                       std::span<double const>{imag, n},
                       std::span<double>{out_real, n},
                       std::span<double>{out_imag, n});
}

int main() {
  double const coeffs[5]{2.0, -5.0, 7.0, -11.0, 13.0};
  double const xs[6]{-1.25, -0.5, 0.0, 0.25, 0.875, 1.5};
  double const batch_xs[17]{-1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
                            0.25,  0.5,  0.75,  1.0,  1.25, 1.5,
                            1.75,  2.0,  2.25,  2.5,  2.75};
  double batch_out[17]{};
  double motzkin_out[17]{};
  double split_real[17]{};
  double split_imag[17]{};
  double split_out_real[17]{};
  double split_out_imag[17]{};
  double complex_coeff_out[2]{};
  float batch_xs_f[17]{};
  float batch_out_f[17]{};
  for (auto i = std::size_t{}; i < 17; ++i) {
    batch_xs_f[i] = static_cast<float>(batch_xs[i]);
  }
  polygnition_codegen_complex_coeff_real_horner(0.875, complex_coeff_out);
  polygnition_codegen_batch_automatic_degree20(batch_xs, batch_out, 17);
  polygnition_codegen_batch_automatic_float_degree20(batch_xs_f, batch_out_f,
                                                     17);
  polygnition_codegen_batch_preprocessed_motzkin(batch_xs, motzkin_out, 17);
  for (auto i = std::size_t{}; i < 17; ++i) {
    split_real[i] = std::cos(batch_xs[i]);
    split_imag[i] = -std::sin(batch_xs[i]);
  }
  polygnition_codegen_batch_knuth_split_degree12(
      split_real, split_imag, split_out_real, split_out_imag, 17);
  for (auto i = std::size_t{}; i < 17; ++i) {
    auto const expected =
        poly::evaluate(poly::horner, make_literal<20>(), batch_xs[i]);
    if (!close_relative(batch_out[i], expected, 1e-12)) {
      return 10;
    }
    auto const expected_f = poly::evaluate(poly::horner, make_float_literal<20>(),
                                           batch_xs_f[i]);
    if (!close_relative(batch_out_f[i], expected_f, 1e-5)) {
      return 13;
    }
    auto const motzkin_expected = poly::evaluate(
        poly::motzkin, poly::preprocess_motzkin(
                           poly::literal<2.0, -5.0, 7.0, -11.0, 13.0>()),
        batch_xs[i]);
    if (!close_relative(motzkin_out[i], motzkin_expected, 1e-12)) {
      return 11;
    }
    auto const split_expected = poly::evaluate(
        poly::knuth, make_literal<12>(),
        std::complex<double>{split_real[i], split_imag[i]});
    if (!close_relative(split_out_real[i], split_expected.real(), 1e-12) ||
        !close_relative(split_out_imag[i], split_expected.imag(), 1e-12)) {
      return 12;
    }
  }
  auto const complex_coeff_expected = poly::evaluate(
      poly::horner,
      poly::polynomial_t<std::complex<double>, 6>{
          std::complex<double>{1.0, -0.25}, std::complex<double>{-2.0, 0.5},
          std::complex<double>{3.0, -0.75}, std::complex<double>{-4.0, 1.0},
          std::complex<double>{5.0, -1.25}, std::complex<double>{-6.0, 1.5},
          std::complex<double>{7.0, -1.75}},
      0.875);
  if (!close_relative(complex_coeff_out[0], complex_coeff_expected.real(),
                      1e-12) ||
      !close_relative(complex_coeff_out[1], complex_coeff_expected.imag(),
                      1e-12)) {
    return 14;
  }
  for (auto const x : xs) {
    auto const static_motzkin = polygnition_codegen_static_motzkin(x);
    auto const algebra_static_motzkin =
        polygnition_codegen_algebra_static_motzkin(x);
    auto const runtime_motzkin = polygnition_codegen_runtime_motzkin(coeffs, x);
    auto const automatic_degree12 = polygnition_codegen_automatic_degree12(x);
    auto const automatic_degree20 = polygnition_codegen_automatic_degree20(x);
    auto const automatic_degree48 = polygnition_codegen_automatic_degree48(x);
    auto const horner_degree20 = polygnition_codegen_horner_degree20(x);
    auto const static_bridge_degree20 =
        polygnition_codegen_static_bridge_degree20(x);
    auto const horner_degree48 = poly::evaluate(poly::horner, make_literal<48>(), x);
    if (std::abs(static_motzkin - runtime_motzkin) > 1e-10) {
      return 1;
    }
    if (std::abs(static_motzkin - algebra_static_motzkin) > 1e-10) {
      return 8;
    }
    if (std::abs(automatic_degree12 -
                 poly::evaluate(poly::horner, make_literal<12>(), x)) >
        1e-8) {
      return 2;
    }
    if (std::abs(automatic_degree20 - horner_degree20) > 1e-7) {
      return 3;
    }
    if (std::abs(static_bridge_degree20 - horner_degree20) > 1e-7) {
      return 9;
    }
    if (!close_relative(automatic_degree48, horner_degree48, 1e-12)) {
      return 4;
    }
    if (!close_relative(polygnition_codegen_dorn2_degree20(x), horner_degree20,
                        1e-12)) {
      return 5;
    }
    if (!close_relative(polygnition_codegen_dorn6_degree48(x), horner_degree48,
                        1e-12)) {
      return 6;
    }
    if (!close_relative(polygnition_codegen_estrin_degree20(x), horner_degree20,
                        1e-12)) {
      return 7;
    }
  }
  return 0;
}
