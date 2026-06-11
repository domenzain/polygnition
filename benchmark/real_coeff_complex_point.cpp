#include <benchmark/benchmark.h>
#include <complex>
#include <polygnition/dev/helpers.hpp>
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;

struct horner_tag {
  template <int N>
  static inline std::complex<double>
  eval(const poly::polynomial_t<double, N> &p,
       const std::complex<double> &z) noexcept {
    return poly::evaluate(poly::horner, p, z);
  }
};

struct knuth_tag {
  template <int N>
  static inline std::complex<double>
  eval(const poly::polynomial_t<double, N> &p,
       const std::complex<double> &z) noexcept {
    return poly::evaluate(poly::knuth, p, z);
  }
};

template <class Algo, int N>
static void bm_complex_eval(benchmark::State &state) {
  constexpr auto p = polygnition::dev::iota_poly<double, N>();

  std::complex<double> z{1.23456789, 2.3456789};
  benchmark::DoNotOptimize(p.data());
  benchmark::DoNotOptimize(z);

  for (auto _ : state) {
    auto r = Algo::template eval<N>(p, z);
    benchmark::DoNotOptimize(r);
  }

  state.counters["n"] = N;
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N + 1));
  state.counters["coeff_per_sec"] =
      benchmark::Counter(static_cast<double>(N + 1),
                         benchmark::Counter::kIsIterationInvariantRate);
}

#define INSTANTIATE(N)                                                         \
  BENCHMARK_TEMPLATE(bm_complex_eval, horner_tag, N)                           \
      ->Name("real_coeff_complex_point/horner/n=" #N);                         \
  BENCHMARK_TEMPLATE(bm_complex_eval, knuth_tag, N)                            \
      ->Name("real_coeff_complex_point/knuth/n=" #N);

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(4)
INSTANTIATE(8)
INSTANTIATE(16)
INSTANTIATE(32)
INSTANTIATE(64)
INSTANTIATE(128)
INSTANTIATE(256)
INSTANTIATE(512)
INSTANTIATE(1024)
