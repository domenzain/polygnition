#include <benchmark/benchmark.h>
#include <polygnition/dev/helpers.hpp>
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;
namespace dev = polygnition::dev;

template <int N>
static void bm_horner_eval(benchmark::State &state) {
  constexpr auto p = dev::iota_poly<double, N>();
  double x = 1.23456789;
  benchmark::DoNotOptimize(p.data());
  benchmark::DoNotOptimize(x);

  for (auto _ : state) {
    auto r = poly::evaluate(poly::horner, p, x);
    benchmark::DoNotOptimize(r);
  }

  state.counters["n"] = N;
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N + 1));
  state.counters["coeff_per_sec"] =
      benchmark::Counter(static_cast<double>(N + 1),
                         benchmark::Counter::kIsIterationInvariantRate);
}

template <int N, int K>
static void bm_dorn_eval(benchmark::State &state) {
  static_assert(K > 1);
  constexpr auto p = dev::iota_poly<double, N>();
  double x = 1.23456789;
  benchmark::DoNotOptimize(p.data());
  benchmark::DoNotOptimize(x);

  for (auto _ : state) {
    auto r = poly::evaluate(poly::dorn<K>, p, x);
    benchmark::DoNotOptimize(r);
  }

  state.counters["n"] = N;
  state.counters["k"] = K;
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N + 1));
  state.counters["coeff_per_sec"] =
      benchmark::Counter(static_cast<double>(N + 1),
                         benchmark::Counter::kIsIterationInvariantRate);
}

static void bm_motzkin_preprocessed_eval(benchmark::State &state) {
  constexpr auto N = 4;
  constexpr auto p = dev::iota_poly<double, N>();
  auto const pre = poly::preprocess_motzkin(p);
  double x = 1.23456789;
  benchmark::DoNotOptimize(p.data());
  benchmark::DoNotOptimize(x);

  for (auto _ : state) {
    auto r = poly::evaluate(poly::motzkin, pre, x);
    benchmark::DoNotOptimize(r);
  }

  state.counters["n"] = N;
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N + 1));
  state.counters["coeff_per_sec"] =
      benchmark::Counter(static_cast<double>(N + 1),
                         benchmark::Counter::kIsIterationInvariantRate);
}

static void bm_motzkin_preprocess_and_eval(benchmark::State &state) {
  constexpr auto N = 4;
  constexpr auto p = dev::iota_poly<double, N>();
  double x = 1.23456789;
  benchmark::DoNotOptimize(p.data());
  benchmark::DoNotOptimize(x);

  for (auto _ : state) {
    auto p_runtime = p;
    benchmark::DoNotOptimize(p_runtime.data());
    auto const pre = poly::preprocess_motzkin(p_runtime);
    auto r = poly::evaluate(poly::motzkin, pre, x);
    benchmark::DoNotOptimize(r);
  }

  state.counters["n"] = N;
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N + 1));
  state.counters["coeff_per_sec"] =
      benchmark::Counter(static_cast<double>(N + 1),
                         benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(bm_horner_eval<4>)->Name("real_coeff_real_point/horner/n=4");
BENCHMARK(bm_motzkin_preprocessed_eval)
    ->Name("real_coeff_real_point/motzkin_preprocessed_eval/n=4");
BENCHMARK(bm_motzkin_preprocess_and_eval)
    ->Name("real_coeff_real_point/motzkin_preprocess_and_eval/n=4");
#define POLYGNITION_BENCH_HEAD_TO_HEAD(N, K)                                 \
  BENCHMARK(bm_horner_eval<N>)                                               \
      ->Name("real_coeff_real_point/head_to_head/horner/n=" #N);             \
  BENCHMARK(bm_dorn_eval<N, K>)                                       \
      ->Name("real_coeff_real_point/head_to_head/dorn/k=" #K          \
             "/n=" #N)

POLYGNITION_BENCH_HEAD_TO_HEAD(1, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(2, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(3, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(4, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(5, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(6, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(7, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(8, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(9, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(10, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(11, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(12, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(13, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(14, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(16, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(18, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(20, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(24, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(28, 2);
POLYGNITION_BENCH_HEAD_TO_HEAD(32, 2);

#define POLYGNITION_BENCH_PLANE(N, K)                                       \
  BENCHMARK(bm_dorn_eval<N, K>)                                      \
      ->Name("real_coeff_real_point/plane/k=" #K "/n=" #N)

// Sweep n,k around the k-way threshold to map the performance plane.
POLYGNITION_BENCH_PLANE(7, 2);
POLYGNITION_BENCH_PLANE(7, 3);
POLYGNITION_BENCH_PLANE(7, 4);
POLYGNITION_BENCH_PLANE(7, 5);
POLYGNITION_BENCH_PLANE(7, 6);
POLYGNITION_BENCH_PLANE(7, 7);
POLYGNITION_BENCH_PLANE(7, 8);

POLYGNITION_BENCH_PLANE(8, 2);
POLYGNITION_BENCH_PLANE(8, 3);
POLYGNITION_BENCH_PLANE(8, 4);
POLYGNITION_BENCH_PLANE(8, 5);
POLYGNITION_BENCH_PLANE(8, 6);
POLYGNITION_BENCH_PLANE(8, 7);
POLYGNITION_BENCH_PLANE(8, 8);

POLYGNITION_BENCH_PLANE(9, 2);
POLYGNITION_BENCH_PLANE(9, 3);
POLYGNITION_BENCH_PLANE(9, 4);
POLYGNITION_BENCH_PLANE(9, 5);
POLYGNITION_BENCH_PLANE(9, 6);
POLYGNITION_BENCH_PLANE(9, 7);
POLYGNITION_BENCH_PLANE(9, 8);

POLYGNITION_BENCH_PLANE(10, 2);
POLYGNITION_BENCH_PLANE(10, 3);
POLYGNITION_BENCH_PLANE(10, 4);
POLYGNITION_BENCH_PLANE(10, 5);
POLYGNITION_BENCH_PLANE(10, 6);
POLYGNITION_BENCH_PLANE(10, 7);
POLYGNITION_BENCH_PLANE(10, 8);

POLYGNITION_BENCH_PLANE(11, 2);
POLYGNITION_BENCH_PLANE(11, 3);
POLYGNITION_BENCH_PLANE(11, 4);
POLYGNITION_BENCH_PLANE(11, 5);
POLYGNITION_BENCH_PLANE(11, 6);
POLYGNITION_BENCH_PLANE(11, 7);
POLYGNITION_BENCH_PLANE(11, 8);

POLYGNITION_BENCH_PLANE(12, 2);
POLYGNITION_BENCH_PLANE(12, 3);
POLYGNITION_BENCH_PLANE(12, 4);
POLYGNITION_BENCH_PLANE(12, 5);
POLYGNITION_BENCH_PLANE(12, 6);
POLYGNITION_BENCH_PLANE(12, 7);
POLYGNITION_BENCH_PLANE(12, 8);

#undef POLYGNITION_BENCH_PLANE

#undef POLYGNITION_BENCH_HEAD_TO_HEAD
