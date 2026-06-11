#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <polygnition/dev/helpers.hpp>
#include <polygnition/poly.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace poly = polygnition::polynomial;
namespace dev = polygnition::dev;

namespace {
template <typename T> inline void do_not_optimize(T const &value) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "g"(value) : "memory");
#else
  (void)value;
#endif
}

struct row_t {
  std::string_view profile;
  std::string_view mode;
  int degree;
  int element_bits;
  int lanes;
  int unroll;
  double ns_per_eval;
};

[[nodiscard]] constexpr auto configured_profile_name() -> std::string_view {
  using enum polygnition::target::profile;
  switch (polygnition::target::configured) {
  case generic:
    return "generic";
  case x86_intel_coffee_lake:
    return "x86_intel_coffee_lake";
  case x86_intel_icelake_avx512:
    return "x86_intel_icelake_avx512";
  case x86_amd_zen3:
    return "x86_amd_zen3";
  }
  return "generic";
}

template <typename T> [[nodiscard]] consteval auto element_bits() -> int {
  return static_cast<int>(sizeof(T) * CHAR_BIT);
}

template <int N, typename T, typename Algorithm>
[[gnu::noinline]] auto scalar_loop(Algorithm algorithm,
                                   poly::polynomial_t<T, N> const &p,
                                   std::span<T const> xs, int reps)
    -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    for (auto const x : xs) {
      sum += static_cast<double>(poly::evaluate(algorithm, p, x));
    }
  }
  return sum;
}

template <int N, typename T>
[[gnu::noinline]] auto scalar_automatic_loop(
    poly::polynomial_t<T, N> const &p, std::span<T const> xs,
    int reps) -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    for (auto const x : xs) {
      sum += static_cast<double>(p(x));
    }
  }
  return sum;
}

template <int N, typename T, typename Algorithm>
[[gnu::noinline]] auto batch_loop(Algorithm algorithm,
                                  poly::polynomial_t<T, N> const &p,
                                  std::span<T const> xs, std::span<T> out,
                                  int reps) -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    auto const produced = poly::evaluate(algorithm, p, xs, out);
    sum += static_cast<double>(std::reduce(produced.begin(), produced.end(), T{}));
  }
  return sum;
}

template <int N, typename T, std::size_t LaneCount, std::size_t Unroll,
          typename Algorithm>
[[gnu::noinline]] auto
batch_width_loop(Algorithm algorithm, poly::polynomial_t<T, N> const &p,
                 std::span<T const> xs, std::span<T> out, int reps) -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    auto const produced = poly::detail::evaluate_batch_blocked_width<
        Algorithm, poly::polynomial_t<T, N>, T, T, LaneCount, Unroll>(
        algorithm, p, xs, out);
    sum += static_cast<double>(std::reduce(produced.begin(), produced.end(), T{}));
  }
  return sum;
}

template <int N>
[[gnu::noinline]] auto complex_scalar_knuth_loop(
    poly::polynomial_t<double, N> const &p,
    std::span<std::complex<double> const> zs, int reps) -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    for (auto const z : zs) {
      auto const y = poly::evaluate(poly::knuth, p, z);
      sum += y.real() + y.imag();
    }
  }
  return sum;
}

template <int N>
[[gnu::noinline]] auto complex_batch_knuth_loop(
    poly::polynomial_t<double, N> const &p,
    std::span<std::complex<double> const> zs,
    std::span<std::complex<double>> out, int reps) -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    auto const produced = poly::evaluate(poly::knuth, p, zs, out);
    for (auto const y : produced) {
      sum += y.real() + y.imag();
    }
  }
  return sum;
}

template <int N>
[[gnu::noinline]] auto complex_batch_knuth_split_loop(
    poly::polynomial_t<double, N> const &p, std::span<double const> real,
    std::span<double const> imag, std::span<double> out_real,
    std::span<double> out_imag, int reps) -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    auto const produced =
        poly::evaluate(poly::knuth, p, real, imag, out_real, out_imag);
    sum += std::reduce(produced.real.begin(), produced.real.end(), 0.0);
    sum += std::reduce(produced.imag.begin(), produced.imag.end(), 0.0);
  }
  return sum;
}

template <typename Fn>
[[nodiscard]] auto measure(Fn &&fn, std::int64_t evaluations) -> double {
  auto const start = std::chrono::steady_clock::now();
  auto const sum = fn();
  auto const stop = std::chrono::steady_clock::now();
  do_not_optimize(sum);
  return std::chrono::duration<double, std::nano>(stop - start).count() /
         static_cast<double>(evaluations);
}

template <int N, typename T>
[[nodiscard]] auto probe(std::span<T const> xs, std::span<T> out,
                         int reps) {
  auto const p = dev::iota_poly<T, N>(T{0.001}, T{0.001});
  auto const evaluations = static_cast<std::int64_t>(xs.size()) * reps;
  do_not_optimize(scalar_loop(poly::horner, p, xs, 1));
  do_not_optimize(scalar_automatic_loop(p, xs, 1));
  do_not_optimize(batch_loop(poly::automatic, p, xs, out, 1));

  auto rows = std::vector<row_t>{};
  rows.reserve(64);
  auto const push = [&](std::string_view mode, int lanes, int unroll,
                        double ns_per_eval) {
    rows.push_back(row_t{configured_profile_name(), mode, N, element_bits<T>(),
                         lanes, unroll, ns_per_eval});
  };

  push("scalar_horner", 1, 1,
       measure([&] { return scalar_loop(poly::horner, p, xs, reps); },
               evaluations));
  push("scalar_auto_latency", 1, 1,
       measure([&] { return scalar_automatic_loop(p, xs, reps); },
               evaluations));
  using batch_key = poly::detail::selection_key<
      poly::polynomial_t<T, N>, T,
      poly::detail::evaluation_intent::throughput>;
  constexpr auto automatic_batch_choice =
      poly::detail::profile_tuning<polygnition::target::configured>::table::
          template select_batch_choice<batch_key,
                                       poly::polynomial_t<T, N>, T>();
  push("batch_auto_throughput", automatic_batch_choice.lanes,
       automatic_batch_choice.unroll,
       measure([&] { return batch_loop(poly::automatic, p, xs, out, reps); },
               evaluations));

  auto const append_candidate = [&]<std::size_t LaneCount, std::size_t Unroll,
                                  typename Algorithm>(std::string_view mode,
                                                      Algorithm algorithm) {
    push(mode, static_cast<int>(LaneCount), static_cast<int>(Unroll),
         measure([&] {
           return batch_width_loop<N, T, LaneCount, Unroll>(algorithm, p, xs,
                                                            out, reps);
         },
                 evaluations));
  };

  auto const append_sweep = [&]<std::size_t LaneCount, std::size_t Unroll>() {
    append_candidate.template operator()<LaneCount, Unroll>(
        "batch_explicit_horner", poly::horner);
    append_candidate.template operator()<LaneCount, Unroll>("batch_dorn2",
                                                            poly::dorn<2>);
    append_candidate.template operator()<LaneCount, Unroll>("batch_dorn4",
                                                            poly::dorn<4>);
    append_candidate.template operator()<LaneCount, Unroll>("batch_dorn5",
                                                            poly::dorn<5>);
    append_candidate.template operator()<LaneCount, Unroll>("batch_dorn6",
                                                            poly::dorn<6>);
  };

  append_sweep.template operator()<2, 1>();
  append_sweep.template operator()<2, 2>();
  append_sweep.template operator()<2, 4>();
  append_sweep.template operator()<2, 8>();
  append_sweep.template operator()<4, 1>();
  append_sweep.template operator()<4, 2>();
  append_sweep.template operator()<4, 4>();
  append_sweep.template operator()<4, 8>();
  append_sweep.template operator()<8, 1>();
  append_sweep.template operator()<8, 2>();
  append_sweep.template operator()<8, 4>();
  append_sweep.template operator()<8, 8>();

  return rows;
}

template <int N>
[[nodiscard]] auto probe_complex(std::span<std::complex<double> const> zs,
                                 std::span<std::complex<double>> out,
                                 std::span<double const> real,
                                 std::span<double const> imag,
                                 std::span<double> out_real,
                                 std::span<double> out_imag, int reps) {
  auto const p = dev::iota_poly<double, N>(0.001, 0.001);
  auto const evaluations = static_cast<std::int64_t>(zs.size()) * reps;
  auto constexpr default_lanes =
      static_cast<int>(poly::detail::default_batch_lanes<double>());
  auto constexpr default_unroll = 4;
  using split_key = poly::detail::selection_key<
      poly::polynomial_t<double, N>, polygnition::split_complex<double>,
      poly::detail::evaluation_intent::throughput>;
  constexpr auto split_choice =
      poly::detail::profile_tuning<polygnition::target::configured>::table::
          template select_batch_choice<split_key, poly::polynomial_t<double, N>,
                                       polygnition::split_complex<double>>();
  do_not_optimize(complex_scalar_knuth_loop(p, zs, 1));
  do_not_optimize(complex_batch_knuth_loop(p, zs, out, 1));
  do_not_optimize(complex_batch_knuth_split_loop(p, real, imag, out_real,
                                                 out_imag, 1));

  return std::array{
      row_t{configured_profile_name(), "complex_scalar_knuth", N, 64, 1, 1,
            measure([&] { return complex_scalar_knuth_loop(p, zs, reps); },
                    evaluations)},
      row_t{configured_profile_name(), "complex_batch_knuth_aos", N, 64,
            default_lanes, default_unroll,
            measure([&] { return complex_batch_knuth_loop(p, zs, out, reps); },
                    evaluations)},
      row_t{configured_profile_name(), "complex_batch_knuth_split", N, 64,
            split_choice.lanes,
            split_choice.unroll,
            measure([&] {
              return complex_batch_knuth_split_loop(p, real, imag, out_real,
                                                    out_imag, reps);
            },
                    evaluations)},
  };
}
} // namespace

int main() {
  auto xs = std::vector<double>(8192);
  auto out = std::vector<double>(xs.size());
  auto xs_float = std::vector<float>(xs.size());
  auto out_float = std::vector<float>(xs.size());
  auto zs = std::vector<std::complex<double>>(xs.size());
  auto complex_out = std::vector<std::complex<double>>(xs.size());
  auto imag = std::vector<double>(xs.size());
  auto split_out_real = std::vector<double>(xs.size());
  auto split_out_imag = std::vector<double>(xs.size());
  for (auto i = std::size_t{}; i < xs.size(); ++i) {
    xs[i] = 0.999 + static_cast<double>(i % 257U) * 1e-6;
    imag[i] = -std::sin(xs[i]);
    xs[i] = std::cos(xs[i]);
    xs_float[i] = static_cast<float>(xs[i]);
    zs[i] = std::complex<double>{xs[i], imag[i]};
  }

  auto constexpr reps = 32;
  std::puts("profile,mode,degree,element_bits,lanes,unroll,ns_per_eval,eval_per_second");
  auto const print = [](auto const &rows) {
    for (auto const &row : rows) {
      std::printf("%.*s,%.*s,%d,%d,%d,%d,%.9f,%.3f\n",
                  static_cast<int>(row.profile.size()), row.profile.data(),
                  static_cast<int>(row.mode.size()), row.mode.data(), row.degree,
                  row.element_bits, row.lanes, row.unroll, row.ns_per_eval,
                  1e9 / row.ns_per_eval);
    }
  };
  print(probe<4, float>(xs_float, out_float, reps));
  print(probe<12, float>(xs_float, out_float, reps));
  print(probe<20, float>(xs_float, out_float, reps));
  print(probe<32, float>(xs_float, out_float, reps));
  print(probe<48, float>(xs_float, out_float, reps));
  print(probe<80, float>(xs_float, out_float, reps));
  print(probe<4, double>(xs, out, reps));
  print(probe<12, double>(xs, out, reps));
  print(probe<20, double>(xs, out, reps));
  print(probe<32, double>(xs, out, reps));
  print(probe<48, double>(xs, out, reps));
  print(probe<80, double>(xs, out, reps));
  print(probe_complex<12>(zs, complex_out, xs, imag, split_out_real,
                          split_out_imag, reps));
  print(probe_complex<32>(zs, complex_out, xs, imag, split_out_real,
                          split_out_imag, reps));
}
