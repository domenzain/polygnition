#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <polygnition/poly.hpp>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <vector>

namespace poly = polygnition::polynomial;

enum class strategy_t : std::uint8_t {
  horner,
  dorn2,
  dorn4,
  dorn5,
  dorn6,
  automatic,
};

struct samples_t {
  strategy_t strategy;
  std::vector<double> ns_per_eval;
};

struct mann_whitney_t {
  double u;
  double p_less;
  double common_language_less;
};

template <typename T> inline void do_not_optimize(T const &value) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "g"(value) : "memory");
#else
  (void)value;
#endif
}

[[nodiscard]] auto name(strategy_t strategy) -> std::string_view {
  switch (strategy) {
  case strategy_t::horner:
    return "horner";
  case strategy_t::dorn2:
    return "dorn2";
  case strategy_t::dorn4:
    return "dorn4";
  case strategy_t::dorn5:
    return "dorn5";
  case strategy_t::dorn6:
    return "dorn6";
  case strategy_t::automatic:
    return "automatic";
  }
  return "unknown";
}

template <int N> [[nodiscard]] consteval auto selected_strategy() {
  using polynomial_type = poly::polynomial_t<double, N>;
  using selected_type =
      poly::detail::selected_algorithm_t<polynomial_type, double>;
  if constexpr (std::is_same_v<selected_type, poly::algorithm::dorn_t<2>>) {
    return strategy_t::dorn2;
  } else if constexpr (std::is_same_v<selected_type,
                                      poly::algorithm::dorn_t<4>>) {
    return strategy_t::dorn4;
  } else if constexpr (std::is_same_v<selected_type,
                                      poly::algorithm::dorn_t<5>>) {
    return strategy_t::dorn5;
  } else if constexpr (std::is_same_v<selected_type,
                                      poly::algorithm::dorn_t<6>>) {
    return strategy_t::dorn6;
  } else {
    return strategy_t::horner;
  }
}

[[nodiscard]] auto median(std::vector<double> values) -> double {
  auto const n = values.size();
  auto const middle = values.begin() + static_cast<std::ptrdiff_t>(n / 2);
  std::ranges::nth_element(values, middle);
  if (n % 2 != 0) {
    return *middle;
  }
  auto const low_middle =
      std::ranges::max_element(std::ranges::subrange(values.begin(), middle));
  return (*low_middle + *middle) * 0.5;
}

[[nodiscard]] auto normal_cdf(double z) -> double {
  return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

// One-sided Mann-Whitney U test. p_less is P(U <= observed U_x) under H0,
// so smaller p_less means x is faster than y when samples are timings.
[[nodiscard]] auto mann_whitney_less(std::vector<double> const &x,
                                     std::vector<double> const &y)
    -> mann_whitney_t {
  struct observation_t {
    double value;
    int group;
  };

  auto observations = std::vector<observation_t>{};
  observations.reserve(x.size() + y.size());
  std::ranges::transform(x, std::back_inserter(observations),
                         [](double value) { return observation_t{value, 0}; });
  std::ranges::transform(y, std::back_inserter(observations),
                         [](double value) { return observation_t{value, 1}; });
  std::ranges::sort(observations, {}, &observation_t::value);

  auto rank_sum_x = 0.0;
  auto tie_sum = 0.0;
  for (auto first = observations.begin(); first != observations.end();) {
    auto const equal_values = std::ranges::subrange(first, observations.end());
    auto const last = std::ranges::find_if_not(
        equal_values, [value = first->value](auto const &observation) {
          return observation.value == value;
        });
    auto const tie_count = static_cast<double>(last - first);
    auto const average_rank =
        (static_cast<double>((first - observations.begin()) + 1) +
         static_cast<double>(last - observations.begin())) *
        0.5;
    rank_sum_x +=
        average_rank * static_cast<double>(std::ranges::count_if(
                           std::ranges::subrange(first, last),
                           [](auto const &o) { return o.group == 0; }));
    tie_sum += (tie_count * tie_count * tie_count) - tie_count;
    first = last;
  }

  auto const n_x = static_cast<double>(x.size());
  auto const n_y = static_cast<double>(y.size());
  auto const n = n_x + n_y;
  auto const u = rank_sum_x - (n_x * (n_x + 1.0) * 0.5);
  auto const mean = n_x * n_y * 0.5;
  auto const tie_correction = n > 1.0 ? tie_sum / (n * (n - 1.0)) : 0.0;
  auto const variance = n_x * n_y * ((n + 1.0) - tie_correction) / 12.0;
  auto const z = variance > 0.0 ? (u - mean + 0.5) / std::sqrt(variance) : 0.0;
  return {.u = u,
          .p_less = variance > 0.0 ? normal_cdf(z) : 1.0,
          .common_language_less = 1.0 - (u / (n_x * n_y))};
}

template <int N> [[nodiscard]] auto make_poly() {
  auto p = poly::polynomial_t<double, N>{};
  for (auto i = 0; i <= N; ++i)
    p[static_cast<std::size_t>(i)] = static_cast<double>(i + 1) * 0.001;
  return p;
}

template <int N, typename Algorithm>
[[gnu::noinline]] auto run(Algorithm algorithm,
                           poly::polynomial_t<double, N> const &p,
                           std::vector<double> const &xs, int reps) -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    for (auto const x : xs) {
      sum += poly::evaluate(algorithm, p, x);
    }
  }
  return sum;
}

template <int N>
[[gnu::noinline]] auto run_automatic(poly::polynomial_t<double, N> const &p,
                                     std::vector<double> const &xs, int reps)
    -> double {
  auto sum = 0.0;
  for (auto r = 0; r < reps; ++r) {
    for (auto const x : xs) {
      sum += p(x);
    }
  }
  return sum;
}

template <int N>
[[nodiscard]] auto run_strategy(strategy_t strategy,
                                poly::polynomial_t<double, N> const &p,
                                std::vector<double> const &xs, int reps)
    -> double {
  switch (strategy) {
  case strategy_t::horner:
    return run(poly::horner, p, xs, reps);
  case strategy_t::dorn2:
    return run(poly::dorn<2>, p, xs, reps);
  case strategy_t::dorn4:
    return run(poly::dorn<4>, p, xs, reps);
  case strategy_t::dorn5:
    return run(poly::dorn<5>, p, xs, reps);
  case strategy_t::dorn6:
    return run(poly::dorn<6>, p, xs, reps);
  case strategy_t::automatic:
    return run_automatic(p, xs, reps);
  }
  return 0.0;
}

template <int N>
[[nodiscard]] auto ns_per_eval(strategy_t strategy,
                               poly::polynomial_t<double, N> const &p,
                               std::vector<double> const &xs, int reps)
    -> double {
  auto const evaluations = static_cast<std::int64_t>(xs.size()) * reps;
  auto const start = std::chrono::steady_clock::now();
  auto const sum = run_strategy(strategy, p, xs, reps);
  auto const stop = std::chrono::steady_clock::now();
  do_not_optimize(sum);
  return std::chrono::duration<double, std::nano>(stop - start).count() /
         static_cast<double>(evaluations);
}

template <int N>
auto probe(std::vector<double> const &xs, int reps, int sample_count) -> void {
  constexpr auto strategies =
      std::array{strategy_t::horner, strategy_t::dorn2, strategy_t::dorn4,
                 strategy_t::dorn5,  strategy_t::dorn6, strategy_t::automatic};
  constexpr auto candidates =
      std::array{strategy_t::horner, strategy_t::dorn2, strategy_t::dorn4,
                 strategy_t::dorn5, strategy_t::dorn6};
  auto const p = make_poly<N>();
  auto rows = std::array<samples_t, strategies.size()>{};
  std::ranges::transform(strategies, rows.begin(), [](auto strategy) {
    return samples_t{.strategy = strategy, .ns_per_eval = {}};
  });
  std::ranges::for_each(rows, [sample_count](auto &row) {
    row.ns_per_eval.reserve(static_cast<std::size_t>(sample_count));
  });

  std::ranges::for_each(strategies, [&](auto strategy) {
    do_not_optimize(run_strategy(strategy, p, xs, 1));
  });

  for (auto sample = 0; sample < sample_count; ++sample) {
    auto order = strategies;
    std::ranges::rotate(order, order.begin() + (sample % order.size()));
    if (sample % 2 != 0) {
      std::ranges::reverse(order);
    }
    for (auto strategy : order) {
      auto const sample_ns = ns_per_eval(strategy, p, xs, reps);
      auto const row = std::ranges::find_if(
          rows, [strategy](auto const &r) { return r.strategy == strategy; });
      row->ns_per_eval.push_back(sample_ns);
    }
  }

  auto const median_for = [&](strategy_t strategy) {
    auto const row = std::ranges::find_if(
        rows, [strategy](auto const &r) { return r.strategy == strategy; });
    return median(row->ns_per_eval);
  };
  auto const samples_for =
      [&](strategy_t strategy) -> std::vector<double> const & {
    auto const row = std::ranges::find_if(
        rows, [strategy](auto const &r) { return r.strategy == strategy; });
    return row->ns_per_eval;
  };

  auto const best = *std::ranges::min_element(candidates, [&](auto a, auto b) {
    return median_for(a) < median_for(b);
  });
  auto const selected = selected_strategy<N>();
  auto const p_best_lt_selected =
      best == selected
          ? mann_whitney_t{.u = 0.0, .p_less = 1.0, .common_language_less = 0.5}
          : mann_whitney_less(samples_for(best), samples_for(selected));
  auto const p_selected_lt_horner =
      selected == strategy_t::horner
          ? mann_whitney_t{.u = 0.0, .p_less = 1.0, .common_language_less = 0.5}
          : mann_whitney_less(samples_for(selected),
                              samples_for(strategy_t::horner));

  auto const best_median = median_for(best);
  auto const selected_median = median_for(selected);
  auto const horner_median = median_for(strategy_t::horner);
  std::printf("%d,%.*s,%.*s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3g,%.3g,%.3f,%.3f,%"
              ".3f,%.3f\n",
              N, static_cast<int>(name(selected).size()), name(selected).data(),
              static_cast<int>(name(best).size()), name(best).data(),
              horner_median, median_for(strategy_t::dorn2),
              median_for(strategy_t::dorn4), median_for(strategy_t::dorn5),
              median_for(strategy_t::dorn6), median_for(strategy_t::automatic),
              p_best_lt_selected.p_less, p_selected_lt_horner.p_less,
              p_best_lt_selected.common_language_less,
              p_selected_lt_horner.common_language_less,
              best_median / selected_median, selected_median / horner_median);
}

int main() {
  auto xs = std::vector<double>(2048);
  for (std::size_t i = 0; i < xs.size(); ++i)
    xs[i] = 0.999 + static_cast<double>(i % 257U) * 1e-6;
  auto const reps = 512;
  auto const sample_count = 31;
  std::puts("# lower ns is better; p-values are one-sided Mann-Whitney U "
            "normal approximations");
  std::puts("degree,selected,best,horner_ns,dorn2_ns,dorn4_ns,dorn5_ns,"
            "dorn6_ns,auto_ns,p_best_lt_selected,p_selected_lt_horner,"
            "cl_best_lt_selected,cl_selected_lt_horner,"
            "ratio_best_to_selected,ratio_selected_to_horner");
  probe<12>(xs, reps, sample_count);
  probe<13>(xs, reps, sample_count);
  probe<14>(xs, reps, sample_count);
  probe<15>(xs, reps, sample_count);
  probe<16>(xs, reps, sample_count);
  probe<17>(xs, reps, sample_count);
  probe<18>(xs, reps, sample_count);
  probe<20>(xs, reps, sample_count);
  probe<24>(xs, reps, sample_count);
  probe<32>(xs, reps, sample_count);
  probe<47>(xs, reps, sample_count);
  probe<48>(xs, reps, sample_count);
  probe<64>(xs, reps, sample_count);
  probe<79>(xs, reps, sample_count);
  probe<80>(xs, reps, sample_count);
}
