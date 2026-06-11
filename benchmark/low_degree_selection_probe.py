#!/usr/bin/env python3
import argparse
import csv
import subprocess
import textwrap
from pathlib import Path


CASES = {
    "floating_real": {
        "coeff_type": "double",
        "var_type": "double",
        "make_poly": "dynamic_double",
        "make_value": "return 0.25 + (static_cast<double>((i * 17U) % 257U) / 256.0);",
        "strategies": ["horner", "dorn2", "dorn4", "dorn5", "dorn6"],
    },
    "floating_static_real": {
        "coeff_type": "double",
        "var_type": "double",
        "make_poly": "static_double",
        "make_value": "return 0.25 + (static_cast<double>((i * 17U) % 257U) / 256.0);",
        "strategies": ["horner", "motzkin", "dorn2", "dorn4", "dorn5", "dorn6"],
    },
    "floating_complex_var": {
        "coeff_type": "double",
        "var_type": "std::complex<double>",
        "make_poly": "dynamic_double",
        "make_value": "return {0.5 + (static_cast<double>((i * 17U) % 257U) / 1024.0), -0.25 + (static_cast<double>((i * 29U) % 251U) / 2048.0)};",
        "strategies": ["horner", "knuth", "dorn2", "dorn4", "dorn5", "dorn6"],
    },
    "integral_real": {
        "coeff_type": "std::int64_t",
        "var_type": "std::int64_t",
        "make_poly": "dynamic_integral",
        "make_value": "return static_cast<std::int64_t>(1 + (i % 3U));",
        "strategies": ["horner", "dorn2", "dorn4", "dorn5", "dorn6"],
    },
    "integral_complex_var": {
        "coeff_type": "std::int64_t",
        "var_type": "std::complex<double>",
        "make_poly": "dynamic_integral",
        "make_value": "return {0.5 + (static_cast<double>((i * 17U) % 257U) / 1024.0), -0.25 + (static_cast<double>((i * 29U) % 251U) / 2048.0)};",
        "strategies": ["knuth"],
    },
    "modular_unsigned": {
        "coeff_type": "std::uint32_t",
        "var_type": "std::uint32_t",
        "make_poly": "dynamic_unsigned",
        "make_value": "return static_cast<std::uint32_t>(1 + (i % 3U));",
        "strategies": ["horner", "dorn2", "dorn4", "dorn5", "dorn6"],
    },
    "modular_unsigned_complex_var": {
        "coeff_type": "std::uint32_t",
        "var_type": "std::complex<double>",
        "make_poly": "dynamic_unsigned",
        "make_value": "return {0.5 + (static_cast<double>((i * 17U) % 257U) / 1024.0), -0.25 + (static_cast<double>((i * 29U) % 251U) / 2048.0)};",
        "strategies": ["knuth"],
    },
    "complex_coeff_real_var": {
        "coeff_type": "std::complex<double>",
        "var_type": "double",
        "make_poly": "dynamic_complex",
        "make_value": "return 0.25 + (static_cast<double>((i * 17U) % 257U) / 256.0);",
        "strategies": ["horner", "dorn2", "dorn4", "dorn5", "dorn6"],
    },
    "complex_coeff_complex_var": {
        "coeff_type": "std::complex<double>",
        "var_type": "std::complex<double>",
        "make_poly": "dynamic_complex",
        "make_value": "return {0.5 + (static_cast<double>((i * 17U) % 257U) / 1024.0), -0.25 + (static_cast<double>((i * 29U) % 251U) / 2048.0)};",
        "strategies": ["horner", "dorn2", "dorn4", "dorn5", "dorn6"],
    },
}

STRATEGY_CASES = {
    "horner": "case strategy_t::horner: return run<N>(poly::horner, p, xs, reps);",
    "knuth": "case strategy_t::knuth: return run<N>(poly::knuth, p, xs, reps);",
    "motzkin": "case strategy_t::motzkin: if constexpr (N == 4) { return run<N>(poly::motzkin, p, xs, reps); } else { return std::numeric_limits<double>::quiet_NaN(); }",
    "dorn2": "case strategy_t::dorn2: return run<N>(poly::dorn<2>, p, xs, reps);",
    "dorn4": "case strategy_t::dorn4: return run<N>(poly::dorn<4>, p, xs, reps);",
    "dorn5": "case strategy_t::dorn5: return run<N>(poly::dorn<5>, p, xs, reps);",
    "dorn6": "case strategy_t::dorn6: return run<N>(poly::dorn<6>, p, xs, reps);",
}

PROFILE_VALUES = {
    "generic": "0",
    "generic_x86_64_fma": "0",
    "x86_intel_coffee_lake": "1",
    "x86_intel_icelake_avx512": "2",
    "x86_amd_zen3": "3",
}


def make_poly_code(kind: str) -> str:
    if kind == "static_double":
        return r'''
template <int N, std::size_t... I>
[[nodiscard]] consteval auto make_poly_impl(std::index_sequence<I...>) {
  return poly::literal<static_cast<double>(I + 1U)...>();
}

template <int N> [[nodiscard]] consteval auto make_poly() {
  return make_poly_impl<N>(std::make_index_sequence<static_cast<std::size_t>(N + 1)>{});
}
'''
    if kind == "dynamic_double":
        return r'''
template <int N> [[nodiscard]] auto make_poly() {
  auto p = poly::polynomial_t<coeff_t, N>{};
  std::iota(p.begin(), p.end(), coeff_t{1});
  return p;
}
'''
    if kind == "dynamic_integral":
        return r'''
template <int N> [[nodiscard]] auto make_poly() {
  auto p = poly::polynomial_t<coeff_t, N>{};
  std::iota(p.begin(), p.end(), coeff_t{1});
  return p;
}
'''
    if kind == "dynamic_unsigned":
        return r'''
template <int N> [[nodiscard]] auto make_poly() {
  auto p = poly::polynomial_t<coeff_t, N>{};
  std::iota(p.begin(), p.end(), coeff_t{1});
  return p;
}
'''
    if kind == "dynamic_complex":
        return r'''
template <int N> [[nodiscard]] auto make_poly() {
  auto p = poly::polynomial_t<coeff_t, N>{};
  for (auto i = std::size_t{}; i <= static_cast<std::size_t>(N); ++i)
    p[i] = coeff_t{static_cast<double>(i + 1U),
                   (static_cast<double>(i % 5U) - 2.0) * 0.125};
  return p;
}
'''
    raise KeyError(kind)


def source_for(case_name: str, config: dict) -> str:
    switch_cases = "\n  ".join(STRATEGY_CASES[s] for s in config["strategies"])
    candidate_entries = ", ".join(f"strategy_t::{s}" for s in config["strategies"])
    return textwrap.dedent(f'''
        #include <algorithm>
        #include <array>
        #include <chrono>
        #include <cmath>
        #include <complex>
        #include <cstdint>
        #include <cstdio>
        #include <iterator>
        #include <limits>
        #include <numeric>
        #include <polygnition/poly.hpp>
        #include <ranges>
        #include <string_view>
        #include <type_traits>
        #include <utility>
        #include <vector>

        namespace poly = polygnition::polynomial;

        using coeff_t = {config["coeff_type"]};
        using var_t = {config["var_type"]};

        enum class strategy_t : std::uint8_t {{
          horner,
          motzkin,
          knuth,
          dorn2,
          dorn4,
          dorn5,
          dorn6,
          automatic,
        }};

        struct samples_t {{
          strategy_t strategy;
          std::vector<double> ns_per_eval;
        }};

        struct mann_whitney_t {{
          double u;
          double p_less;
          double common_language_less;
        }};

        template <typename T> inline void do_not_optimize(T const &value) {{
        #if defined(__GNUC__) || defined(__clang__)
          asm volatile("" : : "g"(value) : "memory");
        #else
          (void)value;
        #endif
        }}

        [[nodiscard]] auto name(strategy_t strategy) -> std::string_view {{
          switch (strategy) {{
          case strategy_t::horner: return "horner";
          case strategy_t::motzkin: return "motzkin";
          case strategy_t::knuth: return "knuth";
          case strategy_t::dorn2: return "dorn2";
          case strategy_t::dorn4: return "dorn4";
          case strategy_t::dorn5: return "dorn5";
          case strategy_t::dorn6: return "dorn6";
          case strategy_t::automatic: return "automatic";
          }}
          return "unknown";
        }}

        template <int N> [[nodiscard]] auto candidate_strategies() {{
          auto candidates = std::vector<strategy_t>{{{candidate_entries}}};
          if constexpr (std::is_same_v<coeff_t, double> and std::is_same_v<var_t, double>) {{
            if constexpr (N != 4) {{
              std::erase(candidates, strategy_t::motzkin);
            }}
          }}
          return candidates;
        }}

        {make_poly_code(config["make_poly"])}

        [[nodiscard]] auto make_value(std::size_t i) -> var_t {{
          {config["make_value"]}
        }}

        [[nodiscard]] auto make_values(std::size_t point_count) {{
          auto xs = std::vector<var_t>(point_count);
          for (auto i = std::size_t{{}}; i < point_count; ++i)
            xs[i] = make_value(i);
          return xs;
        }}

        template <typename T>
        [[nodiscard]] auto sample_value(T const &value) -> double {{
          if constexpr (requires {{ value.real(); value.imag(); }}) {{
            return value.real() + value.imag();
          }} else {{
            return static_cast<double>(value);
          }}
        }}

        template <int N, typename Algorithm, typename P>
        [[gnu::noinline]] auto run(Algorithm algorithm, P const &p,
                                   std::vector<var_t> const &xs, int reps) -> double {{
          auto sum = 0.0;
          for (auto r = 0; r < reps; ++r) {{
            for (auto const &x : xs) {{
              sum += sample_value(poly::evaluate(algorithm, p, x));
            }}
          }}
          return sum;
        }}

        template <int N, typename P>
        [[gnu::noinline]] auto run_automatic(P const &p,
                                             std::vector<var_t> const &xs,
                                             int reps) -> double {{
          auto sum = 0.0;
          for (auto r = 0; r < reps; ++r) {{
            for (auto const &x : xs) {{
              sum += sample_value(p(x));
            }}
          }}
          return sum;
        }}

        template <int N, typename P>
        [[nodiscard]] auto run_strategy(strategy_t strategy, P const &p,
                                        std::vector<var_t> const &xs, int reps) -> double {{
          switch (strategy) {{
          {switch_cases}
          case strategy_t::automatic: return run_automatic<N>(p, xs, reps);
          default: return std::numeric_limits<double>::quiet_NaN();
          }}
          return std::numeric_limits<double>::quiet_NaN();
        }}

        template <int N> [[nodiscard]] consteval auto selected_strategy() {{
          using selected_type = poly::detail::selected_algorithm_t<decltype(make_poly<N>()), var_t>;
          if constexpr (std::is_same_v<selected_type, poly::algorithm::motzkin_t>) {{
            return strategy_t::motzkin;
          }} else if constexpr (std::is_same_v<selected_type, poly::algorithm::knuth_t>) {{
            return strategy_t::knuth;
          }} else if constexpr (std::is_same_v<selected_type, poly::algorithm::dorn_t<2>>) {{
            return strategy_t::dorn2;
          }} else if constexpr (std::is_same_v<selected_type, poly::algorithm::dorn_t<4>>) {{
            return strategy_t::dorn4;
          }} else if constexpr (std::is_same_v<selected_type, poly::algorithm::dorn_t<5>>) {{
            return strategy_t::dorn5;
          }} else if constexpr (std::is_same_v<selected_type, poly::algorithm::dorn_t<6>>) {{
            return strategy_t::dorn6;
          }} else {{
            return strategy_t::horner;
          }}
        }}

        [[nodiscard]] auto median(std::vector<double> values) -> double {{
          std::erase_if(values, [](auto value) {{ return std::isnan(value); }});
          auto const n = values.size();
          if (n == 0) {{ return std::numeric_limits<double>::quiet_NaN(); }}
          auto const middle = values.begin() + static_cast<std::ptrdiff_t>(n / 2);
          std::ranges::nth_element(values, middle);
          if (n % 2 != 0) {{ return *middle; }}
          auto const low_middle = std::ranges::max_element(std::ranges::subrange(values.begin(), middle));
          return (*low_middle + *middle) * 0.5;
        }}

        [[nodiscard]] auto normal_cdf(double z) -> double {{
          return 0.5 * std::erfc(-z / std::sqrt(2.0));
        }}

        [[nodiscard]] auto mann_whitney_less(std::vector<double> const &x,
                                             std::vector<double> const &y) -> mann_whitney_t {{
          struct observation_t {{ double value; int group; }};
          auto observations = std::vector<observation_t>{{}};
          observations.reserve(x.size() + y.size());
          std::ranges::transform(
              x, std::back_inserter(observations),
              [](double value) {{ return observation_t{{value, 0}}; }});
          std::ranges::transform(
              y, std::back_inserter(observations),
              [](double value) {{ return observation_t{{value, 1}}; }});
          std::erase_if(observations,
                        [](auto const &o) {{ return std::isnan(o.value); }});
          std::ranges::sort(observations, {{}}, &observation_t::value);
          auto rank_sum_x = 0.0;
          auto tie_sum = 0.0;
          for (auto first = observations.begin(); first != observations.end();) {{
            auto const equal_values = std::ranges::subrange(first, observations.end());
            auto const last = std::ranges::find_if_not(
                equal_values, [value = first->value](auto const &observation) {{ return observation.value == value; }});
            auto const tie_count = static_cast<double>(last - first);
            auto const average_rank = (static_cast<double>((first - observations.begin()) + 1) +
                                       static_cast<double>(last - observations.begin())) * 0.5;
            rank_sum_x += average_rank * static_cast<double>(std::ranges::count_if(
                std::ranges::subrange(first, last), [](auto const &o) {{ return o.group == 0; }}));
            tie_sum += (tie_count * tie_count * tie_count) - tie_count;
            first = last;
          }}
          auto const n_x = static_cast<double>(std::ranges::count_if(observations, [](auto const &o) {{ return o.group == 0; }}));
          auto const n_y = static_cast<double>(std::ranges::count_if(observations, [](auto const &o) {{ return o.group == 1; }}));
          auto const n = n_x + n_y;
          auto const u = rank_sum_x - (n_x * (n_x + 1.0) * 0.5);
          auto const mean = n_x * n_y * 0.5;
          auto const tie_correction = n > 1.0 ? tie_sum / (n * (n - 1.0)) : 0.0;
          auto const variance = n_x * n_y * ((n + 1.0) - tie_correction) / 12.0;
          auto const z = variance > 0.0 ? (u - mean + 0.5) / std::sqrt(variance) : 0.0;
          return {{.u = u,
                   .p_less = variance > 0.0 ? normal_cdf(z) : 1.0,
                   .common_language_less = (n_x * n_y) > 0.0 ? 1.0 - (u / (n_x * n_y)) : 0.0}};
        }}

        template <int N, typename P>
        [[nodiscard]] auto ns_per_eval(strategy_t strategy, P const &p,
                                       std::vector<var_t> const &xs, int reps) -> double {{
          auto const evaluations = static_cast<std::int64_t>(xs.size()) * reps;
          auto const start = std::chrono::steady_clock::now();
          auto const sum = run_strategy<N>(strategy, p, xs, reps);
          auto const stop = std::chrono::steady_clock::now();
          do_not_optimize(sum);
          return std::chrono::duration<double, std::nano>(stop - start).count() /
                 static_cast<double>(evaluations);
        }}

        template <int N>
        auto probe(std::vector<var_t> const &xs, int reps, int sample_count) -> void {{
          auto candidates = candidate_strategies<N>();
          auto strategies = candidates;
          strategies.push_back(strategy_t::automatic);
          auto const p = make_poly<N>();
          auto rows = std::vector<samples_t>{{}};
          rows.reserve(strategies.size());
          std::ranges::transform(
              strategies, std::back_inserter(rows),
              [](auto strategy) {{ return samples_t{{.strategy = strategy, .ns_per_eval = {{}}}}; }});
          std::ranges::for_each(rows, [sample_count](auto &row) {{ row.ns_per_eval.reserve(static_cast<std::size_t>(sample_count)); }});
          std::ranges::for_each(strategies, [&](auto strategy) {{ do_not_optimize(run_strategy<N>(strategy, p, xs, 1)); }});
          for (auto sample = 0; sample < sample_count; ++sample) {{
            auto order = strategies;
            std::ranges::rotate(order, order.begin() + static_cast<std::ptrdiff_t>(sample % static_cast<int>(order.size())));
            if (sample % 2 != 0) {{ std::ranges::reverse(order); }}
            for (auto strategy : order) {{
              auto const sample_ns = ns_per_eval<N>(strategy, p, xs, reps);
              auto const row = std::ranges::find_if(rows, [strategy](auto const &r) {{ return r.strategy == strategy; }});
              row->ns_per_eval.push_back(sample_ns);
            }}
          }}
          auto const median_for = [&](strategy_t strategy) {{
            auto const row = std::ranges::find_if(rows, [strategy](auto const &r) {{ return r.strategy == strategy; }});
            return row == rows.end() ? std::numeric_limits<double>::quiet_NaN() : median(row->ns_per_eval);
          }};
          auto const samples_for = [&](strategy_t strategy) -> std::vector<double> const & {{
            auto const row = std::ranges::find_if(rows, [strategy](auto const &r) {{ return r.strategy == strategy; }});
            return row->ns_per_eval;
          }};
          auto const selected = selected_strategy<N>();
          auto const best = *std::ranges::min_element(candidates, [&](auto a, auto b) {{ return median_for(a) < median_for(b); }});
          auto const has_horner = std::ranges::find(candidates, strategy_t::horner) != candidates.end();
          auto const best_vs_selected = best == selected ? mann_whitney_t{{.u = 0.0, .p_less = 0.5, .common_language_less = 0.5}} : mann_whitney_less(samples_for(best), samples_for(selected));
          auto const selected_vs_horner = !has_horner ? mann_whitney_t{{.u = 0.0, .p_less = std::numeric_limits<double>::quiet_NaN(), .common_language_less = std::numeric_limits<double>::quiet_NaN()}} : (selected == strategy_t::horner ? mann_whitney_t{{.u = 0.0, .p_less = 0.5, .common_language_less = 0.5}} : mann_whitney_less(samples_for(selected), samples_for(strategy_t::horner)));
          auto const automatic_vs_selected = selected == strategy_t::automatic ? mann_whitney_t{{.u = 0.0, .p_less = 0.5, .common_language_less = 0.5}} : mann_whitney_less(samples_for(strategy_t::automatic), samples_for(selected));
          auto const selected_median = median_for(selected);
          auto const horner_median = has_horner ? median_for(strategy_t::horner) : std::numeric_limits<double>::quiet_NaN();
          auto const automatic_median = median_for(strategy_t::automatic);
          std::printf(
              "{case_name},%d,%.*s,%.*s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\\n",
              N,
              static_cast<int>(name(selected).size()), name(selected).data(),
              static_cast<int>(name(best).size()), name(best).data(),
              median_for(strategy_t::horner), median_for(strategy_t::knuth),
              median_for(strategy_t::motzkin), median_for(strategy_t::dorn2),
              median_for(strategy_t::dorn4), median_for(strategy_t::dorn5),
              median_for(strategy_t::dorn6), automatic_median,
              best_vs_selected.p_less, selected_vs_horner.p_less,
              automatic_vs_selected.p_less, best_vs_selected.common_language_less,
              selected_vs_horner.common_language_less,
              automatic_vs_selected.common_language_less,
              median_for(best) / selected_median,
              selected_median / horner_median,
              automatic_median / selected_median);
        }}

        template <int... N>
        auto probe_all(std::integer_sequence<int, N...>, std::vector<var_t> const &xs,
                       int reps, int sample_count) -> void {{
          (probe<N>(xs, reps, sample_count), ...);
        }}

        int main(int argc, char **argv) {{
          auto const point_count = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : std::size_t{{256}};
          auto const reps = argc > 2 ? std::atoi(argv[2]) : 16;
          auto const sample_count = argc > 3 ? std::atoi(argv[3]) : 17;
          auto const xs = make_values(point_count);
          std::puts("category,degree,selected,best,horner_ns,knuth_ns,motzkin_ns,dorn2_ns,dorn4_ns,dorn5_ns,dorn6_ns,auto_ns,p_best_lt_selected,p_selected_lt_horner,p_auto_lt_selected,cl_best_lt_selected,cl_selected_lt_horner,cl_auto_lt_selected,ratio_best_to_selected,ratio_selected_to_horner,ratio_auto_to_selected");
          probe_all(std::make_integer_sequence<int, 15>{{}}, xs, reps, sample_count);
          return 0;
        }}
    ''')


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--include", default="include")
    parser.add_argument("--build-dir", default="build/low-degree-selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--point-count", type=int, default=256)
    parser.add_argument("--reps", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=17)
    parser.add_argument("--target-profile", choices=sorted(PROFILE_VALUES))
    args = parser.parse_args()

    build_dir = Path(args.build_dir) / args.case
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "probe.cpp"
    binary = build_dir / "probe"
    source.write_text(source_for(args.case, CASES[args.case]))
    profile_define = []
    if args.target_profile is not None:
        profile_define = [
            f"-DPOLYGNITION_TARGET_PROFILE={PROFILE_VALUES[args.target_profile]}"
        ]
    run([
        args.cxx,
        "-std=c++23",
        "-O3",
        "-march=native",
        *profile_define,
        "-I",
        args.include,
        str(source),
        "-o",
        str(binary),
    ])
    completed = subprocess.run(
        [str(binary), str(args.point_count), str(args.reps), str(args.sample_count)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    Path(args.output).write_text(completed.stdout)


if __name__ == "__main__":
    main()
