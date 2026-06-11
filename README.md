# polygnition

`polygnition` is a small, header-only C++23 library for dense polynomials that
need to disappear into the generated code.

Give it coefficients. Call `p(x)`. When the compiler can see the degree,
coefficient type, argument type, and target profile, `polygnition` selects an
evaluation strategy at compile time: Horner for the small and general case,
Dorn decompositions for wider dependency graphs, Knuth's recurrence for real
polynomials at complex points, Motzkin's degree-4 form for safe static quartics,
and multivariate Horner when the coefficients are polynomials too. Estrin and a
compensated Horner path are exposed as explicit strategies when dependency-tree
shape or tighter forward error matters more than the default.

```cpp
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;

constexpr auto p = poly::literal<2.0, -3.0, 0.5, 7.0>(); // 2x³ - 3x² + 0.5x + 7
static_assert(p(2.0) == 12.0);                          // automatic strategy

constexpr auto same_value = poly::evaluate(poly::horner, p, 2.0);
static_assert(same_value == p(2.0));
```

Coefficients are stored in mathematical order: `c_n, c_{n-1}, ..., c_0`.

## Why care this much about polynomial evaluation?

Because in the hot paths below, the polynomial is not a line item. It is the
inner loop.

### 1. Real-time lens models: half a billion evaluations per second

A 3840×2160 stream at 60 Hz contains 497,664,000 pixels per second. If a
wide-angle camera pipeline applies one radial scale polynomial per pixel, the
budget is already about half a billion polynomial evaluations per second before
stereo, 360° seams, supersampling, iterative inverse maps, or multiple streams.

Radial distortion models are naturally polynomial. With `q = x² + y²`, a common
scale has the shape

```text
s(q) = 1 + k1 q + k2 q² + k3 q³ + k4 q⁴
```

`polygnition` lets that become a static degree-4 polynomial. Automatic dispatch
selects the Motzkin path for this exact case when the constexpr Motzkin
precomputation gate says the coefficient shape is numerically safe; otherwise it
falls back to the table's general quartic row.

```cpp
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;

struct normalized_xy {
  double x;
  double y;
};

struct distorted_xy {
  double x;
  double y;
};

[[nodiscard]] constexpr auto distort(normalized_xy const p) -> distorted_xy {
  // s(q) = 1 + k1 q + k2 q² + k3 q³ + k4 q⁴
  // coefficients are high-to-low: k4, k3, k2, k1, 1
  constexpr auto radial_scale = poly::literal<0.001621, -0.009271, 0.026731,
                                              -0.041442, 1.0>();
  auto const q = (p.x * p.x) + (p.y * p.y);
  auto const s = radial_scale(q);
  return {.x = p.x * s, .y = p.y * s};
}
```

If the coefficients are calibrated at runtime, preprocess once and keep the
pixel loop to the evaluation itself:

```cpp
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;

struct radial_model_t {
  poly::motzkin_preprocessed_t<double> scale;

  [[nodiscard]] constexpr auto operator()(double const q) const -> double {
    return scale(q);
  }
};

[[nodiscard]] constexpr auto make_radial_model(double const k1, double const k2,
                                               double const k3, double const k4)
    -> radial_model_t {
  return {.scale = poly::preprocess_motzkin(
              poly::polynomial_t{k4, k3, k2, k1, 1.0})};
}
```

### 2. Filter sweeps: complex polynomial ratios on the unit circle

Interactive EQs, loudspeaker crossovers, control loops, modem filters, and DSP
design tools all ask the same question over and over:

```text
H(e^{-jω}) = B(e^{-jω}) / A(e^{-jω})
```

Each sample on the frequency grid needs numerator and denominator polynomial
evaluations at a complex point. A design-space search can multiply that by
thousands of candidate filters; a responsive UI can do it repeatedly while a
user drags a control.

For real coefficients at a complex point, `polygnition` automatically uses
Knuth's recurrence instead of generic complex Horner evaluation.

```cpp
#include <cmath>
#include <complex>
#include <polygnition/poly.hpp>

namespace poly = polygnition::polynomial;

struct response_t {
  std::complex<double> h;
  double magnitude_db;
};

[[nodiscard]] auto response_at(double const radians_per_sample) -> response_t {
  // freqz-style coefficient order is b0 + b1 z⁻¹ + b2 z⁻².
  // polygnition wants high-to-low powers of u = z⁻¹: b2, b1, b0.
  constexpr auto b = poly::literal<0.206572083826147,
                                   0.413144167652294,
                                   0.206572083826147>();
  constexpr auto a = poly::literal<0.195815712655833,
                                   -0.369527377351241,
                                   1.0>();

  auto const u = std::complex<double>{std::cos(radians_per_sample),
                                      -std::sin(radians_per_sample)};
  auto const h = b(u) / a(u);
  return {.h = h, .magnitude_db = 20.0 * std::log10(std::abs(h))};
}
```

## API sketch

```cpp
#include <polygnition/arithmetic.hpp>
#include <polygnition/poly.hpp>
#include <polygnition/taylor.hpp>
#include <array>
#include <span>
#include <type_traits>

namespace poly = polygnition::polynomial;

constexpr auto static_p = poly::literal<1.0, -2.0, 3.0>();
auto const runtime_p = poly::polynomial_t{1.0, -2.0, 3.0};

auto const automatic_value = static_p(0.25);
auto const zen3_value = poly::on<polygnition::target::profile::x86_amd_zen3>(static_p)(0.25);
auto const horner_value = poly::evaluate(poly::horner, static_p, 0.25);
auto const dorn_value = poly::evaluate(poly::dorn<4>, static_p, 0.25);
auto const estrin_value = poly::evaluate(poly::estrin, static_p, 0.25);
auto const compensated_value = poly::evaluate(poly::compensated, static_p, 0.25);

static_assert(static_p.coeff<2>() == 1.0); // coefficient of x²
auto const c1 = static_p.coeff(1);         // coefficient of x¹, or zero out of range
auto const forward_error_bound = poly::error_bound(static_p, 1.0);

auto const sum = static_p + runtime_p;
auto const product = static_p * runtime_p;
auto const derivative = poly::derivative(static_p);

auto xs = std::array{0.0, 0.25, 0.5, 0.75};
auto ys = std::array<double, xs.size()>{};
poly::evaluate(poly::automatic, static_p, xs, ys); // contiguous ranges are accepted

auto const dividend = poly::polynomial_t{1.0, 0.0, -1.0};
auto const divisor = poly::polynomial_t{1.0, -1.0};
auto const [quotient, rem] = poly::divide(dividend, divisor);

constexpr auto expansion = poly::series_expansion<1>(poly::literal<1, 2, 3>(), 2.0);
static_assert(std::is_same_v<decltype(expansion), poly::taylor_series<double, 1> const>);
static_assert(expansion.derivative<0>() == 11.0);
static_assert(expansion.derivative<1>() == 6.0);
```

## Evaluation strategies

| Strategy | Use it for |
| --- | --- |
| `p(x)` / `poly::evaluate(poly::automatic, p, x)` | Compile-time latency selection by degree, coefficient type, argument type, storage kind, element width, lane count, coefficient shape, and target profile. `poly::on<Profile>(p)(x)` and `poly::evaluate(poly::tuned<Profile>, p, x)` make the profile part of the call type. |
| `poly::horner` | Small degrees, integral rings, custom rings, and the boring reliable baseline. |
| `poly::estrin` | Explicit shorter dependency trees by recursively evaluating coefficient pairs in powers of `x²`. |
| `poly::dorn<K>` | Medium and high-degree floating-point polynomials where extra independent multiply chains expose instruction-level parallelism. |
| `poly::knuth` | Real-coefficient polynomials evaluated at `std::complex<T>` points. |
| `poly::motzkin` | Degree-4 floating-point polynomials, especially static quartics whose constexpr cancellation estimate is safe or runtime quartics preprocessed once with `poly::preprocess_motzkin`. |
| `poly::compensated` | Error-free-transform Horner for users who want a high-accuracy explicit path while staying constexpr-friendly in C++23. |
| `poly::multivariate_horner` | Polynomials whose coefficients are themselves callable polynomials. |

The selection table is data-driven. `selection_key` includes value categories,
storage, intent, degree, element widths, SIMD lane count, and coefficient shape;
Knuth, Motzkin, leading-zero demotion, Dorn, Estrin, and Horner rows all pass
through the same `select_t` customization point. `benchmark/generate_profile_tuning.py`
regenerates the checked-in `include/polygnition/detail/profile_tuning.inc`; the
`test_profile_tuning_generated` ctest diffs the generated output against the
header so probes and tables cannot silently drift.

The built-in target profiles are in `polygnition/target.hpp`:

| `POLYGNITION_TARGET_PROFILE` | Profile |
| ---: | --- |
| `0` | generic / generic x86-64 FMA |
| `1` | Intel Coffee Lake |
| `2` | Intel Ice Lake AVX-512 |
| `3` | AMD Zen 3 |

Example:

```sh
c++ -std=c++23 -O3 -march=native -DPOLYGNITION_TARGET_PROFILE=3 \
  -I include app.cpp
```

## C++23 implementation notes

The public surface remains intentionally small, but the implementation now leans
on C++23 where it removes bespoke machinery: static literals are zero-storage
`static_poly<polynomial_t{...}>` values, tuning-table lookup validates strategy
conformance before falling back to Horner, contiguous range batch overloads adapt
to spans, and small compile-time loops use shared `static_for`/`for_each_index`
helpers. The stable-evaluation algorithms remain conservative: faster forms are
selected only when their conditioning and preprocessing assumptions are explicit.

## Algebra helpers

`polygnition/arithmetic.hpp` provides dense polynomial arithmetic while keeping
the static storage degree visible in the type. The arithmetic operators share
one coefficient-kernel/materialization path for `static_poly` values, runtime
arrays, and mixed operands; they preserve the computed storage degree until
`poly::trim(p)` is requested for static compile-time coefficients. `operator==` is structural: the
static degree and every stored coefficient are part of the representation and
comparison. Use `poly::same_polynomial(a, b)` when extra stored leading zeros
should be ignored.

Batch evaluation is available through the same CPO:
`poly::evaluate(strategy, p, xs, out)` for sized contiguous ranges, or the
explicit `std::span` form when that is clearer. `xs.size() == out.size()` is a
precondition enforced by the library; mismatches throw `std::invalid_argument`
rather than silently truncating. The generic adapter is constrained on scalar
evaluability, so explicit strategies, preprocessed strategy objects, and
user-defined strategy tags all get a batch path without per-strategy boilerplate.
For floating-point spans the implementation blocks through the owned
`lanes<T, W>` type and an unroll factor selected by the throughput table. The
table is keyed by element width, lane count, degree, coefficient shape, and
target profile, and every row is checked against the actual lane value type
before it can be selected. If no tuned batch row applies, automatic batch
evaluation falls back to the validated scalar strategy with the default lane
width for the element type.

`lanes<T, W>` uses compiler vector storage only when the current translation
unit's ISA macros make that width native; otherwise it uses array storage with
the same semantics. That means `POLYGNITION_TARGET_PROFILE` can request an
Ice Lake 8-lane double row while a non-AVX-512 `-march` still compiles and runs
correctly, just without native 512-bit storage.

For real-coefficient filter sweeps at split complex points, Knuth also has a
structure-of-arrays batch entry point:

```cpp
constexpr auto b = poly::literal<0.206572083826147,
                                 0.413144167652294,
                                 0.206572083826147>();
auto wr = std::array<double, 1024>{};
auto wi = std::array<double, 1024>{};
auto hr = std::array<double, 1024>{};
auto hi = std::array<double, 1024>{};
poly::evaluate(poly::knuth, b, wr, wi, hr, hi);
```

`poly::effective_degree(p)` returns the mathematical degree and reports `-1` for
the zero polynomial. `poly::derivative(p)` returns the formal derivative.
`poly::trim(p)` removes stored leading zeroes from `static_poly` values; for
runtime polynomials it is an identity copy because the storage extent is already
part of the value object.

`poly::divide(u, f)` performs quotient/remainder long division over
`poly::field_coefficient` domains: floating-point scalars and complex
floating-point scalars. The divisor is a caller precondition: a
`polynomial_t<T, M>` passed to `divide` must actually have degree `M`, so its
stored leading coefficient must be non-zero.

## Accuracy contracts

`poly::error_bound(p, xmax)` exposes a constexpr Horner forward-error bound of
the form `γ₂ₙ · Σ|cᵢ||xmax|ⁱ`. Literal polynomials can use it in
`static_assert`s, and the test suite checks it against a constexpr double-double oracle.
Static quartic selection also computes Motzkin preprocessing diagnostics in
constant evaluation and only lets the automatic table pick Motzkin for shapes
that pass that cancellation gate.

The `poly::compensated` strategy implements an error-free-transform Horner path
using Dekker `two_prod`/`two_sum`, avoiding a dependency on `std::fma` for
constexpr C++23. `test_accuracy_contracts` keeps a small corpus of randomish,
Wilkinson-like, and clustered-root polynomials and asserts per-strategy accuracy
envelopes. Near clustered roots it uses absolute forward-error envelopes because
ULP counts around zero mostly measure exponent-bin changes rather than useful
relative accuracy.

## Taylor series

`polygnition/taylor.hpp` exposes `poly::taylor_series<T, Order>` and the common
first-order alias `poly::dual<T>`. Coefficients are normalized Taylor
coefficients, stored high-to-low: `[a_N, ..., a_1, value]`, where
`a_k = f⁽ᵏ⁾(x) / k!`.

`poly::series_expansion<Order>(p, x)` evaluates `p` at `x + ε` and returns a
`taylor_series`, preserving the series type through scalar operations and all
explicit evaluation strategies.

## Add it to a CMake project

```cmake
add_subdirectory(polygnition)
target_link_libraries(app PRIVATE polygnition::polygnition)
```

No runtime library is produced; the target is an interface target exporting the
`include/` directory and C++23 requirement.

## Test and benchmark

```sh
cmake -S . -B build -G Ninja -DENABLE_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The lightweight benchmark probes are opt-in:

```sh
cmake -S . -B build-bench -G Ninja -DBENCHMARK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench --target algorithm_selection_probe batch_throughput_probe
./build-bench/benchmark/algorithm_selection_probe
./build-bench/benchmark/batch_throughput_probe
python3 benchmark/generate_profile_tuning.py --check
```

`batch_throughput_probe` prints CSV rows with `profile`, `element_bits`,
`lanes`, and `unroll` columns. To refresh measured throughput rows for a
profile, compile the probe with the matching `POLYGNITION_TARGET_PROFILE`, then
feed its CSV back into the generator:

```sh
./build-bench/benchmark/batch_throughput_probe > batch.csv
python3 benchmark/generate_profile_tuning.py --batch-csv batch.csv \
  --output include/polygnition/detail/profile_tuning.inc
```

Google Benchmark based suites are built only when `benchmark::benchmark` is
available or `POLYGNITION_FETCH_GOOGLE_BENCHMARK=ON` is supplied.

## License

MIT. See [LICENSE](LICENSE).
