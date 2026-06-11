# `polygnition`
A fast `constexpr` oriented dense polynomial library.

# Build
Use CMake with Ninja.

# Test
ENABLE_TESTING=ON will build tests.
Run them with ctest at build dir.

# Performance
BENCHMARK=ON will build benchmarks.
Run them when required to iterate on speed or where comparisons between alternatives.

# Bundle
Create agent bundle with `git bundle create polygnition-agent.bundle refs/heads/dev refs/tags/boost-ext-ut/v2.3.1`.

# Verification
No change is complete without:
```shell
cmake -S . -B build -G Ninja && ninja -C build && ctest --test-dir build
```
