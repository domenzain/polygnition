option(POLYGNITION_FETCH_GOOGLE_BENCHMARK
       "Fetch Google Benchmark when benchmark::benchmark is not available" OFF)

function(polygnition_target_benchmark_options target)
  target_compile_options(${target} PRIVATE -O3)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
      CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(${target} PRIVATE -march=native)
  endif()
endfunction()

add_executable(algorithm_selection_probe algorithm_selection_probe.cpp)
polygnition_target_benchmark_options(algorithm_selection_probe)
target_link_libraries(algorithm_selection_probe PRIVATE polygnition::polygnition)

add_executable(batch_throughput_probe batch_throughput_probe.cpp)
polygnition_target_benchmark_options(batch_throughput_probe)
target_link_libraries(batch_throughput_probe PRIVATE
  polygnition::polygnition
  polygnition::dev
)

if (NOT TARGET benchmark::benchmark AND POLYGNITION_FETCH_GOOGLE_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING off)
  include(FetchContent)
  FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG origin/main
  )
  FetchContent_MakeAvailable(googlebenchmark)
endif()

if (TARGET benchmark::benchmark)
  add_executable(real_coeff_complex_point real_coeff_complex_point.cpp)
  polygnition_target_benchmark_options(real_coeff_complex_point)
  target_link_libraries(real_coeff_complex_point PRIVATE
    polygnition::polygnition
    polygnition::dev
    benchmark::benchmark_main
  )

  add_executable(real_coeff_real_point real_coeff_real_point.cpp)
  polygnition_target_benchmark_options(real_coeff_real_point)
  target_link_libraries(real_coeff_real_point PRIVATE
    polygnition::polygnition
    polygnition::dev
    benchmark::benchmark_main
  )
else()
  message(STATUS
          "Google Benchmark targets skipped; set "
          "POLYGNITION_FETCH_GOOGLE_BENCHMARK=ON or provide "
          "benchmark::benchmark to build them")
endif()
