cmake_minimum_required(VERSION 3.26)
project(polygnition VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_TESTING "Build tests" OFF)
option(BENCHMARK "Build benchmarks" OFF)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

if (ENABLE_TESTING OR BENCHMARK)
  add_library(${PROJECT_NAME}_dev INTERFACE)
  add_library(${PROJECT_NAME}::dev ALIAS ${PROJECT_NAME}_dev)
  target_include_directories(${PROJECT_NAME}_dev INTERFACE internal)
  target_compile_features(${PROJECT_NAME}_dev INTERFACE cxx_std_23)
endif()

if (ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if (BENCHMARK)
  add_subdirectory(benchmark)
endif()
