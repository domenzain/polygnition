include(FetchContent)

if (EXISTS "${PROJECT_SOURCE_DIR}/.git")
  set(POLYGNITION_UT_DEFAULT_REPOSITORY "file://${PROJECT_SOURCE_DIR}")
  set(POLYGNITION_UT_DEFAULT_TAG "boost-ext-ut/v2.3.1")
else()
  set(POLYGNITION_UT_DEFAULT_REPOSITORY "https://github.com/boost-ext/ut.git")
  set(POLYGNITION_UT_DEFAULT_TAG "v2.3.1")
endif()

set(POLYGNITION_UT_REPOSITORY "${POLYGNITION_UT_DEFAULT_REPOSITORY}"
    CACHE STRING "Boost.UT repository used by FetchContent")
set(POLYGNITION_UT_TAG "${POLYGNITION_UT_DEFAULT_TAG}"
    CACHE STRING "Boost.UT tag used by FetchContent")

FetchContent_Declare(
  ut
  GIT_REPOSITORY ${POLYGNITION_UT_REPOSITORY}
  GIT_TAG ${POLYGNITION_UT_TAG}
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ut)

function(polygnition_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE polygnition::polygnition Boost::ut)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

polygnition_add_test(test_polynomial_api)
polygnition_add_test(test_evaluation)
polygnition_add_test(test_polynomial_arithmetic)
polygnition_add_test(test_taylor_series)
polygnition_add_test(test_knuth_eve)
