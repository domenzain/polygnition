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
  if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Boost.UT guards constructor-priority setup only through this platform macro;
    # GCC on Darwin rejects constructor priorities as well.
    target_compile_definitions(${name} PRIVATE __EMSCRIPTEN__)
  endif()
  target_link_libraries(${name} PRIVATE
    polygnition::polygnition
    polygnition::dev
    Boost::ut
  )
  add_test(NAME ${name} COMMAND ${name})
endfunction()

polygnition_add_test(test_polynomial_api)
polygnition_add_test(test_polynomial_arithmetic)
polygnition_add_test(test_evaluation)
polygnition_add_test(test_knuth_eve)
polygnition_add_test(test_accuracy_contracts)
polygnition_add_test(test_algorithm_selection)
polygnition_add_test(test_taylor_series)

find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
  add_test(
    NAME test_profile_tuning_generated
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/check_profile_tuning_generated.py
            --python ${Python3_EXECUTABLE}
            --script ${PROJECT_SOURCE_DIR}/benchmark/generate_profile_tuning.py
            --inc ${PROJECT_SOURCE_DIR}/include/polygnition/detail/profile_tuning.inc
  )
endif()
