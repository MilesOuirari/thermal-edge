cmake_minimum_required(VERSION 3.16)
project(edge-thermal-ai VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# ── TFLite (optional — stubs used when not available) ─────────────────
find_library(TFLITE_LIB tensorflowlite
    PATHS /usr/local/lib /opt/tflite/lib)

add_library(tflite_engine STATIC inference/tflite_engine.cpp)
target_include_directories(tflite_engine PUBLIC inference)

if(TFLITE_LIB)
    target_compile_definitions(tflite_engine PRIVATE HAVE_TFLITE)
    target_include_directories(tflite_engine PRIVATE /opt/tflite/include)
    target_link_libraries(tflite_engine PRIVATE ${TFLITE_LIB})
    message(STATUS "TFLite: found at ${TFLITE_LIB}")
else()
    message(STATUS "TFLite: not found, using inference stub")
endif()

# ── Main pipeline executable ──────────────────────────────────────────
add_executable(thermal_guardian pipeline/thermal_guardian.cpp)
target_link_libraries(thermal_guardian PRIVATE tflite_engine)

# ── Tests ─────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    find_package(Catch2 QUIET)
    if(Catch2_FOUND)
        enable_testing()
        add_executable(test_inference tests/test_inference.cpp)
        target_link_libraries(test_inference PRIVATE tflite_engine Catch2::Catch2)
        add_test(NAME inference COMMAND test_inference)
    endif()
endif()
