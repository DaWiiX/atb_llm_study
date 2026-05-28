cmake_minimum_required(VERSION 3.16)
project(atb_cpp_llm_engine LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Ascend paths ──────────────────────────────────────────
if(NOT DEFINED ATB_DIR)
    set(ATB_DIR "/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1")
endif()
if(NOT DEFINED ASCEND_TOOLKIT_DIR)
    set(ASCEND_TOOLKIT_DIR "/usr/local/Ascend/cann-9.0.0/aarch64-linux")
endif()

# ── Third-party (cJSON) ──────────────────────────────────
add_library(thirdparty STATIC utils/cJSON.c)
target_include_directories(thirdparty PUBLIC utils)

# ── Core library ──────────────────────────────────────────
file(GLOB_RECURSE ENGINE_SRCS
    src/core/*.cpp
    src/io/*.cpp
    src/log/*.cpp
    src/engine/*.cpp
    src/ops/*.cpp
    src/components/*.cpp
    src/layers/*.cpp
    src/models/*.cpp
    src/adapters/*.cpp
)

add_library(atb_llm_engine SHARED ${ENGINE_SRCS})

target_include_directories(atb_llm_engine
    PUBLIC
        include
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)

target_link_directories(atb_llm_engine
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)

target_link_libraries(atb_llm_engine
    PRIVATE
        thirdparty
        ascendcl
        atb
)

# ── Tests ─────────────────────────────────────────────────
enable_testing()

add_executable(test_core tests/test_core.cpp)
target_include_directories(test_core
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(test_core
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_core PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_core COMMAND test_core)

add_executable(test_text_ops tests/test_text_ops.cpp)
target_include_directories(test_text_ops
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(test_text_ops
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_text_ops PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_text_ops COMMAND test_text_ops)

add_executable(test_vision_ops tests/test_vision_ops.cpp)
target_include_directories(test_vision_ops
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(test_vision_ops
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_vision_ops PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_vision_ops COMMAND test_vision_ops)

add_executable(test_text_model tests/test_text_model.cpp)
target_include_directories(test_text_model
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(test_text_model
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_text_model PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_text_model COMMAND test_text_model)

add_executable(test_e2e tests/test_e2e.cpp)
target_include_directories(test_e2e
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(test_e2e
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_e2e PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_e2e COMMAND test_e2e)

add_executable(test_consistency tests/test_consistency.cpp)
target_include_directories(test_consistency
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(test_consistency
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_consistency PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_consistency COMMAND test_consistency)

add_executable(benchmark tests/benchmark.cpp)
target_include_directories(benchmark
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(benchmark
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(benchmark PRIVATE atb_llm_engine atb ascendcl)

add_executable(bench_mm tests/bench_mm.cpp)
target_include_directories(bench_mm
    PRIVATE
        src
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(bench_mm
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(bench_mm PRIVATE atb_llm_engine atb ascendcl)

# ── Install ───────────────────────────────────────────────
install(TARGETS atb_llm_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/atb_llm DESTINATION include)
