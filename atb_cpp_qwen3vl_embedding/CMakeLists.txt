cmake_minimum_required(VERSION 3.16)
project(atb_cpp_llm_engine LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ───────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Build type" FORCE)

# Common flags applied to all configs
add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -DDEBUG)
    add_compile_definitions(ATB_DEBUG=1)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2 -DNDEBUG)
    add_compile_definitions(ATB_RELEASE=1)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Ascend paths ──────────────────────────────────────────
# Try find_package(CANN) first; fall back to manual paths.
find_package(CANN QUIET)
if(CANN_FOUND)
    message(STATUS "Found CANN via find_package")
    set(ATB_DIR "${CANN_ATB_DIR}" CACHE PATH "ATB install dir")
    set(ASCEND_TOOLKIT_DIR "${CANN_TOOLKIT_DIR}" CACHE PATH "Ascend toolkit dir")
else()
    if(NOT DEFINED ATB_DIR)
        set(ATB_DIR "/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1" CACHE PATH "ATB install dir")
    endif()
    if(NOT DEFINED ASCEND_TOOLKIT_DIR)
        set(ASCEND_TOOLKIT_DIR "/usr/local/Ascend/cann-9.0.0/aarch64-linux" CACHE PATH "Ascend toolkit dir")
    endif()
endif()

# ── Third-party (cJSON) ──────────────────────────────────
add_library(thirdparty STATIC utils/cJSON.c)
target_include_directories(thirdparty PUBLIC utils)

# ── Core library (explicit source list) ──────────────────
set(ENGINE_SRCS
    # core
    src/core/npu_tensor.cpp
    src/core/tensor_allocator.cpp
    src/core/context_manager.cpp
    src/core/buffer_pool.cpp
    src/core/graph_builder.cpp
    # io
    src/io/safetensors_reader.cpp
    src/io/weight_loader.cpp
    src/io/weight_helpers.cpp
    src/io/json_config.cpp
    # log
    src/log/logger.cpp
    # engine
    src/engine/runtime_impl.cpp
    src/engine/model_registry.cpp
    src/engine/llm_engine.cpp
    # ops
    src/ops/split_op.cpp
    src/ops/transpose_op.cpp
    src/ops/set_value_op.cpp
    src/ops/rope_op.cpp
    src/ops/softmax_op.cpp
    src/ops/self_attention_op.cpp
    src/ops/rms_norm_op.cpp
    src/ops/linear_op.cpp
    src/ops/layer_norm_op.cpp
    src/ops/gather_op.cpp
    src/ops/reduce_op.cpp
    src/ops/concat_op.cpp
    src/ops/elewise_op.cpp
    src/ops/activation_op.cpp
    # components/common
    src/components/common/self_attention_graph.cpp
    src/components/common/swiglu_mlp_graph.cpp
    src/components/common/gqa_attention_builder.cpp
    src/components/common/mlp_builder.cpp
    src/components/common/rms_norm_graph.cpp
    src/components/common/mrope.cpp
    src/components/common/deepstack_fusion.cpp
    # components/text
    src/components/text/decoder_layer_graph.cpp
    # components/vision
    src/components/vision/vision_mlp_graph.cpp
    src/components/vision/vision_merger_graph.cpp
    src/components/vision/vision_block_graph.cpp
    src/components/vision/vision_attention_graph.cpp
    src/components/vision/patch_embed_graph.cpp
    src/components/vision/deepstack_graph.cpp
    src/components/vision/pos_embed_interp.cpp
    # runners
    src/runners/vision_runner.cpp
    src/runners/text_runner.cpp
    # families
    src/families/base_model.cpp
    # adapters
    src/adapters/qwen3vl_embedding/qwen3vl_model.cpp
    src/adapters/qwen3vl_embedding/qwen3vl_weights.cpp
    src/adapters/qwen3vl_embedding/qwen3vl_preprocess.cpp
    src/adapters/qwen3vl_embedding/register.cpp
    src/adapters/qwen3vl_embedding/qwen3vl_config.cpp
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

# ── Helper: add a test/bench executable ───────────────────
function(add_atb_test name source)
    add_executable(${name} ${source})
    target_include_directories(${name}
        PRIVATE
            src
            utils
            ${ATB_DIR}/include
            ${ASCEND_TOOLKIT_DIR}/include
    )
    target_link_directories(${name}
        PRIVATE
            ${ATB_DIR}/lib
            ${ASCEND_TOOLKIT_DIR}/lib64
    )
    target_link_libraries(${name} PRIVATE atb_llm_engine atb ascendcl)
    # Register as CTest target unless it's a benchmark
    if(NOT name MATCHES "^bench_")
        add_test(NAME ${name} COMMAND ${name})
    endif()
endfunction()

# ── Tests ─────────────────────────────────────────────────
enable_testing()

add_atb_test(test_core                  tests/test_core.cpp)
add_atb_test(test_text_ops             tests/test_text_ops.cpp)
add_atb_test(test_vision_ops           tests/test_vision_ops.cpp)
add_atb_test(test_text_model           tests/test_text_model.cpp)
add_atb_test(test_e2e                  tests/test_e2e.cpp)
add_atb_test(test_consistency          tests/test_consistency.cpp)
add_atb_test(test_accuracy             tests/test_accuracy.cpp)
add_atb_test(test_forward_error_paths  tests/test_forward_error_paths.cpp)

# doctest-based IO/Adapter tests (needs tests/ for doctest.h)
add_executable(test_io_adapters tests/test_io_adapters.cpp)
target_include_directories(test_io_adapters
    PRIVATE
        src
        utils
        tests
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_link_directories(test_io_adapters
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_io_adapters PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_io_adapters COMMAND test_io_adapters)

# Benchmarks (not registered as CTest tests)
add_atb_test(benchmark                 tests/benchmark.cpp)

# ── Install ───────────────────────────────────────────────
install(TARGETS atb_llm_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/atb_llm DESTINATION include)
