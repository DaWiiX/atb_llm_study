cmake_minimum_required(VERSION 3.16)
project(atb_cpp_llm_engine LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
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
        set(ATB_DIR "$ENV{HOME}/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1" CACHE PATH "ATB install dir")
    endif()
    if(NOT DEFINED ASCEND_TOOLKIT_DIR)
        set(ASCEND_TOOLKIT_DIR "$ENV{HOME}/Ascend/cann-9.0.0/aarch64-linux" CACHE PATH "Ascend toolkit dir")
    endif()
endif()

# ── Third-party (cJSON) ──────────────────────────────────
# Static lib gets linked into the SHARED atb_llm_engine, so PIC is required;
# otherwise aarch64 link fails with "R_AARCH64_ADR_PREL_PG_HI21 ... -fPIC".
add_library(thirdparty STATIC utils/cJSON.c)
set_target_properties(thirdparty PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(thirdparty PUBLIC utils)

# ── Core library (explicit source list) ──────────────────
set(ENGINE_SRCS
    # core
    src/core/npu_tensor.cpp
    src/core/tensor_allocator.cpp
    src/core/context_manager.cpp
    src/core/buffer_pool.cpp
    src/core/graph_builder.cpp
    src/core/debug_dump.cpp
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
    src/engine/embedder.cpp
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
    src/ops/index_add_op.cpp
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
    src/components/vision/pos_embed_npu_graph.cpp
    src/components/vision/vis_rope_npu_graph.cpp
    src/components/vision/aclnn_bicubic_resize.cpp
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

# `utils/` holds vendored third-party headers (safetensors.hh, cJSON).
# ATB / CANN SDK headers ship with their own -Wpedantic / -Wunused-parameter
# noise. Mark all of these as SYSTEM so GCC suppresses warnings from inside
# them — we only want diagnostics for our own code.
target_include_directories(atb_llm_engine
    PUBLIC
        include
    PRIVATE
        src
)
target_include_directories(atb_llm_engine SYSTEM
    PRIVATE
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
        opapi
        atb
)

# ── Helper: add a test/bench executable ───────────────────
function(add_atb_test name source)
    add_executable(${name} ${source})
    target_include_directories(${name}
        PRIVATE
            src
            tests
    )
    target_include_directories(${name} SYSTEM
        PRIVATE
            utils
            ${ATB_DIR}/include
            ${ASCEND_TOOLKIT_DIR}/include
    )
    # Test fixtures intentionally:
    #   - ignore fread() results when probing test data files
    #     (REQUIRE() on the parsed values is what we care about);
    #   - keep main(argc, argv) signatures even when unused;
    #   - mix size_t loop counters with signed shape values;
    #   - leave optional Load* helpers defined for future cases.
    # Silence these in tests so the build output stays clean.
    target_compile_options(${name} PRIVATE
        -Wno-unused-result
        -Wno-unused-parameter
        -Wno-sign-compare
        -Wno-unused-function
    )
    target_link_directories(${name}
        PRIVATE
            ${ATB_DIR}/lib
            ${ASCEND_TOOLKIT_DIR}/lib64
    )
    target_link_libraries(${name} PRIVATE atb_llm_engine atb ascendcl)
    # Register as CTest target unless it's a benchmark (names starting with "bench")
    if(NOT name MATCHES "^bench")
        add_test(NAME ${name} COMMAND ${name})
        # Auto-derive a CTest label from the source path:
        #   tests/level1_cpu_pure/foo.cpp  →  label "level1_cpu_pure"
        # This is what powers `build_and_test.sh level1_cpu_pure` (uses
        # `ctest -L`). Keep .cpp files under tests/levelN_<tag>/ to inherit
        # the label automatically — no manual config needed when adding tests.
        get_filename_component(_test_dir "${source}" DIRECTORY)
        get_filename_component(_test_label "${_test_dir}" NAME)
        if(_test_label MATCHES "^level[0-9]")
            set_tests_properties(${name} PROPERTIES LABELS "${_test_label}")
        endif()
        # Propagate critical env vars from .env to ctest subprocesses.
        # This ensures C++ tests can access ASCEND_PLATFORM, model dirs, etc.
        # even when ctest is invoked from a shell that hasn't sourced .env.
        set(_test_env "")
        if(DEFINED ENV{ASCEND_PLATFORM})
            list(APPEND _test_env "ASCEND_PLATFORM=$ENV{ASCEND_PLATFORM}")
        endif()
        if(DEFINED ENV{QWEN3VL_EMB_MODEL_DIR})
            list(APPEND _test_env "QWEN3VL_EMB_MODEL_DIR=$ENV{QWEN3VL_EMB_MODEL_DIR}")
        endif()
        if(_test_env)
            set_tests_properties(${name} PROPERTIES ENVIRONMENT "${_test_env}")
        endif()
    endif()
endfunction()

# ── Tests ─────────────────────────────────────────────────
enable_testing()

add_atb_test(test_core                  tests/level0_framework/test_core.cpp)
add_atb_test(test_config_wiring        tests/level0_framework/test_config_wiring.cpp)
add_atb_test(test_weight_precision     tests/level0_framework/test_weight_precision.cpp)
add_atb_test(test_model_registry       tests/level0_framework/test_model_registry.cpp)
add_atb_test(test_text_ops             tests/level2_op_precision/test_text_ops.cpp)
add_atb_test(test_vision_ops           tests/level2_op_precision/test_vision_ops.cpp)
add_atb_test(test_text_model           tests/level3_integration/test_text_model.cpp)
add_atb_test(test_e2e                  tests/level4_e2e/test_e2e.cpp)
add_atb_test(test_accuracy             tests/level4_e2e/test_accuracy.cpp)
add_atb_test(test_stage_precision      tests/level4_e2e/test_stage_precision.cpp)
add_atb_test(test_forward_error_paths  tests/level4_e2e/test_forward_error_paths.cpp)
add_atb_test(test_sync_safety        tests/level4_e2e/test_sync_safety.cpp)
add_atb_test(test_path_c_raw_image   tests/level4_e2e/test_path_c_raw_image.cpp)
add_atb_test(test_engine_vs_official tests/level4_e2e/test_engine_vs_official.cpp)
add_atb_test(test_vision_stages       tests/level3_integration/test_vision_stages.cpp)
add_atb_test(test_pos_embed_npu_graph tests/level2_op_precision/test_pos_embed_npu_graph.cpp)
add_atb_test(test_vis_rope_npu_graph  tests/level2_op_precision/test_vis_rope_npu_graph.cpp)
add_atb_test(test_index_add_npu       tests/level2_op_precision/test_index_add_npu.cpp)
add_atb_test(test_aclnn_bicubic_spike tests/level2_op_precision/test_aclnn_bicubic_spike.cpp)

# Level 1 CPU pure-function tests (no NPU dependency for the logic,
# but link against engine library for the implementation).
add_atb_test(test_mrope_cpu         tests/level1_cpu_pure/test_mrope_cpu.cpp)
add_atb_test(test_vision_rope_cpu   tests/level1_cpu_pure/test_vision_rope_cpu.cpp)
add_atb_test(test_preprocess_cpu    tests/level1_cpu_pure/test_preprocess_cpu.cpp)
add_atb_test(test_pos_embed_cpu     tests/level1_cpu_pure/test_pos_embed_cpu.cpp)
add_atb_test(test_float_utils       tests/level1_cpu_pure/test_float_utils.cpp)
add_atb_test(test_base_model_utils  tests/level1_cpu_pure/test_base_model_utils.cpp)
add_atb_test(test_causal_mask_fp16  tests/level1_cpu_pure/test_causal_mask_fp16.cpp)
add_atb_test(test_embedder_utils    tests/level1_cpu_pure/test_embedder_utils.cpp)
add_atb_test(test_embedder_invariants tests/level1_cpu_pure/test_embedder_invariants.cpp)
add_atb_test(test_bin_format       tests/level1_cpu_pure/test_bin_format.cpp)

# Level 2 per-op precision tests (Wave 2)
add_atb_test(test_rms_norm_precision            tests/level2_op_precision/test_rms_norm_precision.cpp)
add_atb_test(test_layer_norm_precision          tests/level2_op_precision/test_layer_norm_precision.cpp)
add_atb_test(test_linear_precision              tests/level2_op_precision/test_linear_precision.cpp)
add_atb_test(test_activation_precision          tests/level2_op_precision/test_activation_precision.cpp)
add_atb_test(test_elewise_precision             tests/level2_op_precision/test_elewise_precision.cpp)
add_atb_test(test_split_concat_precision        tests/level2_op_precision/test_split_concat_precision.cpp)
add_atb_test(test_softmax_precision             tests/level2_op_precision/test_softmax_precision.cpp)
add_atb_test(test_gather_reduce_precision       tests/level2_op_precision/test_gather_reduce_precision.cpp)
add_atb_test(test_transpose_set_value_precision tests/level2_op_precision/test_transpose_set_value_precision.cpp)
add_atb_test(test_patch_transpose_spike          tests/level2_op_precision/test_patch_transpose_spike.cpp)

# Level 2 advanced precision tests (Wave 3)
add_atb_test(test_rope_precision                tests/level2_op_precision/test_rope_precision.cpp)
add_atb_test(test_self_attention_precision      tests/level2_op_precision/test_self_attention_precision.cpp)

add_atb_test(test_swiglu_mlp_precision          tests/level2_op_precision/test_swiglu_mlp_precision.cpp)
add_atb_test(test_text_decoder_layer_precision  tests/level2_op_precision/test_text_decoder_layer_precision.cpp)
add_atb_test(test_vision_attention_precision    tests/level2_op_precision/test_vision_attention_precision.cpp)
add_atb_test(test_vision_mlp_precision          tests/level2_op_precision/test_vision_mlp_precision.cpp)
add_atb_test(test_vision_block_precision        tests/level2_op_precision/test_vision_block_precision.cpp)
add_atb_test(test_patch_embed_precision         tests/level2_op_precision/test_patch_embed_precision.cpp)
add_atb_test(test_vision_merger_precision       tests/level2_op_precision/test_vision_merger_precision.cpp)

# Level 3 integration tests (Wave 4)
add_atb_test(test_deepstack              tests/level3_integration/test_deepstack.cpp)
add_atb_test(test_deepstack_npu_tensor   tests/level3_integration/test_deepstack_npu_tensor.cpp)
add_atb_test(test_vision_runner_full     tests/level3_integration/test_vision_runner_full.cpp)
add_atb_test(test_text_runner_full       tests/level3_integration/test_text_runner_full.cpp)

# doctest-based IO/Adapter tests (needs tests/ for doctest.h)
add_executable(test_io_adapters tests/level0_framework/test_io_adapters.cpp)
target_include_directories(test_io_adapters
    PRIVATE
        src
        tests
)
target_include_directories(test_io_adapters SYSTEM
    PRIVATE
        utils
        ${ATB_DIR}/include
        ${ASCEND_TOOLKIT_DIR}/include
)
target_compile_options(test_io_adapters PRIVATE
    -Wno-unused-result
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-unused-function
)
target_link_directories(test_io_adapters
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_io_adapters PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_io_adapters COMMAND test_io_adapters)
# Match the label scheme used by add_atb_test() above.
set_tests_properties(test_io_adapters PROPERTIES LABELS "level0_framework")

# Benchmark mode smoke test (doctest, needs NPU + full model).
# Spawns the benchmark binary, runs --mode text, parses output to verify
# BENCH_RESULT machine lines + human-readable stage table both present.
# Prevents regression of §9.8 (benchmark output quality never validated).
add_executable(test_benchmark_modes tests/test_benchmark_modes.cpp)
target_include_directories(test_benchmark_modes
    PRIVATE
        src
        tests
)
target_compile_options(test_benchmark_modes PRIVATE
    -Wno-unused-result
    -Wno-unused-parameter
    -Wno-sign-compare
    -Wno-unused-function
)
target_link_directories(test_benchmark_modes
    PRIVATE
        ${ATB_DIR}/lib
        ${ASCEND_TOOLKIT_DIR}/lib64
)
target_link_libraries(test_benchmark_modes PRIVATE atb_llm_engine atb ascendcl)
add_test(NAME test_benchmark_modes COMMAND test_benchmark_modes)
set_tests_properties(test_benchmark_modes PROPERTIES LABELS "level4_e2e;needs_npu")
# Propagate env vars (ASCEND_PLATFORM, model dir) to the test subprocess
# so the spawned benchmark can locate the model checkpoint — mirrors the
# ENVIRONMENT logic inside add_atb_test() above.
# LOG_LEVEL=1 forces INFO output so diagnostic banners (otherwise muted at
# the default WARN level) are captured alongside the printf-based report
# tables asserted by the test.
set(_bm_env "LOG_LEVEL=1")
if(DEFINED ENV{ASCEND_PLATFORM})
    list(APPEND _bm_env "ASCEND_PLATFORM=$ENV{ASCEND_PLATFORM}")
endif()
if(DEFINED ENV{QWEN3VL_EMB_MODEL_DIR})
    list(APPEND _bm_env "QWEN3VL_EMB_MODEL_DIR=$ENV{QWEN3VL_EMB_MODEL_DIR}")
endif()
set_tests_properties(test_benchmark_modes PROPERTIES ENVIRONMENT "${_bm_env}")

# Benchmarks (not registered as CTest tests)
add_atb_test(benchmark                 tests/benchmark.cpp)  # stays at tests/ root

# ── Test scheduling ───────────────────────────────────────
# Each E2E (and test_vision_stages) test allocates a 10–16 GB NPU buffer
# pool and loads the full 2B model. Running two of them concurrently under
# `ctest -j4` blows past the device HBM budget and trips
# `aclrtMalloc error=207001`. Share a single "npu_e2e" resource lock so
# they serialize against each other while unrelated CPU / lightweight
# tests still run in parallel.
# Must come AFTER all add_atb_test() calls — set_tests_properties()
# requires the targets to already exist.
set_tests_properties(
    test_e2e
    test_accuracy
    test_stage_precision
    test_forward_error_paths
    test_sync_safety
    test_path_c_raw_image
    test_engine_vs_official
    test_vision_stages
    test_benchmark_modes
    PROPERTIES RESOURCE_LOCK "npu_e2e"
)

# ── Reference-data dependency labels ──────────────────────
# Tests that fopen() /tmp/{cpu_,preprocess_,stage_,posembed_npu_,visrope_npu_,bicubic_,official_pv_,official_embed_,official_tokens_}*.bin
# files produced by tests/python_reference/gen_all.py. When those .bin
# files are absent, every test listed below correctly returns a non-zero exit code
# (true failure), so ctest will report them as FAIL — not PASS.
#
# The "needs_refdata" label is defense-in-depth: build_and_test.sh's
# --no-refdata mode passes `ctest -LE "^needs_refdata$"` to proactively
# exclude these tests when reference data is known to be unavailable
# (either explicitly via the flag, or via auto-fallback from
# --no-refresh-refdata when sentinel files are missing). This avoids
# noisy, expected failures that would distract from real regressions.
#
# Maintenance contract: when a new test reads a /tmp/*.bin file, list it
# here. Discovered via: grep -lE '/tmp/(cpu_|preprocess_|stage_|posembed_npu_|bicubic_|official_pv_|official_embed_|official_tokens_|visrope_npu_)' \
#                          tests/level*/*.cpp | grep -v test_config_wiring
set(REFDATA_DEPENDENT_TESTS
    # Level 0 framework tests
    test_io_adapters
    # Level 1 CPU pure-function tests
    test_float_utils
    test_mrope_cpu
    test_pos_embed_cpu
    test_preprocess_cpu
    test_vision_rope_cpu
    # Level 2 op-precision tests
    test_rms_norm_precision
    test_layer_norm_precision
    test_linear_precision
    test_activation_precision
    test_elewise_precision
    test_split_concat_precision
    test_softmax_precision
    test_gather_reduce_precision
    test_transpose_set_value_precision
    test_rope_precision
    test_self_attention_precision
    test_swiglu_mlp_precision
    test_text_decoder_layer_precision
    test_vision_attention_precision
    test_vision_mlp_precision
    test_vision_block_precision
    test_patch_embed_precision
    test_vision_merger_precision
    test_aclnn_bicubic_spike
    test_pos_embed_npu_graph
    test_vis_rope_npu_graph
    # Level 3 integration
    test_vision_stages
    # Level 4 E2E
    test_stage_precision
    test_engine_vs_official
)
foreach(_t ${REFDATA_DEPENDENT_TESTS})
    # APPEND so we don't clobber the levelN_* label set by add_atb_test().
    set_property(TEST ${_t} APPEND PROPERTY LABELS "needs_refdata")
endforeach()

# ── Install ───────────────────────────────────────────────
install(TARGETS atb_llm_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/atb_llm DESTINATION include)
