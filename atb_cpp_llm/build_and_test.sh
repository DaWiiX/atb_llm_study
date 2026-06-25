#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# atb_cpp_llm — build & test driver
#
# What it does:
#   1. Loads `.env` from the repo root (or this project dir).
#   2. Sources Ascend CANN/ATB env scripts when present.
#   3. Configures + builds the project via CMake.
#   4. Handles Python reference data the C++ tests read from /tmp/
#      (regenerate / reuse / skip — see "Reference data" below).
#   5. Runs CTest only when an NPU + npu-smi are present.
#
# Behaviour on a host without NPU/ATB runtime (e.g. CI dev box):
#   - Still loads .env and sources whatever Ascend scripts exist.
#   - Tries to configure + build (will fail at link time without ATB
#     libs — that is expected; we surface the failure).
#   - Skips test execution and prints how to run it on an NPU host.
#
# Usage (full build + test):
#   bash atb_cpp_llm/build_and_test.sh
#   bash atb_cpp_llm/build_and_test.sh --debug
#   bash atb_cpp_llm/build_and_test.sh --clean
#   bash atb_cpp_llm/build_and_test.sh --no-test               # build only
#
# Reference data (three modes; see testing-guide.md § 一·五 #8 for the why):
#   (default)                regenerate ALL /tmp/*.bin reference data
#   --no-refresh-refdata     reuse existing /tmp/*.bin (faster); auto-fallback
#                            to --no-refdata if any sentinel files are missing
#   --no-refdata             skip generation AND exclude the 28 tests that
#                            read /tmp/*.bin (ctest -LE needs_refdata),
#                            printing exactly which tests get skipped
#   --refresh-refdata        explicit alias for the default (legacy compat)
#
# Fast iteration (skip build, run only what you want):
#   bash atb_cpp_llm/build_and_test.sh --test-only                       # rerun all tests, no build
#   bash atb_cpp_llm/build_and_test.sh --test-only level1_cpu_pure       # one level
#   bash atb_cpp_llm/build_and_test.sh --test-only level1_cpu_pure level3_integration
#   bash atb_cpp_llm/build_and_test.sh --test-only test_bin_format test_text_model
#   bash atb_cpp_llm/build_and_test.sh --test-only level1_cpu_pure test_text_model
#       (mixing both: level filter AND name filter — ctest intersects them)
#   bash atb_cpp_llm/build_and_test.sh --list            # list registered tests + labels
#   bash atb_cpp_llm/build_and_test.sh --test-only -v test_vision_stages  # verbose
#
# Positional args are auto-classified:
#   - matches a known level dir (level0_framework, level1_cpu_pure,
#     level2_op_precision, level3_integration, level4_e2e) → ctest -L
#   - otherwise → ctest -R   (test-name regex; exact-match anchored)
# ─────────────────────────────────────────────────────────────────────
set -o pipefail

# ── Known test levels (kept in sync with tests/levelN_*/ dirs) ──────
# Anything matching one of these as a positional arg is treated as a
# LABEL filter (ctest -L). Everything else is treated as a NAME filter
# (ctest -R). These names must match the directory names under tests/
# AND the labels emitted by CMakeLists.txt's add_atb_test() helper.
KNOWN_LEVELS=(
    level0_framework
    level1_cpu_pure
    level2_op_precision
    level3_integration
    level4_e2e
)

# ── Reference-data sentinels ────────────────────────────────────────
# One representative file per generator in gen_all.py (keep this list
# in sync with gen_all.py's GENERATORS table). Used to detect missing
# reference data so --no-refresh-refdata can fall back to --no-refdata
# instead of letting tests run with stale/absent .bin files.
REFDATA_SENTINELS=(
    /tmp/cpu_op_rms_norm_medium_input.bin
    /tmp/cpu_vision_merger_main_x.bin
    /tmp/stage_L0_pixel_values.bin
    /tmp/stage_L3_rope_sin.bin
    /tmp/stage_pixels.bin
    /tmp/stage_final_text_only.bin
    /tmp/posembed_npu_case_tiny_4x4.bin
    /tmp/visrope_npu_case_tiny_4x4.bin
)

# ── CLI arg parsing ─────────────────────────────────────────────────
BUILD_TYPE="Release"
CLEAN=0
NO_TEST=0
# Refdata mode is one of: refresh (default) | reuse | none.
# Set by --refresh-refdata / --no-refresh-refdata / --no-refdata.
REFDATA_MODE="refresh"
REFDATA_FLAG_SEEN=""   # remembers which flag set REFDATA_MODE, for conflict detection
TEST_ONLY=0
LIST_ONLY=0
VERBOSE=0
LABEL_FILTERS=()
NAME_FILTERS=()

is_known_level() {
    local needle="$1"
    for lvl in "${KNOWN_LEVELS[@]}"; do
        if [ "$lvl" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

# Set REFDATA_MODE if no conflicting flag has been seen yet.
set_refdata_mode() {
    local new_mode="$1"
    local new_flag="$2"
    if [ -n "$REFDATA_FLAG_SEEN" ] && [ "$REFDATA_FLAG_SEEN" != "$new_flag" ]; then
        echo "[build_and_test] ERROR: conflicting refdata flags: $REFDATA_FLAG_SEEN and $new_flag" >&2
        echo "[build_and_test] Pick at most one of --refresh-refdata / --no-refresh-refdata / --no-refdata." >&2
        exit 2
    fi
    REFDATA_MODE="$new_mode"
    REFDATA_FLAG_SEEN="$new_flag"
}

for arg in "$@"; do
    case "$arg" in
        --debug)                  BUILD_TYPE="Debug" ;;
        --release)                BUILD_TYPE="Release" ;;
        --clean)                  CLEAN=1 ;;
        --no-test)                NO_TEST=1 ;;
        --no-refdata)             set_refdata_mode "none"    "--no-refdata" ;;
        --no-refresh-refdata)     set_refdata_mode "reuse"   "--no-refresh-refdata" ;;
        --refresh-refdata)        set_refdata_mode "refresh" "--refresh-refdata" ;;
        --test-only|-t)           TEST_ONLY=1 ;;
        --list|-l)                LIST_ONLY=1 ;;
        --verbose|-v)             VERBOSE=1 ;;
        -h|--help)
            sed -n '2,50p' "$0"
            exit 0
            ;;
        --*|-*)
            echo "[build_and_test] Unknown flag: $arg" >&2
            echo "[build_and_test] Run with --help for usage." >&2
            exit 2
            ;;
        *)
            # Positional: classify as level (LABEL) or test name (NAME).
            if is_known_level "$arg"; then
                LABEL_FILTERS+=("$arg")
            else
                NAME_FILTERS+=("$arg")
            fi
            ;;
    esac
done

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

log() { echo "[build_and_test] $*"; }

# ── 1. Load .env (repo root first, then project dir) ────────────────
ENV_FILE=""
for d in "$REPO_ROOT" "$SCRIPT_DIR"; do
    if [ -f "$d/.env" ]; then
        ENV_FILE="$d/.env"
        break
    fi
done
if [ -n "$ENV_FILE" ]; then
    log "Loading $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    log "No .env file found (repo root or $SCRIPT_DIR). Tests will use defaults."
fi

# Expand OFFICIAL_EMBED_CASES into reference-data sentinels.
# These check that gen_official_embedding.py has been run before C++ tests
# that read /tmp/official_embed_mm_*.bin and /tmp/official_tokens_mm_*.bin.
OFFICIAL_CASES="${OFFICIAL_EMBED_CASES:-416x672,720x1280,1080x1920,1440x2560}"
IFS=',' read -ra _OC_TOKENS <<< "$OFFICIAL_CASES"
for _oc in "${_OC_TOKENS[@]}"; do
    _oc_trimmed="$(echo "$_oc" | tr -d ' ')"
    [ -z "$_oc_trimmed" ] && continue
    REFDATA_SENTINELS+=("/tmp/official_embed_mm_${_oc_trimmed}.bin")
    REFDATA_SENTINELS+=("/tmp/official_tokens_mm_${_oc_trimmed}.bin")
done

# ── 2. Source Ascend env scripts when available ─────────────────────
# Probe two common install roots in order:
#   1. ~/Ascend/        — non-root install (recommended)
#   2. /usr/local/Ascend/ — root install (default CANN installer location)
ASCEND_OK=1
ASCEND_ROOTS=("$HOME/Ascend" "/usr/local/Ascend")
for ROOT in "${ASCEND_ROOTS[@]}"; do
    for f in \
        "$ROOT/ascend-toolkit/set_env.sh" \
        "$ROOT/cann/set_env.sh" \
        "$ROOT/nnal/atb/latest/atb/set_env.sh" \
        "$ROOT/nnal/atb/set_env.sh"
    do
        if [ -f "$f" ]; then
            # The ATB set_env.sh accepts --cxx_abi=1; tolerate either form.
            # shellcheck disable=SC1090
            if [[ "$f" == */atb/set_env.sh* ]]; then
                source "$f" --cxx_abi=1 2>/dev/null || source "$f"
            else
                source "$f"
            fi
            log "Sourced $f"
        fi
    done
done
# Resolve actual ATB install dir: prefer ATB_HOME_PATH (set by set_env.sh),
# then fall back to the first cxx_abi_1 dir found under the probed roots.
if [ -n "${ATB_HOME_PATH:-}" ] && [ -d "$ATB_HOME_PATH" ]; then
    ATB_CXX_ABI_DIR="$ATB_HOME_PATH"
else
    ATB_CXX_ABI_DIR=""
    for ROOT in "${ASCEND_ROOTS[@]}"; do
        if [ -d "$ROOT/nnal/atb/latest/atb/cxx_abi_1" ]; then
            ATB_CXX_ABI_DIR="$ROOT/nnal/atb/latest/atb/cxx_abi_1"
            break
        fi
    done
fi

if [ -n "$ATB_CXX_ABI_DIR" ]; then
    export ATB_BUILD_DEPENDENCY_PATH="$ATB_CXX_ABI_DIR"
    log "ATB_BUILD_DEPENDENCY_PATH=$ATB_BUILD_DEPENDENCY_PATH"
else
    ASCEND_OK=0
    log "WARNING: ATB cxx_abi_1 dir not found — CANN/ATB runtime is missing."
    log "         Build may fail at link time. To run on this host, install"
    log "         the Ascend CANN + ATB packages, or copy build artefacts to"
    log "         an NPU-equipped machine and re-run the tests there."
fi

# Resolve Ascend toolkit dir for CMake (needs acl/acl.h and lib64/)
ASCEND_TOOLKIT_DIR=""
TOOLKIT_CANDIDATES=("${ASCEND_HOME_PATH:-}")
for ROOT in "${ASCEND_ROOTS[@]}"; do
    TOOLKIT_CANDIDATES+=(
        "$ROOT/cann-9.0.0/aarch64-linux"
        "$ROOT/ascend-toolkit/latest"
        "$ROOT/cann/latest"
    )
done
for candidate in "${TOOLKIT_CANDIDATES[@]}"; do
    if [ -n "$candidate" ] && [ -d "$candidate" ]; then
        ASCEND_TOOLKIT_DIR="$candidate"
        break
    fi
done

# ── --list shortcut: just enumerate tests + labels and exit ─────────
if [ "$LIST_ONLY" = "1" ]; then
    if [ ! -d "$BUILD_DIR" ]; then
        log "ERROR: $BUILD_DIR does not exist. Run a regular build first:"
        log "       bash build_and_test.sh --no-test"
        exit 1
    fi
    log "Registered tests (with labels):"
    ctest --test-dir "$BUILD_DIR" -N --print-labels || true
    ctest --test-dir "$BUILD_DIR" -N
    exit 0
fi

# ── 3. Configure + build  (skipped under --test-only) ───────────────
if [ "$TEST_ONLY" = "1" ]; then
    if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        log "ERROR: --test-only requires an existing build in $BUILD_DIR"
        log "       Run a full build first:  bash build_and_test.sh --no-test"
        exit 1
    fi
    log "--test-only: skipping CMake configure + build"
else
    if [ "$CLEAN" = "1" ] && [ -d "$BUILD_DIR" ]; then
        log "Removing $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
    mkdir -p "$BUILD_DIR"

    CMAKE_EXTRA_ARGS=()
    if [ -n "$ATB_CXX_ABI_DIR" ]; then
        CMAKE_EXTRA_ARGS+=(-DATB_DIR="$ATB_CXX_ABI_DIR")
    fi
    if [ -n "$ASCEND_TOOLKIT_DIR" ]; then
        CMAKE_EXTRA_ARGS+=(-DASCEND_TOOLKIT_DIR="$ASCEND_TOOLKIT_DIR")
    fi

    log "Configuring CMake (build type: $BUILD_TYPE)..."
    if ! cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        "${CMAKE_EXTRA_ARGS[@]}"; then
        log "ERROR: CMake configure failed."
        exit 1
    fi

    log "Building (parallel jobs: $(nproc))..."
    if ! cmake --build "$BUILD_DIR" -j"$(nproc)"; then
        log "ERROR: Build failed."
        if [ "$ASCEND_OK" = "0" ]; then
            log "Hint: Ascend/ATB libraries were not found. The build is expected"
            log "      to fail at link time without them. Run this script on an"
            log "      NPU host with CANN+ATB installed to complete the build."
        fi
        exit 1
    fi
    log "Build succeeded."
fi

# ── 4. Decide whether to run tests ──────────────────────────────────
if [ "$NO_TEST" = "1" ]; then
    log "--no-test passed; skipping test execution."
    exit 0
fi
if ! command -v npu-smi &>/dev/null; then
    log "npu-smi not found — skipping test execution."
    log "To run tests on an NPU host: ctest --test-dir $BUILD_DIR --output-on-failure -j4"
    exit 0
fi

# ── 5. Echo effective model dir + run CTest ─────────────────────────
if [ -n "${QWEN3VL_EMB_MODEL_DIR:-}" ]; then
    log "QWEN3VL_EMB_MODEL_DIR=$QWEN3VL_EMB_MODEL_DIR"
else
    log "QWEN3VL_EMB_MODEL_DIR is not set — tests will fall back to the"
    log "  hard-coded path inside tests/test_env.h."
fi

# ── 5a. Handle Python reference data (refresh / reuse / none) ───────
# Three-state semantics — see testing-guide.md § 一·五 #8 for the why.
# All paths may end up setting CTEST_EXCLUDE_LABEL=needs_refdata so the
# 28 tests that fopen() /tmp/*.bin don't silently SKIP-and-pass when
# their data is absent.

count_missing_refdata() {
    local missing=0
    for f in "${REFDATA_SENTINELS[@]}"; do
        [ -s "$f" ] || missing=$((missing + 1))
    done
    echo $missing
}

CTEST_EXCLUDE_LABEL=""

case "$REFDATA_MODE" in
    refresh)
        log "Regenerating ALL Python reference data (default; pass --no-refresh-refdata to reuse /tmp/*.bin)..."
        if ! python3 "$SCRIPT_DIR/tests/python_reference/gen_all.py"; then
            log "WARN: reference-data generation had failures (gen_all.py exited non-zero)."
            log "      Falling back to --no-refdata: excluding tests that need reference data."
            log "      Fix the failing generators and re-run without --no-refresh-refdata to test those tests."
            log "      Hint: check the gen_all.py output above for which generator(s) failed."
            REFDATA_MODE="none"
        fi
        ;;
    reuse)
        missing=$(count_missing_refdata)
        if [ "$missing" -gt 0 ]; then
            log "WARN: --no-refresh-refdata requested, but $missing/${#REFDATA_SENTINELS[@]} sentinel files are missing under /tmp/."
            log "      Falling back to --no-refdata behaviour: excluding the 28 tests that need reference data."
            log "      Re-run without --no-refresh-refdata (default behaviour regenerates) to test those tests."
            REFDATA_MODE="none"
        else
            log "Reusing existing reference data in /tmp/ (--no-refresh-refdata; all ${#REFDATA_SENTINELS[@]} sentinels present)."
        fi
        ;;
    none)
        log "--no-refdata passed; skipping reference-data generation and excluding dependent tests."
        ;;
esac

# In "none" mode (either explicit or via fallback), list which tests are being
# skipped so the user knows exactly what's NOT running.
if [ "$REFDATA_MODE" = "none" ]; then
    SKIPPED_TESTS=$(ctest --test-dir "$BUILD_DIR" -N -L "^needs_refdata$" 2>/dev/null \
        | awk '/Test #/ {print $NF}')
    if [ -n "$SKIPPED_TESTS" ]; then
        skipped_count=$(echo "$SKIPPED_TESTS" | wc -l)
        log "  Excluding $skipped_count tests labelled needs_refdata:"
        echo "$SKIPPED_TESTS" | xargs -n4 | sed 's/^/    /'
    fi
    CTEST_EXCLUDE_LABEL="needs_refdata"

    # Edge case: user explicitly named a needs_refdata test in NAME_FILTERS.
    # The combined -R + -LE filter would silently empty-match; warn them.
    if [ "${#NAME_FILTERS[@]}" -gt 0 ] && [ -n "$SKIPPED_TESTS" ]; then
        for name in "${NAME_FILTERS[@]}"; do
            if echo "$SKIPPED_TESTS" | grep -qx "$name"; then
                log "WARN: '$name' requires reference data and will be excluded by --no-refdata."
                log "      Drop --no-refdata (or use --no-refresh-refdata) to run it."
            fi
        done
    fi
fi

log "Detected NPU; running CTest..."
npu-smi info | head -20 || true

# Assemble ctest filter args. -L and -R together = AND (test must satisfy both).
CTEST_FILTER_ARGS=()
if [ "${#LABEL_FILTERS[@]}" -gt 0 ]; then
    # Anchor each level name so "level1" does NOT match "level10" if ever added.
    label_regex="^($(IFS='|'; echo "${LABEL_FILTERS[*]}"))$"
    CTEST_FILTER_ARGS+=(-L "$label_regex")
    log "Label filter: $label_regex"
fi
if [ "${#NAME_FILTERS[@]}" -gt 0 ]; then
    # Anchor names so "test_e2e" doesn't accidentally match "test_e2e_extra".
    name_regex="^($(IFS='|'; echo "${NAME_FILTERS[*]}"))$"
    CTEST_FILTER_ARGS+=(-R "$name_regex")
    log "Name filter:  $name_regex"
fi
if [ "${#LABEL_FILTERS[@]}" -gt 0 ] && [ "${#NAME_FILTERS[@]}" -gt 0 ]; then
    log "Note: -L and -R combine with AND — a test must match BOTH to run."
fi
if [ -n "$CTEST_EXCLUDE_LABEL" ]; then
    # AND-combined with any -L/-R the user gave: must match BOTH user filters
    # AND NOT match the excluded label. Empty intersections are common and
    # legal here (e.g. --no-refdata level1_cpu_pure → 0 tests run).
    CTEST_FILTER_ARGS+=(-LE "^${CTEST_EXCLUDE_LABEL}$")
fi

CTEST_VERBOSE_ARGS=()
if [ "$VERBOSE" = "1" ]; then
    CTEST_VERBOSE_ARGS+=(-V)
fi

if ctest --test-dir "$BUILD_DIR" --output-on-failure -j4 \
        "${CTEST_FILTER_ARGS[@]}" "${CTEST_VERBOSE_ARGS[@]}"; then
    log "All selected tests passed."
else
    log "Some tests FAILED — see ctest output above for details."
    exit 1
fi
