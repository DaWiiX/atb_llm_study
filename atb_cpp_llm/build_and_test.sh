#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# atb_cpp_llm — one-shot build & test driver
#
# What it does:
#   1. Loads `.env` from the repo root (or this project dir).
#   2. Sources Ascend CANN/ATB env scripts when present.
#   3. Configures + builds the project via CMake.
#   4. Runs CTest only when an NPU + npu-smi are present.
#
# Behaviour on a host without NPU/ATB runtime (e.g. CI dev box):
#   - Still loads .env and sources whatever Ascend scripts exist.
#   - Tries to configure + build (will fail at link time without ATB
#     libs — that is expected; we surface the failure).
#   - Skips test execution and prints how to run it on an NPU host.
#
# Usage:
#   bash atb_cpp_llm/build_and_test.sh
#   bash atb_cpp_llm/build_and_test.sh --debug
#   bash atb_cpp_llm/build_and_test.sh --clean
#   bash atb_cpp_llm/build_and_test.sh --no-test  # build only
# ─────────────────────────────────────────────────────────────────────
set -o pipefail

# ── CLI arg parsing ─────────────────────────────────────────────────
BUILD_TYPE="Release"
CLEAN=0
NO_TEST=0
for arg in "$@"; do
    case "$arg" in
        --debug)   BUILD_TYPE="Debug" ;;
        --release) BUILD_TYPE="Release" ;;
        --clean)   CLEAN=1 ;;
        --no-test) NO_TEST=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "[build_and_test] Unknown arg: $arg" >&2
            exit 2
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

# ── 2. Source Ascend env scripts when available ─────────────────────
ASCEND_OK=1
for f in \
    ~/Ascend/ascend-toolkit/set_env.sh \
    ~/Ascend/cann/set_env.sh \
    ~/Ascend/nnal/atb/latest/atb/set_env.sh \
    ~/Ascend/nnal/atb/set_env.sh
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
# Resolve actual ATB install dir: prefer ATB_HOME_PATH (set by set_env.sh),
# then fall back to ~/Ascend/nnal/atb/latest/atb/cxx_abi_1.
# NOTE: ~ must NOT be inside quotes, or it won't expand.
if [ -n "${ATB_HOME_PATH:-}" ] && [ -d "$ATB_HOME_PATH" ]; then
    ATB_CXX_ABI_DIR="$ATB_HOME_PATH"
elif [ -d ~/Ascend/nnal/atb/latest/atb/cxx_abi_1 ]; then
    ATB_CXX_ABI_DIR="$(eval echo ~/Ascend/nnal/atb/latest/atb/cxx_abi_1)"
else
    ATB_CXX_ABI_DIR=""
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
for candidate in \
    "${ASCEND_HOME_PATH:-}" \
    ~/Ascend/cann-9.0.0/aarch64-linux \
    ~/Ascend/ascend-toolkit/latest \
    ~/Ascend/cann/latest
do
    if [ -n "$candidate" ] && [ -d "$candidate" ]; then
        ASCEND_TOOLKIT_DIR="$candidate"
        break
    fi
done

# ── 3. Configure + build ────────────────────────────────────────────
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

log "Detected NPU; running CTest..."
npu-smi info | head -20 || true
ctest --test-dir "$BUILD_DIR" --output-on-failure -j4
log "All tests passed."