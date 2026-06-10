#!/usr/bin/env bash
# ============================================================================
# verify_baseline.sh — Pre-change regression script for ATB C++ LLM engine
# ============================================================================
#
# Workflow:
#   1. Record current commit + stash dirty working tree (unless --no-stash)
#   2. Build the benchmark target
#   3. Generate chat-templated token files for C++ compare mode
#   4. Run C++ benchmark --mode compare (saves 13 output .bin files)
#   5. Run Python engine on identical inputs, compare cosine vs C++ outputs
#   6. Print PASS/FAIL summary table (threshold: cosine >= 0.99)
#   7. Restore stashed changes
#
# The 13 test combinations (matching C++ benchmark --mode compare):
#   5x TEXT_ONLY:   S = 100, 512, 1024, 2048, 4096
#   4x IMAGE_ONLY:  416x672, 720x1280, 1080x1920, 1440x2560
#   4x MULTIMODAL:  416x672, 720x1280, 1080x1920, 1440x2560
#
# Usage:
#   bash scripts/verify_baseline.sh              # full check
#   bash scripts/verify_baseline.sh --no-stash   # skip stash (clean tree)
#   bash scripts/verify_baseline.sh --no-build   # skip rebuild (already built)
#
# Exit codes:
#   0 — all cosine values >= 0.99 (baseline clean)
#   1 — one or more cosine values < 0.99 (baseline broken)
#   2 — script error (build failure, missing model, no NPU, etc.)
# ============================================================================

set -euo pipefail

# ── Colour helpers ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Paths (absolute, works from any directory) ───────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
BENCHMARK_BIN="$BUILD_DIR/benchmark"
TOKEN_GEN_SCRIPT="$REPO_ROOT/atb_python_qwen3vl_embedding/tests/gen_baseline_tokens.py"

# Load .env so QWEN3VL_EMB_MODEL_DIR is set just like build_and_test.sh does.
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090,SC1091
    source "$REPO_ROOT/.env"
    set +a
fi
MODEL_DIR="${QWEN3VL_EMB_MODEL_DIR:?QWEN3VL_EMB_MODEL_DIR is not set — add it to $REPO_ROOT/.env (see .env.example)}"

# ── Timing bookkeeping ────────────────────────────────────────────────────
BUILD_DURATION=0
BENCH_START=0

# ── Threshold ────────────────────────────────────────────────────────────
COSINE_THRESHOLD=0.99

# ── CLI flags ────────────────────────────────────────────────────────────
NO_STASH=0
NO_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --no-stash) NO_STASH=1 ;;
        --no-build) NO_BUILD=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown argument: $arg"
            echo "Usage: $0 [--no-stash] [--no-build]"
            exit 2
            ;;
    esac
done

# ── Logging helpers ──────────────────────────────────────────────────────
log()      { echo -e "${CYAN}[verify]${NC} $*"; }
success()  { echo -e "${GREEN}[PASS]${NC}  $*"; }
warn()     { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()    { echo -e "${RED}[FAIL]${NC}  $*" >&2; }

# ══════════════════════════════════════════════════════════════════════════
# Step 0 — Pre-flight checks
# ══════════════════════════════════════════════════════════════════════════

log "Pre-flight checks..."

# Model checkpoint
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    warn "Model checkpoint not found at $MODEL_DIR"
    warn "Set QWEN3VL_EMB_MODEL_DIR env var or place model at the default path."
    warn "Skipping baseline verification (CI-friendly)."
    exit 0
fi
log "Model checkpoint: $MODEL_DIR"

# NPU check (best-effort — the actual benchmark run will catch missing NPU)
if command -v npu-smi &>/dev/null; then
    log "NPU detected ($(npu-smi info 2>/dev/null | head -1 || echo 'unknown'))"
else
    warn "npu-smi not found — NPU may not be available."
    warn "Benchmark will fail if ATB/ACL runtime is not initialized."
fi

# Python check (needed for token generation and comparison)
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    warn "python3 not found — cannot run token generation or comparison."
    warn "Skipping baseline verification."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 1 — Save git state
# ══════════════════════════════════════════════════════════════════════════

log "Saving git state..."
CURRENT_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
log "  Commit: ${CURRENT_COMMIT:0:12}"

STASHED=0
if [ "$NO_STASH" -eq 0 ]; then
    if ! git -C "$PROJECT_DIR" diff-index --quiet HEAD -- 2>/dev/null; then
        log "  Working tree is dirty — stashing changes..."
        if ! git -C "$PROJECT_DIR" stash push -m "verify_baseline auto stash ($(date -Iseconds))"; then
            error "git stash failed"
            exit 2
        fi
        STASHED=1
        log "  Stashed: $(git -C "$PROJECT_DIR" stash list -1 --format='%s')"
    else
        log "  Working tree is clean."
    fi
else
    log "  --no-stash: skipping stash (assuming clean tree)."
fi

# ── Cleanup trap (restore stash on exit) ────────────────────────────────
cleanup() {
    local exit_code=$?
    if [ "$STASHED" -eq 1 ]; then
        log "Restoring stashed changes..."
        if git -C "$PROJECT_DIR" stash pop --quiet 2>/dev/null; then
            log "  Stash restored."
        else
            warn "  Failed to pop stash — you may need to restore manually:"
            warn "    cd $PROJECT_DIR && git stash pop"
        fi
    fi
    exit $exit_code
}
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════════════════
# Step 2 — Build benchmark
# ══════════════════════════════════════════════════════════════════════════

if [ "$NO_BUILD" -eq 1 ]; then
    log "--no-build: skipping build step."
    if [ ! -x "$BENCHMARK_BIN" ]; then
        error "Benchmark binary not found at $BENCHMARK_BIN despite --no-build."
        error "Build it first: cd $BUILD_DIR && cmake --build . --target benchmark -j\$(nproc)"
        exit 2
    fi
    log "  Benchmark binary: $BENCHMARK_BIN"
else
    log "Building benchmark target..."
    if [ ! -d "$BUILD_DIR" ]; then
        error "Build directory not found at $BUILD_DIR"
        error "Run cmake configure first: see build_and_test.sh"
        exit 2
    fi

    BUILD_START=$(date +%s)
    if ! cmake --build "$BUILD_DIR" --target benchmark -j"$(nproc)" 2>&1; then
        error "Build failed — restoring stash and exiting."
        exit 2
    fi
    BUILD_END=$(date +%s)
    BUILD_DURATION=$((BUILD_END - BUILD_START))
    log "  Build succeeded in ${BUILD_DURATION}s"
fi

if [ ! -x "$BENCHMARK_BIN" ]; then
    error "Benchmark binary not found at $BENCHMARK_BIN after build."
    exit 2
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 3 — Generate token files
# ══════════════════════════════════════════════════════════════════════════

log "Generating token files for C++ compare mode..."
if [ -f "$TOKEN_GEN_SCRIPT" ]; then
    if ! "$PYTHON_BIN" "$TOKEN_GEN_SCRIPT" 2>&1; then
        warn "Token generation failed — C++ benchmark TEXT mode may fail."
        warn "Continuing with IO and MM modes only."
    else
        log "  Token files generated."
    fi
else
    warn "Token generation script not found at $TOKEN_GEN_SCRIPT"
    warn "C++ benchmark TEXT mode will need pre-existing token files."
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 4 — Run C++ benchmark (compare mode)
# ══════════════════════════════════════════════════════════════════════════

log "Running C++ benchmark --mode compare ..."
log "  (13 combinations: 5 TEXT + 4 IO + 4 MM, each iter=3 warmup=2)"
log "  This may take several minutes..."

BENCH_START=$(date +%s)

# Capture both stdout and stderr; tee to log for debugging
BENCH_LOG="/tmp/verify_baseline_bench_$$.log"
set +e
"$BENCHMARK_BIN" --mode compare --iter 3 --warmup 2 > "$BENCH_LOG" 2>&1
BENCH_EXIT=$?
set -e

BENCH_END=$(date +%s)
BENCH_DURATION=$((BENCH_END - BENCH_START))
log "  C++ benchmark finished in ${BENCH_DURATION}s (exit code: $BENCH_EXIT)"

if [ "$BENCH_EXIT" -ne 0 ]; then
    error "C++ benchmark exited with code $BENCH_EXIT"
    warn "Benchmark log saved to: $BENCH_LOG"
    warn "Last 30 lines:"
    tail -30 "$BENCH_LOG" >&2
    exit 2
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 5 — Python comparison: run Python engine on identical inputs
# ══════════════════════════════════════════════════════════════════════════

log "Running Python comparison (C++ vs Python cosine similarity)..."
log "  Loading Qwen3VLEngine and comparing 13 output pairs..."

COMPARE_START=$(date +%s)

# Write a temporary Python comparison script.
# This is inlined to keep verify_baseline.sh self-contained.
COMPARE_SCRIPT="/tmp/verify_baseline_compare_$$.py"
cat > "$COMPARE_SCRIPT" << 'PYEOF'
import sys, os, struct, time, glob
import numpy as np
import torch
import torch.nn.functional as F

# ── Paths (passed via env from the shell wrapper above) ─────────
MODEL_DIR = os.environ["QWEN3VL_EMB_MODEL_DIR"]
REPO_ROOT = os.environ["REPO_ROOT"]
BIN_DIR = "/tmp"

# ── Add Python package to path ──────────────────────────────────
PY_PKG = os.path.join(REPO_ROOT, "atb_python_qwen3vl_embedding")
if os.path.isdir(PY_PKG):
    sys.path.insert(0, REPO_ROOT)
else:
    print(f"[ERROR] Python package not found at {PY_PKG}", file=sys.stderr)
    sys.exit(2)

# ── Constants (must match C++ benchmark --mode compare) ─────────
TEXT_SEQ_LENGTHS = [100, 512, 1024, 2048, 4096]
RESOLUTIONS = [(416, 672), (720, 1280), (1080, 1920), (1440, 2560)]
COSINE_THRESHOLD = float(os.environ.get("COSINE_THRESHOLD", "0.99"))

# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def load_token_ids(path: str):
    """Load C++-format token ID binary: [int32 count] [int64 * count]."""
    with open(path, "rb") as f:
        (count,) = struct.unpack("<i", f.read(4))
        fmt = f"<{count}q"
        return list(struct.unpack(fmt, f.read(count * 8)))

def load_pooler(path: str) -> torch.Tensor:
    """Load pooler output: [int64 dim] [float32 * dim]."""
    with open(path, "rb") as f:
        (dim,) = struct.unpack("<q", f.read(8))
        data = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return torch.from_numpy(data.copy()).float()

def load_pixel_values(path: str) -> torch.Tensor:
    """Load C++-preprocessed pixel_values: [int32 count] [uint16 * count]."""
    with open(path, "rb") as f:
        (count,) = struct.unpack("<i", f.read(4))
        raw = f.read(count * 2)
    pv_fp16 = np.frombuffer(raw, dtype=np.float16)
    return torch.from_numpy(pv_fp16.astype(np.float32))

def load_grid_thw(pv_path: str, engine) -> torch.Tensor:
    """Recompute grid_thw from SmartResize using engine config.

    Reads the .bin header to extract the original resolution encoded
    in the filename (e.g., /tmp/cpp_pv_416x672.bin).
    Falls back to a per-resolution lookup if SmartResize is not importable.
    """
    # Determine original WxH from filename
    # Format: /tmp/cpp_pv_{W}x{H}.bin
    basename = os.path.basename(pv_path)  # cpp_pv_416x672.bin
    parts = basename.replace(".bin", "").split("_")
    if len(parts) >= 3:
        wh = parts[-1]  # 416x672
        ws, hs = wh.split("x")
        orig_w, orig_h = int(ws), int(hs)
    else:
        return None

    try:
        from atb_python_qwen3vl_embedding.preprocess import smart_resize
        factor = int(engine.patch_size * engine.merge_size)
        new_h, new_w = smart_resize(orig_h, orig_w, factor=factor,
                                    min_pixels=int(engine.pp_min_px),
                                    max_pixels=int(engine.pp_max_px))
    except ImportError:
        new_w, new_h = orig_w, orig_h

    grid_h = new_h // engine.patch_size
    grid_w = new_w // engine.patch_size
    return torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

# ══════════════════════════════════════════════════════════════════
# Main comparison
# ══════════════════════════════════════════════════════════════════

def main():
    print("[compare] Setting up ATB buffer size...", file=sys.stderr)
    from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)  # 10GB

    print("[compare] Loading Qwen3VLEngine...", file=sys.stderr)
    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    t0 = time.time()
    engine = Qwen3VLEngine(MODEL_DIR)
    t1 = time.time()
    print(f"[compare] Engine loaded in {t1 - t0:.1f}s", file=sys.stderr)

    results = []
    all_pass = True
    lowest_cosine = 1.0
    first_failure = None

    # ──────────────────────────────────────────────────────────────
    # 1. TEXT_ONLY (5 cases)
    # ──────────────────────────────────────────────────────────────
    for S in TEXT_SEQ_LENGTHS:
        label = f"TEXT {S}"
        cpp_path = os.path.join(BIN_DIR, f"cpp_text_only_{S}.bin")
        tok_path = os.path.join(BIN_DIR, f"tokens_chat_text_only_{S}.bin")

        if not os.path.exists(cpp_path):
            results.append((label, S, 0, None, "SKIP", "C++ output missing"))
            continue

        try:
            input_ids = load_token_ids(tok_path)
        except (FileNotFoundError, struct.error) as e:
            results.append((label, S, 0, None, "SKIP", f"Token file: {e}"))
            continue

        input_ids_t = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            out = engine.forward(input_ids_t, None, None)
        attn_mask = torch.ones_like(input_ids_t)
        pooled = engine.embedding_pooling(out, attn_mask).cpu().float()

        cpp = load_pooler(cpp_path)
        cos = F.cosine_similarity(pooled.squeeze(0), cpp, dim=0).item()

        if cos < lowest_cosine:
            lowest_cosine = cos
        status = "PASS" if cos >= COSINE_THRESHOLD else "FAIL"
        if status == "FAIL":
            if first_failure is None:
                first_failure = label
            all_pass = False
        elapsed = 0  # Not timing individual cases here

        results.append((label, S, 0, cos, status, ""))

    # ──────────────────────────────────────────────────────────────
    # 2. IMAGE_ONLY (4 cases)
    # ──────────────────────────────────────────────────────────────
    for W, H in RESOLUTIONS:
        label = f"IO {W}x{H}"
        cpp_path = os.path.join(BIN_DIR, f"cpp_io_{W}x{H}.bin")
        tok_path = os.path.join(BIN_DIR, f"tokens_chat_io_{W}x{H}.bin")
        pv_path  = os.path.join(BIN_DIR, f"cpp_pv_{W}x{H}.bin")

        if not os.path.exists(cpp_path):
            results.append((label, 0, 0, None, "SKIP", "C++ output missing"))
            continue
        if not os.path.exists(pv_path):
            results.append((label, 0, 0, None, "SKIP", "pixel_values missing"))
            continue

        try:
            # Load tokens (or fallback to bare image tokens)
            if os.path.exists(tok_path):
                input_ids = load_token_ids(tok_path)
            else:
                grid_thw_t = load_grid_thw(pv_path, engine)
                if grid_thw_t is not None:
                    total_patches = int(torch.prod(grid_thw_t, dim=1).sum())
                    merger_tokens = total_patches // (engine.merge_size ** 2)
                    input_ids = [engine.img_tok] * merger_tokens
                else:
                    results.append((label, 0, 0, None, "SKIP", "Cannot determine grid"))
                    continue
        except Exception as e:
            results.append((label, 0, 0, None, "SKIP", str(e)))
            continue

        try:
            pv_fp32 = load_pixel_values(pv_path)
            grid_thw_t = load_grid_thw(pv_path, engine)
            if grid_thw_t is None:
                results.append((label, 0, 0, None, "SKIP", "Cannot compute grid_thw"))
                continue
        except Exception as e:
            results.append((label, 0, 0, None, "SKIP", f"pixel_values: {e}"))
            continue

        input_ids_t = torch.tensor([input_ids], dtype=torch.long)
        S = input_ids_t.shape[1]
        n_vis = int(torch.prod(grid_thw_t, dim=1).sum()) // (engine.merge_size ** 2)

        try:
            with torch.no_grad():
                out = engine.forward(input_ids_t, pv_fp32, grid_thw_t)
            attn_mask = torch.ones_like(input_ids_t)
            pooled = engine.embedding_pooling(out, attn_mask).cpu().float()
        except Exception as e:
            results.append((label, S, n_vis, None, "ERROR", str(e)))
            continue

        cpp = load_pooler(cpp_path)
        cos = F.cosine_similarity(pooled.squeeze(0), cpp, dim=0).item()

        if cos < lowest_cosine:
            lowest_cosine = cos
        status = "PASS" if cos >= COSINE_THRESHOLD else "FAIL"
        if status == "FAIL":
            if first_failure is None:
                first_failure = label
            all_pass = False

        results.append((label, S, n_vis, cos, status, ""))

    # ──────────────────────────────────────────────────────────────
    # 3. MULTIMODAL (4 cases)
    # ──────────────────────────────────────────────────────────────
    for W, H in RESOLUTIONS:
        label = f"MM {W}x{H}"
        cpp_path = os.path.join(BIN_DIR, f"cpp_mm_{W}x{H}.bin")
        tok_path = os.path.join(BIN_DIR, f"tokens_chat_mm_{W}x{H}.bin")
        pv_path  = os.path.join(BIN_DIR, f"cpp_pv_{W}x{H}.bin")

        if not os.path.exists(cpp_path):
            results.append((label, 0, 0, None, "SKIP", "C++ output missing"))
            continue
        if not os.path.exists(pv_path):
            results.append((label, 0, 0, None, "SKIP", "pixel_values missing"))
            continue
        if not os.path.exists(tok_path):
            results.append((label, 0, 0, None, "SKIP", "MM token file missing"))
            continue

        try:
            input_ids = load_token_ids(tok_path)
            pv_fp32 = load_pixel_values(pv_path)
            grid_thw_t = load_grid_thw(pv_path, engine)
            if grid_thw_t is None:
                results.append((label, 0, 0, None, "SKIP", "Cannot compute grid_thw"))
                continue
        except Exception as e:
            results.append((label, 0, 0, None, "SKIP", str(e)))
            continue

        input_ids_t = torch.tensor([input_ids], dtype=torch.long)
        S = input_ids_t.shape[1]
        n_vis = int(torch.prod(grid_thw_t, dim=1).sum()) // (engine.merge_size ** 2)

        try:
            with torch.no_grad():
                out = engine.forward(input_ids_t, pv_fp32, grid_thw_t)
            attn_mask = torch.ones_like(input_ids_t)
            pooled = engine.embedding_pooling(out, attn_mask).cpu().float()
        except Exception as e:
            results.append((label, S, n_vis, None, "ERROR", str(e)))
            continue

        cpp = load_pooler(cpp_path)
        cos = F.cosine_similarity(pooled.squeeze(0), cpp, dim=0).item()

        if cos < lowest_cosine:
            lowest_cosine = cos
        status = "PASS" if cos >= COSINE_THRESHOLD else "FAIL"
        if status == "FAIL":
            if first_failure is None:
                first_failure = label
            all_pass = False

        results.append((label, S, n_vis, cos, status, ""))

    # ──────────────────────────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────────────────────────
    print()
    print("=" * 88)
    print(f"  Baseline Verification Results  (threshold: cosine >= {COSINE_THRESHOLD})")
    print("=" * 88)
    print(f"  {'Mode':<18} {'S':>6} {'VisTok':>8} {'Cosine':>12} {'Status':>8}")
    print(f"  {'─' * 18} {'─' * 6} {'─' * 8} {'─' * 12} {'─' * 8}")

    for label, S, n_vis, cos, status, note in results:
        if cos is not None:
            cos_str = f"{cos:.6f}"
        else:
            cos_str = "—"
        status_str = status
        if status == "PASS":
            status_str = f"\033[32mPASS\033[0m"
        elif status in ("FAIL", "ERROR"):
            status_str = f"\033[31m{status}\033[0m"
        elif status == "SKIP":
            status_str = f"\033[33mSKIP\033[0m"

        line = f"  {label:<18} {S:>6} {n_vis:>8} {cos_str:>12} {status_str}"
        # Use print() directly — colors are handled inline via ANSI codes
        sys.stdout.write(line + "\n")
        if note:
            sys.stdout.write(f"    ({note})\n")

    print(f"  {'─' * 18} {'─' * 6} {'─' * 8} {'─' * 12} {'─' * 8}")
    print(f"  Lowest cosine: {lowest_cosine:.6f}")

    if all_pass:
        print(f"\n  \033[32m\033[1mBASELINE PASS\033[0m — all cosine values >= {COSINE_THRESHOLD}")
    else:
        print(f"\n  \033[31m\033[1mBASELINE FAIL\033[0m — one or more cosine values below {COSINE_THRESHOLD}")
        if first_failure:
            print(f"  First failure: {first_failure}")

    print("=" * 88)

    # Return exit code for caller
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
PYEOF

# Run the comparison script
set +e
COMPARE_EXIT=0
QWEN3VL_EMB_MODEL_DIR="$MODEL_DIR" \
REPO_ROOT="$REPO_ROOT" \
COSINE_THRESHOLD="$COSINE_THRESHOLD" \
"$PYTHON_BIN" "$COMPARE_SCRIPT" 2>&1
COMPARE_EXIT=$?
set -e

COMPARE_END=$(date +%s)
COMPARE_DURATION=$((COMPARE_END - COMPARE_START))
log "  Python comparison finished in ${COMPARE_DURATION}s"

# Clean up temp scripts
rm -f "$COMPARE_SCRIPT" "$BENCH_LOG"

# ══════════════════════════════════════════════════════════════════════════
# Step 6 — Final report
# ══════════════════════════════════════════════════════════════════════════

TOTAL_DURATION=$((COMPARE_END - BENCH_START + BUILD_DURATION))
echo ""
log "Total elapsed: ~${TOTAL_DURATION}s (build + benchmark + compare)"
log "Commit: ${CURRENT_COMMIT:0:12}"

if [ "$COMPARE_EXIT" -eq 0 ]; then
    success "Baseline is clean — safe to proceed with changes."
    # Clean exit; trap will restore stash
    exit 0
else
    error "Baseline is BROKEN — do NOT proceed with changes until this is fixed."
    error "Investigate the failing cases above. Common causes:"
    error "  - Recent changes introduced precision regression"
    error "  - NPU environment mismatch (check ACL/ATB versions)"
    error "  - Token generation produced different tokens than C++ expects"
    error "  - Model checkpoint was updated"
    exit 1
fi
