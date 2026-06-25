"""G5: Sync safety experiment — systematically test which syncs are essential.

Runs the C++ benchmark --mode compare under multiple sync configurations,
each repeated N trials to account for non-deterministic NPU behavior.

Sync configurations tested (per-op sync is OFF by default; ATB_ENABLE_PER_OP_SYNC=1
opts back in. PER_OP=on => per-op Synchronize on; TIMING=on => stage-boundary
timing syncs on, i.e. ATB_SKIP_TIMING_SYNCS unset):
  A: Full sync (baseline)        PER_OP=on   TIMING=on   LAUNCH_BLOCKING=0
  B: No per-op sync              PER_OP=off  TIMING=on   LAUNCH_BLOCKING=0
  C: No timing syncs             PER_OP=on   TIMING=off  LAUNCH_BLOCKING=0
  D: Minimal sync                PER_OP=off  TIMING=off  LAUNCH_BLOCKING=0
  E: CANN launch blocking        PER_OP=off  TIMING=off  LAUNCH_BLOCKING=1

For each config × trial, the C++ benchmark runs --mode compare (13 combos).
Each C++ output .bin is compared against the Python ATB reference via cosine.

Key question: does removing syncs preserve correctness (cosine >= 0.99)
AND improve performance, or does it introduce instability?

Threshold: cosine >= 0.99 (NEVER lowered — see testing-guide.md)

Requires: NPU device, model checkpoint, compiled benchmark binary.
Run: python3 tests/level4_e2e/test_sync_safety.py [--trials N] [--quick]
"""

import os
import subprocess
import sys
import struct
import time
import json
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ────────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR as _RESOLVED_MODEL_DIR, REPO_ROOT  # noqa: E402
PROJECT_DIR = str(REPO_ROOT / "atb_cpp_llm")
BUILD_DIR = os.path.join(PROJECT_DIR, "build")
BENCHMARK_BIN = os.path.join(BUILD_DIR, "benchmark")
MODEL_DIR = _RESOLVED_MODEL_DIR
BIN_DIR = "/tmp"

# ── Sync configurations ──────────────────────────────────────
# Each is (label, env_override_dict)
SYNC_CONFIGS = [
    ("A: full sync (baseline)",     {"ATB_ENABLE_PER_OP_SYNC": "1"}),
    ("B: no per-op sync",           {}),  # per-op off by default
    ("C: no timing syncs",          {"ATB_ENABLE_PER_OP_SYNC": "1",
                                      "ATB_SKIP_TIMING_SYNCS": "1"}),
    ("D: minimal sync",             {"ATB_SKIP_TIMING_SYNCS": "1"}),
    ("E: CANN launch blocking",     {"ATB_SKIP_TIMING_SYNCS": "1",
                                      "ASCEND_LAUNCH_BLOCKING": "1"}),
]

# ── 13 test combinations (must match benchmark --mode compare) ──
TEXT_SEQ_LENGTHS = [100, 512, 1024, 2048, 4096]
RESOLUTIONS = [(416, 672), (720, 1280), (1080, 1920), (1440, 2560)]

# ── Bin file helpers ─────────────────────────────────────────

def load_token_ids(path):
    """Load C++-format token IDs: [int32 count] [int64 * count]."""
    with open(path, "rb") as f:
        count = struct.unpack("<i", f.read(4))[0]
        fmt = f"<{count}q"
        return list(struct.unpack(fmt, f.read(count * 8)))

def load_pooler(path):
    """Load pooler output: [int64 dim] [float32 * dim]."""
    with open(path, "rb") as f:
        dim = struct.unpack("<q", f.read(8))[0]
        data = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return torch.from_numpy(data.copy()).float()

def load_pixel_values(path):
    """Load C++-preprocessed pixel_values: [int32 count] [uint16 * count]."""
    with open(path, "rb") as f:
        count = struct.unpack("<i", f.read(4))[0]
        raw = f.read(count * 2)
    return np.frombuffer(raw, dtype=np.float16).copy()

# ── Reference generation (Python ATB — known correct) ────────

def generate_reference():
    """Run Python ATB engine once to generate reference pooler outputs."""
    print("=" * 70)
    print("Generating Python ATB reference outputs (once)...")
    print("=" * 70)

    sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_DIR),
                                     "atb_python_qwen3vl_embedding"))

    from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine
    from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
    set_atb_buffer_size(10 * 1024 * 1024 * 1024)  # 10 GB

    t0 = time.time()
    engine = Qwen3VLEngine(MODEL_DIR)
    print(f"Engine loaded in {time.time() - t0:.1f}s")

    ref_outputs = {}
    ref_perf = {}

    # ── TEXT_ONLY ──────────────────────────────────────────
    for S in TEXT_SEQ_LENGTHS:
        label = f"TEXT {S}"
        tok_path = os.path.join(BIN_DIR, f"tokens_chat_text_only_{S}.bin")
        try:
            input_ids = load_token_ids(tok_path)
        except (FileNotFoundError, struct.error):
            print(f"  SKIP {label}: token file not found")
            continue
        input_ids_t = torch.tensor([input_ids], dtype=torch.long)
        t1 = time.time()
        with torch.no_grad():
            out = engine.forward(input_ids_t, None, None)
        elapsed = time.time() - t1
        attn_mask = torch.ones_like(input_ids_t)
        pooled = engine.embedding_pooling(out, attn_mask).cpu().float()
        ref_outputs[label] = pooled.squeeze(0)
        ref_perf[label] = elapsed
        print(f"  {label:<18} ({S:>4} tok)  {elapsed*1000:.1f}ms")

    # ── IMAGE_ONLY ─────────────────────────────────────────
    for W, H in RESOLUTIONS:
        label = f"IO {W}x{H}"
        tok_path = os.path.join(BIN_DIR, f"tokens_chat_io_{W}x{H}.bin")
        pv_path = os.path.join(BIN_DIR, f"cpp_pv_{W}x{H}.bin")
        if not os.path.exists(pv_path):
            print(f"  SKIP {label}: pixel_values not found")
            continue
        pv_fp32 = torch.from_numpy(
            load_pixel_values(pv_path).astype(np.float32)
        )

        # Compute grid_thw from SmartResize
        from atb_python_qwen3vl_embedding.preprocess import smart_resize
        factor = engine.patch_size * engine.merge_size
        new_h, new_w = smart_resize(H, W, factor=factor,
                                     min_pixels=engine.pp_min_px,
                                     max_pixels=engine.pp_max_px)
        grid_h, grid_w = new_h // engine.patch_size, new_w // engine.patch_size
        grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

        n_patches = grid_h * grid_w
        merger_tokens = n_patches // (engine.merge_size ** 2)
        input_ids = [engine.img_tok] * merger_tokens
        input_ids_t = torch.tensor([input_ids], dtype=torch.long)

        t1 = time.time()
        with torch.no_grad():
            out = engine.forward(input_ids_t, pv_fp32, grid_thw)
        elapsed = time.time() - t1
        attn_mask = torch.ones_like(input_ids_t)
        pooled = engine.embedding_pooling(out, attn_mask).cpu().float()
        ref_outputs[label] = pooled.squeeze(0)
        ref_perf[label] = elapsed
        S = input_ids_t.shape[1]
        print(f"  {label:<18} ({S:>4} tok, {n_patches:>4} patches)  "
              f"{elapsed*1000:.1f}ms")

    # ── MULTIMODAL ─────────────────────────────────────────
    for W, H in RESOLUTIONS:
        label = f"MM {W}x{H}"
        tok_path = os.path.join(BIN_DIR, f"tokens_chat_mm_{W}x{H}.bin")
        pv_path = os.path.join(BIN_DIR, f"cpp_pv_{W}x{H}.bin")
        if not os.path.exists(tok_path) or not os.path.exists(pv_path):
            print(f"  SKIP {label}: input files not found")
            continue

        input_ids = load_token_ids(tok_path)
        pv_fp32 = torch.from_numpy(
            load_pixel_values(pv_path).astype(np.float32)
        )
        from atb_python_qwen3vl_embedding.preprocess import smart_resize
        factor = engine.patch_size * engine.merge_size
        new_h, new_w = smart_resize(H, W, factor=factor,
                                     min_pixels=engine.pp_min_px,
                                     max_pixels=engine.pp_max_px)
        grid_h, grid_w = new_h // engine.patch_size, new_w // engine.patch_size
        grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
        n_patches = grid_h * grid_w

        input_ids_t = torch.tensor([input_ids], dtype=torch.long)
        t1 = time.time()
        with torch.no_grad():
            out = engine.forward(input_ids_t, pv_fp32, grid_thw)
        elapsed = time.time() - t1
        attn_mask = torch.ones_like(input_ids_t)
        pooled = engine.embedding_pooling(out, attn_mask).cpu().float()
        ref_outputs[label] = pooled.squeeze(0)
        ref_perf[label] = elapsed
        S = input_ids_t.shape[1]
        print(f"  {label:<18} ({S:>4} tok, {n_patches:>4} patches)  "
              f"{elapsed*1000:.1f}ms")

    print(f"\nReference generation complete: {len(ref_outputs)} cases\n")
    return ref_outputs, ref_perf

# ── C++ benchmark runner ─────────────────────────────────────

def run_cpp_benchmark(env_overrides, trial_label):
    """Run C++ benchmark --mode compare with given env vars.
    Returns (success, elapsed_sec, cpp_outputs_dict, stderr_lines).
    cpp_outputs_dict: {label: torch.Tensor} for the 13 cases.
    """
    env = os.environ.copy()
    env.update(env_overrides)

    # Ensure gen_baseline_tokens.py has been run (generate token files)
    # and ensure benchmark binary exists
    if not os.path.exists(BENCHMARK_BIN):
        print(f"    ERROR: benchmark binary not found at {BENCHMARK_BIN}")
        return False, 0, {}, ["build missing"]

    # Clean up previous output .bin files so we don't read stale ones
    for pattern in ["cpp_text_only_*.bin", "cpp_io_*.bin", "cpp_mm_*.bin"]:
        import glob
        for f in glob.glob(os.path.join(BIN_DIR, pattern)):
            os.remove(f)

    t_start = time.time()
    try:
        proc = subprocess.run(
            [BENCHMARK_BIN, "--mode", "compare", "--iter", "2", "--warmup", "1"],
            env=env,
            cwd=BUILD_DIR,
            capture_output=True,
            timeout=600,  # 10 min timeout
        )
    except subprocess.TimeoutExpired:
        return False, 0, {}, ["TIMEOUT (>600s)"]
    except Exception as e:
        return False, 0, {}, [str(e)]

    elapsed = time.time() - t_start
    stderr_lines = proc.stderr.decode("utf-8", errors="replace").splitlines()
    stdout_lines = proc.stdout.decode("utf-8", errors="replace").splitlines()

    if proc.returncode != 0:
        # Collect last error lines
        err_tail = stderr_lines[-20:] if len(stderr_lines) > 20 else stderr_lines
        return False, elapsed, {}, [f"exit={proc.returncode}"] + err_tail

    # Parse C++ output .bin files
    cpp_outputs = {}
    for S in TEXT_SEQ_LENGTHS:
        label = f"TEXT {S}"
        path = os.path.join(BIN_DIR, f"cpp_text_only_{S}.bin")
        try:
            cpp_outputs[label] = load_pooler(path)
        except (FileNotFoundError, struct.error):
            pass

    for W, H in RESOLUTIONS:
        for prefix, mode_label in [("io", "IO"), ("mm", "MM")]:
            label = f"{mode_label} {W}x{H}"
            path = os.path.join(BIN_DIR, f"cpp_{prefix}_{W}x{H}.bin")
            try:
                cpp_outputs[label] = load_pooler(path)
            except (FileNotFoundError, struct.error):
                pass

    return True, elapsed, cpp_outputs, stderr_lines

# ── Cosine comparison ────────────────────────────────────────

def compute_cosine(cpp_tensor, ref_tensor):
    """Compute cosine similarity between two 1D tensors."""
    return F.cosine_similarity(
        cpp_tensor.unsqueeze(0), ref_tensor.unsqueeze(0)
    ).item()

# ── Main experiment ──────────────────────────────────────────

def run_experiment(trials=5, quick=False):
    """Run the full sync safety experiment matrix."""
    print("=" * 70)
    print("G5: Sync Safety Experiment")
    print(f"  Trials per config: {trials}")
    print(f"  Configurations: {len(SYNC_CONFIGS)}")
    print(f"  Threshold: cosine >= 0.99")
    print("=" * 70)

    # ── Step 1: Generate Python reference (once) ────────────
    ref_outputs, ref_perf = generate_reference()
    if len(ref_outputs) == 0:
        print("ERROR: No reference outputs generated — check model/NPU setup")
        return False

    ref_labels = sorted(ref_outputs.keys())
    print(f"Reference cases: {ref_labels}")

    # ── Step 2: Build benchmark ─────────────────────────────
    print("=" * 70)
    print("Building C++ benchmark...")
    t0 = time.time()
    build_env = os.environ.copy()
    proc = subprocess.run(
        ["cmake", "--build", BUILD_DIR, "--target", "benchmark", "-j",
         str(os.cpu_count() or 4)],
        env=build_env, capture_output=True, timeout=300
    )
    if proc.returncode != 0:
        print("BUILD FAILED:")
        print(proc.stderr.decode("utf-8", errors="replace")[-40:])
        return False
    print(f"Build OK ({time.time() - t0:.1f}s)")

    # ── Step 3: Ensure token files exist ────────────────────
    token_gen = os.path.join(
        os.path.dirname(PROJECT_DIR),
        "atb_python_qwen3vl_embedding/tests/gen_baseline_tokens.py"
    )
    if os.path.exists(token_gen):
        subprocess.run(
            ["python3", token_gen],
            env=os.environ.copy(), capture_output=True, timeout=120
        )

    # ── Step 4: Run experiment matrix ───────────────────────
    results = {}  # config_label -> {trials: [...], cosine_history: {...}}

    for cfg_idx, (cfg_label, env_overrides) in enumerate(SYNC_CONFIGS):
        print()
        print("=" * 70)
        print(f"[{cfg_idx + 1}/{len(SYNC_CONFIGS)}] {cfg_label}")
        print(f"  Env: {env_overrides or '(baseline)'}")
        print("=" * 70)

        cfg_trials = []
        cfg_cosines = defaultdict(list)  # label -> [cosine values]
        crashes = 0
        invalid_outputs = 0

        for trial in range(trials):
            trial_label = f"T{trial + 1}/{trials}"
            ok, elapsed, cpp_outputs, stderr = run_cpp_benchmark(
                env_overrides, trial_label
            )

            if not ok:
                crashes += 1
                err_summary = " | ".join(
                    [l for l in stderr if l][:3]
                )
                print(f"  [{trial_label}] CRASH ({elapsed:.1f}s): {err_summary}")
                cfg_trials.append({
                    "trial": trial,
                    "ok": False,
                    "elapsed": elapsed,
                    "cosines": {},
                    "error": err_summary,
                })
                continue

            # Compare against reference
            trial_cosines = {}
            all_pass = True
            lowest_cos = 1.0
            zero_outputs = []

            for label in ref_labels:
                if label not in cpp_outputs:
                    trial_cosines[label] = None
                    all_pass = False
                    continue

                cpp_t = cpp_outputs[label]
                ref_t = ref_outputs[label]

                # Check for all-zero output (sign of sync failure)
                cpp_norm = torch.norm(cpp_t).item()
                if cpp_norm < 1e-6:
                    zero_outputs.append(label)
                    invalid_outputs += 1
                    all_pass = False
                    trial_cosines[label] = 0.0
                    continue

                cos = compute_cosine(cpp_t, ref_t)
                trial_cosines[label] = cos
                cfg_cosines[label].append(cos)
                if cos < lowest_cos:
                    lowest_cos = cos
                if cos < 0.99:
                    all_pass = False

            # Status line
            n_valid = len([c for c in trial_cosines.values()
                          if c is not None and c > 0.0])
            status = "PASS" if all_pass else "FAIL"
            if zero_outputs:
                status += f" ZERO:{','.join(zero_outputs)}"

            print(f"  [{trial_label}] {status}  "
                  f"lowest_cos={lowest_cos:.6f}  "
                  f"({n_valid}/{len(ref_labels)} valid)  "
                  f"{elapsed:.1f}s")

            cfg_trials.append({
                "trial": trial,
                "ok": all_pass,
                "elapsed": elapsed,
                "cosines": trial_cosines,
                "lowest_cosine": lowest_cos,
                "zero_outputs": zero_outputs,
            })

        # ── Config summary ─────────────────────────────────
        n_pass = sum(1 for t in cfg_trials if t["ok"])
        avg_elapsed = (
            sum(t["elapsed"] for t in cfg_trials if t["ok"]) /
            max(n_pass, 1)
        )
        stability = n_pass / trials

        # Compute per-label cosine stats
        cosine_stats = {}
        for label in ref_labels:
            vals = cfg_cosines.get(label, [])
            if vals:
                cosine_stats[label] = {
                    "mean": np.mean(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "std": np.std(vals) if len(vals) > 1 else 0.0,
                    "count": len(vals),
                }
            else:
                cosine_stats[label] = {
                    "mean": None, "min": None, "max": None,
                    "std": None, "count": 0,
                }

        # Overall lowest cosine across all trials
        all_cosines = [
            c for vals in cfg_cosines.values() for c in vals
        ]
        overall_lowest = min(all_cosines) if all_cosines else 0.0

        print(f"  ── Summary ──")
        print(f"  Pass rate: {n_pass}/{trials} ({stability*100:.0f}%)")
        print(f"  Crashes: {crashes}, Invalid outputs: {invalid_outputs}")
        print(f"  Lowest cosine (any trial): {overall_lowest:.6f}")
        print(f"  Avg elapsed (passing): {avg_elapsed:.1f}s")

        results[cfg_label] = {
            "env": env_overrides,
            "trials": cfg_trials,
            "cosine_stats": cosine_stats,
            "stability": stability,
            "avg_elapsed": avg_elapsed,
            "overall_lowest_cosine": overall_lowest,
            "crashes": crashes,
            "invalid_outputs": invalid_outputs,
        }

    # ══════════════════════════════════════════════════════════
    # Final report
    # ══════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("FINAL REPORT: G5 Sync Safety Experiment")
    print("=" * 70)

    # Header
    print(f"\n{'Config':<36} {'Stability':>10} {'Avg Time':>10} "
          f"{'Lowest Cos':>12} {'Verdict':>10}")
    print("-" * 78)

    for cfg_label in [c[0] for c in SYNC_CONFIGS]:
        r = results.get(cfg_label)
        if not r:
            continue
        stab_pct = f"{r['stability']*100:.0f}%"
        time_s = f"{r['avg_elapsed']:.1f}s"
        cos_s = f"{r['overall_lowest_cosine']:.6f}"

        if r['stability'] >= 0.9 and r['overall_lowest_cosine'] >= 0.99:
            verdict = "✅ SAFE"
        elif r['stability'] >= 0.5 and r['overall_lowest_cosine'] >= 0.99:
            verdict = "⚠️ UNSTABLE"
        elif r['stability'] >= 0.9:
            verdict = "❌ LOW-COS"
        else:
            verdict = "💀 BROKEN"

        print(f"{cfg_label:<36} {stab_pct:>10} {time_s:>10} "
              f"{cos_s:>12} {verdict:>10}")

    print("-" * 78)

    # Performance comparison
    baseline_time = None
    for cfg_label in [c[0] for c in SYNC_CONFIGS]:
        r = results.get(cfg_label)
        if r and "baseline" in cfg_label.lower():
            baseline_time = r["avg_elapsed"]
            break

    if baseline_time and baseline_time > 0:
        print(f"\nPerformance vs baseline ({baseline_time:.1f}s):")
        for cfg_label in [c[0] for c in SYNC_CONFIGS]:
            r = results.get(cfg_label)
            if not r or r["avg_elapsed"] <= 0:
                continue
            ratio = baseline_time / r["avg_elapsed"]
            faster = "faster" if ratio >= 1.0 else "slower"
            print(f"  {cfg_label:<50} {r['avg_elapsed']:.1f}s  "
                  f"({ratio:.2f}x {faster})")

    # ── Highlight: ASCEND_LAUNCH_BLOCKING impact ──────────────
    print()
    print("─" * 70)
    print("ASCEND_LAUNCH_BLOCKING analysis:")
    for cfg_label in [c[0] for c in SYNC_CONFIGS]:
        if "LAUNCH_BLOCKING" in cfg_label.upper() or "launch blocking" in cfg_label.lower():
            r = results.get(cfg_label)
            if not r:
                continue
            print(f"  {cfg_label}:")
            print(f"    Stability: {r['stability']*100:.0f}%")
            print(f"    Lowest cos: {r['overall_lowest_cosine']:.6f}")
            print(f"    Elapsed: {r['avg_elapsed']:.1f}s")

            # Check per-label cosine detail
            if r["cosine_stats"]:
                problem_labels = []
                for label, stats in r["cosine_stats"].items():
                    if stats["min"] is not None and stats["min"] < 0.99:
                        problem_labels.append(
                            f"{label} (min={stats['min']:.6f})"
                        )
                if problem_labels:
                    print(f"    Problem cases: {', '.join(problem_labels)}")
                else:
                    print(f"    All cases stable")

    # ── Final verdict ────────────────────────────────────────
    print()
    print("=" * 70)
    all_configs_safe = all(
        r["stability"] >= 0.9 and r["overall_lowest_cosine"] >= 0.99
        for r in results.values()
    )
    if all_configs_safe:
        print("ALL CONFIGS SAFE — sync removal does not affect correctness.")
        print("Recommendation: use minimal sync for best performance.")
    else:
        print("Some configs are UNSAFE.")
        print("Recommendation: see per-config verdict above for guidance.")

    print("=" * 70)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="G5 Sync Safety Experiment")
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per config (default: 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 trials, skip slowest cases")
    args = parser.parse_args()

    if args.quick:
        args.trials = min(args.trials, 2)

    ok = run_experiment(trials=args.trials, quick=args.quick)
    sys.exit(0 if ok else 1)
