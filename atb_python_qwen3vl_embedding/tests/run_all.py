#!/usr/bin/env python3
from __future__ import annotations
"""Test runner: discover and run all Python test scripts in this directory.

Each test script is a standalone program (``if __name__ == "__main__"``)
that communicates results via exit code (0 = pass, non‑zero = fail).
This runner executes them as independent subprocesses to avoid
``set_atb_buffer_size`` conflicts, collects timing and output, and prints
a summary table.

Usage::

    python tests/run_all.py               # run all test_*.py files
    python tests/run_all.py --include '*e2e*'  # only tests matching *e2e*
    python tests/run_all.py --exclude '*310p*' # skip tests matching *310p*
    python tests/run_all.py --verbose      # print full stdout/stderr
    python tests/run_all.py --fail-fast    # stop after first failure
    python tests/run_all.py --list         # only list discovered tests
    python tests/run_all.py --benchmarks   # include benchmark.py

Exit code: 0 if all tests pass, non‑zero otherwise.
"""

import argparse
import concurrent.futures
import fnmatch
import os
import signal
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path


# ── discovery ──────────────────────────────────────────────────────────
def discover_tests(tests_dir: Path, include_benchmarks: bool = False):
    """Return sorted list of (display_name, file_path) tuples.

    *test_*.py* files are always included.  *benchmark.py* is only included
    when ``include_benchmarks`` is ``True``.
    """
    tests_dir = tests_dir.resolve()
    discovered: list[tuple[str, Path]] = []

    for p in sorted(tests_dir.glob("test_*.py")):
        discovered.append((p.stem, p))

    if include_benchmarks:
        bench = tests_dir / "benchmark.py"
        if bench.exists():
            discovered.append(("benchmark", bench))

    return discovered


# ── filtering ──────────────────────────────────────────────────────────
def apply_filters(
    tests: list[tuple[str, Path]],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[tuple[str, Path]]:
    """Filter tests by glob patterns.

    Patterns follow fnmatch rules (``*``, ``?``, ``[...]``).
    Use explicit wildcards for substring matching (e.g. ``*e2e*``).

    *include*: keep only tests whose stem matches at least one pattern.
    *exclude*: drop tests whose stem matches any pattern.
    Exclude is applied after include.
    """
    result = tests

    if include:
        result = [
            (name, path)
            for name, path in result
            if any(fnmatch.fnmatch(name, pat) for pat in include)
        ]

    if exclude:
        result = [
            (name, path)
            for name, path in result
            if not any(fnmatch.fnmatch(name, pat) for pat in exclude)
        ]

    return result


# ── execution ──────────────────────────────────────────────────────────
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

# ── resource groups for parallel execution ───────────────────────────
# Same-group tests run serially; different-group tests can run in parallel.
# "exclusive" — full model (10+ GB HBM), must be completely serial
# "text_model" — only Text DecoderLayer graph, ≈5 GB
# "vision_model" — only VisionBlock graph, ≈5 GB
# "light" (default) — small buffers (≤1 GB), N-way parallel
TEST_RESOURCE_GROUPS = {
    "test_e2e": "exclusive",
    "test_embedder_e2e": "exclusive",
    "test_deepstack_integration": "exclusive",
    "test_pipeline_trace": "exclusive",
    "test_text_model": "exclusive",
    "test_vision_pos_embed": "exclusive",
    "test_text_diagnostics": "text_model",
    "test_vision_diagnostics": "vision_model",
}
# Unlisted tests default to "light"

TEST_TIMEOUT = 600  # 10 minutes per test


def get_current_platform() -> str | None:
    """Read ASCEND_PLATFORM from the package env module.

    Returns '910B', '310P', or None if the import fails.
    """
    try:
        from atb_python_qwen3vl_embedding.env import ASCEND_PLATFORM
        return ASCEND_PLATFORM
    except Exception:
        return None


def run_one_test(
    name: str,
    script_path: Path,
    repo_root: Path,
    verbose: bool,
    running_pids: dict | None = None,
    pids_lock: threading.Lock | None = None,
) -> dict:
    """Run a single test script as a subprocess.

    Returns a dict with keys: name, status, exit_code, elapsed_s, output, pid.
    """
    env = os.environ.copy()
    # Ensure the package is importable from the repo root
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (":" + existing if existing else "")

    start = time.monotonic()
    proc = None

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if running_pids is not None and pids_lock is not None:
            with pids_lock:
                running_pids[name] = proc.pid

        try:
            stdout, stderr = proc.communicate(timeout=TEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            elapsed = time.monotonic() - start
            output = f"[TIMEOUT after {TEST_TIMEOUT}s]"
            if stdout:
                output += "\n[STDOUT]\n" + stdout
            if stderr:
                output += "\n[STDERR]\n" + stderr
            return {
                "name": name,
                "status": STATUS_FAIL,
                "exit_code": -1,
                "elapsed_s": elapsed,
                "output": output,
                "pid": proc.pid,
            }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {
            "name": name,
            "status": STATUS_FAIL,
            "exit_code": -1,
            "elapsed_s": elapsed,
            "output": f"[EXCEPTION] {type(e).__name__}: {e}",
            "pid": proc.pid if proc is not None else None,
        }
    finally:
        if proc is not None and running_pids is not None and pids_lock is not None:
            with pids_lock:
                running_pids.pop(name, None)

    elapsed = time.monotonic() - start
    status = STATUS_PASS if proc.returncode == 0 else STATUS_FAIL
    output = stdout
    if stderr:
        output += "\n[STDERR]\n" + stderr

    return {
        "name": name,
        "status": status,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "output": output,
        "pid": proc.pid,
    }


# ── parallel execution ──────────────────────────────────────────────────

def run_tests_parallel(
    tests: list[tuple[str, Path]],
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[list[dict], bool]:
    """Run tests with resource-group-aware parallelism.

    Tests in the same resource group run serially; tests in different groups
    may run concurrently.  ``light`` group concurrency is capped at
    ``args.parallel``.

    Returns ``(results_in_original_order, any_failed)``.
    """
    max_per_group = {
        "exclusive": 1,
        "text_model": 1,
        "vision_model": 1,
        "light": args.parallel,
    }
    running_per_group = {"exclusive": 0, "text_model": 0, "vision_model": 0, "light": 0}

    pending = list(tests)          # (name, path) — submission order
    running: dict[concurrent.futures.Future, tuple[int, str, str]] = {}
    results_by_name: dict[str, dict] = {}
    original_order = [name for name, _ in tests]
    total = len(tests)
    any_failed = False

    # Track subprocess pids for fail-fast SIGTERM delivery
    running_pids: dict[str, int] = {}
    pids_lock = threading.Lock()

    # Total thread pool size = sum of per-group concurrency caps:
    #   exclusive(1) + text_model(1) + vision_model(1) + light(N)
    max_workers = sum(max_per_group.values())

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        print_header()
        while pending or running:
            # Submit tests whose resource group still has capacity
            while pending:
                name, path = pending[0]
                group = TEST_RESOURCE_GROUPS.get(name, "light")
                if running_per_group[group] < max_per_group[group]:
                    pending.pop(0)
                    running_per_group[group] += 1
                    idx = original_order.index(name)
                    fut = executor.submit(
                        partial(run_one_test, name, path, repo_root,
                                args.verbose, running_pids, pids_lock)
                    )
                    running[fut] = (idx, name, group)
                else:
                    break

            if not running:
                break

            # Wait for at least one running test to finish
            done, _ = concurrent.futures.wait(
                running, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for fut in done:
                idx, name, group = running.pop(fut)
                running_per_group[group] -= 1
                result = fut.result()
                results_by_name[name] = result

                # Progress display with [idx/total] prefix + aligned columns
                print(f"[{idx + 1}/{total}] ", end="")
                print_summary_row(result)

                if args.verbose:
                    print_verbose_output(result)

                if result["status"] == STATUS_FAIL:
                    any_failed = True
                    if args.fail_fast:
                        pending.clear()

            if args.fail_fast and any_failed:
                # Send SIGTERM to all still-running subprocesses
                with pids_lock:
                    for pid in list(running_pids.values()):
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except OSError:
                            pass
                for f in running:
                    f.cancel()
                break
    finally:
        executor.shutdown(wait=not (args.fail_fast and any_failed))

    # Return results in original test-list order
    ordered_results = [
        results_by_name[name] for name in original_order if name in results_by_name
    ]
    return ordered_results, any_failed


# ── reporting ──────────────────────────────────────────────────────────
HEADER_FMT = "{:<42s} {:>6s}  {:>8s}  {:>8s}"
ROW_FMT = "{:<42s} {:>6s}  {:>7.1f}s  {:>8d}"

if sys.stdout.isatty():
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
else:
    GREEN = RED = YELLOW = RESET = ""


def color_status(status: str) -> str:
    if status == STATUS_PASS:
        return f"{GREEN}{status}{RESET}"
    return f"{RED}{status}{RESET}"


def print_header():
    print()
    print(HEADER_FMT.format("Test", "Status", "Time", "Exit"))
    print(HEADER_FMT.format("-" * 40, "-" * 6, "-" * 8, "-" * 8))


def print_summary_row(result: dict):
    print(
        ROW_FMT.format(
            result["name"],
            color_status(result["status"]),
            result["elapsed_s"],
            result["exit_code"],
        )
    )


def print_verbose_output(result: dict):
    if result["output"].strip():
        print(f"\n--- {result['name']} output ---")
        print(result["output"])
        print(f"--- end {result['name']} ---")


def print_final_summary(
    results: list[dict], total_elapsed: float
):
    passed = sum(1 for r in results if r["status"] == STATUS_PASS)
    failed = sum(1 for r in results if r["status"] == STATUS_FAIL)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"Total time: {total_elapsed:.1f}s")
    if failed > 0:
        print(f"\n{YELLOW}Failed tests:{RESET}")
        for r in results:
            if r["status"] == STATUS_FAIL:
                print(f"  {r['name']} (exit {r['exit_code']})")


# ── main ───────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discover and run all Python test scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--include",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Only run tests whose name matches PATTERN (fnmatch, can repeat).",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Skip tests whose name matches PATTERN (fnmatch, can repeat).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full stdout/stderr for every test.",
    )
    p.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop after the first failure.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Only list discovered tests, don't run them.",
    )
    p.add_argument(
        "--benchmarks", "-b",
        action="store_true",
        help="Include benchmark.py in the run.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Maximum concurrent light-group tests (1-8, default 1 = serial).",
    )
    p.add_argument(
        "--no-auto-exclude",
        action="store_true",
        help="Disable automatic platform-based test exclusion.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Clamp --parallel to [1, 8]
    if args.parallel < 1:
        print(f"Warning: --parallel {args.parallel} clamped to 1", file=sys.stderr)
        args.parallel = 1
    elif args.parallel > 8:
        print(f"Warning: --parallel {args.parallel} clamped to 8", file=sys.stderr)
        args.parallel = 8

    # Paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    tests_dir = Path(__file__).resolve().parent

    # Discover
    all_tests = discover_tests(tests_dir, include_benchmarks=args.benchmarks)

    # Auto-exclude tests that don't match the current platform
    if not args.no_auto_exclude:
        platform = get_current_platform()
        if platform == "910B":
            auto_exclude = ["*310p*", "*310P*"]
            if args.exclude:
                args.exclude = list(args.exclude) + auto_exclude
            else:
                args.exclude = auto_exclude

    tests = apply_filters(all_tests, args.include, args.exclude)

    if not tests:
        print("No tests found matching the given filters.")
        if args.include or args.exclude:
            print(f"Available tests: {', '.join(n for n, _ in all_tests)}")
        return 2 if (args.include or args.exclude) else 0

    if args.list:
        print(f"Discovered {len(tests)} test(s):")
        for name, path in tests:
            tag = " [benchmark]" if name == "benchmark" else ""
            print(f"  {name}{tag}  →  {path}")
        return 0

    if args.benchmarks:
        print(f"{YELLOW}Note: benchmark.py is included.{RESET}")

    print(f"Running {len(tests)} test(s) from {tests_dir}")
    print(f"Repo root: {repo_root}")
    print(f"Python:   {sys.executable}")

    start_all = time.monotonic()

    if args.parallel == 1:
        # ── serial mode (original behaviour, backward compatible) ──────
        print_header()
        results: list[dict] = []
        any_failed = False

        for name, path in tests:
            result = run_one_test(name, path, repo_root, args.verbose)
            results.append(result)
            print_summary_row(result)

            if args.verbose:
                print_verbose_output(result)

            if result["status"] == STATUS_FAIL:
                any_failed = True
                if args.fail_fast:
                    print(
                        f"\n{YELLOW}Stopping after first failure"
                        f" (--fail-fast).{RESET}"
                    )
                    break
    else:
        # ── parallel mode (resource-group-aware) ───────────────────────
        print(f"Parallelism: light={args.parallel}, exclusive/text/vision=1")
        results, any_failed = run_tests_parallel(tests, args, repo_root)

    total_elapsed = time.monotonic() - start_all
    print_final_summary(results, total_elapsed)

    if any_failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
