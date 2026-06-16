#!/usr/bin/env python3
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
import fnmatch
import os
import subprocess
import sys
import time
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

TEST_TIMEOUT = 600  # 10 minutes per test


def run_one_test(
    name: str,
    script_path: Path,
    repo_root: Path,
    verbose: bool,
) -> dict:
    """Run a single test script as a subprocess.

    Returns a dict with keys: name, status, exit_code, elapsed_s, output.
    """
    env = os.environ.copy()
    # Ensure the package is importable from the repo root
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (":" + existing if existing else "")

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        output = f"[TIMEOUT after {TEST_TIMEOUT}s]"
        if e.stdout:
            stdout_text = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout
            output += "\n[STDOUT]\n" + stdout_text
        if e.stderr:
            stderr_text = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
            output += "\n[STDERR]\n" + stderr_text
        return {
            "name": name,
            "status": STATUS_FAIL,
            "exit_code": -1,
            "elapsed_s": elapsed,
            "output": output,
        }

    elapsed = time.monotonic() - start
    status = STATUS_PASS if proc.returncode == 0 else STATUS_FAIL
    output = proc.stdout
    if proc.stderr:
        output += "\n[STDERR]\n" + proc.stderr

    return {
        "name": name,
        "status": status,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "output": output,
    }


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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    tests_dir = Path(__file__).resolve().parent

    # Discover
    all_tests = discover_tests(tests_dir, include_benchmarks=args.benchmarks)
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

    print_header()
    results: list[dict] = []
    start_all = time.monotonic()
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
                print(f"\n{YELLOW}Stopping after first failure (--fail-fast).{RESET}")
                break

    total_elapsed = time.monotonic() - start_all
    print_final_summary(results, total_elapsed)

    if any_failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
