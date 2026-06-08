"""
C++ vs Python ATB Engine Performance Comparison.

Uses IDENTICAL inputs (same token IDs, same sequence length, same model)
for a fair side-by-side text-only inference comparison.

Run:
    python tests/compare_bench.py [--iter N] [--warmup M] [--seq S]

Outputs CSV at end for easy copy-paste into spreadsheets.
"""

import sys, os, time, argparse, struct, subprocess, re
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/mnt/workspace/gitCode/atb_llm')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--seq', type=int, default=64, help='Sequence length')
parser.add_argument('--all-seqs', action='store_true',
                    help='Test multiple sequence lengths: 4, 16, 64, 256, 1024')
args = parser.parse_args()

MODEL_DIR = "/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B"
CPP_BUILD_DIR = "/mnt/workspace/gitCode/atb_llm/atb_cpp_llm/build"
CPP_BENCHMARK = os.path.join(CPP_BUILD_DIR, "benchmark")

def stats_ms(arr):
    a = np.array(arr) * 1000
    return {
        'mean': float(a.mean()),       'std': float(a.std()),
        'min':  float(a.min()),        'max':  float(a.max()),
        'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
    }

def make_input_ids(seq_len):
    """Same format as C++ benchmark: [151643, 15339, ..., 15339, 151645]"""
    ids = [151643] + [15339] * (seq_len - 2) + [151645]
    return torch.tensor([ids], dtype=torch.long)

def run_python_benchmark(py_engine, seq_len, n_warmup, n_iter):
    """Run Python ATB engine benchmark. Returns stats dict and raw times (ms)."""
    input_ids = make_input_ids(seq_len)

    for _ in range(n_warmup):
        py_engine.encode(input_ids, normalize=True)
    torch.npu.synchronize()

    times = []
    for i in range(n_iter):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        py_engine.encode(input_ids, normalize=True)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return stats_ms(times), times

def run_cpp_benchmark(seq_len, n_warmup, n_iter):
    """Run C++ benchmark as subprocess. Returns stats dict."""
    cmd = [CPP_BENCHMARK, '--iter', str(n_iter), '--warmup', str(n_warmup),
           '--seq', str(seq_len), '--cmp']

    result = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=CPP_BUILD_DIR, timeout=600)
    # Parse CPP_RESULT line
    for line in result.stdout.split('\n'):
        if line.startswith('CPP_RESULT:'):
            parts = line.split()
            stats = {}
            for p in parts[1:]:
                k, v = p.split('=')
                stats[k] = float(v)
            return stats

    # Fallback: parse from stderr
    stats = {}
    for line in result.stderr.split('\n'):
        m = re.search(r'Mean:\s+([\d.]+)\s+ms', line)
        if m: stats['mean'] = float(m.group(1))
        m = re.search(r'Median:\s+([\d.]+)\s+ms', line)
        if m: stats['median'] = float(m.group(1))
        m = re.search(r'Min:\s+([\d.]+)\s+ms', line)
        if m: stats['min'] = float(m.group(1))
        m = re.search(r'Max:\s+([\d.]+)\s+ms', line)
        if m: stats['max'] = float(m.group(1))
        m = re.search(r'Stddev:\s+([\d.]+)\s+ms', line)
        if m: stats['std'] = float(m.group(1))
        m = re.search(r'P95:\s+([\d.]+)\s+ms', line)
        if m: stats['p95'] = float(m.group(1))
    return stats if stats else None

def print_table(results):
    """Print side-by-side comparison table."""
    hdr = f"{'Seq':>6s}  {'Python(ms)':>12s}  {'C++(ms)':>10s}  {'Speedup':>8s}  {'Faster':>8s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        speedup = r['py_mean'] / r['cpp_mean'] if r['cpp_mean'] > 0 else 0
        faster = "C++" if speedup > 1.0 else ("Python" if speedup < 1.0 else "Equal")
        arrow = f"{speedup:.2f}x" if speedup != 0 else "N/A"
        print(f"{r['seq']:>6d}  {r['py_mean']:>10.2f}   {r['cpp_mean']:>8.2f}   {arrow:>8s}  {faster:>8s}")

    print("\n--- CSV ---")
    print("seq,py_mean_ms,py_p95_ms,cpp_mean_ms,cpp_p95_ms,speedup")
    for r in results:
        speedup = r['py_mean'] / r['cpp_mean'] if r['cpp_mean'] > 0 else 0
        print(f"{r['seq']},{r['py_mean']:.2f},{r['py_p95']:.2f},{r['cpp_mean']:.2f},{r['cpp_p95']:.2f},{speedup:.2f}")

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if not os.path.exists(CPP_BENCHMARK):
    print("ERROR: C++ benchmark binary not found. Build it first:")
    print(f"  cd {CPP_BUILD_DIR} && cmake --build . --target benchmark")
    sys.exit(1)

# Build C++ binary (ensure it's up to date)
print("Building C++ benchmark...")
build_result = subprocess.run(['cmake', '--build', '.', '--target', 'benchmark'],
                               cwd=CPP_BUILD_DIR, capture_output=True, text=True, timeout=120)
if build_result.returncode != 0:
    print("Build failed:")
    print(build_result.stderr)
    sys.exit(1)

# Load Python engine (once, to avoid multi-load overhead)
from atb_python_qwen3vl_embedding.utils import set_atb_buffer_size
from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine

set_atb_buffer_size(10 * 1024 * 1024 * 1024)
py_engine = Qwen3VLEngine(MODEL_DIR)
print(f"Python engine loaded: {py_engine.n_layer} layers, {py_engine.n_layer} text layers, hidden={py_engine.hidden_t}")

seq_lengths = [4, 16, 64, 256, 1024] if args.all_seqs else [args.seq]
results = []

for seq_len in seq_lengths:
    print(f"\n{'='*60}")
    print(f"  Sequence length: {seq_len}")
    print(f"{'='*60}")

    # Python
    py_stats, py_times = run_python_benchmark(py_engine, seq_len, args.warmup, args.iter)
    print(f"  Python: mean={py_stats['mean']:.2f} median={py_stats['median']:.2f} p95={py_stats['p95']:.2f} ms")

    # C++
    cpp_stats = run_cpp_benchmark(seq_len, args.warmup, args.iter)
    if cpp_stats:
        print(f"  C++:    mean={cpp_stats['mean']:.2f} median={cpp_stats['median']:.2f} p95={cpp_stats['p95']:.2f} ms")

        speedup = py_stats['mean'] / cpp_stats['mean'] if cpp_stats['mean'] > 0 else 0
        if speedup > 1.0:
            print(f"  >>> C++ is {speedup:.2f}x FASTER")
        elif speedup < 1.0:
            print(f"  >>> C++ is {1/speedup:.2f}x SLOWER")
        else:
            print(f"  >>> IDENTICAL")

        results.append({'seq': seq_len,
                        'py_mean': py_stats['mean'], 'py_p95': py_stats['p95'],
                        'cpp_mean': cpp_stats['mean'], 'cpp_p95': cpp_stats['p95']})

print_table(results)
print("\nDone.")
