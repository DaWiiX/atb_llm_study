"""
ATB C++ engine vs torch_npu transformers precision & performance comparison.

Runs torch_npu Qwen3VLEmbedding model as golden reference,
then compares C++ ATB engine output against it.

Usage:
    python tests/compare_with_torch.py [--seq S] [--iter N] [--warmup M]

Outputs:
    - Cosine similarity between C++ ATB and torch_npu outputs
    - Side-by-side latency comparison
"""

import sys, os, time, argparse, struct, subprocess
import numpy as np
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _tests_env import MODEL_DIR, REPO_ROOT, CPP_BUILD_DIR, TRANSFORMERS_SRC  # noqa: E402
if TRANSFORMERS_SRC:
    sys.path.insert(0, TRANSFORMERS_SRC)
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch_npu
import torch.nn.functional as F
import safetensors.torch

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=int, default=64)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--warmup', type=int, default=3)
args = parser.parse_args()

CPP_BIN = str(__import__('pathlib').Path(CPP_BUILD_DIR) / 'benchmark')
CPP_SAVE = '/tmp/cpp_text_only.bin'
TORCH_SAVE = '/tmp/torch_embedding.bin'


def sync():
    torch.npu.synchronize()


def now():
    return time.perf_counter()


def make_input_ids(seq_len):
    """Same format as C++ benchmark: [151643, 15339, ..., 15339, 151645]"""
    ids = [151643] + [15339] * (seq_len - 2) + [151645]
    return torch.tensor([ids], dtype=torch.long)


# ═══════════════════════════════════════════════════════════
# 1. torch_npu transformers — golden reference
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("1. Loading torch_npu transformers model (golden reference)")
print("=" * 60)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

cfg = Qwen3VLConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
cfg._attn_implementation = "eager"
cfg.text_config._attn_implementation = "eager"

ref = Qwen3VLModel(cfg).eval().half().npu()

sd = safetensors.torch.load_file(f"{MODEL_DIR}/model.safetensors", device="cpu")
sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
missing, unexpected = ref.load_state_dict(sd, strict=False)
print(f"torch_npu model loaded: {len(missing)} missing, {len(unexpected)} unexpected")
assert len(missing) == 0 and len(unexpected) == 0, "Weight mismatch!"

# ── Inference ──────────────────────────────────────────────

input_ids = make_input_ids(args.seq)
print(f"Input: {input_ids.shape}, seq_len={args.seq}")

# Warmup
for _ in range(args.warmup):
    with torch.no_grad():
        out = ref(input_ids=input_ids.npu(), return_dict=True)
    sync()

# Benchmark
torch_times = []
for i in range(args.iter):
    sync()
    t0 = now()
    with torch.no_grad():
        out = ref(input_ids=input_ids.npu(), return_dict=True)
    sync()
    t1 = now()
    torch_times.append((t1 - t0) * 1000)
    print(f"  torch_npu iter {i}: {torch_times[-1]:.1f} ms")

torch_emb = out.last_hidden_state.float().cpu()

# Pooling: last valid token (last position in sequence)
# Qwen3VLEmbedding uses last-token pooling by default
pooled = torch_emb[0, -1, :]  # last token of batch 0

# L2 normalize (as done by ATB engine)
pooled = F.normalize(pooled, p=2, dim=-1)

torch_a = np.array(torch_times)
print(f"torch_npu (pooled+normalized): mean={torch_a.mean():.1f} median={np.median(torch_a):.1f} ms")
print(f"First 8 values: {pooled[:8].tolist()}")
print(f"Norm: {pooled.norm().item():.4f}")

# Save to file
pooled_np = pooled.numpy().astype(np.float32)
with open(TORCH_SAVE, 'wb') as f:
    dim = np.int64(len(pooled_np))
    f.write(dim.tobytes())
    f.write(pooled_np.tobytes())
print(f"Saved torch_npu output to {TORCH_SAVE}")

# Free reference to free NPU memory for C++ engine
del ref
torch.npu.empty_cache()
sync()

# ═══════════════════════════════════════════════════════════
# 2. C++ ATB Engine
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print("2. Running C++ ATB engine")
print("=" * 60)

if not os.path.exists(CPP_BIN):
    print(f"ERROR: C++ benchmark not found at {CPP_BIN}")
    print("Build it with: cd build && cmake --build . --target benchmark")
    sys.exit(1)

# First, run C++ with --cmp to get timing
cmd = [CPP_BIN, '--iter', str(args.iter), '--warmup', str(args.warmup),
       '--seq', str(args.seq), '--cmp']
print(f"Running: {' '.join(cmd)}")

result = subprocess.run(cmd, capture_output=True, text=True,
                         cwd=os.path.dirname(CPP_BIN), timeout=600)

# Parse C++ benchmark results
cpp_mean = cpp_median = cpp_p95 = 0.0
for line in result.stdout.split('\n'):
    if line.startswith('CPP_RESULT:'):
        for part in line.split()[1:]:
            k, v = part.split('=')
            if k == 'mean': cpp_mean = float(v)
            elif k == 'median': cpp_median = float(v)
            elif k == 'p95': cpp_p95 = float(v)

# Also run test_accuracy to get the output embedding
cons_bin = CPP_BIN.replace('benchmark', 'test_accuracy')
if os.path.exists(cons_bin):
    print(f"Running accuracy test to capture C++ embedding...")
    subprocess.run([cons_bin], capture_output=True, text=True,
                    cwd=os.path.dirname(CPP_BIN), timeout=600)
    cpp_bin_path = '/tmp/cpp_text_only.bin'
    print(f"Expected embedding at {cpp_bin_path}")
else:
    print(f"WARN: test_accuracy not found, skipping output capture")

# ═══════════════════════════════════════════════════════════
# 3. Precision Comparison (if both outputs exist)
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print("3. Precision Comparison")
print("=" * 60)

if os.path.exists(TORCH_SAVE) and os.path.exists(CPP_SAVE):
    with open(TORCH_SAVE, 'rb') as f:
        t_dim = struct.unpack('q', f.read(8))[0]
        t_data = np.frombuffer(f.read(t_dim * 4), dtype=np.float32)
    torch_out = torch.from_numpy(t_data.copy()).float()

    with open(CPP_SAVE, 'rb') as f:
        c_dim = struct.unpack('q', f.read(8))[0]
        c_data = np.frombuffer(f.read(c_dim * 4), dtype=np.float32)
    cpp_out = torch.from_numpy(c_data.copy()).float()

    print(f"  torch_npu dim: {t_dim}")
    print(f"  ATB C++   dim: {c_dim}")

    cos_sim = F.cosine_similarity(torch_out, cpp_out, dim=0).item()
    mse = F.mse_loss(torch_out, cpp_out).item()
    max_diff = (torch_out - cpp_out).abs().max().item()

    print(f"\n  Cosine similarity: {cos_sim:.6f}")
    print(f"  MSE:               {mse:.2e}")
    print(f"  Max abs diff:      {max_diff:.2e}")
    print(f"\n  First 8 values:")
    print(f"    torch_npu: {t_data[:8].tolist()}")
    print(f"    ATB C++:   {c_data[:8].tolist()}")

    threshold = 0.99
    if cos_sim > threshold:
        print(f"\n  PRECISION PASS (cosine {cos_sim:.6f} > {threshold})")
    else:
        print(f"\n  PRECISION FAIL (cosine {cos_sim:.6f} <= {threshold})")
else:
    print("  One or both output files missing — skipping precision comparison")
    if not os.path.exists(TORCH_SAVE):
        print(f"    Missing: {TORCH_SAVE}")
    if not os.path.exists(CPP_SAVE):
        print(f"    Missing: {CPP_SAVE}")

# ═══════════════════════════════════════════════════════════
# 4. Performance Comparison
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print("4. Performance Comparison (text-only, S={})".format(args.seq))
print("=" * 60)

if cpp_mean > 0:
    print(f"  torch_npu: {torch_a.mean():.1f} ms (mean)")
    print(f"  ATB C++:   {cpp_mean:.1f} ms (mean)")
    speedup = torch_a.mean() / cpp_mean if cpp_mean > 0 else 0
    print(f"  Speedup:   {speedup:.2f}x {'(C++ faster)' if speedup > 1.0 else '(C++ slower)'}")
else:
    print("  C++ benchmark results not parsed — check output above")
    # Print raw output for debugging
    print("\n  C++ stdout:")
    for line in result.stdout.split('\n')[:10]:
        print(f"    {line}")

print()
print("Done.")
