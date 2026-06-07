"""
C++ ATB vs torch_npu multimodal benchmark comparison.
Uses 4 image resolutions with "Describe the image." prompt.
Generates inputs in Python, saves to binary files, runs C++ benchmark.
"""
import sys, os, time, argparse, struct, subprocess
import numpy as np
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
sys.path.insert(0, '/mnt/workspace/gitCode/transformers/src')

import torch
import torch_npu
import torch.nn.functional as F
import safetensors.torch
from PIL import Image

# ═══════════════════════════════════════════════════════════
def sync(): torch.npu.synchronize()
def now(): return time.perf_counter()

MODEL_DIR = '/mnt/workspace/gitCode/models/Qwen3-VL-Embedding-2B'
CPP_BUILD = '/mnt/workspace/gitCode/atb_llm/atb_cpp_qwen3vl_embedding/build'
INPUT_DIR = '/tmp/mm_inputs'
OUTPUT_DIR = '/tmp/mm_outputs'

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 1. Load models & config
# ═══════════════════════════════════════════════════════════

print("Loading torch_npu model...")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

cfg = Qwen3VLConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
cfg._attn_implementation = "eager"
cfg.text_config._attn_implementation = "eager"
ref = Qwen3VLModel(cfg).eval().half().npu()
sd = safetensors.torch.load_file(f"{MODEL_DIR}/model.safetensors", device="cpu")
sd = {k.removeprefix("model."): v.half() for k, v in sd.items()}
m, u = ref.load_state_dict(sd, strict=False)
assert len(m) == 0 and len(u) == 0
print("torch_npu model loaded OK")
sync()

# ═══════════════════════════════════════════════════════════
# 2. Generate test images + preprocessed inputs
# ═══════════════════════════════════════════════════════════

def compute_vis_tokens(vision_config, grid_thw, spatial_merge_size):
    # grid_thw: [grid_t, grid_h, grid_w]
    # vis tokens = total_patches / (merge_size^2)
    return int(torch.prod(torch.tensor(grid_thw), dim=0).item()) // (spatial_merge_size ** 2)

from atb_python_qwen3vl_embedding.preprocess import preprocess_image
from transformers import AutoProcessor

rms = cfg.vision_config.spatial_merge_size
proc = AutoProcessor.from_pretrained(MODEL_DIR)

resolutions = [(416, 672), (720, 1280), (1080, 1920), (2560, 1440)]
results = []

for w, h in resolutions:
    print(f"\n{'='*60}")
    print(f"  Resolution: {w}x{h}")
    print(f"{'='*60}")

    # Create image
    img = Image.new('RGB', (w, h), color='blue')
    img_arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    pv_raw, grid_thw = preprocess_image(img_arr)
    gth_list = grid_thw.squeeze(0).tolist()

    # Tokenize
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'Describe the image.'}]}]
    tf_in = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                      return_tensors='pt', add_generation_prompt=True)
    input_ids = tf_in['input_ids']
    S = input_ids.shape[1]
    n_vis = compute_vis_tokens(cfg.vision_config, grid_thw.tolist()[0], rms)

    print(f"  grid_thw: {gth_list},  vis_tokens: {n_vis},  S={S}")

    # Save inputs for C++ (binary format)
    tag = f"{w}x{h}"
    pv = pv_raw.numpy().astype(np.float16)
    iids = input_ids.numpy().astype(np.int64)
    gth = grid_thw.numpy().astype(np.int64)

    pv.tofile(f"{INPUT_DIR}/{tag}_pixel_values.bin")
    iids.tofile(f"{INPUT_DIR}/{tag}_input_ids.bin")
    gth.tofile(f"{INPUT_DIR}/{tag}_grid_thw.bin")

    # Save metadata
    with open(f"{INPUT_DIR}/{tag}_meta.bin", 'wb') as f:
        f.write(struct.pack('qqqq', pv.shape[0], pv.shape[1], S, n_vis))

    # ── torch_npu benchmark ──────────────────────────────────
    for _ in range(3):
        with torch.no_grad():
            ref(input_ids=tf_in['input_ids'].npu(),
                pixel_values=tf_in['pixel_values'].half().npu(),
                image_grid_thw=tf_in['image_grid_thw'].npu())
        sync()

    ts_torch = []
    for i in range(10):
        sync(); t0 = now()
        with torch.no_grad():
            out = ref(input_ids=tf_in['input_ids'].npu(),
                      pixel_values=tf_in['pixel_values'].half().npu(),
                      image_grid_thw=tf_in['image_grid_thw'].npu())
        sync(); t1 = now()
        ts_torch.append((t1 - t0) * 1000)

    # Get torch output for precision comparison (pool + normalize)
    with torch.no_grad():
        out_full = ref(input_ids=tf_in['input_ids'].npu(),
                       pixel_values=tf_in['pixel_values'].half().npu(),
                       image_grid_thw=tf_in['image_grid_thw'].npu(),
                       return_dict=True)
    sync()
    torch_pooled = F.normalize(
        out_full.last_hidden_state.float().cpu()[0, -1, :], p=2, dim=-1)
    torch_pooled.numpy().astype(np.float32).tofile(f"{OUTPUT_DIR}/{tag}_torch.bin")

    t_mean = np.mean(ts_torch)
    print(f"  torch_npu: mean={t_mean:.1f} ms  (10 iters, 3 warmup)")

    results.append({
        'tag': tag, 'w': w, 'h': h, 'S': S, 'n_vis': n_vis,
        'torch_mean': t_mean,
        'gth': gth_list,
    })

# Free torch model
del ref
torch.npu.empty_cache()
sync()

# ═══════════════════════════════════════════════════════════
# 3. C++ ATB multimodal benchmark (unified benchmark with per-stage timing)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("C++ ATB Multimodal Benchmark (unified, per-stage timing)")
print(f"{'='*60}")

for r in results:
    tag = r['tag']
    w, h = r['w'], r['h']
    print(f"\n  {tag} (S={r['S']}, vis={r['n_vis']}):")

    # Run the unified C++ benchmark in mm mode with compact output
    sub_result = subprocess.run(
        [f"{CPP_BUILD}/benchmark", '--mode', 'mm', '--iter', '5', '--warmup', '2',
         '--width', str(w), '--height', str(h), '--cmp'],
        capture_output=True, text=True, cwd=CPP_BUILD, timeout=300)

    cpp_e2e_ms = 0
    cpp_stages = {}
    for line in sub_result.stdout.split('\n'):
        if line.startswith('BENCH_RESULT:'):
            for part in line.split()[1:]:
                k, v = part.split('=')
                if k == 'e2e_mean':
                    cpp_e2e_ms = float(v)
                elif k in ('preprocess', 'vision_pos', 'vision_model',
                           'text_embed', 'position_ids', 'text_model',
                           'pooling', 'staged'):
                    cpp_stages[k] = float(v)

    r['cpp_e2e_ms'] = cpp_e2e_ms
    r['cpp_stages'] = cpp_stages
    if cpp_stages:
        print(f"    C++ E2E: {cpp_e2e_ms:.1f} ms  (staged={cpp_stages.get('staged', 0):.1f} ms)")
        print(f"    Vision model: {cpp_stages.get('vision_model', 0):.1f} ms  "
              f"Text model: {cpp_stages.get('text_model', 0):.1f} ms")
    else:
        print(f"    C++ E2E: {cpp_e2e_ms:.1f} ms")

# ═══════════════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*80}")
print(f" Multimodal Performance Comparison")
print(f"{'='*80}")
print(f"{'Resolution':<16} {'S':<6} {'VisTok':<8} {'torch_npu':>10} {'C++ E2E':>10} {'Speedup':>10}")
print(f"{'-'*60}")

for r in results:
    cpp_e2e = r.get('cpp_e2e_ms', 0)
    speedup = r['torch_mean'] / cpp_e2e if cpp_e2e > 0 else 0
    print(f"{r['w']}x{r['h']:<10} {r['S']:<6} {r['n_vis']:<8} "
          f"{r['torch_mean']:>8.1f} ms  {cpp_e2e:>8.1f} ms  "
          f"{speedup:>8.2f}x")
print(f"\nDone.")
