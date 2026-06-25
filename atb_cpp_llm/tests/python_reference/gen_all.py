#!/usr/bin/env python3
"""One-shot driver: generate every binary reference file the C++ tests need.

The C++ Level-1/Level-2 precision tests and several Level-3/Level-4 stage
tests read `.bin` files from /tmp. Each file is produced by one of the
generator scripts in this directory. They cannot share a Python process
(each one calls `set_atb_buffer_size()`, which must only be invoked once),
so we spawn an independent subprocess per generator.

Usage:
    python atb_cpp_llm/tests/python_reference/gen_all.py
    python atb_cpp_llm/tests/python_reference/gen_all.py --skip-fresh   # skip generators whose outputs already exist

Reads QWEN3VL_EMB_MODEL_DIR from the environment (inherited by every child).

Exit code is 0 only when every generator succeeds.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent

sys.path.insert(0, str(HERE.parent))
from _tests_env import ASCEND_PLATFORM  # noqa: E402

# Generator → list of sentinel files it produces (used by --skip-fresh).
# We only check one or two representative paths per generator; if those are
# present we assume the full batch is, too. Re-run without --skip-fresh to
# rebuild from scratch.
GENERATORS = [
    ("gen_cpu_reference.py",          ["/tmp/cpu_op_rms_norm_medium_input.bin", "/tmp/cpu_vision_merger_main_x.bin"]),
    ("gen_official_pixel_values.py",  ["/tmp/official_pv_prod_416x672.bin"]),
    ("gen_official_embedding.py",     ["/tmp/official_embed_mm_416x672.bin", "/tmp/official_tokens_mm_416x672.bin"]),
    ("gen_stage_reference.py",        ["/tmp/stage_L0_pixel_values.bin", "/tmp/stage_L3_rope_sin.bin"]),
    ("test_stage_reference.py",       ["/tmp/stage_pixels.bin", "/tmp/stage_final_text_only.bin"]),
    ("gen_pos_embed_npu_reference.py", ["/tmp/posembed_npu_case_tiny_4x4.bin"]),
    ("gen_vis_rope_npu_reference.py",  ["/tmp/visrope_npu_case_tiny_4x4.bin"]),
]


def already_fresh(sentinels):
    return all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in sentinels)


def run_one(script: str) -> int:
    """Run a single generator in its own process; stream its output live."""
    path = HERE / script
    print(f"\n[gen_all] ▶ {script}", flush=True)
    t0 = time.monotonic()
    rc = subprocess.call([sys.executable, str(path)])
    dt = time.monotonic() - t0
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"[gen_all] ◀ {script} — {status} in {dt:.1f}s", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-fresh", action="store_true",
                    help="skip a generator if its sentinel outputs already exist")
    args = ap.parse_args()

    if "QWEN3VL_EMB_MODEL_DIR" not in os.environ:
        print("[gen_all] WARNING: QWEN3VL_EMB_MODEL_DIR is not set; "
              "generators will fall back to the hard-coded default path.",
              file=sys.stderr)

    failures = []
    for script, sentinels in GENERATORS:
        if args.skip_fresh and already_fresh(sentinels):
            print(f"[gen_all] ◇ skip {script} (outputs already present)")
            continue
        rc = run_one(script)
        if rc != 0:
            failures.append(script)
            # Continue running remaining generators — one failure
            # shouldn't block others (e.g. stage refdata can be
            # regenerated later; op-level refdata is independent).

    if failures:
        platform = ASCEND_PLATFORM
        print(f"\n[gen_all] ❌ {len(failures)}/{len(GENERATORS)} generators failed "
              f"(platform: {platform}):")
        for s in failures:
            print(f"           - {s}")
        print("\n[gen_all] 💡 Recover by re-running the failed generator directly:")
        for s in failures:
            print(f"           python tests/python_reference/{s}")
        print("[gen_all] 💡 For 310P Python ATB issues, the generator may need")
        print("[gen_all]    to fall back to torch_npu or CPU (see refdata_fallback.py).")

        if platform == "310P":
            print("\n[gen_all] 🔧 310P-specific diagnostics:")
            print("[gen_all]    1. Check mask format: make_causal_mask_nz_npu() MUST set")
            print("[gen_all]       FRACTAL_NZ format (acl_format=29) on the NPU tensor.")
            print("[gen_all]       Without this, ATB SelfAttention fails 'call operation setup fail'.")
            print("[gen_all]    2. test_stage_reference.py: TEXT_ONLY has transformers CPU fallback;")
            print("[gen_all]       IMAGE_ONLY/IMAGE_AND_TEXT are skipped if ATB fails.")
            print("[gen_all]    3. gen_stage_reference.py: vision stages (L1-L3) should work on 310P")
            print("[gen_all]       (no SelfAttention mask involved). If they fail, check ATB log.")
            print("[gen_all]    4. ATB log: cat $(ls -rt ~/ascend/log/atb/ | tail -n 1)")
            print("[gen_all]    5. When a generator writes NO .bin file, C++ tests dependent on it")
            print("[gen_all]       are automatically excluded by build_and_test.sh (--no-refdata).")

        sys.exit(1)
    print("\n[gen_all] ✅ All reference data generated.")


if __name__ == "__main__":
    main()
