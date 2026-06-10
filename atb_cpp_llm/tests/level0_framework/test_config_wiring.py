"""Verify C++ config dump matches Python config loading.

Reads the JSON dump written by the C++ test_config_wiring and diffs it
against config values loaded directly from the model's config.json and
preprocessor_config.json.

CI-friendly: skips if either the model checkpoint or the C++ dump is missing.

The key regression guard is vis_epsilon — if someone changes the C++ JSON key
from "layer_norm_eps" back to "initializer_range", the epsilon jumps from 1e-6
to 0.02 (20 000x too large) and this test will fail.
"""
import json
import os
import sys


def load_python_config(model_dir):
    """Load config the same way the C++ LoadQwen3VLConfig does.

    Reads config.json and preprocessor_config.json, extracting every field
    that Qwen3VLConfig carries.  Defaults match the struct initializers in
    qwen3vl_config.h.
    """
    cfg = {}
    try:
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        pp_path = os.path.join(model_dir, "preprocessor_config.json")
        with open(pp_path, "r") as f:
            pp = json.load(f)
    except FileNotFoundError as e:
        print(f"SKIP: {e}")
        return None

    # ── Top-level ────────────────────────────────────────────
    cfg["image_token_id"] = config.get("image_token_id", 151655)
    cfg["vision_start_token_id"] = config.get(
        "vision_start_token_id", 151652
    )

    # ── Text config ──────────────────────────────────────────
    text = config.get("text_config", {})
    cfg["text_hidden_size"] = text.get("hidden_size", 2048)
    cfg["text_num_heads"] = text.get("num_attention_heads", 16)
    cfg["text_num_kv_heads"] = text.get("num_key_value_heads", 8)
    # head_dim from config or derived
    if "head_dim" in text:
        cfg["text_head_dim"] = text["head_dim"]
    else:
        cfg["text_head_dim"] = cfg["text_hidden_size"] // cfg["text_num_heads"]
    cfg["text_intermediate_size"] = text.get("intermediate_size", 6144)
    cfg["text_num_layers"] = text.get("num_hidden_layers", 28)
    cfg["text_rms_norm_eps"] = text.get("rms_norm_eps", 1e-6)
    cfg["text_rope_theta"] = text.get("rope_theta", 5000000.0)
    cfg["text_vocab_size"] = text.get("vocab_size", 151936)

    # ── Vision config ────────────────────────────────────────
    vis = config.get("vision_config", {})
    cfg["vis_hidden_size"] = vis.get("hidden_size", 1024)
    cfg["vis_num_heads"] = vis.get("num_heads", 16)
    cfg["vis_intermediate_size"] = vis.get("intermediate_size", 4096)
    cfg["vis_depth"] = vis.get("depth", 24)
    cfg["vis_in_channels"] = vis.get("in_channels", 3)
    cfg["vis_temporal_patch_size"] = vis.get("temporal_patch_size", 2)
    cfg["vis_patch_size"] = vis.get("patch_size", 16)
    cfg["vis_spatial_merge_size"] = vis.get("spatial_merge_size", 2)
    cfg["vis_num_position_embeddings"] = vis.get(
        "num_position_embeddings", 2304
    )
    cfg["vis_out_hidden_size"] = vis.get("out_hidden_size", 2048)

    # ── THIS IS THE KEY REGRESSION GUARD ─────────────────────
    # The C++ code MUST read "layer_norm_eps" (value ~1e-6), NOT
    # "initializer_range" (value 0.02).  There is no layer_norm_eps
    # in the real vision_config, so the default 1e-6 is used.
    # If someone wires this to initializer_range, the value becomes
    # 0.02 and the comparison below will catch it.
    cfg["vis_epsilon"] = vis.get("layer_norm_eps", 1e-6)
    # BUG (this is what we guard against):
    # cfg["vis_epsilon"] = vis.get("initializer_range", 1e-6)  # 0.02 — 20000x off!

    # ── Embedding output ─────────────────────────────────────
    cfg["normalize"] = True  # Python default; matches struct initializer

    # ── Preprocessor config ─────────────────────────────────
    cfg["pp_patch_size"] = pp.get("patch_size", 16)
    cfg["pp_temporal_patch_size"] = pp.get("temporal_patch_size", 2)
    cfg["pp_merge_size"] = pp.get("merge_size", 2)
    cfg["pp_min_pixels"] = pp.get("min_pixels", 4096)
    cfg["pp_max_pixels"] = pp.get("max_pixels", 1310720)

    # ── Derived ──────────────────────────────────────────────
    cfg["vis_head_dim"] = cfg["vis_hidden_size"] // cfg["vis_num_heads"]

    return cfg


def compare_configs():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from _tests_env import MODEL_DIR as model_dir  # noqa: E402

    py_cfg = load_python_config(model_dir)
    if py_cfg is None:
        print("SKIP: model config not found (CI-friendly)")
        return True

    cpp_dump_path = "/tmp/cpp_config_dump.json"
    try:
        with open(cpp_dump_path, "r") as f:
            cpp_cfg = json.load(f)
    except FileNotFoundError:
        print(
            "FAIL: /tmp/cpp_config_dump.json not found — "
            "run the C++ test_config_wiring first"
        )
        return False

    print("=== Config wiring comparison: C++ vs Python ===\n")

    mismatches = []
    for key in py_cfg:
        py_val = py_cfg[key]
        cpp_val = cpp_cfg.get(key)
        if cpp_val is None:
            mismatches.append(
                f"MISSING: {key} (Python={py_val}, C++=N/A)"
            )
            continue

        # Float comparison with relative tolerance
        if isinstance(py_val, float):
            tol = 1e-6 * max(1.0, abs(py_val))
            if abs(py_val - cpp_val) > tol:
                mismatches.append(
                    f"MISMATCH: {key} "
                    f"(Python={py_val}, C++={cpp_val})"
                )
            else:
                print(f"  OK  {key}: {py_val}")
        else:
            if py_val != cpp_val:
                mismatches.append(
                    f"MISMATCH: {key} "
                    f"(Python={py_val}, C++={cpp_val})"
                )
            else:
                print(f"  OK  {key}: {py_val}")

    # ── Spotlight: vis_epsilon is the critical anti-regression check ──
    py_eps = py_cfg["vis_epsilon"]
    cpp_eps = cpp_cfg.get("vis_epsilon", "MISSING")
    print(
        f"\n  vis_epsilon: Python={py_eps}, C++={cpp_eps}"
    )
    if py_eps != cpp_eps:
        print(
            "     ^^^ WAS THIS READ FROM 'initializer_range' (0.02) "
            "INSTEAD OF 'layer_norm_eps' (1e-6)?"
        )
        mismatches.append(
            "CRITICAL: vis_epsilon mismatch — check JSON key in "
            "qwen3vl_config.cpp!"
        )

    if mismatches:
        print(f"\n*** {len(mismatches)} MISMATCH(ES) ***")
        for m in mismatches:
            print(f"  - {m}")
        return False
    else:
        print(f"\n=== All {len(py_cfg)} fields match — PASS ===")
        return True


if __name__ == "__main__":
    ok = compare_configs()
    sys.exit(0 if ok else 1)
