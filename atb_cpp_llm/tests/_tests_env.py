"""Thin shim that lets every script under `atb_cpp_llm/tests/` reach the
canonical .env-driven configuration in `atb_python_qwen3vl_embedding.env`
without each script having to bootstrap `sys.path` itself.

Why this file exists:
  - `atb_python_qwen3vl_embedding/env.py` is the single source of truth for
    config (reads `.env`, raises on missing required vars, exposes typed
    constants). It is intentionally part of the Python package, so it lives
    under `atb_python_qwen3vl_embedding/` rather than at the repo root.
  - To import from it, `sys.path` must contain the repo root.
  - Hand-rolling the same five lines of `sys.path.insert(...)` at the top
    of every test script (and getting the parent-count right) is exactly
    what produced the hard-coded-path mess this file is replacing.

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tests/
    from _tests_env import MODEL_DIR, REPO_ROOT  # noqa: E402

    # Optional values (may be None if not set in .env):
    from _tests_env import TRANSFORMERS_SRC, QWEN3VL_EMB_SRC, CPP_BUILD_DIR
"""
import sys
from pathlib import Path

# Locate the repo root: this file → tests/_tests_env.py → tests/ → atb_cpp_llm/ → repo
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-export everything from the canonical config module.
from atb_python_qwen3vl_embedding.env import (  # noqa: E402
    QWEN3VL_EMB_MODEL_DIR as MODEL_DIR,
    QWEN3VL_EMB_SRC,
    QWEN3VL_EMB_TRANSFORMERS_SRC as TRANSFORMERS_SRC,
    REPO_ROOT,
    get_env,
)

#: C++ build output dir for diagnostics that need to exec `./benchmark` etc.
#: Derived from REPO_ROOT — no env var needed.
CPP_BUILD_DIR: str = str(REPO_ROOT / "atb_cpp_llm" / "build")

__all__ = [
    "MODEL_DIR",
    "QWEN3VL_EMB_SRC",
    "TRANSFORMERS_SRC",
    "REPO_ROOT",
    "CPP_BUILD_DIR",
    "get_env",
]
