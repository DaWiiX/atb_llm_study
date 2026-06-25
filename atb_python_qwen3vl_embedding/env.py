"""Central config — values read from os.environ first, then a .env file.

Public API:
    - read .env once at module import into ``_DOTENV``
    - ``get_env(name, default=None, required=False)`` for ad-hoc lookups
    - Module-level constants (e.g. ``QWEN3VL_EMB_MODEL_DIR``) for callers,
      defined below using ``get_env``. Add new project-wide variables in the
      same section so consumers can do
      ``from atb_python_qwen3vl_embedding.env import FOO`` and the failure
      mode (missing-and-required) surfaces at import time with a clear hint.

The .env file lives at the repo root by convention (gitignored; see
``.env.example``). It is auto-discovered by walking up from this file;
override with the ``ATB_DOTENV_PATH`` shell variable when needed.

Parser supports KEY=VALUE per line, ``#`` comments, optional surrounding
quotes. No interpolation, no exports — keep it boring.
"""
import os
from pathlib import Path
from typing import Optional


# ── .env discovery and parsing ──────────────────────────────────────

_DOTENV_FILENAME = ".env"
_SEARCH_DEPTH = 6   # this file + 5 parents covers any realistic repo layout


def _find_dotenv() -> Optional[Path]:
    override = os.environ.get("ATB_DOTENV_PATH")
    if override:
        p = Path(override).expanduser().resolve()
        return p if p.is_file() else None
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents][:_SEARCH_DEPTH]:
        p = d / _DOTENV_FILENAME
        if p.is_file():
            return p
    return None


def read_dotenv(path: Path) -> dict:
    """Parse a .env file into a ``{key: value}`` dict. No os.environ writes."""
    out: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


#: Resolved .env path, or None if no .env file is present.
DOTENV_PATH: Optional[Path] = _find_dotenv()

#: Parsed .env contents. Empty dict when no file is found.
_DOTENV: dict = read_dotenv(DOTENV_PATH) if DOTENV_PATH else {}


# ── Generic accessor ────────────────────────────────────────────────

def get_env(name: str, default: Optional[str] = None,
            required: bool = False) -> Optional[str]:
    """Resolve a config value with the precedence os.environ > .env > default.

    Args:
        name:     environment variable name.
        default:  value when neither shell nor .env defines it.
        required: when True, raise RuntimeError if no value is resolved.
    """
    v = os.environ.get(name)
    if v is None:
        v = _DOTENV.get(name)
    if v is None:
        v = default
    if required and not v:
        where = str(DOTENV_PATH) if DOTENV_PATH else "a .env file at the repo root"
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            f"Set it in your shell or add `{name}=...` to {where}."
        )
    return v


# ── Project-wide config variables ───────────────────────────────────
# Add new entries below. Importers do
#     from atb_python_qwen3vl_embedding.env import QWEN3VL_EMB_MODEL_DIR
# and required-but-missing failures surface at that import line.

#: Local Qwen3-VL-Embedding-2B checkpoint directory.
QWEN3VL_EMB_MODEL_DIR: str = get_env("QWEN3VL_EMB_MODEL_DIR", required=True)

#: Path to a checked-out copy of the Qwen3-VL-Embedding reference repo root.
#: Official-reference generators append `/src` before importing the official
#: `models.*` package. Optional — only official-reference generators need it.
QWEN3VL_EMB_SRC: Optional[str] = get_env("QWEN3VL_EMB_SRC")

#: Path to a checked-out huggingface/transformers (the `src/` dir).
#: Optional — only diagnostics that import via the reference transformers
#: impl (compare_with_torch.py, compare_multimodal.py) read it.
QWEN3VL_EMB_TRANSFORMERS_SRC: Optional[str] = get_env("QWEN3VL_EMB_TRANSFORMERS_SRC")

#: Ascend NPU platform identifier. Valid values: "910B" (Atlas A2), "310P" (Atlas推理系列).
#: Controls which ATB operator paths are used where platform support differs.
ASCEND_PLATFORM: str = get_env("ASCEND_PLATFORM", default="910B")

#: Resolutions for the official embedding gate (path C).
#: Comma-separated HxW pairs. Users may add/remove via .env or shell env.
OFFICIAL_EMBED_CASES: str = get_env("OFFICIAL_EMBED_CASES",
                                    default="416x672,720x1280,1080x1920,1440x2560")

#: Repository root, derived from this file's location. Useful for tests
#: that need to add the repo to sys.path without hard-coding the path.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
