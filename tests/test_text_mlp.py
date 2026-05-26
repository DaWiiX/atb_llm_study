"""Test Qwen3VLTextMLP: transformers vs ATB graph."""
import sys
import os

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import torch
import torch_npu  # noqa: needed for .npu()

from atb_python_model import data_utils, utils
from atb_python_model.text_mlp import build_mlp
from atb_python_model.transformers_runner import run_mlp


def test_mlp(B=1, S=16, seed=42):
    print("\n=== Qwen3VLTextMLP ===")
    config = data_utils.make_config(intermediate_size=512)

    gen_data = data_utils.generate_base_data(config, B=B, S=S, seed=seed)
    gen_data["config"] = config
    ref_out, weights = run_mlp(gen_data, seed=seed)

    _, graph_op, _ = build_mlp(intermediate_size=512)

    utils.set_atb_buffer_size(100 * 1024 * 1024)
    atb_out = graph_op.forward([
        gen_data["hidden_states"].half().npu(),
        weights["gate_proj.weight"].half().npu(),
        weights["up_proj.weight"].half().npu(),
        weights["down_proj.weight"].half().npu(),
    ])[0].cpu().float()
    return utils.compare_tensors(ref_out, atb_out, label="MLP")


if __name__ == "__main__":
    ok = test_mlp()
    exit(0 if ok else 1)
