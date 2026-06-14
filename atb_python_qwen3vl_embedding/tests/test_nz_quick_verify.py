"""Quick verification that make_causal_mask_nz_npu() works correctly.

Usage:
    python tests/test_nz_quick_verify.py
    ASCEND_PLATFORM=310P python tests/test_nz_quick_verify.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch_npu  # noqa: F401

from atb_python_qwen3vl_embedding.utils import make_causal_mask_nz_npu


def main():
    print(f"Platform: {os.getenv('ASCEND_PLATFORM', '910B')}")
    print(f"Device: {torch.npu.get_device_name(0)}")

    all_ok = True
    for S, label in [(3, "TEXT_ONLY"), (4, "S=4"), (8, "S=8"),
                     (16, "S=16"), (880, "IMAGE_ONLY")]:
        try:
            mask = make_causal_mask_nz_npu(S)
            fmt = torch_npu.get_npu_format(mask)
            n1 = (S + 15) // 16
            s_pad = n1 * 16
            expected_shape = (1, n1, s_pad, 16)

            shape_ok = mask.shape == expected_shape
            format_ok = fmt.name == "FRACTAL_NZ"
            dtype_ok = mask.dtype == torch.float16

            if shape_ok and format_ok and dtype_ok:
                print(f"  {label} S={S}: shape={mask.shape}, format={fmt.name} ✅")
            else:
                all_ok = False
                print(f"  {label} S={S}: shape={mask.shape} (expect {expected_shape}), "
                      f"format={fmt.name}, dtype={mask.dtype} ❌")
        except Exception as e:
            all_ok = False
            print(f"  {label} S={S}: ERROR — {e} ❌")

    if all_ok:
        print("\n✅ All checks passed")
    else:
        print("\n❌ Some checks failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
