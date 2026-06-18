"""Minimal verification of forward() input validation (no NPU required).

Each case asserts that the correct exception type is raised with a
message mentioning forward().
"""
import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
from pathlib import Path


class TestForwardValidation(unittest.TestCase):
    """Test all 9 validation rules on Qwen3VLEngine._validate_inputs."""

    @classmethod
    def setUpClass(cls):
        """Build a minimal engine-like object with mocked ATB/graph internals.

        We patch enough of the constructor so we never touch NPU, ATB, or
        torch_npu.  Only the Python weight tensors that _validate_inputs
        reads (embed_w) need real torch.Tensor values.
        """
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from atb_python_qwen3vl_embedding.engine import Qwen3VLEngine

        # Create an engine instance bypassing ATB/NPU init entirely.
        obj = Qwen3VLEngine.__new__(Qwen3VLEngine)
        obj._closed = False
        obj._lock = MagicMock()  # context-manager no-op
        # embed_w shape = (vocab_size, hidden_size); use 151936 as typical
        obj.embed_w = torch.empty(151936, 2048)
        # Attributes needed by pixel_values ↔ grid_thw length consistency check
        obj.patch_size = 14
        obj.tp = 2
        obj.v_cfg = {"in_channels": 3}
        cls.engine = obj

    # ── Happy path ─────────────────────────────────────────────────────

    def test_valid_text_only(self):
        """Normal text-only input passes all checks."""
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        self.engine._validate_inputs(input_ids, None, None)

    def test_valid_vision(self):
        """Normal vision input passes all checks."""
        input_ids = torch.zeros((1, 100), dtype=torch.int64)
        # grid_thw [[1, 3, 3]] → expected_patches = 9
        # 1D pixel_values: 9 * 14*14*3*2 = 10584
        pv = torch.randn(10584, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        self.engine._validate_inputs(input_ids, pv, thw)

    # ── Check 1: input_ids dtype ───────────────────────────────────────

    def test_input_ids_wrong_dtype(self):
        input_ids = torch.zeros((1, 10), dtype=torch.float32)
        with self.assertRaises(TypeError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("forward()", str(ctx.exception))
        self.assertIn("int64", str(ctx.exception))

    # ── Check 2: input_ids shape ───────────────────────────────────────

    def test_input_ids_1d(self):
        input_ids = torch.zeros(10, dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("2D", str(ctx.exception))

    def test_input_ids_3d(self):
        input_ids = torch.zeros((2, 3, 4), dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("2D", str(ctx.exception))

    # ── Check 3: batch_size ────────────────────────────────────────────

    def test_batch_size_greater_than_1(self):
        input_ids = torch.zeros((3, 10), dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("batch_size", str(ctx.exception))

    # ── Check 4: pixel_values / image_grid_thw consistency ─────────────

    def test_pv_set_thw_none(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, None)
        self.assertIn("pixel_values", str(ctx.exception))

    def test_pv_none_thw_set(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, thw)
        self.assertIn("image_grid_thw", str(ctx.exception))

    # ── Check 5: pixel_values dtype ────────────────────────────────────

    def test_pixel_values_wrong_dtype(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float16)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        with self.assertRaises(TypeError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("float32", str(ctx.exception))

    # ── Check 6: pixel_values shape ────────────────────────────────────

    def test_pixel_values_3d(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(2, 3, 4, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("pixel_values", str(ctx.exception))

    # ── Check 7: image_grid_thw dtype ──────────────────────────────────

    def test_image_grid_thw_wrong_dtype(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.float32)
        with self.assertRaises(TypeError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("int64", str(ctx.exception))

    # ── Check 8: image_grid_thw shape ──────────────────────────────────

    def test_image_grid_thw_1d(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([1, 3, 3], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("image_grid_thw", str(ctx.exception))

    def test_image_grid_thw_wrong_last_dim(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("image_grid_thw", str(ctx.exception))

    # ── Check 9: token_id range ────────────────────────────────────────

    def test_token_id_negative(self):
        input_ids = torch.tensor([[-5, 0, 10]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("token_ids", str(ctx.exception))

    def test_token_id_exceeds_vocab(self):
        input_ids = torch.tensor([[0, 151936]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("token_ids", str(ctx.exception))

    def test_token_id_at_boundary_valid(self):
        """Max allowed value = vocab_size - 1."""
        input_ids = torch.tensor([[0, 151935]], dtype=torch.int64)
        self.engine._validate_inputs(input_ids, None, None)

    # ── P0-1: empty input_ids ────────────────────────────────────────────

    def test_input_ids_empty(self):
        """Empty sequence (1, 0) raises ValueError, not RuntimeError."""
        input_ids = torch.empty((1, 0), dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("cannot be empty", str(ctx.exception))

    # ── P0-2: grid_thw value range ───────────────────────────────────────

    def test_grid_thw_zero_t(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[0, 3, 3]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("(T) must be >= 1", str(ctx.exception))

    def test_grid_thw_zero_h(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[1, 0, 3]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("(H) must be >= 1", str(ctx.exception))

    def test_grid_thw_zero_w(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 0]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("(W) must be >= 1", str(ctx.exception))

    def test_grid_thw_negative(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[1, 3, -1]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("must be >= 1", str(ctx.exception))

    # ── P0-3: pixel_values length ↔ grid_thw consistency ─────────────────

    def test_pixel_values_1d_length_mismatch(self):
        """1D pixel_values length doesn't match expected patches * patch_dim."""
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        # grid_thw [[1, 3, 3]] → expected_patches=9, expected_len=10584
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("does not match", str(ctx.exception))

    def test_pixel_values_2d_length_mismatch(self):
        """2D pixel_values shape[0] doesn't match expected patches."""
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        # grid_thw [[1, 3, 3]] → expected_patches=9
        pv = torch.randn(5, 1176, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("does not match", str(ctx.exception))

    # ── P0-4: non-Tensor inputs ──────────────────────────────────────────

    def test_pixel_values_not_tensor(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        with self.assertRaises(TypeError) as ctx:
            self.engine._validate_inputs(input_ids, [1.0, 2.0], thw)
        self.assertIn("pixel_values must be a torch.Tensor", str(ctx.exception))

    def test_image_grid_thw_not_tensor(self):
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        with self.assertRaises(TypeError) as ctx:
            self.engine._validate_inputs(input_ids, pv, [[1, 3, 3]])
        self.assertIn("image_grid_thw must be a torch.Tensor", str(ctx.exception))

    # ── P1-5: empty image_grid_thw ───────────────────────────────────────

    def test_image_grid_thw_zero_rows(self):
        """image_grid_thw with 0 rows should raise ValueError."""
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.empty((0, 3), dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("0 rows", str(ctx.exception))

    # ── P1-6: validation order (dtype before shape) ──────────────────────

    def test_input_ids_dtype_and_shape_combo(self):
        """Both dtype and shape are wrong → TypeError (dtype checked first)."""
        input_ids = torch.zeros((3,), dtype=torch.float32)
        with self.assertRaises(TypeError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("int64", str(ctx.exception))

    # ── P1-7: non-CPU input_ids ──────────────────────────────────────────

    def test_input_ids_not_cpu(self):
        """input_ids on non-CPU device raises ValueError."""
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        try:
            input_ids = input_ids.to("meta")
        except Exception:
            self.skipTest("meta device not available")
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, None, None)
        self.assertIn("must be on CPU", str(ctx.exception))

    # ── P2-10: boundary cases ────────────────────────────────────────────

    def test_pixel_values_0d_scalar(self):
        """0D scalar pixel_values raises ValueError."""
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.tensor(1.0, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("pixel_values", str(ctx.exception))

    def test_pixel_values_2d_valid(self):
        """Legal 2D pixel_values (N, patch_dim) passes validation."""
        input_ids = torch.zeros((1, 100), dtype=torch.int64)
        # grid_thw [[1, 3, 3]] → expected_patches=9, patch_dim=1176
        pv = torch.randn(9, 1176, dtype=torch.float32)
        thw = torch.tensor([[1, 3, 3]], dtype=torch.int64)
        self.engine._validate_inputs(input_ids, pv, thw)

    def test_image_grid_thw_3d(self):
        """3D image_grid_thw raises ValueError."""
        input_ids = torch.zeros((1, 10), dtype=torch.int64)
        pv = torch.randn(100, dtype=torch.float32)
        thw = torch.tensor([[[1, 3, 3]]], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            self.engine._validate_inputs(input_ids, pv, thw)
        self.assertIn("image_grid_thw", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
