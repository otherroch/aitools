"""Tests for face_blender.py."""

import numpy as np
import pytest

from chararep.config import PipelineConfig
from chararep.face_blender import FaceBlender
from chararep.face_detector import TrackedFace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**kw) -> PipelineConfig:
    defaults = dict(
        blend_mode="alpha",
        mask_blur_kernel=5,
        mask_erode_pixels=0,
    )
    defaults.update(kw)
    return PipelineConfig(**defaults)


def _solid(h, w, color=(128, 0, 0)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _make_tracked(bbox, identity_label="hero", landmarks=None) -> TrackedFace:
    if landmarks is None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        landmarks = np.array(
            [
                [cx - 10, cy - 10],
                [cx + 10, cy - 10],
                [cx, cy],
                [cx - 8, cy + 10],
                [cx + 8, cy + 10],
            ],
            dtype=np.float32,
        )
    return TrackedFace(
        track_id=0,
        bbox=np.array(bbox, dtype=np.float32),
        landmarks=landmarks,
        identity_label=identity_label,
    )


# ---------------------------------------------------------------------------
# _alpha_blend static method
# ---------------------------------------------------------------------------

class TestAlphaBlend:
    def test_all_zeros_mask(self):
        orig = _solid(100, 100, (255, 0, 0))
        swap = _solid(100, 100, (0, 0, 255))
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = FaceBlender._alpha_blend(orig, swap, mask)
        np.testing.assert_array_equal(result, orig)

    def test_all_ones_mask(self):
        orig = _solid(100, 100, (255, 0, 0))
        swap = _solid(100, 100, (0, 0, 255))
        mask = np.full((100, 100), 255, dtype=np.uint8)
        result = FaceBlender._alpha_blend(orig, swap, mask)
        np.testing.assert_array_equal(result, swap)

    def test_half_mask_blends(self):
        orig = _solid(100, 100, (0, 0, 0))
        swap = _solid(100, 100, (200, 0, 0))
        mask = np.full((100, 100), 128, dtype=np.uint8)
        result = FaceBlender._alpha_blend(orig, swap, mask)
        # Blended value should be roughly half of 200
        assert 90 <= int(result[50, 50, 0]) <= 110

    def test_output_dtype_is_uint8(self):
        orig = _solid(50, 50)
        swap = _solid(50, 50, (0, 200, 0))
        mask = np.full((50, 50), 128, dtype=np.uint8)
        result = FaceBlender._alpha_blend(orig, swap, mask)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# FaceBlender.blend_all – no-op when no labelled faces
# ---------------------------------------------------------------------------

class TestBlendAllNoOp:
    def test_no_identity_returns_swapped(self):
        cfg = _make_cfg()
        blender = FaceBlender(cfg)
        original = _solid(100, 100, (255, 0, 0))
        swapped = _solid(100, 100, (0, 255, 0))
        tf = TrackedFace(
            track_id=0,
            bbox=np.array([10, 10, 50, 50], dtype=np.float32),
            landmarks=np.zeros((5, 2)),
            identity_label=None,
        )
        result = blender.blend_all(original, swapped, [tf], frame_idx=0)
        np.testing.assert_array_equal(result, swapped)

    def test_empty_faces_returns_swapped(self):
        cfg = _make_cfg()
        blender = FaceBlender(cfg)
        original = _solid(100, 100, (255, 0, 0))
        swapped = _solid(100, 100, (0, 255, 0))
        result = blender.blend_all(original, swapped, [], frame_idx=0)
        np.testing.assert_array_equal(result, swapped)


# ---------------------------------------------------------------------------
# FaceBlender._build_mask
# ---------------------------------------------------------------------------

class TestBuildMask:
    def test_mask_nonzero_inside_bbox(self):
        cfg = _make_cfg(mask_erode_pixels=0, mask_blur_kernel=0)
        blender = FaceBlender(cfg)
        h, w = 200, 200
        bbox = np.array([50, 50, 150, 150], dtype=np.float32)
        lm = np.array(
            [[80, 80], [120, 80], [100, 100], [85, 130], [115, 130]],
            dtype=np.float32,
        )
        mask = blender._build_mask((h, w), bbox, lm)
        assert mask.dtype == np.uint8
        assert mask.shape == (h, w)
        assert mask[100, 100] > 0  # centre of bbox

    def test_mask_without_landmarks_uses_ellipse(self):
        cfg = _make_cfg(mask_erode_pixels=0, mask_blur_kernel=0)
        blender = FaceBlender(cfg)
        bbox = np.array([20, 20, 80, 80], dtype=np.float32)
        mask = blender._build_mask((100, 100), bbox, None)
        assert mask[50, 50] > 0

    def test_mask_with_erode(self):
        cfg = _make_cfg(mask_erode_pixels=5, mask_blur_kernel=0)
        blender = FaceBlender(cfg)
        bbox = np.array([10, 10, 90, 90], dtype=np.float32)
        lm = np.array(
            [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], dtype=np.float32
        )
        mask = blender._build_mask((100, 100), bbox, lm)
        assert mask is not None


# ---------------------------------------------------------------------------
# FaceBlender seamless mode (single face)
# ---------------------------------------------------------------------------

class TestBlendAllSeamless:
    def test_single_face_seamless(self):
        cfg = _make_cfg(blend_mode="seamless", mask_erode_pixels=0)
        blender = FaceBlender(cfg)
        h, w = 200, 200
        original = _solid(h, w, (100, 100, 100))
        swapped = _solid(h, w, (50, 50, 50))
        tf = _make_tracked([40, 40, 160, 160])
        result = blender.blend_all(original, swapped, [tf], frame_idx=0)
        assert result.shape == (h, w, 3)

    def test_two_faces_seamless(self):
        """Two distant faces – multi-face Poisson path."""
        cfg = _make_cfg(blend_mode="seamless", mask_erode_pixels=0, mask_blur_kernel=3)
        blender = FaceBlender(cfg)
        h, w = 200, 400
        original = _solid(h, w, (100, 100, 100))
        swapped = _solid(h, w, (50, 50, 50))
        tf1 = _make_tracked([10, 50, 80, 150])
        tf2 = _make_tracked([300, 50, 390, 150])
        result = blender.blend_all(original, swapped, [tf1, tf2], frame_idx=0)
        assert result.shape == (h, w, 3)


# ---------------------------------------------------------------------------
# FaceBlender alpha mode with labelled face
# ---------------------------------------------------------------------------

class TestBlendAllAlpha:
    def test_alpha_mode_produces_output(self):
        cfg = _make_cfg(blend_mode="alpha")
        blender = FaceBlender(cfg)
        h, w = 200, 200
        original = _solid(h, w, (255, 0, 0))
        swapped = _solid(h, w, (0, 0, 255))
        tf = _make_tracked([40, 40, 160, 160])
        result = blender.blend_all(original, swapped, [tf], frame_idx=0)
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8
