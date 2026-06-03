"""Tests for face_blender.py."""

import types

import cv2
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


def _make_dense_brow_face(landmarks: np.ndarray):
    left_eye, right_eye = landmarks[0], landmarks[1]
    mouth_mid = (landmarks[3] + landmarks[4]) * 0.5
    eye_mid = (left_eye + right_eye) * 0.5
    eye_dist = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
    face_h = max(float(mouth_mid[1] - eye_mid[1]), eye_dist * 0.9)

    def _brow_arc(center: np.ndarray) -> np.ndarray:
        xs = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        ys = -face_h * (0.14 + 0.05 * (1.0 - xs * xs))
        return np.column_stack(
            [
                center[0] + xs * eye_dist * 0.24,
                center[1] + ys,
            ]
        ).astype(np.float32)

    dense = np.full((106, 2), np.nan, dtype=np.float32)
    dense[0:8] = _brow_arc(left_eye)
    dense[16:24] = _brow_arc(right_eye)
    return types.SimpleNamespace(landmark_2d_106=dense)


def _make_tracked(bbox, identity_label="hero", landmarks=None, face_obj=None) -> TrackedFace:
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
        face_obj=face_obj,
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


class TestHybridBlendOne:
    def test_tiny_inner_mask_falls_back_to_alpha(self, monkeypatch):
        original = _solid(64, 64, (10, 20, 30))
        swapped = _solid(64, 64, (220, 210, 200))
        soft_mask = np.zeros((64, 64), dtype=np.uint8)
        soft_mask[32, 32] = 255

        monkeypatch.setattr(
            cv2,
            "seamlessClone",
            lambda *a, **kw: pytest.fail("seamlessClone should not be called"),
        )

        result = FaceBlender._hybrid_blend_one(original, swapped, soft_mask)
        expected = FaceBlender._alpha_blend(original, swapped, soft_mask)
        np.testing.assert_array_equal(result, expected)


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
        result, mask = blender.blend_all(original, swapped, [tf], frame_idx=0)
        np.testing.assert_array_equal(result, swapped)

    def test_empty_faces_returns_swapped(self):
        cfg = _make_cfg()
        blender = FaceBlender(cfg)
        original = _solid(100, 100, (255, 0, 0))
        swapped = _solid(100, 100, (0, 255, 0))
        result, mask = blender.blend_all(original, swapped, [], frame_idx=0)
        np.testing.assert_array_equal(result, swapped)


# ---------------------------------------------------------------------------
# FaceBlender._build_mask
# ---------------------------------------------------------------------------

class TestBuildMask:
    pass
