"""Tests for face_enhancer.py."""

import types
import cv2
import numpy as np
import pytest

from chararep.config import PipelineConfig
from chararep.face_enhancer import FaceEnhancer, _GfpganBackend, _OnnxCodeFormerBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(enable=True, weight=0.7, model="gfpgan", model_path=None) -> PipelineConfig:
    return PipelineConfig(
        enable_face_enhancement=enable,
        enhancement_weight=weight,
        enhancement_model=model,
        enhance_model_path=model_path,
    )


def _frame(h=100, w=100) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_landmarks(x1=10, y1=10, x2=50, y2=50) -> np.ndarray:
    cx = (x1 + x2) / 2.0
    eye_y = y1 + (y2 - y1) * 0.35
    mouth_y = y1 + (y2 - y1) * 0.72
    return np.array(
        [
            [x1 + (x2 - x1) * 0.28, eye_y],
            [x1 + (x2 - x1) * 0.72, eye_y],
            [cx, y1 + (y2 - y1) * 0.52],
            [x1 + (x2 - x1) * 0.34, mouth_y],
            [x1 + (x2 - x1) * 0.66, mouth_y],
        ],
        dtype=np.float32,
    )


def _make_tracked_face(
    x1=10,
    y1=10,
    x2=50,
    y2=50,
    label="villain",
    landmarks=None,
    track_id=0,
):
    """Create a minimal TrackedFace-like object for testing."""
    return types.SimpleNamespace(
        track_id=track_id,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        identity_label=label,
        landmarks=landmarks,
    )


# ---------------------------------------------------------------------------
# FaceEnhancer – disabled path
# ---------------------------------------------------------------------------

class TestFaceEnhancerDisabled:
    def test_not_available_when_disabled(self):
        cfg = _make_cfg(enable=False)
        enhancer = FaceEnhancer(cfg)
        assert not enhancer.available

    def test_enhance_returns_original_when_disabled(self):
        cfg = _make_cfg(enable=False)
        enhancer = FaceEnhancer(cfg)
        frame = _frame()
        result = enhancer.enhance(frame, frame_idx=0)
        np.testing.assert_array_equal(result, frame)

    def test_enhance_crop_returns_original_when_disabled(self):
        cfg = _make_cfg(enable=False)
        enhancer = FaceEnhancer(cfg)
        crop = _frame(64, 64)
        result = enhancer.enhance_crop(crop)
        np.testing.assert_array_equal(result, crop)


# ---------------------------------------------------------------------------
# FaceEnhancer – enabled with gfpgan stub
# ---------------------------------------------------------------------------

class TestFaceEnhancerEnabled:
    def test_available_when_gfpgan_loads(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        assert enhancer.available

    def test_enhance_returns_frame(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        frame = _frame()
        result = enhancer.enhance(frame, frame_idx=0)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_enhance_crop_returns_crop(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        crop = _frame(64, 64)
        result = enhancer.enhance_crop(crop)
        assert result.shape == crop.shape

    def test_enhance_crop_custom_weight(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        crop = _frame(64, 64)
        result = enhancer.enhance_crop(crop, weight=0.5)
        assert result.shape == crop.shape

    def test_enhance_fallback_on_exception(self, monkeypatch):
        """If the restorer raises, enhance() returns the original frame."""
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)

        def _raise(*a, **kw):
            raise RuntimeError("mock failure")

        monkeypatch.setattr(enhancer._backend._restorer, "enhance", _raise)
        frame = _frame()
        result = enhancer.enhance(frame, frame_idx=0)
        np.testing.assert_array_equal(result, frame)

    def test_enhance_crop_fallback_on_exception(self, monkeypatch):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)

        def _raise(*a, **kw):
            raise RuntimeError("mock failure")

        monkeypatch.setattr(enhancer._backend._restorer, "enhance", _raise)
        crop = _frame(64, 64)
        result = enhancer.enhance_crop(crop)
        np.testing.assert_array_equal(result, crop)


# ---------------------------------------------------------------------------
# FaceEnhancer – enhance_faces (crop-based path)
# ---------------------------------------------------------------------------

class TestFaceEnhancerEnhanceFaces:
    def test_enhance_faces_modifies_face_region(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        frame = _frame(200, 200)
        faces = [_make_tracked_face(20, 20, 80, 80, label="hero")]
        result = enhancer.enhance_faces(frame, faces, frame_idx=0)
        assert result.shape == frame.shape

    def test_enhance_faces_skips_unlabelled(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        frame = _frame(200, 200)
        original = frame.copy()
        faces = [_make_tracked_face(20, 20, 80, 80, label=None)]
        result = enhancer.enhance_faces(frame, faces, frame_idx=0)
        np.testing.assert_array_equal(result, original)

    def test_enhance_faces_empty_list(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        frame = _frame(100, 100)
        original = frame.copy()
        result = enhancer.enhance_faces(frame, [], frame_idx=0)
        np.testing.assert_array_equal(result, original)

    def test_enhance_faces_when_disabled(self):
        cfg = _make_cfg(enable=False)
        enhancer = FaceEnhancer(cfg)
        frame = _frame(100, 100)
        original = frame.copy()
        faces = [_make_tracked_face()]
        result = enhancer.enhance_faces(frame, faces, frame_idx=0)
        np.testing.assert_array_equal(result, original)

    def test_enhance_faces_uses_soft_mask_pasteback(self, monkeypatch):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        frame = np.full((200, 200, 3), 200, dtype=np.uint8)
        face = _make_tracked_face(
            60,
            50,
            140,
            150,
            label="hero",
            landmarks=_make_landmarks(60, 50, 140, 150),
        )

        monkeypatch.setattr(
            enhancer._backend,
            "enhance_crop",
            lambda crop, weight=0.7: np.zeros_like(crop),
        )

        box = enhancer._propose_enhancement_box(face, 200, 200)
        result = enhancer.enhance_faces(frame.copy(), [face], frame_idx=0)

        x1, y1, x2, y2 = box
        assert result[y1 + 2, x1 + 2, 0] >= 180
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        assert result[center_y, center_x, 0] < 160

    def test_enhance_faces_prefers_aligned_crop_when_backend_supports_it(self, monkeypatch):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        enhancer._backend.aligned_face_size = 64

        seen = {}

        def _enhance(crop, weight=0.7):
            seen["shape"] = crop.shape
            return np.clip(crop.astype(np.int16) + 20, 0, 255).astype(np.uint8)

        monkeypatch.setattr(enhancer._backend, "enhance_crop", _enhance)

        frame = np.full((180, 180, 3), 100, dtype=np.uint8)
        face = _make_tracked_face(
            50,
            40,
            130,
            140,
            label="hero",
            landmarks=_make_landmarks(55, 45, 125, 135),
        )

        result = enhancer.enhance_faces(frame.copy(), [face], frame_idx=0)

        assert seen["shape"] == (64, 64, 3)
        assert result.mean() > frame.mean()


class TestEnhancementStabilizationHelpers:
    def test_estimate_enhancement_affine_maps_landmarks_to_template(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        landmarks = _make_landmarks(60, 50, 140, 150)

        affine = enhancer._estimate_enhancement_affine(landmarks, 128)

        assert affine is not None
        transformed = cv2.transform(landmarks[np.newaxis, :, :], affine)[0]
        template = enhancer._aligned_template(128)
        assert np.max(np.abs(transformed - template)) < 5.0

    def test_propose_enhancement_box_recenters_on_landmarks(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        face = _make_tracked_face(
            40,
            40,
            100,
            120,
            landmarks=_make_landmarks(70, 40, 130, 120),
        )

        x1, _, x2, _ = enhancer._propose_enhancement_box(face, 200, 200)
        box_cx = (x1 + x2) / 2.0
        raw_bbox_cx = float((face.bbox[0] + face.bbox[2]) / 2.0)
        landmark_cx = float(face.landmarks[:, 0].mean())

        assert abs(box_cx - landmark_cx) < abs(raw_bbox_cx - landmark_cx)

    def test_stabilize_enhancement_box_limits_jump(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)

        first = enhancer._stabilize_enhancement_box(1, 0, (40, 40, 140, 180), 300, 300)
        second = enhancer._stabilize_enhancement_box(1, 1, (100, 40, 200, 180), 300, 300)

        first_cx = (first[0] + first[2]) / 2.0
        second_cx = (second[0] + second[2]) / 2.0
        assert second_cx - first_cx < 60
        assert second_cx - first_cx > 0

    def test_stabilize_enhancement_residual_damps_low_frequency(self):
        cfg = _make_cfg(enable=True)
        enhancer = FaceEnhancer(cfg)
        base = np.full((32, 32, 3), 40, dtype=np.float32)
        zero = np.zeros((32, 32, 3), dtype=np.float32)

        first = enhancer._stabilize_enhancement_residual(5, 0, base)
        second = enhancer._stabilize_enhancement_residual(5, 1, zero)

        assert first.mean() == pytest.approx(40.0)
        assert 0.0 < second.mean() < 40.0


# ---------------------------------------------------------------------------
# FaceEnhancer._padded_box
# ---------------------------------------------------------------------------

class TestPaddedBox:
    def test_basic_padding(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        x1, y1, x2, y2 = FaceEnhancer._padded_box(bbox, 500, 500, pad=0.5)
        assert x1 < 100
        assert y1 < 100
        assert x2 > 200
        assert y2 > 200

    def test_clamps_to_frame(self):
        bbox = np.array([0, 0, 50, 50], dtype=np.float32)
        x1, y1, x2, y2 = FaceEnhancer._padded_box(bbox, 60, 60, pad=0.5)
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 60
        assert y2 <= 60


# ---------------------------------------------------------------------------
# FaceEnhancer – gfpgan import error
# ---------------------------------------------------------------------------

class TestFaceEnhancerImportError:
    def test_not_available_when_gfpgan_missing(self, monkeypatch):
        import sys
        # Temporarily remove the gfpgan stub to simulate ImportError
        original = sys.modules.pop("gfpgan", None)
        try:
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "gfpgan":
                    raise ImportError("No module named 'gfpgan'")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", mock_import)
            cfg = _make_cfg(enable=True)
            enhancer = FaceEnhancer(cfg)
            assert not enhancer.available
        finally:
            if original is not None:
                sys.modules["gfpgan"] = original


# ---------------------------------------------------------------------------
# FaceEnhancer – codeformer_onnx backend
# ---------------------------------------------------------------------------

class TestOnnxCodeFormerBackend:
    def test_not_available_without_model_path(self):
        cfg = _make_cfg(enable=True, model="codeformer_onnx", model_path=None)
        enhancer = FaceEnhancer(cfg)
        assert not enhancer.available

    def test_enhance_faces_noop_when_unavailable(self):
        cfg = _make_cfg(enable=True, model="codeformer_onnx", model_path=None)
        enhancer = FaceEnhancer(cfg)
        frame = _frame(200, 200)
        original = frame.copy()
        faces = [_make_tracked_face(20, 20, 80, 80, label="hero")]
        result = enhancer.enhance_faces(frame, faces, frame_idx=0)
        np.testing.assert_array_equal(result, original)

    def test_enhance_crop_returns_original_when_unavailable(self):
        cfg = _make_cfg(enable=True, model="codeformer_onnx", model_path=None)
        enhancer = FaceEnhancer(cfg)
        crop = _frame(64, 64)
        result = enhancer.enhance_crop(crop)
        np.testing.assert_array_equal(result, crop)


# ---------------------------------------------------------------------------
# CLI --enhance-model / --enhance-model-path
# ---------------------------------------------------------------------------

class TestEnhanceModelConfig:
    def test_default_model_is_gfpgan(self):
        cfg = _make_cfg(enable=True)
        assert cfg.enhancement_model == "gfpgan"

    def test_codeformer_onnx_model_set(self):
        cfg = _make_cfg(enable=True, model="codeformer_onnx", model_path="/tmp/cf.onnx")
        assert cfg.enhancement_model == "codeformer_onnx"
        assert cfg.enhance_model_path == "/tmp/cf.onnx"

    def test_enhance_model_path_default_none(self):
        cfg = PipelineConfig()
        assert cfg.enhance_model_path is None
