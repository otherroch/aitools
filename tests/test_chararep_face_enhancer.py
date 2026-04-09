"""Tests for face_enhancer.py."""

import types
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


def _make_tracked_face(x1=10, y1=10, x2=50, y2=50, label="villain"):
    """Create a minimal TrackedFace-like object for testing."""
    return types.SimpleNamespace(
        track_id=0,
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        identity_label=label,
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
