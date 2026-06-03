"""Tests for face_swapper.py — covers both inswapper and simswap paths."""

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from chararep.config import PipelineConfig
from chararep.face_swapper import (
    FaceSwapper,
    _ARCFACE_112_V1,
    _WARP_TEMPLATES,
    _detect_model_type,
    _get_simswap_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face(kps=None, embedding=None):
    """Return a minimal InsightFace-like Face stub."""
    face = types.SimpleNamespace(
        kps=kps if kps is not None else np.array(
            [[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
            dtype=np.float32,
        ),
        embedding=embedding if embedding is not None else np.random.randn(512).astype(np.float32),
    )
    return face


def _make_cfg(tmp_path, model_name="inswapper_128.onnx", **kw):
    """Create a PipelineConfig pointing to a real (empty) model file."""
    model_file = tmp_path / model_name
    model_file.write_bytes(b"")
    return PipelineConfig(swap_model_path=str(model_file), **kw)


def _make_aligned_eye_landmarks(
    crop_size=256,
    template_name="arcface_128",
    left_aperture=10.0,
    right_aperture=3.5,
):
    """Build synthetic aligned landmarks with per-eye aperture differences."""
    template = _WARP_TEMPLATES[template_name] * float(crop_size)
    left_eye, right_eye = template[:2]
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    horiz = eye_dist * 0.16

    def _eye_ring(center, aperture):
        return np.array(
            [
                center + [-horiz, 0.0],
                center + [-horiz * 0.55, -aperture * 0.85],
                center + [0.0, -aperture],
                center + [horiz * 0.55, -aperture * 0.85],
                center + [horiz, 0.0],
                center + [horiz * 0.55, aperture * 0.85],
                center + [0.0, aperture],
                center + [-horiz * 0.55, aperture * 0.85],
            ],
            dtype=np.float32,
        )

    points = np.zeros((106, 2), dtype=np.float32)
    left_ring = _eye_ring(left_eye, left_aperture)
    right_ring = _eye_ring(right_eye, right_aperture)
    points[: len(left_ring)] = left_ring
    points[16 : 16 + len(right_ring)] = right_ring
    return points


def _make_aligned_eye_brow_landmarks(
    crop_size=256,
    template_name="arcface_128",
    left_aperture=10.0,
    right_aperture=3.5,
):
    """Build synthetic aligned landmarks with brows above each eye."""
    points = _make_aligned_eye_landmarks(
        crop_size=crop_size,
        template_name=template_name,
        left_aperture=left_aperture,
        right_aperture=right_aperture,
    )
    template = _WARP_TEMPLATES[template_name] * float(crop_size)
    left_eye, right_eye, _, mouth_left, mouth_right = template
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (mouth_left + mouth_right) * 0.5
    eye_dist = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
    mid_height = max(float(mouth_mid[1] - eye_mid[1]), eye_dist * 0.85)

    brow_x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    brow_y = -mid_height * (0.48 + 0.10 * (1.0 - brow_x * brow_x))
    points[24:32] = np.column_stack(
        [
            left_eye[0] + brow_x * eye_dist * 0.28,
            left_eye[1] + brow_y,
        ]
    ).astype(np.float32)
    points[32:40] = np.column_stack(
        [
            right_eye[0] + brow_x * eye_dist * 0.28,
            right_eye[1] + brow_y,
        ]
    ).astype(np.float32)
    return points


# ---------------------------------------------------------------------------
# _detect_model_type
# ---------------------------------------------------------------------------

class TestDetectModelType:
    def test_inswapper(self):
        assert _detect_model_type("inswapper_128.onnx") == "inswapper"

    def test_inswapper_fp16(self):
        assert _detect_model_type("inswapper_128_fp16.onnx") == "inswapper"

    def test_simswap_256(self):
        assert _detect_model_type("simswap_256.onnx") == "simswap"

    def test_simswap_unofficial_512(self):
        assert _detect_model_type("simswap_unofficial_512.onnx") == "simswap"

    def test_simswap_case_insensitive(self):
        assert _detect_model_type("/models/SimSwap_256.onnx") == "simswap"

    def test_unknown_defaults_to_inswapper(self):
        assert _detect_model_type("some_other_model.onnx") == "inswapper"


# ---------------------------------------------------------------------------
# _get_simswap_params
# ---------------------------------------------------------------------------

class TestGetSimswapParams:
    def test_simswap_256_params(self):
        params = _get_simswap_params("simswap_256.onnx")
        assert params["size"] == 256
        assert params["mean"].shape == (3,)
        assert params["std"].shape == (3,)
        # ImageNet normalisation
        assert pytest.approx(params["mean"][0], abs=1e-4) == 0.485

    def test_simswap_unofficial_512_params(self):
        params = _get_simswap_params("simswap_unofficial_512.onnx")
        assert params["size"] == 512
        np.testing.assert_array_equal(params["mean"], [0, 0, 0])
        np.testing.assert_array_equal(params["std"], [1, 1, 1])

    def test_unknown_returns_defaults(self):
        params = _get_simswap_params("unknown_model.onnx")
        assert params["size"] == 256


# ---------------------------------------------------------------------------
# FaceSwapper — inswapper path (uses insightface.model_zoo stub)
# ---------------------------------------------------------------------------

class TestFaceSwapperInswapper:
    def test_loads_inswapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "inswapper_128.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "inswapper"
        assert swapper._ort_session is None
        assert swapper._embedding_converter is None

    def test_swap_returns_same_shape(self, tmp_path):
        cfg = _make_cfg(tmp_path, "inswapper_128.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        target_face = _make_face()
        result = swapper.swap(frame, source_face, target_face)
        assert result.shape == frame.shape

    def test_swap_multiple_applies_all_pairs(self, tmp_path):
        cfg = _make_cfg(tmp_path, "inswapper_128.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pairs = [(_make_face(), _make_face()), (_make_face(), _make_face())]
        result = swapper.swap_multiple(frame, pairs, frame_idx=0)
        assert result.shape == frame.shape

    def test_model_not_found_raises(self):
        cfg = PipelineConfig(swap_model_path="/nonexistent/inswapper_128.onnx")
        with pytest.raises(FileNotFoundError):
            FaceSwapper(cfg)

    def test_no_path_no_default_raises(self, tmp_path, monkeypatch):
        """When no path is given and no default exists, raise FileNotFoundError."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cfg = PipelineConfig(swap_model_path=None)
        with pytest.raises(FileNotFoundError, match="inswapper model not found"):
            FaceSwapper(cfg)

    def test_default_path_detection(self, tmp_path, monkeypatch):
        """Falls back to ~/.insightface/models/inswapper_128.onnx if it exists."""
        models_dir = tmp_path / ".insightface" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "inswapper_128.onnx").write_bytes(b"")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cfg = PipelineConfig(swap_model_path=None)
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "inswapper"

    def test_buffalo_l_fallback_path(self, tmp_path, monkeypatch):
        """Falls back to ~/.insightface/models/buffalo_l/inswapper_128.onnx."""
        buffalo_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
        buffalo_dir.mkdir(parents=True)
        (buffalo_dir / "inswapper_128.onnx").write_bytes(b"")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cfg = PipelineConfig(swap_model_path=None)
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "inswapper"


# ---------------------------------------------------------------------------
# FaceSwapper — simswap path (uses onnxruntime.InferenceSession stub)
# ---------------------------------------------------------------------------

class TestFaceSwapperSimswap:
    def test_loads_simswap_unofficial_512(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "simswap"
        assert swapper._model is None
        assert swapper._ort_session is not None
        assert swapper._simswap_params is not None
        assert swapper._simswap_params["size"] == 512

    def test_loads_simswap_256(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_256.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "simswap"
        # The stub returns shape [1,3,256,256] for simswap_256, confirming
        # that ONNX-derived size (256) matches the filename-derived size.
        assert swapper._simswap_params["size"] == 256

    def test_onnx_size_overrides_filename_size(self, tmp_path, monkeypatch):
        """When ONNX image input reports a size different from the filename,
        __init__ updates _simswap_params['size'] to the ONNX-reported value."""
        import sys

        # Build a replacement InferenceSession stub that always reports
        # 512×512 for its image input regardless of filename.
        _orig_session = sys.modules["onnxruntime"].InferenceSession

        class _Session512(_orig_session):
            def get_inputs(self):
                return [
                    types.SimpleNamespace(name="source", shape=[1, 512]),
                    types.SimpleNamespace(name="target", shape=[1, 3, 512, 512]),
                ]

            def run(self, out, feed):
                tgt = feed.get("target")
                return [tgt if tgt is not None else np.zeros((1, 3, 512, 512), dtype=np.float32)]

        monkeypatch.setattr(sys.modules["onnxruntime"], "InferenceSession", _Session512)

        # Filename implies size=256 via _get_simswap_params, but ONNX says 512.
        cfg = _make_cfg(tmp_path, "simswap_256.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._simswap_params["size"] == 512

    def test_swap_returns_same_shape(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        target_face = _make_face()
        result = swapper.swap(frame, source_face, target_face)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_swap_multiple_all_pairs(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pairs = [(_make_face(), _make_face()), (_make_face(), _make_face())]
        result = swapper.swap_multiple(frame, pairs, frame_idx=5)
        assert result.shape == frame.shape

    def test_embedding_converter_missing_logs_warning(self, tmp_path, caplog):
        import logging
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path="/nonexistent/crossface.onnx",
        )
        with caplog.at_level(logging.WARNING):
            swapper = FaceSwapper(cfg)
        assert swapper._embedding_converter is None
        assert any("not found" in r.message for r in caplog.records)

    def test_embedding_converter_loaded_when_present(self, tmp_path):
        converter_file = tmp_path / "crossface_simswap.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)
        assert swapper._embedding_converter is not None


# ---------------------------------------------------------------------------
# FaceSwapper internal helpers
# ---------------------------------------------------------------------------

class TestWarpFace:
    def setup_method(self):
        """Create a minimal simswap swapper to exercise helpers."""
        pass

    def _make_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        return FaceSwapper(cfg)

    def test_warp_returns_correct_size(self, tmp_path):
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        kps = np.array(
            [[200, 250], [350, 250], [275, 350], [210, 450], [340, 450]],
            dtype=np.float32,
        )
        crop, M = swapper._warp_face(frame, kps, size=512)
        assert crop.shape == (512, 512, 3)
        assert M.shape == (2, 3)

    def test_warp_affine_matrix_is_valid(self, tmp_path):
        swapper = self._make_swapper(tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kps = np.array(
            [[100, 120], [180, 120], [140, 170], [105, 220], [175, 220]],
            dtype=np.float32,
        )
        _, M = swapper._warp_face(frame, kps, size=256)
        # Matrix should have finite values
        assert np.all(np.isfinite(M))

    def test_similarity_transform_is_stable_under_small_landmark_perturbation(self, tmp_path):
        swapper = self._make_swapper(tmp_path)
        template = (_WARP_TEMPLATES["arcface_128"] * 256.0).astype(np.float32)
        base_kps = np.array(
            [[100, 120], [180, 120], [140, 170], [105, 220], [175, 220]],
            dtype=np.float32,
        )
        perturbed_kps = base_kps.copy()
        perturbed_kps[3] += np.array([2.0, -1.0], dtype=np.float32)
        perturbed_kps[4] += np.array([-2.0, 1.0], dtype=np.float32)

        M1 = swapper._estimate_similarity_transform(base_kps, template)
        M2 = swapper._estimate_similarity_transform(perturbed_kps, template)

        assert M1 is not None
        assert M2 is not None
        scale1 = float(np.sqrt(abs(np.linalg.det(M1[:, :2]))))
        scale2 = float(np.sqrt(abs(np.linalg.det(M2[:, :2]))))
        assert abs(scale1 - scale2) < 0.015

    def test_warp_raises_on_degenerate_landmarks(self, tmp_path):
        """All-identical landmarks make RANSAC return None → RuntimeError."""
        swapper = self._make_swapper(tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # All keypoints at the same location → degenerate
        kps = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Face alignment failed"):
            swapper._warp_face(frame, kps, size=256)


class TestFilterLandmarks:
    def _make_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        return FaceSwapper(cfg)

    def test_accepts_row_major_landmarks(self, tmp_path):
        swapper = self._make_swapper(tmp_path)
        kps = np.array(
            [[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
            dtype=np.float32,
        )
        filtered = swapper._filter_landmarks(kps)
        assert filtered.shape == (5, 2)
        np.testing.assert_array_equal(filtered, kps)

    def test_transposes_legacy_column_major_landmarks(self, tmp_path):
        swapper = self._make_swapper(tmp_path)
        kps = np.array(
            [[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
            dtype=np.float32,
        )
        filtered = swapper._filter_landmarks(kps.T)
        assert filtered.shape == (5, 2)
        np.testing.assert_array_equal(filtered, kps)


class TestPrepareCropFrame:
    def test_output_shape_and_dtype(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        crop = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        tensor = swapper._prepare_crop_frame(crop)
        assert tensor.shape == (1, 3, 512, 512)
        assert tensor.dtype == np.float32

    def test_normalization_zero_mean_unit_std(self, tmp_path):
        """For simswap_unofficial_512 (mean=0, std=1), values are in [0, 1]."""
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        crop = np.full((512, 512, 3), 128, dtype=np.uint8)
        tensor = swapper._prepare_crop_frame(crop)
        # 128 / 255 ≈ 0.502; with mean=0, std=1 → values ≈ 0.502
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_normalization_imagenet(self, tmp_path):
        """For simswap_256 (ImageNet mean/std), output can be negative."""
        cfg = _make_cfg(tmp_path, "simswap_256.onnx")
        swapper = FaceSwapper(cfg)
        crop = np.zeros((256, 256, 3), dtype=np.uint8)
        tensor = swapper._prepare_crop_frame(crop)
        # pixel=0 → (0 - mean)/std is negative for ImageNet mean
        assert tensor.min() < 0.0


class TestNormalizeCropFrame:
    def test_output_shape_and_dtype(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        output = np.random.rand(3, 512, 512).astype(np.float32)
        result = swapper._normalize_crop_frame(output)
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    def test_clipping_to_valid_range(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        # Values outside [0,1] should be clipped
        output = np.full((3, 64, 64), 2.0, dtype=np.float32)
        result = swapper._normalize_crop_frame(output)
        assert result.max() == 255
        output_neg = np.full((3, 64, 64), -1.0, dtype=np.float32)
        result_neg = swapper._normalize_crop_frame(output_neg)
        assert result_neg.max() == 0


class TestPasteBack:
    pass
