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
    def test_feature_core_mask_is_stronger_on_eyes_than_forehead(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)

        mask = swapper._build_feature_core_mask(256, "arcface_128")
        template = _WARP_TEMPLATES["arcface_128"] * 256.0
        left_eye, right_eye, _, mouth_left, mouth_right = template
        eye_mid = (left_eye + right_eye) * 0.5
        mouth_mid = (mouth_left + mouth_right) * 0.5
        eye_dist = np.linalg.norm(right_eye - left_eye)
        mid_height = max(float(mouth_mid[1] - eye_mid[1]), float(eye_dist) * 0.85)

        eye_y = int(round(float(left_eye[1])))
        eye_x = int(round(float(left_eye[0])))
        forehead_y = int(round(float(eye_mid[1] - mid_height * 0.60)))
        forehead_x = int(round(float(eye_mid[0])))

        assert mask[eye_y, eye_x] > mask[forehead_y, forehead_x] + 0.2

    def test_feature_core_mask_tracks_live_eye_aperture(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)

        aligned_landmarks = _make_aligned_eye_landmarks()
        mask = swapper._build_feature_core_mask(
            256,
            "arcface_128",
            aligned_landmarks=aligned_landmarks,
        )
        template = _WARP_TEMPLATES["arcface_128"] * 256.0
        left_eye, right_eye = template[:2]

        def _eye_energy(center):
            cx = int(round(float(center[0])))
            cy = int(round(float(center[1])))
            roi = mask[cy - 14 : cy + 15, cx - 18 : cx + 19]
            return float(roi.sum())

        assert _eye_energy(left_eye) > _eye_energy(right_eye) * 1.18

    def test_feature_core_mask_tracks_live_brow_height(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)

        plain_mask = swapper._build_feature_core_mask(256, "arcface_128")
        aligned_landmarks = _make_aligned_eye_brow_landmarks()
        mask = swapper._build_feature_core_mask(
            256,
            "arcface_128",
            aligned_landmarks=aligned_landmarks,
        )
        template = _WARP_TEMPLATES["arcface_128"] * 256.0
        left_eye, right_eye, _, mouth_left, mouth_right = template
        eye_mid = (left_eye + right_eye) * 0.5
        mouth_mid = (mouth_left + mouth_right) * 0.5
        eye_dist = np.linalg.norm(right_eye - left_eye)
        mid_height = max(float(mouth_mid[1] - eye_mid[1]), float(eye_dist) * 0.85)

        brow_y = int(round(float(left_eye[1] - mid_height * 0.56)))
        brow_x = int(round(float(left_eye[0])))
        plain_roi = plain_mask[brow_y - 3 : brow_y + 4, brow_x - 14 : brow_x + 15]
        live_roi = mask[brow_y - 3 : brow_y + 4, brow_x - 14 : brow_x + 15]

        assert float(live_roi.mean()) > float(plain_roi.mean()) + 0.03

    def test_output_shape_preserved(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float32)  # identity — crop maps to top-left
        result = swapper._paste_back(frame, crop, M)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_degenerate_matrix_returns_original_frame(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        # Zero matrix → determinant = 0 → degenerate
        M_zero = np.zeros((2, 3), dtype=np.float32)
        result = swapper._paste_back(frame, crop, M_zero)
        np.testing.assert_array_equal(result, frame)

    def test_paste_back_keeps_eye_core_stronger_than_forehead(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        crop = np.full((256, 256, 3), 200, dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float32)

        result = swapper._paste_back(frame, crop, M, template_name="arcface_128")
        template = _WARP_TEMPLATES["arcface_128"] * 256.0
        left_eye, right_eye, _, mouth_left, mouth_right = template
        eye_mid = (left_eye + right_eye) * 0.5
        mouth_mid = (mouth_left + mouth_right) * 0.5
        eye_dist = np.linalg.norm(right_eye - left_eye)
        mid_height = max(float(mouth_mid[1] - eye_mid[1]), float(eye_dist) * 0.85)

        eye_y = int(round(float(left_eye[1])))
        eye_x = int(round(float(left_eye[0])))
        forehead_y = int(round(float(eye_mid[1] - mid_height * 0.60)))
        forehead_x = int(round(float(eye_mid[0])))

        assert int(result[eye_y, eye_x, 0]) > int(result[forehead_y, forehead_x, 0]) + 20

    def test_paste_back_tracks_live_eye_aperture(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        crop = np.full((256, 256, 3), 200, dtype=np.uint8)
        aligned_landmarks = _make_aligned_eye_landmarks(left_aperture=10.0, right_aperture=3.0)

        result = swapper._paste_back(
            frame,
            crop,
            np.eye(2, 3, dtype=np.float32),
            template_name="arcface_128",
            aligned_landmarks=aligned_landmarks,
        )
        template = _WARP_TEMPLATES["arcface_128"] * 256.0
        left_eye, right_eye = template[:2]

        def _eye_mean(center):
            cx = int(round(float(center[0])))
            cy = int(round(float(center[1])))
            roi = result[cy - 14 : cy + 15, cx - 18 : cx + 19, 0]
            return float(roi.mean())

        assert _eye_mean(left_eye) > _eye_mean(right_eye) + 6.0

    def test_paste_back_tracks_live_brow_height(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        crop = np.full((256, 256, 3), 200, dtype=np.uint8)

        plain = swapper._paste_back(
            frame,
            crop,
            np.eye(2, 3, dtype=np.float32),
            template_name="arcface_128",
        )
        aligned_landmarks = _make_aligned_eye_brow_landmarks()
        result = swapper._paste_back(
            frame,
            crop,
            np.eye(2, 3, dtype=np.float32),
            template_name="arcface_128",
            aligned_landmarks=aligned_landmarks,
        )
        template = _WARP_TEMPLATES["arcface_128"] * 256.0
        left_eye, right_eye, _, mouth_left, mouth_right = template
        eye_mid = (left_eye + right_eye) * 0.5
        mouth_mid = (mouth_left + mouth_right) * 0.5
        eye_dist = np.linalg.norm(right_eye - left_eye)
        mid_height = max(float(mouth_mid[1] - eye_mid[1]), float(eye_dist) * 0.85)

        brow_y = int(round(float(left_eye[1] - mid_height * 0.56)))
        brow_x = int(round(float(left_eye[0])))
        plain_roi = plain[brow_y - 3 : brow_y + 4, brow_x - 14 : brow_x + 15, 0]
        live_roi = result[brow_y - 3 : brow_y + 4, brow_x - 14 : brow_x + 15, 0]

        assert float(live_roi.mean()) > float(plain_roi.mean()) + 8.0


class TestPrepareSourceEmbedding:
    def test_output_shape_and_normalized(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        face = _make_face(embedding=np.random.randn(512).astype(np.float32))
        emb = swapper._prepare_source_embedding(face)
        assert emb.shape == (1, 512)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_zero_embedding_handled_safely(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        face = _make_face(embedding=np.zeros(512, dtype=np.float32))
        emb = swapper._prepare_source_embedding(face)
        assert emb.shape == (1, 512)
        # Should not raise; result will be all zeros
        assert not np.any(np.isnan(emb))

    def test_none_embedding_returns_none(self, tmp_path):
        """Face with no embedding attribute returns None without raising."""
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        face = types.SimpleNamespace(embedding=None)
        result = swapper._prepare_source_embedding(face)
        assert result is None

    def test_missing_embedding_attr_returns_none(self, tmp_path):
        """Face object without an embedding attribute returns None."""
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        face = types.SimpleNamespace()
        result = swapper._prepare_source_embedding(face)
        assert result is None

    def test_embedding_converter_receives_rank4_input(self, tmp_path, monkeypatch):
        """Embedding converter must receive a rank-4 tensor (N, 512, 1, 1).

        The real ``simswap_arcface_model.onnx`` / ``crossface_simswap.onnx``
        id-encoder expects rank 4.  Passing rank 2 raises
        ``onnxruntime.InvalidArgument: Invalid rank for input: input
        Got: 2 Expected: 4``.
        """
        converter_file = tmp_path / "crossface_simswap.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)

        received_shapes = []
        original_run = swapper._embedding_converter.run

        def capturing_run(output_names, input_feed):
            received_shapes.append(input_feed["input"].shape)
            return original_run(output_names, input_feed)

        monkeypatch.setattr(swapper._embedding_converter, "run", capturing_run)

        face = _make_face(embedding=np.random.randn(512).astype(np.float32))
        emb = swapper._prepare_source_embedding(face)

        assert len(received_shapes) == 1
        assert len(received_shapes[0]) == 4, (
            f"Embedding converter must receive rank-4 input; got rank {len(received_shapes[0])}"
        )
        assert received_shapes[0] == (1, 512, 1, 1)
        # Final output must still be (1, 512) and L2-normalised
        assert emb.shape == (1, 512)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_embedding_converter_failure_returns_none(self, tmp_path, monkeypatch, caplog):
        """An ONNX error from the embedding converter is caught and returns None."""
        import logging
        converter_file = tmp_path / "crossface_simswap.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)

        def _failing_run(output_names, input_feed):
            raise RuntimeError("InvalidArgument: Invalid rank for input: input Got: 4 Expected: 2")

        monkeypatch.setattr(swapper._embedding_converter, "run", _failing_run)

        face = _make_face(embedding=np.random.randn(512).astype(np.float32))
        with caplog.at_level(logging.WARNING):
            result = swapper._prepare_source_embedding(face)
        assert result is None
        assert any("embedding converter failed" in r.message for r in caplog.records)

    def test_arcface_image_encoder_uses_crop(self, tmp_path, monkeypatch):
        """Image-encoder mode: converter receives (1,3,112,112) crop, not embedding."""
        converter_file = tmp_path / "simswap_arcface_model.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)

        # Make the stub look like an ArcFace image encoder
        monkeypatch.setattr(
            swapper._embedding_converter,
            "get_inputs",
            lambda: [types.SimpleNamespace(name="input", shape=[1, 3, 112, 112])],
        )
        # Re-detect mode after patching
        swapper._converter_mode = swapper._detect_converter_mode()
        assert swapper._converter_mode == "image"

        received_feeds = []
        def _capturing_run(output_names, input_feed):
            received_feeds.append(input_feed)
            # Return a fake 512-dim embedding
            return [np.random.randn(1, 512).astype(np.float32)]

        monkeypatch.setattr(swapper._embedding_converter, "run", _capturing_run)

        crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        face = types.SimpleNamespace(arcface_crop=crop)
        emb = swapper._prepare_source_embedding(face)

        assert len(received_feeds) == 1
        inp = received_feeds[0]["input"]
        assert inp.shape == (1, 3, 112, 112), f"Expected (1,3,112,112), got {inp.shape}"
        assert inp.dtype == np.float32
        # Values should be in [-1, 1] range (ArcFace normalisation)
        assert inp.min() >= -1.0 - 1e-5
        assert inp.max() <= 1.0 + 1e-5
        # Output should be (1, 512) and L2-normalised
        assert emb is not None
        assert emb.shape == (1, 512)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_arcface_image_encoder_missing_crop_returns_none(self, tmp_path, monkeypatch, caplog):
        """Image-encoder mode returns None when arcface_crop is not on the face."""
        import logging
        converter_file = tmp_path / "simswap_arcface_model.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)
        monkeypatch.setattr(
            swapper._embedding_converter,
            "get_inputs",
            lambda: [types.SimpleNamespace(name="input", shape=[1, 3, 112, 112])],
        )
        swapper._converter_mode = swapper._detect_converter_mode()

        face = types.SimpleNamespace()  # no arcface_crop
        with caplog.at_level(logging.WARNING):
            result = swapper._prepare_source_embedding(face)
        assert result is None
        assert any("arcface_crop" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# FaceSwapper._detect_converter_mode
# ---------------------------------------------------------------------------

class TestDetectConverterMode:
    def test_no_converter_returns_embedding(self, tmp_path):
        """No converter loaded → always 'embedding' mode."""
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._converter_mode == "embedding"

    def test_image_encoder_shape_detected(self, tmp_path, monkeypatch):
        """Converter with (N,3,112,112) input → 'image' mode."""
        converter_file = tmp_path / "simswap_arcface_model.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)
        monkeypatch.setattr(
            swapper._embedding_converter,
            "get_inputs",
            lambda: [types.SimpleNamespace(name="input", shape=[1, 3, 112, 112])],
        )
        assert swapper._detect_converter_mode() == "image"

    def test_feature_converter_shape_detected(self, tmp_path, monkeypatch):
        """Converter with (N,512,1,1) input → 'embedding' mode."""
        converter_file = tmp_path / "crossface_simswap.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)
        monkeypatch.setattr(
            swapper._embedding_converter,
            "get_inputs",
            lambda: [types.SimpleNamespace(name="input", shape=[1, 512, 1, 1])],
        )
        assert swapper._detect_converter_mode() == "embedding"

    def test_missing_shape_attr_defaults_to_embedding(self, tmp_path, monkeypatch):
        """Converter whose inputs have no 'shape' attr → safe 'embedding' fallback."""
        converter_file = tmp_path / "unknown_converter.onnx"
        converter_file.write_bytes(b"")
        cfg = _make_cfg(
            tmp_path,
            "simswap_unofficial_512.onnx",
            embedding_converter_path=str(converter_file),
        )
        swapper = FaceSwapper(cfg)
        monkeypatch.setattr(
            swapper._embedding_converter,
            "get_inputs",
            lambda: [types.SimpleNamespace(name="input")],  # no shape attribute
        )
        assert swapper._detect_converter_mode() == "embedding"


# ---------------------------------------------------------------------------
# FaceSwapper._build_feed_dict — named and positional input mapping
# ---------------------------------------------------------------------------

class TestBuildFeedDict:
    def _make_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        return FaceSwapper(cfg)

    def test_named_inputs_matched_correctly(self, tmp_path):
        """Standard 'source'/'target' names are mapped as expected."""
        swapper = self._make_swapper(tmp_path)
        src = np.zeros((1, 512), dtype=np.float32)
        tgt = np.zeros((1, 3, 512, 512), dtype=np.float32)
        feed = swapper._build_feed_dict(src, tgt)
        assert "source" in feed
        assert "target" in feed
        # Values must equal the inputs (identity check not valid after reshape)
        np.testing.assert_array_equal(feed["source"].ravel(), src.ravel())
        np.testing.assert_array_equal(feed["target"], tgt)

    def test_positional_fallback_for_unknown_names(self, tmp_path, monkeypatch):
        """When inputs lack shape metadata, the last-resort positional fallback is used."""
        swapper = self._make_swapper(tmp_path)
        # Patch get_inputs to return non-standard names without shape metadata
        inp0 = types.SimpleNamespace(name="latent_id")
        inp1 = types.SimpleNamespace(name="face_image")
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.zeros((1, 512), dtype=np.float32)
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_feed_dict(src, tgt)
        # No shape metadata → last-resort positional: first input gets embedding
        np.testing.assert_array_equal(feed.get("latent_id").ravel(), src.ravel())
        np.testing.assert_array_equal(feed.get("face_image"), tgt)

    def test_official_256_layout_crop_first(self, tmp_path, monkeypatch):
        """Official SimSwap-256 layout: 'input' (image crop) is inputs[0], embedding is inputs[1].

        The shape-based classifier must route the embedding and crop correctly
        even when the model's positional order is [crop, embedding].
        """
        swapper = self._make_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="input", shape=[1, 3, 512, 512])   # image crop
        inp1 = types.SimpleNamespace(name="latent_id", shape=[1, 512, 1, 1]) # identity embedding
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 512), dtype=np.float32)
        tgt = np.zeros((1, 3, 512, 512), dtype=np.float32)
        feed = swapper._build_feed_dict(src, tgt)
        # Embedding should go to "latent_id", reshaped to (1, 512, 1, 1)
        assert "latent_id" in feed
        assert feed["latent_id"].shape == (1, 512, 1, 1)
        np.testing.assert_array_equal(feed["latent_id"].ravel(), src.ravel())
        # Crop should go to "input"
        assert "input" in feed
        np.testing.assert_array_equal(feed["input"], tgt)

    def test_rank4_embedding_for_rank4_input(self, tmp_path, monkeypatch):
        """Source input with rank-4 expected shape gets embedding reshaped to (1,512,1,1)."""
        swapper = self._make_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="latent_id", shape=[1, 512, 1, 1])
        inp1 = types.SimpleNamespace(name="face_image", shape=[1, 3, 256, 256])
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 512), dtype=np.float32)
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_feed_dict(src, tgt)
        assert feed["latent_id"].shape == (1, 512, 1, 1)
        np.testing.assert_array_equal(feed["latent_id"].ravel(), src.ravel())

    def test_rank2_embedding_for_rank2_input(self, tmp_path, monkeypatch):
        """Source input with rank-2 expected shape keeps embedding as (1,512)."""
        swapper = self._make_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="source", shape=[1, 512])
        inp1 = types.SimpleNamespace(name="target", shape=[1, 3, 256, 256])
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 512), dtype=np.float32)
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_feed_dict(src, tgt)
        assert feed["source"].shape == (1, 512)
        np.testing.assert_array_equal(feed["source"], src)

    def test_no_shape_metadata_returns_embedding_unchanged(self, tmp_path, monkeypatch):
        """When model input has no shape metadata, embedding is passed through unchanged."""
        swapper = self._make_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="latent_id")  # no shape attribute
        inp1 = types.SimpleNamespace(name="face_image")
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 512), dtype=np.float32)
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_feed_dict(src, tgt)
        assert feed["latent_id"].shape == (1, 512)
        np.testing.assert_array_equal(feed["latent_id"], src)


# ---------------------------------------------------------------------------
# FaceSwapper._build_image_feed_dict — named and positional input mapping
# for image-source models (uniface / blendswap)
# ---------------------------------------------------------------------------

class TestBuildImageFeedDict:
    def _make_uniface_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        return FaceSwapper(cfg)

    def test_build_image_feed_dict_uses_named_inputs_when_present(self, tmp_path, monkeypatch):
        """When the model has 'source' and 'target' inputs, they are used as-is."""
        swapper = self._make_uniface_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="source", shape=[1, 3, 256, 256])
        inp1 = types.SimpleNamespace(name="target", shape=[1, 3, 256, 256])
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 3, 256, 256), dtype=np.float32)
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_image_feed_dict(src, tgt)
        assert "source" in feed
        assert "target" in feed
        np.testing.assert_array_equal(feed["source"], src)
        np.testing.assert_array_equal(feed["target"], tgt)

    def test_build_image_feed_dict_uses_positional_for_generic_names(self, tmp_path, monkeypatch):
        """input_0 / input_1 (real uniface export) → positional mapping."""
        swapper = self._make_uniface_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="input_0", shape=[1, 3, 256, 256])
        inp1 = types.SimpleNamespace(name="input_1", shape=[1, 3, 256, 256])
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 3, 256, 256), dtype=np.float32) * 0.5
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_image_feed_dict(src, tgt)
        # First input receives the source portrait
        assert "input_0" in feed
        np.testing.assert_array_equal(feed["input_0"], src)
        # Second input receives the target crop
        assert "input_1" in feed
        np.testing.assert_array_equal(feed["input_1"], tgt)

    def test_build_image_feed_dict_uses_positional_for_arbitrary_names(self, tmp_path, monkeypatch):
        """Any non-standard input names fall back to positional assignment."""
        swapper = self._make_uniface_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="x", shape=[1, 3, 256, 256])
        inp1 = types.SimpleNamespace(name="y", shape=[1, 3, 256, 256])
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 3, 256, 256), dtype=np.float32)
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_image_feed_dict(src, tgt)
        np.testing.assert_array_equal(feed["x"], src)
        np.testing.assert_array_equal(feed["y"], tgt)

    def test_build_image_feed_dict_falls_back_to_positional_on_partial_match(self, tmp_path, monkeypatch):
        """Only having 'source' but not 'target' (or vice-versa) uses positional."""
        swapper = self._make_uniface_swapper(tmp_path)
        inp0 = types.SimpleNamespace(name="source", shape=[1, 3, 256, 256])
        inp1 = types.SimpleNamespace(name="input_1", shape=[1, 3, 256, 256])
        monkeypatch.setattr(swapper._ort_session, "get_inputs", lambda: [inp0, inp1])

        src = np.ones((1, 3, 256, 256), dtype=np.float32)
        tgt = np.zeros((1, 3, 256, 256), dtype=np.float32)
        feed = swapper._build_image_feed_dict(src, tgt)
        # Neither "target" is present → positional: first→source, second→target
        np.testing.assert_array_equal(feed["source"], src)
        np.testing.assert_array_equal(feed["input_1"], tgt)


# ---------------------------------------------------------------------------
# End-to-end swap tests for uniface with input_0/input_1 model names
# ---------------------------------------------------------------------------

class TestUnifaceGenericInputNames:
    """Integration tests confirming the fix for uniface ONNX exports that use
    generic input names (input_0 / input_1) instead of source / target."""

    def test_swap_succeeds_with_input_0_input_1(self, tmp_path, monkeypatch):
        """Full uniface swap pipeline works when model inputs are input_0/input_1."""
        import sys
        _orig_session = sys.modules["onnxruntime"].InferenceSession

        class _GenericNameSession(_orig_session):
            def get_inputs(self):
                return [
                    types.SimpleNamespace(name="input_0", shape=[1, 3, 256, 256]),
                    types.SimpleNamespace(name="input_1", shape=[1, 3, 256, 256]),
                ]

            def run(self, out, feed):
                img = feed.get("input_1", np.zeros((1, 3, 256, 256), dtype=np.float32))
                return [img]

        monkeypatch.setattr(sys.modules["onnxruntime"], "InferenceSession", _GenericNameSession)

        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = types.SimpleNamespace(
            kps=np.array([[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
                         dtype=np.float32),
        )
        portrait_crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        target_face = types.SimpleNamespace(portrait_crop_ffhq=portrait_crop)

        result = swapper.swap(frame, source_face, target_face)
        # Swap must complete (not return the original frame unchanged)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_swap_does_not_raise_missing_inputs_error(self, tmp_path, monkeypatch, caplog):
        """Verifies the original bug is fixed: no 'missing from input feed' error."""
        import logging
        import sys
        _orig_session = sys.modules["onnxruntime"].InferenceSession

        class _GenericNameSession(_orig_session):
            def get_inputs(self):
                return [
                    types.SimpleNamespace(name="input_0", shape=[1, 3, 256, 256]),
                    types.SimpleNamespace(name="input_1", shape=[1, 3, 256, 256]),
                ]

            def run(self, out, feed):
                # Raise the exact error the user reported if wrong names are used
                if "source" in feed or "target" in feed:
                    raise ValueError(
                        "Required inputs (['input_0', 'input_1']) are missing "
                        "from input feed (['source', 'target'])."
                    )
                img = feed.get("input_1", np.zeros((1, 3, 256, 256), dtype=np.float32))
                return [img]

        monkeypatch.setattr(sys.modules["onnxruntime"], "InferenceSession", _GenericNameSession)

        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = types.SimpleNamespace(
            kps=np.array([[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
                         dtype=np.float32),
        )
        portrait_crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        target_face = types.SimpleNamespace(portrait_crop_ffhq=portrait_crop)

        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, source_face, target_face)

        assert result.shape == frame.shape
        assert not any("missing from input feed" in r.message for r in caplog.records), (
            "ONNX inference should not fail with 'missing from input feed'. "
            f"Warnings logged: {[r.message for r in caplog.records]}"
        )



class TestSwapSimswapGuards:
    def _make_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        return FaceSwapper(cfg)

    def test_none_kps_returns_frame_unchanged(self, tmp_path, caplog):
        """Face with kps=None → swap is skipped, original frame returned."""
        import logging
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face_no_kps = types.SimpleNamespace(kps=None)
        portrait = _make_face()
        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, face_no_kps, portrait)
        np.testing.assert_array_equal(result, frame)
        assert any("no keypoints" in r.message for r in caplog.records)

    def test_missing_kps_attr_returns_frame_unchanged(self, tmp_path, caplog):
        """Face object without kps attribute → swap is skipped."""
        import logging
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face_no_attr = types.SimpleNamespace()
        portrait = _make_face()
        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, face_no_attr, portrait)
        np.testing.assert_array_equal(result, frame)

    def test_ransac_failure_returns_frame_unchanged(self, tmp_path, caplog, monkeypatch):
        """RuntimeError from _warp_face is caught; original frame is returned."""
        import logging
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        target_face = _make_face()

        def _always_fail(*a, **kw):
            raise RuntimeError("Face alignment failed: RANSAC returned None")

        monkeypatch.setattr(swapper, "_warp_face", _always_fail)
        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, source_face, target_face)
        np.testing.assert_array_equal(result, frame)
        assert any("alignment failed" in r.message for r in caplog.records)

    def test_none_embedding_returns_frame_unchanged(self, tmp_path, caplog):
        """Portrait face with no embedding → swap skipped, frame unchanged."""
        import logging
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        portrait_no_emb = types.SimpleNamespace(
            kps=np.zeros((5, 2), dtype=np.float32),
            embedding=None,
        )
        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, source_face, portrait_no_emb)
        np.testing.assert_array_equal(result, frame)
        assert any("no embedding" in r.message for r in caplog.records)

    def test_onnx_error_returns_frame_unchanged(self, tmp_path, caplog, monkeypatch):
        """Exception from ort_session.run is caught; original frame is returned."""
        import logging
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        target_face = _make_face()

        def _raise_onnx(*a, **kw):
            raise RuntimeError("ONNX inference error: missing input")

        monkeypatch.setattr(swapper._ort_session, "run", _raise_onnx)
        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, source_face, target_face)
        np.testing.assert_array_equal(result, frame)
        assert any("inference failed" in r.message for r in caplog.records)

    def test_swap_multiple_skips_bad_face_continues_others(self, tmp_path, monkeypatch):
        """If one face in a multi-swap fails, other swaps still complete."""
        swapper = self._make_swapper(tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        good_face = _make_face()
        bad_face = types.SimpleNamespace(kps=None, embedding=np.ones(512, dtype=np.float32))
        portrait = _make_face()

        pairs = [(bad_face, portrait), (good_face, portrait)]
        # Should not raise even with a bad face in the list
        result = swapper.swap_multiple(frame, pairs, frame_idx=42)
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# _detect_model_type — new model families
# ---------------------------------------------------------------------------

class TestDetectModelTypeNewModels:
    def test_uniface_256(self):
        assert _detect_model_type("uniface_256.onnx") == "uniface"

    def test_uniface_case_insensitive(self):
        assert _detect_model_type("/models/Uniface_256.onnx") == "uniface"

    def test_hyperswap_1a(self):
        assert _detect_model_type("hyperswap_1a_256.onnx") == "hyperswap"

    def test_hyperswap_1b(self):
        assert _detect_model_type("hyperswap_1b_256.onnx") == "hyperswap"

    def test_hyperswap_1c(self):
        assert _detect_model_type("hyperswap_1c_256.onnx") == "hyperswap"

    def test_hyperswap_case_insensitive(self):
        assert _detect_model_type("/models/HyperSwap_1a_256.onnx") == "hyperswap"

    def test_blendswap_256(self):
        assert _detect_model_type("blendswap_256.onnx") == "blendswap"

    def test_blendswap_case_insensitive(self):
        assert _detect_model_type("/models/BlendSwap_256.onnx") == "blendswap"

    def test_uniface_takes_priority_over_simswap(self):
        """uniface check precedes simswap in _detect_model_type."""
        assert _detect_model_type("uniface_256.onnx") != "simswap"

    def test_hyperswap_takes_priority_over_inswapper(self):
        assert _detect_model_type("hyperswap_1a_256.onnx") != "inswapper"


# ---------------------------------------------------------------------------
# _get_simswap_params — new model entries
# ---------------------------------------------------------------------------

class TestGetSimswapParamsNewModels:
    def test_uniface_256_params(self):
        params = _get_simswap_params("uniface_256.onnx")
        assert params["size"] == 256
        assert params["template"] == "ffhq_512"
        assert params["source_type"] == "image"
        assert params["source_crop_attr"] == "portrait_crop_ffhq"
        np.testing.assert_allclose(params["mean"], [0.5, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(params["std"], [0.5, 0.5, 0.5], atol=1e-6)

    def test_hyperswap_1a_256_params(self):
        params = _get_simswap_params("hyperswap_1a_256.onnx")
        assert params["size"] == 256
        assert params["template"] == "arcface_128"
        assert params["source_type"] == "embedding_norm"
        np.testing.assert_allclose(params["mean"], [0.5, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(params["std"], [0.5, 0.5, 0.5], atol=1e-6)

    def test_hyperswap_1b_256_params(self):
        params = _get_simswap_params("hyperswap_1b_256.onnx")
        assert params["size"] == 256
        assert params["template"] == "arcface_128"
        assert params["source_type"] == "embedding_norm"

    def test_hyperswap_1c_256_params(self):
        params = _get_simswap_params("hyperswap_1c_256.onnx")
        assert params["size"] == 256
        assert params["template"] == "arcface_128"
        assert params["source_type"] == "embedding_norm"

    def test_blendswap_256_params(self):
        params = _get_simswap_params("blendswap_256.onnx")
        assert params["size"] == 256
        assert params["template"] == "ffhq_512"
        assert params["source_type"] == "image"
        assert params["source_crop_attr"] == "portrait_crop_arcv2"
        np.testing.assert_array_equal(params["mean"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(params["std"], [1.0, 1.0, 1.0])

    def test_simswap_params_have_template_and_source_type(self):
        """Existing simswap params include the new template/source_type fields."""
        params = _get_simswap_params("simswap_256.onnx")
        assert params.get("template") == "arcface_112_v1"
        assert params.get("source_type") == "embedding"

    def test_simswap_unofficial_512_params_have_template(self):
        params = _get_simswap_params("simswap_unofficial_512.onnx")
        assert params.get("template") == "arcface_112_v1"
        assert params.get("source_type") == "embedding"


# ---------------------------------------------------------------------------
# FaceSwapper loading — new ONNX model types
# ---------------------------------------------------------------------------

class TestFaceSwapperNewModels:
    def test_loads_uniface_256(self, tmp_path):
        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "uniface"
        assert swapper._model is None
        assert swapper._ort_session is not None
        assert swapper._simswap_params["size"] == 256
        assert swapper._simswap_params["template"] == "ffhq_512"

    def test_loads_hyperswap_1a_256(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "hyperswap"
        assert swapper._model is None
        assert swapper._ort_session is not None
        assert swapper._simswap_params["size"] == 256
        assert swapper._simswap_params["template"] == "arcface_128"

    def test_loads_hyperswap_1b_256(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1b_256.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "hyperswap"

    def test_loads_hyperswap_1c_256(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1c_256.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "hyperswap"

    def test_loads_blendswap_256(self, tmp_path):
        cfg = _make_cfg(tmp_path, "blendswap_256.onnx")
        swapper = FaceSwapper(cfg)
        assert swapper._model_type == "blendswap"
        assert swapper._model is None
        assert swapper._ort_session is not None
        assert swapper._simswap_params["size"] == 256
        assert swapper._simswap_params["template"] == "ffhq_512"


# ---------------------------------------------------------------------------
# _warp_face — template selection
# ---------------------------------------------------------------------------

class TestWarpFaceTemplates:
    def _make_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "simswap_unofficial_512.onnx")
        return FaceSwapper(cfg)

    def _valid_kps(self):
        return np.array(
            [[200, 250], [350, 250], [275, 350], [210, 450], [340, 450]],
            dtype=np.float32,
        )

    def test_arcface_128_template_returns_correct_size(self, tmp_path):
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        crop, M = swapper._warp_face(frame, self._valid_kps(), size=256,
                                     template_name="arcface_128")
        assert crop.shape == (256, 256, 3)
        assert M.shape == (2, 3)

    def test_ffhq_512_template_returns_correct_size(self, tmp_path):
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        crop, M = swapper._warp_face(frame, self._valid_kps(), size=256,
                                     template_name="ffhq_512")
        assert crop.shape == (256, 256, 3)
        assert M.shape == (2, 3)

    def test_unknown_template_falls_back_to_arcface_112_v1(self, tmp_path):
        """Unknown template name silently falls back to arcface_112_v1."""
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        crop, M = swapper._warp_face(frame, self._valid_kps(), size=256,
                                     template_name="nonexistent_template")
        assert crop.shape == (256, 256, 3)

    def test_default_template_matches_arcface_112_v1_behaviour(self, tmp_path):
        """Calling _warp_face without template_name gives the same result as
        explicitly passing 'arcface_112_v1'."""
        swapper = self._make_swapper(tmp_path)
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        kps = self._valid_kps()
        crop_default, M_default = swapper._warp_face(frame, kps, size=256)
        crop_explicit, M_explicit = swapper._warp_face(
            frame, kps, size=256, template_name="arcface_112_v1"
        )
        np.testing.assert_array_equal(crop_default, crop_explicit)
        np.testing.assert_array_almost_equal(M_default, M_explicit)


# ---------------------------------------------------------------------------
# _normalize_crop_frame — de-normalisation for hyperswap / uniface
# ---------------------------------------------------------------------------

class TestNormalizeCropFrameNewModels:
    def test_hyperswap_denormalizes_output(self, tmp_path):
        """For hyperswap (mean=0.5, std=0.5) the output is de-normalised."""
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)
        # A tensor of all zeros: after de-normalisation → 0*0.5+0.5=0.5 → 127
        output = np.zeros((3, 256, 256), dtype=np.float32)
        result = swapper._normalize_crop_frame(output)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8
        # 0 * 0.5 + 0.5 = 0.5 → clipped → * 255 = 127
        assert result.min() == 127
        assert result.max() == 127

    def test_uniface_denormalizes_output(self, tmp_path):
        """For uniface (mean=0.5, std=0.5) the output is de-normalised."""
        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        swapper = FaceSwapper(cfg)
        output = np.zeros((3, 256, 256), dtype=np.float32)
        result = swapper._normalize_crop_frame(output)
        assert result.min() == 127
        assert result.max() == 127

    def test_blendswap_does_not_denormalize(self, tmp_path):
        """For blendswap (mean=0, std=1) no de-normalisation is applied."""
        cfg = _make_cfg(tmp_path, "blendswap_256.onnx")
        swapper = FaceSwapper(cfg)
        # Values of 2.0 should be clipped to 1.0 → 255
        output = np.full((3, 256, 256), 2.0, dtype=np.float32)
        result = swapper._normalize_crop_frame(output)
        assert result.max() == 255

    def test_hyperswap_denorm_then_clip(self, tmp_path):
        """Values that after de-normalisation exceed [0,1] are clipped."""
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)
        # 10.0 * 0.5 + 0.5 = 5.5, clipped to 1.0 → 255
        output = np.full((3, 64, 64), 10.0, dtype=np.float32)
        result = swapper._normalize_crop_frame(output)
        assert result.max() == 255
        # -10.0 * 0.5 + 0.5 = -4.5, clipped to 0.0 → 0
        output_neg = np.full((3, 64, 64), -10.0, dtype=np.float32)
        result_neg = swapper._normalize_crop_frame(output_neg)
        assert result_neg.max() == 0


# ---------------------------------------------------------------------------
# _prepare_embedding_norm
# ---------------------------------------------------------------------------

class TestPrepareEmbeddingNorm:
    def _make_hyperswap_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        return FaceSwapper(cfg)

    def test_uses_normed_embedding_when_present(self, tmp_path):
        swapper = self._make_hyperswap_swapper(tmp_path)
        normed = np.random.randn(512).astype(np.float32)
        normed /= np.linalg.norm(normed)
        face = types.SimpleNamespace(normed_embedding=normed)
        result = swapper._prepare_embedding_norm(face)
        assert result is not None
        assert result.shape == (1, 512)
        np.testing.assert_allclose(result.ravel(), normed, atol=1e-6)

    def test_falls_back_to_embedding_when_normed_missing(self, tmp_path):
        swapper = self._make_hyperswap_swapper(tmp_path)
        raw = np.random.randn(512).astype(np.float32)
        face = types.SimpleNamespace(embedding=raw)
        result = swapper._prepare_embedding_norm(face)
        assert result is not None
        assert result.shape == (1, 512)
        # Manually normalised
        expected = raw / np.linalg.norm(raw)
        np.testing.assert_allclose(result.ravel(), expected, atol=1e-5)

    def test_zero_embedding_no_nan(self, tmp_path):
        swapper = self._make_hyperswap_swapper(tmp_path)
        face = types.SimpleNamespace(embedding=np.zeros(512, dtype=np.float32))
        result = swapper._prepare_embedding_norm(face)
        assert result is not None
        assert not np.any(np.isnan(result))

    def test_no_embedding_returns_none(self, tmp_path, caplog):
        import logging
        swapper = self._make_hyperswap_swapper(tmp_path)
        face = types.SimpleNamespace()
        with caplog.at_level(logging.WARNING):
            result = swapper._prepare_embedding_norm(face)
        assert result is None
        assert any("no embedding" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# _prepare_source_frame
# ---------------------------------------------------------------------------

class TestPrepareSourceFrame:
    def _make_uniface_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        return FaceSwapper(cfg)

    def _make_blendswap_swapper(self, tmp_path):
        cfg = _make_cfg(tmp_path, "blendswap_256.onnx")
        return FaceSwapper(cfg)

    def test_uniface_uses_portrait_crop_ffhq(self, tmp_path):
        swapper = self._make_uniface_swapper(tmp_path)
        crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        face = types.SimpleNamespace(portrait_crop_ffhq=crop)
        tensor = swapper._prepare_source_frame(face)
        assert tensor is not None
        assert tensor.shape == (1, 3, 256, 256)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_uniface_missing_crop_returns_none(self, tmp_path, caplog):
        import logging
        swapper = self._make_uniface_swapper(tmp_path)
        face = types.SimpleNamespace()  # no portrait_crop_ffhq
        with caplog.at_level(logging.WARNING):
            result = swapper._prepare_source_frame(face)
        assert result is None
        assert any("portrait" in r.message.lower() or "source crop" in r.message.lower()
                   for r in caplog.records)

    def test_blendswap_uses_portrait_crop_arcv2(self, tmp_path):
        swapper = self._make_blendswap_swapper(tmp_path)
        crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        face = types.SimpleNamespace(portrait_crop_arcv2=crop)
        tensor = swapper._prepare_source_frame(face)
        assert tensor is not None
        assert tensor.shape == (1, 3, 112, 112)
        assert tensor.dtype == np.float32

    def test_blendswap_falls_back_to_arcface_crop(self, tmp_path):
        """blendswap falls back to arcface_crop when portrait_crop_arcv2 is absent."""
        swapper = self._make_blendswap_swapper(tmp_path)
        crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        face = types.SimpleNamespace(arcface_crop=crop)  # no portrait_crop_arcv2
        tensor = swapper._prepare_source_frame(face)
        assert tensor is not None
        assert tensor.shape == (1, 3, 112, 112)

    def test_blendswap_missing_all_crops_returns_none(self, tmp_path, caplog):
        import logging
        swapper = self._make_blendswap_swapper(tmp_path)
        face = types.SimpleNamespace()
        with caplog.at_level(logging.WARNING):
            result = swapper._prepare_source_frame(face)
        assert result is None

    def test_bgr_to_rgb_conversion(self, tmp_path):
        """Source tensor is converted from BGR to RGB."""
        swapper = self._make_uniface_swapper(tmp_path)
        # Pure blue in BGR = (255, 0, 0) → after BGR→RGB becomes (0, 0, 1.0) in CHW
        crop = np.zeros((256, 256, 3), dtype=np.uint8)
        crop[:, :, 0] = 255  # blue channel in BGR
        face = types.SimpleNamespace(portrait_crop_ffhq=crop)
        tensor = swapper._prepare_source_frame(face)
        # Channel 0 (R) should be near 0, channel 2 (B) should be near 1
        assert tensor is not None
        assert tensor[0, 0].max() < 0.01   # R channel ≈ 0
        assert tensor[0, 2].min() > 0.99   # B channel ≈ 1


# ---------------------------------------------------------------------------
# End-to-end swap tests for new model families
# ---------------------------------------------------------------------------

class TestSwapNewModels:
    def test_hyperswap_swap_returns_same_shape(self, tmp_path):
        cfg = _make_cfg(tmp_path, "hyperswap_1a_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        # Portrait face: no normed_embedding → falls back to raw embedding
        target_face = _make_face()
        result = swapper.swap(frame, source_face, target_face)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_hyperswap_uses_normed_embedding(self, tmp_path):
        """When normed_embedding is on the portrait face, it is used directly."""
        cfg = _make_cfg(tmp_path, "hyperswap_1b_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        normed = np.random.randn(512).astype(np.float32)
        normed /= np.linalg.norm(normed)
        target_face = types.SimpleNamespace(
            kps=np.array([[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
                         dtype=np.float32),
            normed_embedding=normed,
        )
        result = swapper.swap(frame, source_face, target_face)
        assert result.shape == frame.shape

    def test_uniface_swap_with_portrait_crop(self, tmp_path):
        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        target_face = types.SimpleNamespace(
            kps=np.array([[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
                         dtype=np.float32),
            portrait_crop_ffhq=crop,
        )
        result = swapper.swap(frame, source_face, target_face)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_uniface_swap_skipped_when_no_crop(self, tmp_path, caplog):
        """Swap is skipped (frame unchanged) when portrait_crop_ffhq is absent."""
        import logging
        cfg = _make_cfg(tmp_path, "uniface_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        target_face = types.SimpleNamespace(
            kps=np.array([[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
                         dtype=np.float32),
        )
        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, source_face, target_face)
        np.testing.assert_array_equal(result, frame)

    def test_blendswap_swap_with_arcface_crop(self, tmp_path):
        cfg = _make_cfg(tmp_path, "blendswap_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        target_face = types.SimpleNamespace(
            kps=np.array([[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
                         dtype=np.float32),
            arcface_crop=crop,  # fallback attribute
        )
        result = swapper.swap(frame, source_face, target_face)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_hyperswap_no_embedding_skips(self, tmp_path, caplog):
        """Portrait face with no embedding skips the swap."""
        import logging
        cfg = _make_cfg(tmp_path, "hyperswap_1c_256.onnx")
        swapper = FaceSwapper(cfg)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_face = _make_face()
        target_face = types.SimpleNamespace(
            kps=np.array([[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
                         dtype=np.float32),
        )  # no embedding or normed_embedding
        with caplog.at_level(logging.WARNING):
            result = swapper.swap(frame, source_face, target_face)
        np.testing.assert_array_equal(result, frame)
        assert any("no embedding" in r.message.lower() for r in caplog.records)

    def test_all_new_models_swap_multiple(self, tmp_path):
        """swap_multiple works for all new model types."""
        for model_name in [
            "hyperswap_1a_256.onnx",
            "hyperswap_1b_256.onnx",
            "hyperswap_1c_256.onnx",
        ]:
            cfg = _make_cfg(tmp_path, model_name)
            swapper = FaceSwapper(cfg)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            pairs = [(_make_face(), _make_face()), (_make_face(), _make_face())]
            result = swapper.swap_multiple(frame, pairs, frame_idx=0)
            assert result.shape == frame.shape, f"Failed for {model_name}"
