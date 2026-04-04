"""
tests/test_ref.py

Unit tests for vicrop.ref – reference-photo quality scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from vicrop.ref import (
    DEFAULT_REF_THRESH,
    _eye_aspect_ratio,
    _face_fill_score,
    _frontality_score,
    _lighting_score,
    _sharpness_score,
    _single_face_score,
    collect_ref_photos,
    score_reference_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_array(h: int = 100, w: int = 100, value: int = 128) -> np.ndarray:
    """Uniform RGB image."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _frontal_landmarks() -> dict[str, list[tuple[int, int]]]:
    """Synthetic landmarks resembling a frontal face."""
    return {
        "left_eye": [(30, 40), (33, 37), (37, 37), (40, 40), (37, 43), (33, 43)],
        "right_eye": [(60, 40), (63, 37), (67, 37), (70, 40), (67, 43), (63, 43)],
        "nose_tip": [(46, 58), (48, 60), (50, 62), (52, 60), (54, 58)],
        "chin": [
            (20, 45), (22, 55), (25, 65), (30, 73), (35, 78),
            (40, 82), (45, 84), (50, 85), (55, 84),
            (60, 82), (65, 78), (70, 73), (75, 65), (78, 55),
            (80, 45), (80, 38), (80, 30),
        ],
    }


def _profile_landmarks() -> dict[str, list[tuple[int, int]]]:
    """Synthetic landmarks resembling a profile (turned far right)."""
    lm = _frontal_landmarks()
    # Shift left eye way to the right → nose much closer to right eye
    lm["left_eye"] = [(65, 40), (67, 37), (69, 37), (71, 40), (69, 43), (67, 43)]
    return lm


# ---------------------------------------------------------------------------
# _eye_aspect_ratio
# ---------------------------------------------------------------------------


class TestEyeAspectRatio:
    def test_open_eye(self):
        # Wide open: large vertical, normal horizontal
        pts = [(0, 10), (3, 5), (7, 5), (10, 10), (7, 15), (3, 15)]
        ear = _eye_aspect_ratio(pts)
        assert ear > 0.3

    def test_closed_eye(self):
        # Closed: negligible vertical
        pts = [(0, 10), (3, 10), (7, 10), (10, 10), (7, 10), (3, 10)]
        ear = _eye_aspect_ratio(pts)
        assert ear == pytest.approx(0.0, abs=0.01)

    def test_too_few_points_returns_zero(self):
        assert _eye_aspect_ratio([(0, 0), (1, 1)]) == 0.0

    def test_zero_horizontal_returns_zero(self):
        pts = [(5, 5)] * 6
        assert _eye_aspect_ratio(pts) == 0.0


# ---------------------------------------------------------------------------
# _frontality_score
# ---------------------------------------------------------------------------


class TestFrontalityScore:
    def test_frontal_face_scores_high(self):
        score = _frontality_score(_frontal_landmarks())
        assert score > 0.6

    def test_profile_face_scores_lower(self):
        frontal = _frontality_score(_frontal_landmarks())
        profile = _frontality_score(_profile_landmarks())
        assert profile < frontal

    def test_empty_landmarks_returns_zero(self):
        assert _frontality_score({}) == 0.0

    def test_missing_chin_returns_zero(self):
        lm = _frontal_landmarks()
        del lm["chin"]
        assert _frontality_score(lm) == 0.0


# ---------------------------------------------------------------------------
# _sharpness_score
# ---------------------------------------------------------------------------


class TestSharpnessScore:
    def test_uniform_image_is_blurry(self):
        img = _make_rgb_array(value=128)
        assert _sharpness_score(img) == pytest.approx(0.0, abs=0.01)

    def test_noisy_image_scores_higher(self):
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert _sharpness_score(img) > 0.3


# ---------------------------------------------------------------------------
# _lighting_score
# ---------------------------------------------------------------------------


class TestLightingScore:
    def test_very_dark_image(self):
        img = _make_rgb_array(value=10)
        assert _lighting_score(img) < 0.4

    def test_very_bright_image(self):
        img = _make_rgb_array(value=250)
        assert _lighting_score(img) < 0.4

    def test_well_lit_image(self):
        # Mid-range luminance + some contrast
        rng = np.random.RandomState(0)
        img = (rng.normal(130, 40, (100, 100, 3)).clip(0, 255)).astype(np.uint8)
        assert _lighting_score(img) > 0.5


# ---------------------------------------------------------------------------
# _face_fill_score
# ---------------------------------------------------------------------------


class TestFaceFillScore:
    def test_large_face(self):
        # Face occupies 40% of frame → should cap at 1.0
        score = _face_fill_score((10, 70, 70, 10), 100, 100)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_tiny_face(self):
        # Tiny face occupies ~1%
        score = _face_fill_score((45, 55, 55, 45), 100, 100)
        assert score < 0.5

    def test_zero_frame(self):
        assert _face_fill_score((0, 1, 1, 0), 0, 0) == 0.0


# ---------------------------------------------------------------------------
# score_reference_quality
# ---------------------------------------------------------------------------


class TestScoreReferenceQuality:
    def test_without_landmarks(self):
        frame = _make_rgb_array(100, 100, 128)
        face_region = _make_rgb_array(50, 50, 128)
        score = score_reference_quality(
            frame, (10, 60, 60, 10), None, face_region,
        )
        assert 0.0 <= score <= 1.0

    def test_with_landmarks_returns_bounded(self):
        frame = _make_rgb_array(100, 100, 128)
        face_region = _make_rgb_array(50, 50, 128)
        score = score_reference_quality(
            frame, (10, 60, 60, 10), _frontal_landmarks(), face_region,
        )
        assert 0.0 <= score <= 1.0

    def test_frontal_beats_profile(self):
        frame = _make_rgb_array(100, 100, 128)
        face_region = _make_rgb_array(50, 50, 128)
        frontal = score_reference_quality(
            frame, (10, 60, 60, 10), _frontal_landmarks(), face_region,
        )
        profile = score_reference_quality(
            frame, (10, 60, 60, 10), _profile_landmarks(), face_region,
        )
        assert frontal > profile

    def test_multiple_faces_returns_zero(self):
        frame = _make_rgb_array(100, 100, 128)
        face_region = _make_rgb_array(50, 50, 128)
        score = score_reference_quality(
            frame, (10, 60, 60, 10), _frontal_landmarks(), face_region,
            face_count=2,
        )
        assert score == 0.0

    def test_single_face_explicit_returns_nonzero(self):
        frame = _make_rgb_array(100, 100, 128)
        face_region = _make_rgb_array(50, 50, 128)
        score = score_reference_quality(
            frame, (10, 60, 60, 10), _frontal_landmarks(), face_region,
            face_count=1,
        )
        assert score > 0.0

    def test_zero_faces_returns_zero(self):
        frame = _make_rgb_array(100, 100, 128)
        face_region = _make_rgb_array(50, 50, 128)
        score = score_reference_quality(
            frame, (10, 60, 60, 10), None, face_region,
            face_count=0,
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# _single_face_score
# ---------------------------------------------------------------------------


class TestSingleFaceScore:
    def test_one_face_returns_one(self):
        assert _single_face_score(1) == 1.0

    def test_two_faces_returns_zero(self):
        assert _single_face_score(2) == 0.0

    def test_zero_faces_returns_zero(self):
        assert _single_face_score(0) == 0.0

    def test_large_count_returns_zero(self):
        assert _single_face_score(10) == 0.0


# ---------------------------------------------------------------------------
# collect_ref_photos
# ---------------------------------------------------------------------------


class TestCollectRefPhotos:
    def test_moves_files_into_ref_subfolder(self, tmp_path):
        paths = [tmp_path / "a.png", tmp_path / "c.png", tmp_path / "b.png"]
        for p in paths:
            p.write_bytes(b"fake")

        result = collect_ref_photos(tmp_path, paths)

        assert result == tmp_path / "ref"
        assert result.is_dir()
        moved = sorted(f.name for f in result.iterdir())
        assert moved == ["a.png", "b.png", "c.png"]
        # originals should be gone
        for p in paths:
            assert not p.exists()

    def test_empty_list_returns_none(self, tmp_path):
        assert collect_ref_photos(tmp_path, []) is None
        assert not (tmp_path / "ref").exists()


# ---------------------------------------------------------------------------
# DEFAULT_REF_THRESH
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_thresh_value(self):
        assert DEFAULT_REF_THRESH == 0.8
