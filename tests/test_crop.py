"""
tests/test_crop.py

Unit tests for portrait_prep.crop
"""

from __future__ import annotations

import numpy as np
import pytest

from portrait_prep.crop import crop_folder
from portrait_prep.cpcap import infer_original_stem


# ---------------------------------------------------------------------------
# infer_original_stem (shared logic lives in cpcap but crop uses same rules)
# ---------------------------------------------------------------------------

class TestInferOriginalStem:
    def test_aug_suffix_stripped(self):
        assert infer_original_stem("photo_aug000") == "photo"

    def test_orig_suffix_stripped(self):
        assert infer_original_stem("photo_orig") == "photo"

    def test_plain_stem_unchanged(self):
        assert infer_original_stem("photo") == "photo"

    def test_double_digit_aug(self):
        assert infer_original_stem("face01_aug99") == "face01"

    def test_underscore_in_original_name(self):
        assert infer_original_stem("my_photo_aug001") == "my_photo"


# ---------------------------------------------------------------------------
# crop_folder – mocked face_recognition
# ---------------------------------------------------------------------------

class TestCropFolder:
    def test_empty_directory(self, tmp_path):
        result = _run_crop_folder_mocked(tmp_path / "in", tmp_path / "out", faces=[])
        assert result["faces"] == 0
        assert result["images_processed"] == 0

    def test_no_face_found(self, tmp_path):
        _make_png(tmp_path / "in" / "photo.png")
        result = _run_crop_folder_mocked(tmp_path / "in", tmp_path / "out", faces=[])
        assert result["faces"] == 0

    def test_one_face_found_no_classify(self, tmp_path):
        _make_png(tmp_path / "in" / "photo.png")
        result = _run_crop_folder_mocked(
            tmp_path / "in",
            tmp_path / "out",
            faces=[((10, 90, 90, 10),)],
            classify=False,
        )
        assert result["faces"] == 1

    def test_classify_creates_person_dirs(self, tmp_path):
        _make_png(tmp_path / "in" / "photoA.png")
        _make_png(tmp_path / "in" / "photoB.png")

        # Two images with one face each; same encoding → same person
        enc = np.zeros(128, dtype=np.float64)
        result = _run_crop_folder_mocked(
            tmp_path / "in",
            tmp_path / "out",
            faces=[((10, 90, 90, 10),), ((10, 90, 90, 10),)],
            encodings=[[enc], [enc]],
            classify=True,
        )
        # Both faces should be assigned to the same person
        assert result["persons"] == 1

    def test_two_different_persons(self, tmp_path):
        _make_png(tmp_path / "in" / "photoA.png")
        _make_png(tmp_path / "in" / "photoB.png")

        enc_a = np.zeros(128, dtype=np.float64)
        enc_b = np.ones(128, dtype=np.float64)  # very different
        result = _run_crop_folder_mocked(
            tmp_path / "in",
            tmp_path / "out",
            faces=[((10, 90, 90, 10),), ((10, 90, 90, 10),)],
            encodings=[[enc_a], [enc_b]],
            classify=True,
        )
        assert result["persons"] == 2


# ---------------------------------------------------------------------------
# Internal helpers for mocked tests
# ---------------------------------------------------------------------------


def _make_png(path, size=(100, 100)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(path)


def _run_crop_folder_mocked(
    input_dir,
    output_dir,
    faces: list,
    encodings: list | None = None,
    classify: bool = False,
):
    """Run crop_folder with face_recognition fully mocked out."""
    from unittest.mock import MagicMock, patch
    import numpy as np

    # Build per-image face data
    if encodings is None:
        encodings = [
            [np.zeros(128, dtype=np.float64)] * len(f)
            for f in faces
        ]

    call_counter = {"n": 0}

    def mock_face_locations(image, model="hog"):
        idx = call_counter["n"]
        call_counter["n"] += 1
        if idx < len(faces):
            return list(faces[idx])
        return []

    def mock_face_encodings(image, locations):
        idx = call_counter["n"] - 1  # already incremented in face_locations
        if idx < len(encodings):
            return list(encodings[idx])
        return []

    def mock_load_image_file(path):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def mock_face_distance(known, unknown):
        dists = [np.linalg.norm(k - unknown) for k in known]
        return np.array(dists, dtype=np.float64)

    fr_mock = MagicMock()
    fr_mock.load_image_file.side_effect = mock_load_image_file
    fr_mock.face_locations.side_effect = mock_face_locations
    fr_mock.face_encodings.side_effect = mock_face_encodings
    fr_mock.face_distance.side_effect = mock_face_distance

    with patch("portrait_prep.crop._load_face_recognition", return_value=fr_mock):
        return crop_folder(input_dir, output_dir, classify=classify)
