"""
tests/test_crop.py

Unit tests for portrait_prep.crop
"""

from __future__ import annotations

import numpy as np
import pytest

from portrait_prep.crop import crop_folder
from portrait_prep.face_utils import cluster_faces, load_reference_encodings
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


# ---------------------------------------------------------------------------
# load_reference_encodings
# ---------------------------------------------------------------------------


class TestLoadReferenceEncodings:
    def test_loads_encodings_from_identity_dirs(self, tmp_path):
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "classified"
        (ref_dir / "alice").mkdir(parents=True)
        _make_png(ref_dir / "alice" / "ref1.png")
        _make_png(ref_dir / "alice" / "ref2.png")
        (ref_dir / "bob").mkdir(parents=True)
        _make_png(ref_dir / "bob" / "ref3.png")

        enc_a = np.zeros(128, dtype=np.float64)
        enc_b = np.ones(128, dtype=np.float64)
        call_counter = {"n": 0}

        def mock_face_locations(image, model="hog"):
            return [(10, 90, 90, 10)]

        def mock_face_encodings(image, locations):
            idx = call_counter["n"]
            call_counter["n"] += 1
            return [enc_a] if idx < 2 else [enc_b]

        fr_mock = MagicMock()
        fr_mock.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        fr_mock.face_locations.side_effect = mock_face_locations
        fr_mock.face_encodings.side_effect = mock_face_encodings

        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            encodings, names = load_reference_encodings(ref_dir)

        assert len(encodings) == 3
        assert names.count("alice") == 2
        assert names.count("bob") == 1

    def test_empty_classified_dir(self, tmp_path):
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "classified"
        ref_dir.mkdir()

        fr_mock = MagicMock()
        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            encodings, names = load_reference_encodings(ref_dir)

        assert len(encodings) == 0
        assert len(names) == 0

    def test_no_face_in_reference_image(self, tmp_path):
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "classified"
        (ref_dir / "alice").mkdir(parents=True)
        _make_png(ref_dir / "alice" / "noface.png")

        fr_mock = MagicMock()
        fr_mock.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        fr_mock.face_locations.return_value = []
        fr_mock.face_encodings.return_value = []

        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            encodings, _ = load_reference_encodings(ref_dir)

        assert len(encodings) == 0

    def test_max_per_identity_limits_loaded_encodings(self, tmp_path):
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "classified"
        (ref_dir / "alice").mkdir(parents=True)
        _make_png(ref_dir / "alice" / "ref1.png")
        _make_png(ref_dir / "alice" / "ref2.png")
        _make_png(ref_dir / "alice" / "ref3.png")

        fr_mock = MagicMock()
        fr_mock.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        fr_mock.face_locations.return_value = [(10, 90, 90, 10)]
        fr_mock.face_encodings.return_value = [np.zeros(128)]

        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            encodings, names = load_reference_encodings(ref_dir, max_per_identity=2)

        assert len(encodings) == 2
        assert names.count("alice") == 2


# ---------------------------------------------------------------------------
# cluster_faces with reference encodings
# ---------------------------------------------------------------------------


class TestClusterFacesWithReferences:
    def test_new_face_matches_reference_uses_original_name(self, tmp_path):
        """A new face close to a reference encoding uses the reference folder name."""
        from unittest.mock import MagicMock, patch

        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc_new = np.zeros(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)  # identical → distance 0

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([0.0])

        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            result = cluster_faces(
                [(staging / "face1.png", enc_new)],
                tmp_path,
                tolerance=0.6,
                reference_encodings=[ref_enc],
                reference_names=["alice"],
            )

        assert "alice" in result
        assert (tmp_path / "alice" / "face1.png").exists()

    def test_unknown_face_gets_person_nn_label(self, tmp_path):
        """A face far from all references creates a new person_NN folder."""
        from unittest.mock import MagicMock, patch

        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc_new = np.ones(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([1.5])

        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            result = cluster_faces(
                [(staging / "face1.png", enc_new)],
                tmp_path,
                tolerance=0.6,
                reference_encodings=[ref_enc],
                reference_names=["alice"],
            )

        assert "person_01" in result
        assert (tmp_path / "person_01" / "face1.png").exists()
        assert "alice" not in result

    def test_person_nn_numbering_avoids_reference_collision(self, tmp_path):
        """Auto-generated labels start after any person_NN in the references."""
        from unittest.mock import MagicMock, patch

        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc_new = np.ones(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([1.5])

        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            result = cluster_faces(
                [(staging / "face1.png", enc_new)],
                tmp_path,
                tolerance=0.6,
                reference_encodings=[ref_enc],
                reference_names=["person_05"],
            )

        # Should skip person_01..05 and start at person_06
        assert "person_06" in result
        assert (tmp_path / "person_06" / "face1.png").exists()

    def test_no_folder_created_for_unmatched_reference(self, tmp_path):
        """Reference identity folder is NOT created if no new photos match it."""
        from unittest.mock import MagicMock, patch

        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc_new = np.ones(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([1.5])

        with patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            cluster_faces(
                [(staging / "face1.png", enc_new)],
                tmp_path,
                tolerance=0.6,
                reference_encodings=[ref_enc],
                reference_names=["alice"],
            )

        # alice had no match → no alice/ folder
        assert not (tmp_path / "alice").exists()


# ---------------------------------------------------------------------------
# crop_folder with classified_path
# ---------------------------------------------------------------------------


class TestCropFolderWithClassifiedPath:
    def test_classified_path_seeds_clustering(self, tmp_path):
        from unittest.mock import MagicMock, patch

        ref_dir = tmp_path / "classified" / "alice"
        ref_dir.mkdir(parents=True)
        _make_png(ref_dir / "ref.png")

        in_dir = tmp_path / "in"
        _make_png(in_dir / "photo.png")

        enc_ref = np.zeros(128, dtype=np.float64)
        enc_new = np.zeros(128, dtype=np.float64)

        loc_calls = {"n": 0}

        def mock_face_locations(image, model="hog"):
            loc_calls["n"] += 1
            return [(10, 90, 90, 10)]

        def mock_face_encodings(image, locations):
            idx = loc_calls["n"] - 1
            return [enc_ref] if idx == 0 else [enc_new]

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

        out_dir = tmp_path / "out"
        with patch("portrait_prep.crop._load_face_recognition", return_value=fr_mock), \
             patch("portrait_prep.face_utils.load_face_recognition", return_value=fr_mock):
            result = crop_folder(
                in_dir, out_dir, classify=True,
                classified_path=tmp_path / "classified",
            )

        assert result["persons"] == 1
        assert (out_dir / "alice").exists()
