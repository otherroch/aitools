"""
tests/test_caption.py

Unit tests for portrait_prep.caption
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from portrait_prep.caption import (
    _load_labels,
    _preprocess,
    caption_image,
    _caption_folder_impl,
    DEFAULT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tags_csv(path: Path, rows: list[dict]) -> Path:
    """Write a minimal WD14-style tags CSV."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag_id", "name", "category", "count"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def make_image(path: Path, size: tuple = (64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(path)
    return path


# ---------------------------------------------------------------------------
# _load_labels
# ---------------------------------------------------------------------------


class TestLoadLabels:
    def test_parses_general_and_rating_categories(self, tmp_path):
        csv_path = make_tags_csv(
            tmp_path / "tags.csv",
            [
                {"tag_id": 0, "name": "1girl", "category": 0, "count": 1000},
                {"tag_id": 1, "name": "safe", "category": 9, "count": 500},
                {"tag_id": 2, "name": "blue_eyes", "category": 0, "count": 800},
            ],
        )
        tag_names, rating_indices, general_indices = _load_labels(csv_path)

        assert tag_names == ["1girl", "safe", "blue_eyes"]
        assert rating_indices == [1]
        assert general_indices == [0, 2]

    def test_empty_csv_returns_empty_lists(self, tmp_path):
        csv_path = make_tags_csv(tmp_path / "tags.csv", [])
        tag_names, rating_indices, general_indices = _load_labels(csv_path)

        assert tag_names == []
        assert rating_indices == []
        assert general_indices == []


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_shape(self, tmp_path):
        img_path = make_image(tmp_path / "img.png", size=(200, 150))
        arr = _preprocess(img_path, size=448)

        assert arr.shape == (1, 448, 448, 3)
        assert arr.dtype == np.float32

    def test_rgba_image_handled(self, tmp_path):
        path = tmp_path / "rgba.png"
        Image.new("RGBA", (100, 100), (255, 0, 0, 128)).save(path)
        arr = _preprocess(path, size=32)

        assert arr.shape == (1, 32, 32, 3)


# ---------------------------------------------------------------------------
# caption_image
# ---------------------------------------------------------------------------


class TestCaptionImage:
    def _make_session(self, probs: list[float]):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.get_inputs.return_value[0].name = "input"
        session.run.return_value = [np.array([probs], dtype=np.float32)]
        return session

    def test_tags_above_threshold_included(self, tmp_path):
        img = make_image(tmp_path / "img.png")
        probs = [0.9, 0.1, 0.8]  # indices 0 and 2 above default threshold
        session = self._make_session(probs)

        result = caption_image(
            img,
            session,
            tag_names=["1girl", "safe", "blue_eyes"],
            general_indices=[0, 2],
            threshold=DEFAULT_THRESHOLD,
        )

        assert "1girl" in result
        assert "blue eyes" in result  # underscores replaced
        assert "safe" not in result

    def test_prefix_prepended(self, tmp_path):
        img = make_image(tmp_path / "img.png")
        session = self._make_session([0.9])

        result = caption_image(
            img,
            session,
            tag_names=["1girl"],
            general_indices=[0],
            threshold=0.5,
            prefix="ohwx man",
        )

        assert result.startswith("ohwx man")

    def test_empty_prefix_no_leading_comma(self, tmp_path):
        img = make_image(tmp_path / "img.png")
        session = self._make_session([0.9])

        result = caption_image(
            img, session, tag_names=["1girl"], general_indices=[0], threshold=0.5, prefix=""
        )

        assert not result.startswith(",")

    def test_no_tags_above_threshold_returns_prefix_only(self, tmp_path):
        img = make_image(tmp_path / "img.png")
        session = self._make_session([0.1])

        result = caption_image(
            img,
            session,
            tag_names=["1girl"],
            general_indices=[0],
            threshold=0.9,
            prefix="myperson",
        )

        assert result == "myperson"

    def test_include_ratings(self, tmp_path):
        img = make_image(tmp_path / "img.png")
        session = self._make_session([0.1, 0.95])

        result = caption_image(
            img,
            session,
            tag_names=["1girl", "safe"],
            general_indices=[0],
            rating_indices=[1],
            threshold=0.5,
            include_ratings=True,
        )

        assert "safe" in result


# ---------------------------------------------------------------------------
# caption_folder / _caption_folder_impl
# ---------------------------------------------------------------------------


def _make_session_mock(probs: list[float]):
    session_mock = MagicMock()
    session_mock.get_inputs.return_value = [MagicMock()]
    session_mock.get_inputs.return_value[0].name = "input"
    session_mock.run.return_value = [np.array([probs], dtype=np.float32)]
    return session_mock


def _make_ort_mock(session_mock):
    ort_mock = MagicMock()
    ort_mock.InferenceSession.return_value = session_mock
    return ort_mock


class TestCaptionFolder:
    def _run(self, tmp_path, src, probs=(0.9, 0.1), **kwargs):
        tags_csv = make_tags_csv(
            tmp_path / "tags.csv",
            [
                {"tag_id": 0, "name": "1girl", "category": 0, "count": 1000},
                {"tag_id": 1, "name": "safe", "category": 9, "count": 500},
            ],
        )
        dummy_model = tmp_path / "model.onnx"
        dummy_model.write_bytes(b"fake")
        session_mock = _make_session_mock(list(probs))
        ort_mock = _make_ort_mock(session_mock)

        with patch("portrait_prep.caption._download_model", return_value=(dummy_model, tags_csv)):
            return _caption_folder_impl(
                ort=ort_mock,
                input_dir=src,
                output_dir=kwargs.pop("output_dir", None),
                prefix=kwargs.pop("prefix", ""),
                threshold=kwargs.pop("threshold", DEFAULT_THRESHOLD),
                model_repo="fake/model",
                include_ratings=kwargs.pop("include_ratings", False),
                skip_existing=kwargs.pop("skip_existing", True),
            )

    def test_creates_txt_file(self, tmp_path):
        src = tmp_path / "images"
        make_image(src / "photo.png")

        stats = self._run(tmp_path, src, prefix="test_prefix")

        assert (src / "photo.txt").exists()
        assert "test_prefix" in (src / "photo.txt").read_text()

    def test_skip_existing(self, tmp_path):
        src = tmp_path / "images"
        make_image(src / "photo.png")
        (src / "photo.txt").write_text("existing caption")

        stats = self._run(tmp_path, src, skip_existing=True)

        assert stats["skipped"] == 1
        assert stats["captioned"] == 0

    def test_output_dir_separate(self, tmp_path):
        src = tmp_path / "images"
        dst = tmp_path / "captions"
        make_image(src / "photo.png")

        self._run(tmp_path, src, output_dir=dst)

        assert (dst / "photo.txt").exists()
        assert not (src / "photo.txt").exists()
