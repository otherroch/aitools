"""
tests/test_convert.py

Unit tests for portrait_prep.convert
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from portrait_prep.convert import convert_folder, SUPPORTED_EXTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_image(path: Path, color: tuple = (128, 64, 32), size: tuple = (100, 100)) -> Path:
    """Create a minimal RGB PNG at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSupportedExts:
    def test_jpg_included(self):
        assert ".jpg" in SUPPORTED_EXTS

    def test_png_included(self):
        assert ".png" in SUPPORTED_EXTS

    def test_heic_included_when_pillow_heif_available(self):
        # heic support depends on pillow_heif being installed in the environment
        try:
            import pillow_heif  # noqa: F401
            assert ".heic" in SUPPORTED_EXTS
        except ImportError:
            pytest.skip("pillow_heif not installed")


class TestConvertFolder:
    def test_empty_directory_returns_zeros(self, tmp_path):
        converted, skipped = convert_folder(tmp_path / "in", tmp_path / "out")
        assert converted == 0
        assert skipped == 0

    def test_converts_png_to_png(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_image(src / "photo.png")

        converted, skipped = convert_folder(src, dst)

        assert converted == 1
        assert skipped == 0
        assert (dst / "photo.png").exists()

    def test_converts_jpg_to_png(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        img_path = src / "photo.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (50, 50), (10, 20, 30)).save(img_path, format="JPEG")

        converted, skipped = convert_folder(src, dst)

        assert converted == 1
        assert (dst / "photo.png").exists()

    def test_skip_existing_true(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_image(src / "photo.png")
        # Pre-create output
        make_image(dst / "photo.png")

        converted, skipped = convert_folder(src, dst, skip_existing=True)

        assert converted == 0
        assert skipped == 1

    def test_skip_existing_false_overwrites(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_image(src / "photo.png", color=(1, 2, 3))
        make_image(dst / "photo.png", color=(200, 200, 200))

        converted, skipped = convert_folder(src, dst, skip_existing=False)

        assert converted == 1

    def test_mirrors_subfolder_structure(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_image(src / "subfolder" / "nested.png")

        convert_folder(src, dst)

        assert (dst / "subfolder" / "nested.png").exists()

    def test_invalid_file_is_skipped_gracefully(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        bad = src / "broken.jpg"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not an image")

        converted, skipped = convert_folder(src, dst)

        assert converted == 0
        assert skipped == 1

    def test_non_image_files_are_ignored(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        src.mkdir(parents=True)
        (src / "readme.txt").write_text("hello")
        (src / "data.json").write_text("{}")

        converted, skipped = convert_folder(src, dst)

        assert converted == 0
        assert skipped == 0

    def test_multiple_images(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        for name in ["a.png", "b.png", "c.png"]:
            make_image(src / name)

        converted, skipped = convert_folder(src, dst)

        assert converted == 3
