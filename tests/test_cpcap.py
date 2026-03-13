"""
tests/test_cpcap.py

Unit tests for portrait_prep.cpcap
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from portrait_prep.cpcap import copy_captions, infer_original_stem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_png(path: Path, size: tuple = (32, 32)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def make_caption(path: Path, content: str = "a photo of a person") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# infer_original_stem
# ---------------------------------------------------------------------------


class TestInferOriginalStem:
    def test_aug_000_suffix(self):
        assert infer_original_stem("photo_aug000") == "photo"

    def test_aug_single_digit(self):
        assert infer_original_stem("face_aug9") == "face"

    def test_orig_suffix(self):
        assert infer_original_stem("photo_orig") == "photo"

    def test_no_suffix(self):
        assert infer_original_stem("photo") == "photo"

    def test_underscores_in_stem(self):
        assert infer_original_stem("my_face_aug001") == "my_face"

    def test_orig_with_underscores(self):
        assert infer_original_stem("my_photo_orig") == "my_photo"


# ---------------------------------------------------------------------------
# copy_captions
# ---------------------------------------------------------------------------


class TestCopyCaptions:
    def test_missing_source_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            copy_captions(tmp_path / "nonexistent", tmp_path / "aug")

    def test_missing_aug_dir_raises(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        with pytest.raises(FileNotFoundError):
            copy_captions(src, tmp_path / "nonexistent")

    def test_empty_aug_dir(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"
        src.mkdir()
        aug.mkdir()

        stats = copy_captions(src, aug)

        assert stats["created"] == 0

    def test_copies_caption_for_aug_image(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"

        make_caption(src / "photo.txt", "a person smiling")
        make_png(aug / "photo_aug000.png")

        stats = copy_captions(src, aug)

        assert stats["created"] == 1
        assert (aug / "photo_aug000.txt").exists()
        assert (aug / "photo_aug000.txt").read_text() == "a person smiling"

    def test_copies_caption_for_orig_image(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"

        make_caption(src / "face.txt", "original caption")
        make_png(aug / "face_orig.png")

        stats = copy_captions(src, aug)

        assert stats["created"] == 1
        assert (aug / "face_orig.txt").read_text() == "original caption"

    def test_skips_when_caption_exists(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"

        make_caption(src / "photo.txt", "new caption")
        make_png(aug / "photo_aug000.png")
        make_caption(aug / "photo_aug000.txt", "existing caption")

        stats = copy_captions(src, aug, skip_existing=True)

        assert stats["skipped_existing"] == 1
        assert stats["created"] == 0
        # Content should remain unchanged
        assert (aug / "photo_aug000.txt").read_text() == "existing caption"

    def test_skips_when_no_source_caption(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"
        src.mkdir()

        make_png(aug / "photo_aug000.png")

        stats = copy_captions(src, aug)

        assert stats["skipped_no_source"] == 1
        assert stats["created"] == 0

    def test_dry_run_does_not_write(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"

        make_caption(src / "photo.txt", "test")
        make_png(aug / "photo_aug000.png")

        stats = copy_captions(src, aug, dry_run=True)

        assert stats["created"] == 1  # counted but not written
        assert not (aug / "photo_aug000.txt").exists()

    def test_preserves_subfolder_structure(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"

        make_caption(src / "set1" / "portrait.txt", "sub caption")
        make_png(aug / "set1" / "portrait_aug000.png")

        stats = copy_captions(src, aug)

        assert stats["created"] == 1
        assert (aug / "set1" / "portrait_aug000.txt").exists()

    def test_multiple_augmented_images_one_source(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"

        make_caption(src / "face.txt", "a face")
        for i in range(5):
            make_png(aug / f"face_aug{i:03d}.png")

        stats = copy_captions(src, aug)

        assert stats["created"] == 5
        for i in range(5):
            assert (aug / f"face_aug{i:03d}.txt").exists()

    def test_custom_caption_extension(self, tmp_path):
        src = tmp_path / "src"
        aug = tmp_path / "aug"

        caption = src / "photo.cap"
        caption.parent.mkdir(parents=True)
        caption.write_text("custom ext caption")
        make_png(aug / "photo_aug000.png")

        stats = copy_captions(src, aug, caption_ext=".cap")

        assert stats["created"] == 1
        assert (aug / "photo_aug000.cap").exists()
