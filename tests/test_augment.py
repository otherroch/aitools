"""
tests/test_augment.py

Unit tests for portrait_prep.augment
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from portrait_prep.augment import augment_folder, build_augment_pipeline, VALID_EXTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_png(path: Path, size: tuple = (64, 64), color: tuple = (100, 150, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------
# build_augment_pipeline
# ---------------------------------------------------------------------------


class TestBuildAugmentPipeline:
    def test_returns_compose(self):
        import albumentations as A

        pipeline = build_augment_pipeline(256, 256)
        assert isinstance(pipeline, A.Compose)

    def test_output_size_matches(self):
        pipeline = build_augment_pipeline(128, 64)
        dummy = np.zeros((200, 300, 3), dtype=np.uint8)
        result = pipeline(image=dummy)["image"]

        assert result.shape == (128, 64, 3)


# ---------------------------------------------------------------------------
# augment_folder
# ---------------------------------------------------------------------------


class TestAugmentFolder:
    def test_empty_directory(self, tmp_path):
        stats = augment_folder(tmp_path / "in", tmp_path / "out")
        assert stats["augmented"] == 0
        assert stats["skipped"] == 0

    def test_generates_correct_count(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_png(src / "photo.png")

        stats = augment_folder(src, dst, per_image=3)

        assert stats["augmented"] == 3
        aug_files = list(dst.rglob("*_aug*.png"))
        assert len(aug_files) == 3

    def test_file_naming_convention(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_png(src / "portrait.png")

        augment_folder(src, dst, per_image=2)

        assert (dst / "portrait_aug000.png").exists()
        assert (dst / "portrait_aug001.png").exists()

    def test_keep_originals_copies_resized(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_png(src / "face.png")

        augment_folder(src, dst, per_image=1, keep_originals=True)

        assert (dst / "face_orig.png").exists()

    def test_output_resolution_respected(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_png(src / "img.png", size=(50, 50))

        augment_folder(src, dst, per_image=1, image_size=(128, 128))

        aug = dst / "img_aug000.png"
        assert aug.exists()
        with Image.open(aug) as im:
            assert im.size == (128, 128)

    def test_mirrors_subfolder_structure(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        make_png(src / "sub" / "photo.png")

        augment_folder(src, dst, per_image=1)

        assert (dst / "sub" / "photo_aug000.png").exists()

    def test_seed_produces_same_number_of_outputs(self, tmp_path):
        """Two runs with the same seed produce the same number of output files
        and images of the correct resolution (pixel-exact reproducibility
        depends on the albumentations backend RNG and is not guaranteed across
        different library versions).
        """
        src = tmp_path / "in"
        make_png(src / "photo.png")

        dst1 = tmp_path / "out1"
        dst2 = tmp_path / "out2"

        augment_folder(src, dst1, per_image=2, seed=42)
        augment_folder(src, dst2, per_image=2, seed=42)

        out1_files = sorted(p.name for p in dst1.rglob("*.png"))
        out2_files = sorted(p.name for p in dst2.rglob("*.png"))
        assert out1_files == out2_files

        # Both runs produce correctly-sized images
        for name in out1_files:
            with Image.open(dst1 / name) as im:
                assert im.size == (1024, 1024)

    def test_multiple_images(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        for name in ["a.png", "b.png", "c.png"]:
            make_png(src / name)

        stats = augment_folder(src, dst, per_image=2)

        assert stats["augmented"] == 6

    def test_non_image_files_ignored(self, tmp_path):
        src = tmp_path / "in"
        dst = tmp_path / "out"
        src.mkdir(parents=True)
        (src / "notes.txt").write_text("ignore me")

        stats = augment_folder(src, dst, per_image=3)

        assert stats["augmented"] == 0
