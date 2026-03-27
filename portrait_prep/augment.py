#!/usr/bin/env python3
"""
augment.py

Task 4 – Data augmentation for portrait images using Albumentations.

Ported from aug.py with a clean function-level API.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

logger = logging.getLogger(__name__)

VALID_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def build_augment_pipeline(height: int, width: int) -> A.Compose:
    """Return an Albumentations pipeline for identity-preserving portrait augmentation."""
    return A.Compose(
        [
            # Always resize to target resolution
            A.Resize(height=height, width=width),

            # Pose / framing
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                fit_output=False,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),

            # Brightness / contrast / gamma
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ],
                p=0.9,
            ),

            # Colour channel / saturation shifts
            A.OneOf(
                [
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
                ],
                p=0.3,
            ),

            # Blur variations
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),

            # Noise
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0.0, 0.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.01, 0.05), p=1.0),
                ],
                p=0.2,
            ),

            # JPEG-style compression artefacts
            A.ImageCompression(quality_range=(70, 100), compression_type="jpeg", p=0.3),
        ]
    )


def augment_folder(
    input_dir: Path,
    output_dir: Path,
    per_image: int = 5,
    image_size: tuple[int, int] = (1024, 1024),
    keep_originals: bool = False,
    seed: int = 4051888,
) -> dict[str, int]:
    """Augment every image in *input_dir* and write results to *output_dir*.

    Args:
        input_dir:      Source directory (searched recursively).
        output_dir:     Output directory (folder structure mirrored).
        per_image:      Number of augmented variants to produce per source image.
        image_size:     Output ``(height, width)`` in pixels.
        keep_originals: When True, also copy a resized original with ``_orig`` suffix.
        seed:           Random seed for reproducibility.

    Returns:
        Summary dict with keys ``augmented`` and ``skipped``.
    """
    random.seed(seed)
    np.random.seed(seed)

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]

    if not images:
        logger.warning("No images found in %s", input_dir)
        return {"augmented": 0, "skipped": 0}

    logger.debug(
        "augment_folder: found %d image(s)  per_image=%d size=%s keep_originals=%s seed=%d",
        len(images), per_image, image_size, keep_originals, seed,
    )

    h, w = image_size
    aug = build_augment_pipeline(height=h, width=w)

    augmented = 0
    skipped = 0

    for img_path in images:
        rel = img_path.relative_to(input_dir)
        out_subdir = output_dir / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        logger.debug("augment: processing %s", img_path.name)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("Could not read image %s, skipping.", img_path)
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if keep_originals:
            resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
            resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            orig_out = out_subdir / f"{img_path.stem}_orig.png"
            cv2.imwrite(str(orig_out), resized_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            logger.debug("augment: saved original copy to %s", orig_out)

        for i in range(per_image):
            try:
                aug_rgb = aug(image=img_rgb)["image"]
                aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
                out_path = out_subdir / f"{img_path.stem}_aug{i:03d}.png"
                cv2.imwrite(str(out_path), aug_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                logger.debug("augment: saved variant %d → %s", i, out_path)
                augmented += 1
            except Exception as exc:
                logger.error("Augmentation failed for %s (variant %d): %s", img_path, i, exc)
                skipped += 1

    return {"augmented": augmented, "skipped": skipped}
