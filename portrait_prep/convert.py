#!/usr/bin/env python3
"""
convert.py

Task 1 – Convert HEIC / JPG (and other common formats) images to PNG.

Supports HEIC natively via pillow_heif. Mirrors source folder structure into
the output directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HEIF_AVAILABLE = False

logger = logging.getLogger(__name__)

# Extensions that we will attempt to open and re-save as PNG
SUPPORTED_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
if _HEIF_AVAILABLE:
    SUPPORTED_EXTS |= {".heic", ".heif"}


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """Convert all supported images in *input_dir* to PNG in *output_dir*.

    Args:
        input_dir:     Source directory (searched recursively).
        output_dir:    Destination directory (folder structure mirrored).
        skip_existing: If True, skip files whose output PNG already exists.

    Returns:
        A ``(converted, skipped)`` count tuple.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(
        "convert_folder: input=%s output=%s skip_existing=%s",
        input_dir,
        output_dir,
        skip_existing,
    )

    candidates = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not candidates:
        logger.warning("No supported images found in %s", input_dir)
        return 0, 0

    logger.debug("convert_folder: found %d candidate(s) to process", len(candidates))

    converted = 0
    skipped = 0

    for src in candidates:
        rel = src.relative_to(input_dir)
        dst_dir = output_dir / rel.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / (src.stem + ".png")

        if skip_existing and dst.exists():
            logger.debug("Skipping (exists): %s", dst)
            skipped += 1
            continue

        try:
            with Image.open(src) as img:
                logger.debug("convert: %s  original size=%s mode=%s", src.name, img.size, img.mode)
                img = img.convert("RGB")
                img.save(dst, format="PNG", optimize=False, compress_level=3)
            logger.info("Converted: %s → %s", src.name, dst)
            converted += 1
        except Exception as exc:
            logger.error("Failed to convert %s: %s", src, exc)
            skipped += 1

    return converted, skipped
