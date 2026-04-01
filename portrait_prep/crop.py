#!/usr/bin/env python3
"""
crop.py

Task 2 – Face-crop portraits and classify each identity into its own sub-folder.

Detection is performed with face_recognition (dlib HOG/CNN backend).
Each unique face cluster found across the source folder is written to:

    output_dir/<person_N>/original_name_faceM.png

When ``--cluster`` is NOT requested (default) each face is placed in a flat
output folder and the caller decides the sub-folder naming.  Pass
``classify=True`` to enable automatic person-clustering via face embedding
k-medoids (requires scikit-learn).
"""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
from PIL import Image

from portrait_prep.face_utils import (
    DEFAULT_CROP_SIZE,
    DEFAULT_MARGIN_RATIO,
    cluster_faces as _cluster_faces_shared,
    load_face_recognition as _load_face_recognition,
    load_reference_encodings,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}


def crop_faces_from_image(
    image_path: Path,
    output_dir: Path,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    model: str = "hog",
) -> list[tuple[Path, np.ndarray]]:
    """Detect, crop, and save each face found in *image_path*.

    Args:
        image_path:   Source image file.
        output_dir:   Directory in which cropped face PNGs are saved.
        margin_ratio: Fraction of face bbox added as margin on each side.
        crop_size:    Output square resolution (pixels).
        model:        face_recognition detection model – "hog" (fast) or "cnn".

    Returns:
        List of ``(saved_path, face_encoding)`` tuples for every face saved.
    """
    fr = _load_face_recognition()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("crop_faces_from_image: %s  margin_ratio=%.2f crop_size=%d model=%s",
                 image_path.name, margin_ratio, crop_size, model)

    image = fr.load_image_file(str(image_path))
    face_locations = fr.face_locations(image, model=model)
    face_encodings = fr.face_encodings(image, face_locations)

    if not face_locations:
        logger.warning("No face found in %s", image_path.name)
        return []

    logger.debug("crop_faces_from_image: detected %d face(s) in %s", len(face_locations), image_path.name)

    h_img, w_img = image.shape[:2]
    results: list[tuple[Path, np.ndarray]] = []

    for i, ((top, right, bottom, left), encoding) in enumerate(
        zip(face_locations, face_encodings)
    ):
        face_h = bottom - top
        face_w = right - left
        margin_h = int(face_h * margin_ratio)
        margin_w = int(face_w * margin_ratio)

        crop_top = max(0, top - margin_h)
        crop_bottom = min(h_img, bottom + margin_h)
        crop_left = max(0, left - margin_w)
        crop_right = min(w_img, right + margin_w)

        face_arr = image[crop_top:crop_bottom, crop_left:crop_right]
        pil_img = Image.fromarray(face_arr).resize(
            (crop_size, crop_size), Image.LANCZOS
        )

        out_name = f"{image_path.stem}_face{i + 1}.png"
        out_path = output_dir / out_name
        pil_img.save(out_path)
        logger.info("Saved face crop: %s", out_path)
        logger.debug(
            "crop: face %d bbox=(top=%d right=%d bottom=%d left=%d) crop=(%d,%d,%d,%d)",
            i + 1, top, right, bottom, left,
            crop_top, crop_bottom, crop_left, crop_right,
        )
        results.append((out_path, encoding))

    return results


def _cluster_faces(
    all_results: list[tuple[Path, np.ndarray]],
    output_dir: Path,
    tolerance: float = 0.6,
    reference_encodings: list[np.ndarray] | None = None,
    reference_names: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Group saved face crops by identity using face distance clustering.

    Delegates to :func:`portrait_prep.face_utils.cluster_faces`.
    """
    fr = _load_face_recognition()
    return _cluster_faces_shared(
        all_results, output_dir, tolerance=tolerance, fr=fr,
        reference_encodings=reference_encodings,
        reference_names=reference_names,
    )


def crop_folder(
    input_dir: Path,
    output_dir: Path,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    classify: bool = True,
    tolerance: float = 0.6,
    model: str = "hog",
    classified_path: Path | None = None,
    classified_max: int = 0,
) -> dict[str, int]:
    """Crop all faces found in *input_dir* and write them to *output_dir*.

    Args:
        input_dir:       Source directory (searched recursively).
        output_dir:      Destination directory.
        margin_ratio:    Margin added around each detected face bbox.
        crop_size:       Output square resolution in pixels.
        classify:        If True, cluster faces by identity, creating
                         identity sub-folders.
        tolerance:       Face-distance threshold for identity clustering.
        model:           face_recognition detection model ("hog" or "cnn").
        classified_path: Optional path to a directory of pre-classified
                         reference photos.  Each sub-folder is treated as a
                         known identity; new faces matching a reference are
                         placed in a folder with the same name.
        classified_max:  Maximum reference images to load per identity.
                         ``0`` means no limit.

    Returns:
        Summary dict with keys ``faces``, ``images_processed``, ``persons``.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not images:
        logger.warning("No images found in %s", input_dir)
        return {"faces": 0, "images_processed": 0, "persons": 0}

    logger.debug(
        "crop_folder: found %d image(s) in %s  classify=%s tolerance=%.2f model=%s",
        len(images), input_dir, classify, tolerance, model,
    )

    # Flat staging dir when classify=True (moved after clustering)
    staging_dir = output_dir / "_staging" if classify else output_dir

    all_results: list[tuple[Path, np.ndarray]] = []
    for img_path in images:
        results = crop_faces_from_image(
            img_path,
            staging_dir,
            margin_ratio=margin_ratio,
            crop_size=crop_size,
            model=model,
        )
        all_results.extend(results)

    persons = 0
    if classify and all_results:
        ref_enc: list[np.ndarray] | None = None
        ref_names: list[str] | None = None
        if classified_path is not None:
            ref_enc, ref_names = load_reference_encodings(
                classified_path, model=model,
                max_per_identity=classified_max,
            )
        person_dirs = _cluster_faces(
            all_results, output_dir, tolerance=tolerance,
            reference_encodings=ref_enc,
            reference_names=ref_names,
        )
        persons = len(person_dirs)
        # Remove staging dir if empty
        try:
            staging_dir.rmdir()
        except OSError:
            pass

    return {
        "faces": len(all_results),
        "images_processed": len(images),
        "persons": persons,
    }
