#!/usr/bin/env python3
"""
vicrop.crop

Extract face-cropped PNG frames from video files.

Reads video files using OpenCV, samples frames at a configurable interval,
detects faces in each frame with face_recognition, and saves a cropped face
region to the output directory.  Optionally clusters face crops by identity
into ``person_NN`` sub-folders (same greedy nearest-neighbour approach used
by portrait_prep.crop).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS: set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv",
}

DEFAULT_EVERY_N_FRAMES: int = 30
DEFAULT_MARGIN_RATIO: float = 0.4
DEFAULT_CROP_SIZE: int = 1024


def _load_face_recognition():
    try:
        import face_recognition
        return face_recognition
    except ImportError as exc:
        raise ImportError(
            "face_recognition is required for vicrop.\n"
            "Install it with: pip install face_recognition"
        ) from exc


def _cluster_faces(
    all_results: list[tuple[Path, np.ndarray]],
    output_dir: Path,
    tolerance: float = 0.6,
) -> dict[int, list[Path]]:
    """Group saved face crops by identity using greedy nearest-neighbour clustering.

    Each unique identity is assigned an integer label starting from 1 and its
    crops are moved into ``output_dir/person_NN/``.

    Returns a mapping ``{person_id: [list of face image paths]}``.
    """
    fr = _load_face_recognition()

    known_encodings: list[np.ndarray] = []
    known_labels: list[int] = []
    next_label = 1
    label_map: dict[Path, int] = {}

    for face_path, encoding in all_results:
        if not known_encodings:
            known_encodings.append(encoding)
            known_labels.append(next_label)
            label_map[face_path] = next_label
            next_label += 1
            continue

        distances = fr.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= tolerance:
            label_map[face_path] = known_labels[best_idx]
        else:
            known_encodings.append(encoding)
            known_labels.append(next_label)
            label_map[face_path] = next_label
            next_label += 1

    person_dirs: dict[int, list[Path]] = {}
    for face_path, label in label_map.items():
        person_dir = output_dir / f"person_{label:02d}"
        person_dir.mkdir(parents=True, exist_ok=True)
        dest = person_dir / face_path.name
        face_path.rename(dest)
        person_dirs.setdefault(label, []).append(dest)

    return person_dirs


def crop_video(
    video_path: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    model: str = "hog",
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Extract face-cropped frames from a single video file.

    Frames are sampled every *every_n* frames.  Detected faces are cropped with
    a fractional *margin_ratio* padding, resized to *crop_size* × *crop_size*,
    and saved as PNG files inside a sub-directory named after the video stem.

    Args:
        video_path:    Path to the input video file.
        output_dir:    Root directory where cropped images are saved.
        every_n:       Process every N-th frame (default: 30).
        margin_ratio:  Fractional padding around each detected face bbox.
        crop_size:     Output square resolution in pixels (default: 1024).
        model:         face_recognition detection model – ``"hog"`` (fast) or
                       ``"cnn"`` (accurate).
        classify:      If True, cluster faces by identity into
                       ``person_NN`` sub-folders.
        tolerance:     Face-distance threshold for identity clustering.
        skip_existing: Skip the video if its output sub-directory already
                       contains PNG files.

    Returns:
        Summary dict with keys ``frames_processed``, ``faces``, ``persons``.
    """
    fr = _load_face_recognition()

    output_dir = output_dir.resolve()
    video_stem_dir = output_dir / video_path.stem

    logger.debug(
        "crop_video: %s  every_n=%d margin_ratio=%.2f crop_size=%d model=%s classify=%s",
        video_path.name, every_n, margin_ratio, crop_size, model, classify,
    )

    if skip_existing and video_stem_dir.exists() and any(video_stem_dir.rglob("*.png")):
        logger.info("Skipping (already processed): %s", video_path.name)
        return {"frames_processed": 0, "faces": 0, "persons": 0}

    staging_dir = video_stem_dir / "_staging" if classify else video_stem_dir
    staging_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return {"frames_processed": 0, "faces": 0, "persons": 0}

    frame_idx = 0
    frames_processed = 0
    all_results: list[tuple[Path, np.ndarray]] = []

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                face_locations = fr.face_locations(frame_rgb, model=model)
                face_encodings = fr.face_encodings(frame_rgb, face_locations)

                logger.debug(
                    "crop_video: frame %d  detected %d face(s)",
                    frame_idx, len(face_locations),
                )

                h_img, w_img = frame_rgb.shape[:2]

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

                    face_arr = frame_rgb[crop_top:crop_bottom, crop_left:crop_right]
                    pil_img = Image.fromarray(face_arr).resize(
                        (crop_size, crop_size), Image.LANCZOS
                    )

                    out_name = f"frame{frame_idx:06d}_face{i + 1}.png"
                    out_path = staging_dir / out_name
                    pil_img.save(out_path)
                    logger.debug("Saved face crop: %s", out_path)
                    all_results.append((out_path, encoding))

                frames_processed += 1

            frame_idx += 1
    finally:
        cap.release()

    persons = 0
    if classify and all_results:
        person_dirs = _cluster_faces(all_results, video_stem_dir, tolerance=tolerance)
        persons = len(person_dirs)
        try:
            staging_dir.rmdir()
        except OSError:
            pass

    return {
        "frames_processed": frames_processed,
        "faces": len(all_results),
        "persons": persons,
    }


def crop_folder(
    input_dir: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    model: str = "hog",
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Process all video files in *input_dir*, extracting face-cropped frames.

    Args:
        input_dir:     Source directory (searched recursively for video files).
        output_dir:    Destination directory.
        every_n:       Process every N-th frame from each video.
        margin_ratio:  Fractional margin around each detected face bbox.
        crop_size:     Output square resolution in pixels.
        model:         face_recognition detection model (``"hog"`` or ``"cnn"``).
        classify:      If True, cluster faces by identity into
                       ``person_NN`` sub-folders.
        tolerance:     Face-distance threshold for identity clustering.
        skip_existing: Skip videos whose output sub-directory already has PNGs.

    Returns:
        Aggregate summary dict with keys ``videos_processed``,
        ``frames_processed``, ``faces``, ``persons``.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ]

    if not videos:
        logger.warning("No video files found in %s", input_dir)
        return {"videos_processed": 0, "frames_processed": 0, "faces": 0, "persons": 0}

    logger.debug(
        "crop_folder: found %d video(s) in %s  every_n=%d classify=%s",
        len(videos), input_dir, every_n, classify,
    )

    total: dict[str, int] = {
        "videos_processed": 0,
        "frames_processed": 0,
        "faces": 0,
        "persons": 0,
    }

    for video_path in videos:
        logger.info("Processing video: %s", video_path.name)
        stats = crop_video(
            video_path,
            output_dir,
            every_n=every_n,
            margin_ratio=margin_ratio,
            crop_size=crop_size,
            model=model,
            classify=classify,
            tolerance=tolerance,
            skip_existing=skip_existing,
        )
        total["videos_processed"] += 1
        total["frames_processed"] += stats["frames_processed"]
        total["faces"] += stats["faces"]
        total["persons"] += stats["persons"]

    return total
