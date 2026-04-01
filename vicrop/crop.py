#!/usr/bin/env python3
"""
vicrop.crop

Extract face-cropped PNG frames from video files.

Reads video files using OpenCV, samples frames at a configurable interval,
detects faces in each frame with face_recognition, and saves a cropped face
region to the output directory.  Optionally clusters face crops by identity
into ``person_NN`` sub-folders (same greedy nearest-neighbour approach used
by portrait_prep.crop).  When ``ref_thresh > 0`` each face crop is also
scored for reference-photo quality and a ``reflist.txt`` is written to each
identity folder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from portrait_prep.face_utils import (
    DEFAULT_CROP_SIZE,
    DEFAULT_MARGIN_RATIO,
    cluster_faces as _cluster_faces_shared,
    load_face_recognition as _load_face_recognition,
)
from vicrop.ref import (
    DEFAULT_REF_THRESH,
    collect_ref_photos,
    score_reference_quality,
)

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS: set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv",
}

DEFAULT_EVERY_N_FRAMES: int = 30


def _cluster_faces(
    all_results: list[tuple[Path, np.ndarray]],
    output_dir: Path,
    tolerance: float = 0.6,
) -> dict[int, list[Path]]:
    """Group saved face crops by identity using greedy nearest-neighbour clustering.

    Delegates to :func:`portrait_prep.face_utils.cluster_faces`.
    """
    fr = _load_face_recognition()
    return _cluster_faces_shared(all_results, output_dir, tolerance=tolerance, fr=fr)


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
    ref_thresh: float = DEFAULT_REF_THRESH,
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
        ref_thresh:    Minimum quality score (0–1) for a face crop to be
                       listed as a reference photo.  ``0`` disables the
                       analysis entirely.

    Returns:
        Summary dict with keys ``frames_processed``, ``faces``,
        ``persons``, ``ref_photos``.
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
        return {"frames_processed": 0, "faces": 0, "persons": 0, "ref_photos": 0}

    staging_dir = video_stem_dir / "_staging" if classify else video_stem_dir
    staging_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return {"frames_processed": 0, "faces": 0, "persons": 0, "ref_photos": 0}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_sample = max(1, (total_frames + every_n - 1) // every_n) if total_frames > 0 else 0
    logger.info(
        "Video %s: %d total frames, sampling every %d → ~%d frames to process",
        video_path.name, total_frames, every_n, frames_to_sample,
    )

    do_ref = ref_thresh > 0

    frame_idx = 0
    frames_processed = 0
    faces_detected = 0
    all_results: list[tuple[Path, np.ndarray]] = []
    ref_scores: dict[str, float] = {}  # filename → quality score

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % 1000 == 0:
                if total_frames > 0:
                    pct = frame_idx * 100.0 / total_frames
                    logger.info(
                        "[%5.1f%%] frame %d / %d  faces so far: %d",
                        pct, frame_idx, total_frames, faces_detected,
                    )
                else:
                    logger.info(
                        "frame %d  faces so far: %d", frame_idx, faces_detected,
                    )
                
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

                    faces_detected += 1

                    if do_ref:
                        lm_list = fr.face_landmarks(
                            frame_rgb, [(top, right, bottom, left)],
                        )
                        lm = lm_list[0] if lm_list else None
                        ref_scores[out_name] = score_reference_quality(
                            frame_rgb,
                            (top, right, bottom, left),
                            lm,
                            face_arr,
                            face_count=len(face_locations),
                        )

                frames_processed += 1

            frame_idx += 1
    finally:
        cap.release()

    persons = 0
    total_refs = 0
    logger.info(
        "Finished processing video: %s  frames processed: %d  faces detected: %d",
        video_path.name, frames_processed, faces_detected,
    )

    if classify and all_results:
        person_dirs = _cluster_faces(all_results, video_stem_dir, tolerance=tolerance)
        persons = len(person_dirs)
        try:
            staging_dir.rmdir()
        except OSError:
            pass

        # Move reference photos into ref/ sub-folder per person
        if do_ref:
            for _pid, paths in person_dirs.items():
                ref_paths = [
                    p for p in paths
                    if ref_scores.get(p.name, 0.0) >= ref_thresh
                ]
                if ref_paths:
                    collect_ref_photos(paths[0].parent, ref_paths)
                    total_refs += len(ref_paths)
    elif not classify and do_ref and all_results:
        ref_paths = [
            path for path, _ in all_results
            if ref_scores.get(path.name, 0.0) >= ref_thresh
        ]
        if ref_paths:
            collect_ref_photos(video_stem_dir, ref_paths)
            total_refs += len(ref_paths)

    return {
        "frames_processed": frames_processed,
        "faces": len(all_results),
        "persons": persons,
        "ref_photos": total_refs,
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
    ref_thresh: float = DEFAULT_REF_THRESH,
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
        ref_thresh:    Minimum quality score (0–1) for reference-photo
                       selection.  ``0`` disables the analysis.

    Returns:
        Aggregate summary dict with keys ``videos_processed``,
        ``frames_processed``, ``faces``, ``persons``, ``ref_photos``.
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
        return {"videos_processed": 0, "frames_processed": 0, "faces": 0, "persons": 0, "ref_photos": 0}

    logger.debug(
        "crop_folder: found %d video(s) in %s  every_n=%d classify=%s",
        len(videos), input_dir, every_n, classify,
    )

    total: dict[str, int] = {
        "videos_processed": 0,
        "frames_processed": 0,
        "faces": 0,
        "persons": 0,
        "ref_photos": 0,
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
            ref_thresh=ref_thresh,
        )
        total["videos_processed"] += 1
        total["frames_processed"] += stats["frames_processed"]
        total["faces"] += stats["faces"]
        total["persons"] += stats["persons"]
        total["ref_photos"] += stats["ref_photos"]

    return total
