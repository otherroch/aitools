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
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from vicrop.ref import (
    DEFAULT_REF_THRESH,
    collect_ref_photos,
    score_reference_quality,
)

if 0:
    from face_ops.backend import FaceBackend

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS: set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv",
}

DEFAULT_EVERY_N_FRAMES: int = 30
DEFAULT_MARGIN_RATIO: float = 0.4
DEFAULT_CROP_SIZE: int = 1024


def _default_backend():
    """Create a default dlib backend when none is provided."""
    from face_ops import backend_for_model

    return backend_for_model("dlib")


def crop_video(
    video_path: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = True,
    ref_thresh: float = DEFAULT_REF_THRESH,
    classified_path: Path | None = None,
    classified_max: int = 0,
    crop_height: int | None = None,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """Extract face-cropped frames from a single video file.

    Frames are sampled every *every_n* frames.  Detected faces are cropped with
    a fractional *margin_ratio* padding, resized to *crop_size* x *crop_size*
    (or *crop_height* x *crop_height* if *crop_height* is given), and saved
    as PNG files inside a sub-directory named after the video stem.

    Args:
        video_path:      Path to the input video file.
        output_dir:      Root directory where cropped images are saved.
        every_n:         Process every N-th frame (default: 30).
        margin_ratio:    Fractional padding around each detected face bbox.
        crop_size:       Output square resolution in pixels (default: 1024).
        classify:        If True, cluster faces by identity into
                         identity sub-folders.
        tolerance:       Face-distance threshold for identity clustering.
        skip_existing:   Skip the video if its output sub-directory already
                         contains PNG files.
        ref_thresh:      Minimum quality score (0-1) for a face crop to be
                         listed as a reference photo.  ``0`` disables the
                         analysis entirely.
        classified_path: Optional path to a directory of pre-classified
                         reference photos used to seed identity clustering.
        classified_max:  Maximum reference images to load per identity.
                         ``0`` means no limit.
        crop_height:     If given, resize each face crop to a square of
                         this size instead of *crop_size*.
        backend:         :class:`FaceBackend` instance for detection, encoding,
                         and clustering.  When *None*, a default dlib backend
                         is created.

    Returns:
        Summary dict with keys ``frames_processed``, ``faces``,
        ``persons``, ``ref_photos``.
    """
    if backend is None:
        backend = _default_backend()

    output_dir = output_dir.resolve()
    video_stem_dir = output_dir / video_path.stem

    logger.debug(
        "crop_video: %s  every_n=%d margin_ratio=%.2f crop_size=%d classify=%s",
        video_path.name, every_n, margin_ratio, crop_size, classify,
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
        "Video %s: %d total frames, sampling every %d -> ~%d frames to process",
        video_path.name, total_frames, every_n, frames_to_sample,
    )

    do_ref = ref_thresh > 0

    frame_idx = 0
    frames_processed = 0
    faces_detected = 0
    all_results: list[tuple[Path, np.ndarray]] = []
    ref_scores: dict[str, float] = {}  # filename -> quality score

    debug_logging = logger.isEnabledFor(logging.DEBUG)

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
                face_locations = backend.detect_faces(frame_rgb)
                face_encodings = backend.encode_faces(frame_rgb, face_locations)

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
                    if crop_height is not None:
                        out_w = crop_size if crop_size is not None else face_arr.shape[1]
                        pil_img = Image.fromarray(face_arr).resize(
                            (out_w, crop_height), Image.LANCZOS
                        )
                    else:
                        out_s = crop_size if crop_size is not None else face_arr.shape[0]
                        pil_img = Image.fromarray(face_arr).resize(
                            (out_s, out_s), Image.LANCZOS
                        )

                    out_name = f"frame{frame_idx:06d}_face{i + 1}.png"
                    out_path = staging_dir / out_name
                    pil_img.save(out_path)
                    logger.debug("Saved face crop: %s", out_path)
                    all_results.append((out_path, encoding))

                    faces_detected += 1

                    if do_ref:
                        lm_list = backend.face_landmarks(
                            frame_rgb, [(top, right, bottom, left)],
                        )
                        lm = lm_list[0] if lm_list else None
                        ref_scores[out_name] = score_reference_quality(
                            frame_rgb,
                            (top, right, bottom, left),
                            lm,
                            face_arr,
                            face_count=len(face_locations),
                            name=out_name if debug_logging else None,
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
        ref_enc: list[np.ndarray] | None = None
        ref_names: list[str] | None = None
        if classified_path is not None:
            ref_enc, ref_names = backend.load_reference_encodings(
                classified_path,
                max_per_identity=classified_max,
            )
        person_dirs = backend.cluster_faces(
            all_results, video_stem_dir, tolerance=tolerance,
            reference_encodings=ref_enc,
            reference_names=ref_names,
        )
        persons = len(person_dirs)
        try:
            staging_dir.rmdir()
        except OSError:
            pass

        # Move reference photos into ref/ sub-folder per person
        if do_ref:
            for _pid, paths in person_dirs.items():
                ref_paths = []
                for p in paths:
                    if p.name in ref_scores and ref_scores[p.name] >= ref_thresh:
                        ref_paths.append(p)
                        logger.debug(
                          "Selected reference photo: %s  score=%.3f",
                          p.name, ref_scores[p.name],
                        )
                if ref_paths:
                    collect_ref_photos(ref_paths[0].parent, ref_paths)
                    total_refs += len(ref_paths)
    elif not classify and do_ref and all_results:
        ref_paths = []
        for path, _ in all_results:
            if path.name in ref_scores and ref_scores[path.name] >= ref_thresh:
                ref_paths.append(path)
                logger.debug(
                    "Selected reference photo: %s  score=%.3f",
                    path.name, ref_scores[path.name],
                )

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
    margin_ratio: float = 0.4,
    crop_size: int = 1024,
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = True,
    ref_thresh: float = DEFAULT_REF_THRESH,
    classified_path: Path | None = None,
    classified_max: int = 0,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """Process all video files in *input_dir*, extracting face-cropped frames.

    Args:
        input_dir:       Source directory (searched recursively for video files).
        output_dir:      Destination directory.
        every_n:         Process every N-th frame from each video.
        margin_ratio:    Fractional margin around each detected face bbox.
        crop_size:       Output square resolution in pixels.
        classify:        If True, cluster faces by identity into
                         identity sub-folders.
        tolerance:       Face-distance threshold for identity clustering.
        skip_existing:   Skip videos whose output sub-directory already has PNGs.
        ref_thresh:      Minimum quality score (0-1) for reference-photo
                         selection.  ``0`` disables the analysis.
        classified_path: Optional path to a directory of pre-classified
                         reference photos used to seed identity clustering.
        classified_max:  Maximum reference images to load per identity.
                         ``0`` means no limit.
        backend:         :class:`FaceBackend` instance.  When *None*, a
                         default dlib backend is created.

    Returns:
        Aggregate summary dict with keys ``videos_processed``,
        ``frames_processed``, ``faces``, ``persons``, ``ref_photos``.
    """
    if backend is None:
        backend = _default_backend()

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
        "crop_folder: found %d video(s) in %s",
        len(videos), input_dir,
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
            classify=classify,
            tolerance=tolerance,
            skip_existing=skip_existing,
            ref_thresh=ref_thresh,
            classified_path=classified_path,
            classified_max=classified_max,
            backend=backend,
        )
        total["videos_processed"] += 1
        total["frames_processed"] += stats["frames_processed"]
        total["faces"] += stats["faces"]
        total["persons"] += stats["persons"]
        total["ref_photos"] += stats["ref_photos"]

    return total


def extract_frames(
    video_path_or_dir: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    crop_size: int | None = None,
    crop_height: int | None = None,
    skip_existing: bool = False,
) -> dict[str, int]:
    """Extract raw frames from video files without face detection.

    Reads video files using OpenCV, samples frames at a configurable interval,
    and saves each frame as a PNG file to the output directory.

    Args:
        video_path_or_dir: Path to a single video file or a directory of videos.
        output_dir:        Root directory where extracted frames are saved.
        every_n:         Process every N-th frame (default: 30).
        crop_size:       If given, resize each frame to a square of this size.
        crop_height:     If given and crop_size is not, resize each frame to
                         a square of this size (takes precedence over crop_size).
        skip_existing:   If True, skip a video if its output sub-directory
                         already contains PNG files.

    Returns:
        Summary dict with keys ``videos_processed``, ``frames_extracted``.
    """
    if crop_height is not None:
        crop_size = crop_height

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if input is a single file or a directory
    if video_path_or_dir.is_file():
        if video_path_or_dir.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            videos = [video_path_or_dir]
        else:
            logger.warning("Unsupported file type: %s", video_path_or_dir)
            return {"videos_processed": 0, "frames_extracted": 0}
    elif video_path_or_dir.is_dir():
        videos = [
            p for p in video_path_or_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
        ]
    else:
        logger.warning("Invalid path: %s", video_path_or_dir)
        return {"videos_processed": 0, "frames_extracted": 0}

    if not videos:
        logger.warning("No video files found")
        return {"videos_processed": 0, "frames_extracted": 0}

    total_stats: dict[str, int] = {
        "videos_processed": 0,
        "frames_extracted": 0,
    }

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video: %s", video_path)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            continue

        stem_dir = output_dir / video_path.stem
        if skip_existing and any(stem_dir.rglob("*.png")):
            logger.info("Skipping (already processed): %s", video_path.name)
            cap.release()
            continue

        stem_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        frames_extracted = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                if crop_size:
                    pil_img = pil_img.resize(
                        (crop_size, crop_size), Image.LANCZOS
                    )
                out_name = f"frame{frame_idx:06d}.png"
                out_path = stem_dir / out_name
                pil_img.save(out_path)
                frames_extracted += 1

            frame_idx += 1

        cap.release()
        total_stats["videos_processed"] += 1
        total_stats["frames_extracted"] += frames_extracted

    return total_stats


# Alias for backwards compatibility
_extract_frames_single = extract_frames
