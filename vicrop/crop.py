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

if TYPE_CHECKING:
    from face_ops.backend import FaceBackend

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_RATIO: float = 0.4
DEFAULT_CROP_SIZE: int = 1024

SUPPORTED_VIDEO_EXTS: set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv",
}

DEFAULT_EVERY_N_FRAMES: int = 30


def _default_backend():
    """Create a default dlib backend when none is provided."""
    from face_ops import backend_for_model

    return backend_for_model("dlib")


MIN_SEGMENT_LENGTH: float = 2.0  # seconds


def _crop_video_segments(
    video_path: Path,
    output_dir: Path,
    backend: FaceBackend,
    segment_length: float = 30.0,
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate video segments where each segment contains a single unique person.

    The approach reads the input video frame-by-frame, detects faces, assigns each
    face to a person identity (via clustering), and groups consecutive frames
    belonging to the same person into segments.  When the dominant person changes
    or the maximum *segment_length* (in seconds) is exceeded, the current segment
    is written as an MP4 file and a new segment begins.

    Args:
        video_path:      Path to the input video file.
        output_dir:      Directory where segment MP4s are saved.
        backend:         FaceBackend instance for detection & encoding.
        segment_length:  Maximum segment duration in seconds (default: 30).
        classify:        When True, cluster faces to resolve identities.
        tolerance:       Face-distance threshold for identity clustering.
        skip_existing:   Skip if the output directory already contains .mp4 files.

    Returns:
        Summary dict with keys ``frames_processed``, ``faces``, ``persons``,
        ``ref_photos``, ``segments``.
    """
    segment_length = max(segment_length, MIN_SEGMENT_LENGTH)

    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and any(output_dir.rglob("*.mp4")):
        logger.info("Skipping (already processed): %s", video_path.name)
        return {"frames_processed": 0, "faces": 0, "persons": 0, "ref_photos": 0, "segments": 0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return {"frames_processed": 0, "faces": 0, "persons": 0, "ref_photos": 0, "segments": 0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        logger.warning("Unable to determine FPS for %s; assuming 30 fps", video_path.name)
        fps = 30.0

    max_frames_per_segment = int(segment_length * fps)
    logger.info(
        "Video %s: %s fps, %d×%d, %d total frames → max %d frames/segment (%.1fs)",
        video_path.name, fps, w, h, total_frames, max_frames_per_segment, segment_length,
    )

    # ---- Phase 1: collect per-frame face info (frame_idx → list of encodings) ----
    logger.info("Phase 1: detecting faces in %s …", video_path.name)
    frame_faces: list[list[np.ndarray]] = []  # frame_idx → [encoding, …]

    frame_idx = 0
    frames_processed = 0
    total_faces = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_locations = backend.detect_faces(frame_rgb)
        face_encodings = backend.encode_faces(frame_rgb, face_locations)

        if face_encodings:
            frame_faces.append(face_encodings)
            total_faces += len(face_encodings)
        else:
            frame_faces.append([])  # no faces

        if frame_idx % every_n_opt == 0:
            frames_processed += 1

        if frame_idx % 1000 == 0:
            pct = frame_idx * 100.0 / total_frames if total_frames else 0
            logger.info("[%5.1f%%] frame %d  faces so far: %d", pct, frame_idx, total_faces)

        frame_idx += 1

    cap.release()
    logger.info(
        "Phase 1 done: %d frames read, %d faces total",
        len(frame_faces), total_faces,
    )

    if not total_faces:
        logger.info("No faces found in %s — no segments to write.", video_path.name)
        return {"frames_processed": frames_processed, "faces": 0, "persons": 0, "ref_photos": 0, "segments": 0}

    # ---- Phase 2: assign identities via greedy clustering ----
    logger.info("Phase 2: assigning identities (tolerance=%.2f) …", tolerance)
    # Build a flat list of (frame_idx, face_idx, encoding)
    flat: list[tuple[int, int, np.ndarray]] = []
    for fi, encs in enumerate(frame_faces):
        for j, enc in enumerate(encs):
            flat.append((fi, j, enc))

    # Map: person_id → sorted list of (frame_idx, face_idx)
    person_assignment: dict[int, list[tuple[int, int]]] = {}
    # Also per-frame dominant person
    frame_person: list[int | None] = [None] * len(frame_faces)
    next_person_id = 0
    person_enc_seeds: list[np.ndarray] = []  # person_id → seed encoding

    for fi, j, enc in flat:
        assigned = False
        for pid in range(next_person_id):
            if np.linalg.norm(enc - person_enc_seeds[pid]) <= tolerance:
                person_assignment.setdefault(pid, []).append((fi, j))
                if frame_person[fi] is None:
                    frame_person[fi] = pid
                assigned = True
                break
        if not assigned:
            person_assignment[next_person_id] = [(fi, j)]
            person_enc_seeds.append(enc)
            frame_person[fi] = next_person_id
            next_person_id += 1

    # For frames with multiple faces, pick the one whose person appears most in that frame
    for fi, encs in enumerate(frame_faces):
        if len(encs) == 0:
            continue
        # count person occurrences in this frame
        person_counts: dict[int, int] = {}
        for j, enc in enumerate(encs):
            pid = frame_person[fi] if frame_person[fi] is not None else _find_person(enc, person_enc_seeds, person_assignment, tolerance)
            if pid is not None:
                person_counts[pid] = person_counts.get(pid, 0) + 1
        if person_counts:
            frame_person[fi] = max(person_counts, key=person_counts.get)

    num_persons = next_person_id
    logger.info("Phase 2 done: %d unique persons identified", num_persons)

    # ---- Phase 3: split into segments & write MP4 ----
    logger.info("Phase 3: writing video segments …")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    segments_written = 0
    segment_start = 0
    segment_person = frame_person[0]
    segment_frame_count = 0

    def _flush_segment(end_exclusive: int) -> None:
        nonlocal segments_written
        seg_frames = end_exclusive - segment_start
        if seg_frames < int(MIN_SEGMENT_LENGTH * fps):
            return  # too short
        seg_person_id = segment_person
        seg_label = f"person_{seg_person_id:02d}" if seg_person_id is not None else "no_face"
        out_name = f"{seg_label}_{segments_written:04d}.mp4"
        out_path = output_dir / out_name
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        cap2 = cv2.VideoCapture(str(video_path))
        for _ in range(segment_start):
            cap2.read()
        for _ in range(seg_frames):
            ret, frame = cap2.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        cap2.release()
        segments_written += 1
        logger.debug("Segment %s written (%d frames, %.1fs)", out_name, seg_frames, seg_frames / fps)

    for fi in range(1, len(frame_person)):
        segment_frame_count = fi - segment_start
        if (
            frame_person[fi] != segment_person
            or segment_frame_count >= max_frames_per_segment
        ):
            _flush_segment(fi)
            segment_start = fi
            segment_person = frame_person[fi]
            segment_frame_count = 0

    # flush last segment
    _flush_segment(len(frame_person))

    logger.info("Phase 3 done: %d segments written to %s", segments_written, output_dir)

    return {
        "frames_processed": frames_processed,
        "faces": total_faces,
        "persons": num_persons,
        "ref_photos": 0,
        "segments": segments_written,
    }


def _find_person(
    enc: np.ndarray,
    seeds: list[np.ndarray],
    assignment: dict[int, list[tuple[int, int]]],
    tolerance: float,
) -> int | None:
    """Return the person id whose seed encoding is closest to *enc* within *tolerance*."""
    best_pid = None
    best_dist = tolerance + 1
    for pid, seed in enumerate(seeds):
        d = np.linalg.norm(enc - seed)
        if d < best_dist:
            best_dist = d
            best_pid = pid
    return best_pid


# every_n for segment mode: read every frame (every_n=1)
every_n_opt = 1


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
    backend: FaceBackend | None = None,
    output_type: str = "photo",
    segment_length: float = 30.0,
) -> dict[str, int]:
    """Extract face-cropped frames from a single video file.

    Frames are sampled every *every_n* frames.  Detected faces are cropped with
    a fractional *margin_ratio* padding, resized to *crop_size* × *crop_size*,
    and saved as PNG files inside a sub-directory named after the video stem.

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
        ref_thresh:      Minimum quality score (0–1) for a face crop to be
                         listed as a reference photo.  ``0`` disables the
                         analysis entirely.
        classified_path: Optional path to a directory of pre-classified
                         reference photos used to seed identity clustering.
        classified_max:  Maximum reference images to load per identity.
                         ``0`` means no limit.
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
        "crop_video: %s  every_n=%d margin_ratio=%.2f crop_size=%d classify=%s output_type=%s",
        video_path.name, every_n, margin_ratio, crop_size, classify, output_type,
    )

    # Video output mode – delegate to the segment generator
    if output_type == "video":
        return _crop_video_segments(
            video_path=video_path,
            output_dir=video_stem_dir,
            backend=backend,
            segment_length=segment_length,
            classify=classify,
            tolerance=tolerance,
            skip_existing=skip_existing,
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
                            name = out_name if debug_logging else None,
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
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = True,
    ref_thresh: float = DEFAULT_REF_THRESH,
    classified_path: Path | None = None,
    classified_max: int = 0,
    backend: FaceBackend | None = None,
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
        ref_thresh:      Minimum quality score (0–1) for reference-photo
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
