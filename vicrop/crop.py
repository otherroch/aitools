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

When ``output_type="video"``, the tool extracts video segments containing
only one person (one face) per segment.
"""

from __future__ import annotations

import logging
import math
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
DEFAULT_SEGMENT_LENGTH: float = 30.0
MIN_SEGMENT_LENGTH: float = 2.0


def _default_backend():
    """Create a default dlib backend when none is provided."""
    from face_ops import backend_for_model
    return backend_for_model("dlib")


def _crop_video_video_mode(
    video_path: Path,
    output_dir: Path,
    backend: FaceBackend,
    segment_length_secs: float,
    every_n: int,
    margin_ratio: float,
    crop_size: int,
    tolerance: float = 0.6,
    skip_existing: bool = True,
    classified_path: Path | None = None,
    classified_max: int = 0,
) -> dict:
    """Extract video segments with only one person per segment.
    
    Args:
        video_path: Path to input video.
        output_dir: Root output directory.
        backend: FaceBackend for detection/encoding.
        segment_length_secs: Max segment length in seconds.
        every_n: Sample every N-th frame.
        margin_ratio: Margin ratio around faces.
        crop_size: Output crop size.
        tolerance: Face distance threshold for clustering.
        skip_existing: Skip if output exists.
        classified_path: Pre-classified reference photos path.
        classified_max: Max reference images per identity.
    """
    output_dir = output_dir.resolve()
    video_stem_dir = output_dir / video_path.stem / "segments"
    video_stem_dir.mkdir(parents=True, exist_ok=True)
    
    if skip_existing and any(video_stem_dir.rglob("*.mp4")):
        logger.info("Skipping (already processed): %s", video_path.name)
        return {"videos_processed": 1, "segments": 0, "frames_processed": 0}
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return {"videos_processed": 1, "segments": 0, "frames_processed": 0}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_interval = max(1, int(every_n))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    segment_frames_secs = max(segment_length_secs, MIN_SEGMENT_LENGTH)
    max_frames_per_segment = int(segment_frames_secs * fps)
    
    logger.info(
        "Video %s: fps=%.2f, max_seg_frames=%d, total_frames=%d, every_n=%d",
        video_path.name, fps, max_frames_per_segment, total_frames, frame_interval
    )
    
    # Collect all face data: frame_idx -> list of (bbox, encoding, pil_img)
    all_face_data: dict[int, list] = {}
    frames_processed = 0
    
    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            if frame_idx % 1000 == 0:
                pct = (frame_idx * 100.0 / total_frames) if total_frames > 0 else 0
                logger.info("[%5.1f%%] frame %d / %d", pct, frame_idx, total_frames)
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                face_locations = backend.detect_faces(frame_rgb)
                face_encodings = backend.encode_faces(frame_rgb, face_locations)
                
                h_img, w_img = frame_rgb.shape[:2]
                frame_faces = []
                
                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    face_h = bottom - top
                    face_w = right - left
                    margin_h = int(face_h * margin_ratio)
                    margin_w = int(face_w * margin_ratio)
                    
                    crop_top = max(0, top - margin_h)
                    crop_bottom = min(h_img, bottom + margin_h)
                    crop_left = max(0, left - margin_w)
                    crop_right = min(w_img, right + margin_w)
                    
                    face_arr = frame_rgb[crop_top:crop_bottom, crop_left:crop_right]
                    pil_img = Image.fromarray(face_arr).resize((crop_size, crop_size), Image.LANCZOS)
                    frame_faces.append((crop_top, crop_right, crop_bottom, crop_left, encoding, pil_img))
                
                if face_locations:  # Only process frames with faces
                    all_face_data[frame_idx] = frame_faces
                frames_processed += 1
            
            frame_idx += 1
    finally:
        cap.release()
    
    # Cluster faces into identities using all results
    all_results = []
    for frame_idx, faces in all_face_data.items():
        for (top, right, bottom, left, encoding, pil_img) in faces:
            out_path = video_stem_dir / f"frame{frame_idx:06d}_face0.png"
            pil_img.save(out_path)
            all_results.append((out_path, encoding))
    
    # Use backend clustering
    ref_enc, ref_names = None, None
    if classified_path is not None:
        ref_enc, ref_names = backend.load_reference_encodings(classified_path, max_per_identity=classified_max)
    
    person_dirs = {}
    if all_results:
        person_dirs = backend.cluster_faces(
            all_results, video_stem_dir, tolerance=tolerance,
            reference_encodings=ref_enc, reference_names=ref_names,
        )
    
    # Build face_id -> person_id mapping from clustering
    face_to_person: dict[str, str] = {}
    person_id_counter = 0
    person_to_faces: dict[str, list] = {}
    
    if person_dirs:
        for pid, paths in person_dirs.items():
            if pid.startswith("person_"):
                actual_id = pid
            else:
                actual_id = pid
            person_id_counter += 1
            person_to_faces[actual_id] = []
            for p in paths:
                # Extract frame and face numbers from filename
                # frame000001_face1.png -> frame=000001, face=1
                stem = p.stem  # e.g. "frame000001_face1"
                parts = stem.rsplit("_face", 1)
                if len(parts) == 2:
                    face_key = f"{parts[0]}_{parts[1]}"
                    face_to_person[face_key] = actual_id
                    person_to_faces[actual_id].append((frame_idx, face_key))
    
    # Build segment groups: group frames where same person appears
    # For each person, find consecutive frame sequences
    segments_created = 0
    
    # Process each person's frames
    for person_id, frame_face_list in person_to_faces.items():
        if not frame_face_list:
            continue
        
        # Sort by frame index
        frame_face_list.sort(key=lambda x: int(x[0].replace("frame", "")))
        
        # Group into segments by time
        segments = _group_into_segments(frame_face_list, fps, max_frames_per_segment)
        
        for seg_frames_info in segments:
            if not seg_frames_info:
                continue
            
            seg_start_frame = seg_frames_info[0][0]
            seg_end_frame = seg_frames_info[-1][0]
            
            seg_name = f"{person_id}_seg{segments_created:04d}"
            seg_dir = video_stem_dir / seg_name
            seg_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames from video for this segment
            seg_cap = cv2.VideoCapture(str(video_path))
            if not seg_cap.isOpened():
                continue
            
            output_frames = []
            for frame_idx_val in seg_frames_info:
                seg_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_val)
                ret, frame_bgr = seg_cap.read()
                if ret:
                    output_frames.append(frame_bgr)
            
            if output_frames:
                out_video_path = seg_dir / f"{seg_name}.mp4"
                h, w = output_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
                for frame in output_frames:
                    writer.write(frame)
                writer.release()
                segments_created += 1
                logger.info("Created segment: %s (%d frames)", out_video_path, len(output_frames))
            
            seg_cap.release()
    
    return {
        "videos_processed": 1,
        "segments": segments_created,
        "frames_processed": frames_processed,
    }


def _group_into_segments(
    frame_face_list: list,
    fps: float,
    max_frames_per_segment: int
) -> list[list[int]]:
    """Group frame indices into segments based on max duration.
    
    Args:
        frame_face_list: List of (frame_idx, ...) tuples.
        fps: Frames per second.
        max_frames_per_segment: Max frames per segment.
    
    Returns:
        List of lists of frame indices.
    """
    if not frame_face_list:
        return []
    
    segments = []
    current_segment = [frame_face_list[0][0]]
    
    for i in range(1, len(frame_face_list)):
        frame_idx = frame_face_list[i][0]
        time_diff = (frame_idx - current_segment[0]) / fps if fps > 0 else 0
        
        if time_diff > max_frames_per_segment and len(current_segment) >= 2:
            segments.append(current_segment)
            current_segment = [frame_idx]
        else:
            current_segment.append(frame_idx)
    
    if current_segment:
        segments.append(current_segment)
    
    return segments


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
    segment_length: float = DEFAULT_SEGMENT_LENGTH,
) -> dict:
    """Extract face-cropped frames from a single video file.

    Args:
        video_path: Path to the input video file.
        output_dir: Root directory where cropped images are saved.
        every_n: Process every N-th frame (default: 30).
        margin_ratio: Fractional padding around each detected face bbox.
        crop_size: Output square resolution in pixels (default: 1024).
        classify: If True, cluster faces by identity.
        tolerance: Face-distance threshold for identity clustering.
        skip_existing: Skip the video if output already exists.
        ref_thresh: Minimum quality score for reference-photo selection.
        classified_path: Optional pre-classified reference photos path.
        classified_max: Max reference images per identity.
        backend: FaceBackend instance. When None, default dlib backend.
        output_type: "photo" for photos, "video" for video segments.
        segment_length: Max segment length in seconds (video mode only).

    Returns:
        Summary dict with results.
    """
    if backend is None:
        backend = _default_backend()

    if output_type == "video":
        return _crop_video_video_mode(
            video_path, output_dir, backend, segment_length,
            every_n, margin_ratio, crop_size, tolerance,
            skip_existing, classified_path, classified_max,
        )

    # Original photo mode
    output_dir = output_dir.resolve()
    video_stem_dir = output_dir / video_path.stem

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
    all_results = []
    ref_scores = {}

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
        ref_enc = None
        ref_names = None
        if classified_path is not None:
            ref_enc, ref_names = backend.load_reference_encodings(
                classified_path, max_per_identity=classified_max,
            )
        person_dirs = backend.cluster_faces(
            all_results, video_stem_dir, tolerance=tolerance,
            reference_encodings=ref_enc, reference_names=ref_names,
        )
        persons = len(person_dirs)
        try:
            staging_dir.rmdir()
        except OSError:
            pass

        if do_ref:
            for _pid, paths in person_dirs.items():
                ref_paths = []
                for p in paths:
                    if p.name in ref_scores and ref_scores[p.name] >= ref_thresh:
                        ref_paths.append(p)
                if ref_paths:
                    collect_ref_photos(ref_paths[0].parent, ref_paths)
                    total_refs += len(ref_paths)
    elif not classify and do_ref and all_results:
        ref_paths = []
        for path, _ in all_results:
            if path.name in ref_scores and ref_scores[path.name] >= ref_thresh:
                ref_paths.append(path)
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
    output_type: str = "photo",
    segment_length: float = DEFAULT_SEGMENT_LENGTH,
) -> dict:
    """Process all video files in input_dir, extracting face-cropped frames.

    Args:
        input_dir: Source directory (searched recursively).
        output_dir: Destination directory.
        every_n: Process every N-th frame from each video.
        margin_ratio: Fractional margin around each detected face bbox.
        crop_size: Output square resolution in pixels.
        classify: If True, cluster faces by identity.
        tolerance: Face-distance threshold for identity clustering.
        skip_existing: Skip videos whose output already has PNGs.
        ref_thresh: Minimum quality score for reference-photo selection.
        classified_path: Optional pre-classified reference photos path.
        classified_max: Max reference images per identity.
        backend: FaceBackend instance. When None, default dlib backend.
        output_type: "photo" or "video".
        segment_length: Max segment length in seconds (video mode only).

    Returns:
        Aggregate summary dict.
    """
    if backend is None:
        backend = _default_backend()

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ]

    if not videos:
        logger.warning("No video files found in %s", input_dir)
        return {
            "videos_processed": 0, "frames_processed": 0, "faces": 0,
            "persons": 0, "ref_photos": 0, "segments": 0,
        }

    logger.debug(
        "crop_folder: found %d video(s) in %s  every_n=%d classify=%s",
        len(videos), input_dir, every_n, classify,
    )

    total = {
        "videos_processed": 0, "frames_processed": 0, "faces": 0,
        "persons": 0, "ref_photos": 0, "segments": 0,
    }

    for video_path in videos:
        logger.info("Processing video: %s", video_path.name)
        stats = crop_video(
            video_path, output_dir,
            every_n=every_n, margin_ratio=margin_ratio, crop_size=crop_size,
            classify=classify, tolerance=tolerance, skip_existing=skip_existing,
            ref_thresh=ref_thresh, classified_path=classified_path,
            classified_max=classified_max, backend=backend,
            output_type=output_type, segment_length=segment_length,
        )
        total["videos_processed"] += 1
        total["frames_processed"] += stats.get("frames_processed", 0)
        total["faces"] += stats.get("faces", 0)
        total["persons"] += stats.get("persons", 0)
        total["ref_photos"] += stats.get("ref_photos", 0)
        total["segments"] += stats.get("segments", 0)

    return total