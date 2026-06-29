#!/usr/bin/env python3
"""
vicrop.segment

Extracts single-person video segments from a video file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vicrop.crop import SUPPORTED_VIDEO_EXTS, DEFAULT_EVERY_N_FRAMES

if TYPE_CHECKING:
    from face_ops.backend import FaceBackend

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEGMENT_LENGTH: float = 30.0
DEFAULT_MIN_SEGMENT_LENGTH: float = 2.0


def _default_backend() -> "FaceBackend":
    from face_ops import backend_for_model

    return backend_for_model("dlib")


# (top, right, bottom, left) — same convention as face_ops
_BBox = tuple[int, int, int, int]


class _Segment:
    """Internal holder for a candidate video segment."""

    __slots__ = ("start_frame", "end_frame", "anchor_enc", "person_id", "sample_bboxes")

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        anchor_enc: np.ndarray | None,
        person_id: int = 0,
        sample_bboxes: list[tuple[int, _BBox]] | None = None,
    ) -> None:
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.anchor_enc = anchor_enc
        self.person_id = person_id
        self.sample_bboxes: list[tuple[int, _BBox]] = sample_bboxes if sample_bboxes is not None else []


def _build_raw_segments(
    frame_records: list[tuple[int, np.ndarray | None, "_BBox | None"]],
    every_n: int,
    tolerance: float,
    backend: "FaceBackend",
    extract_only: bool = False,
) -> list[_Segment]:
    """Convert per-sampled-frame records into contiguous single-person segments.

    If extract_only is True, segments are not grouped by identity.
    """
    segments: list[_Segment] = []
    seg_start: int | None = None
    seg_end: int | None = None
    anchor_enc: np.ndarray | None = None
    seg_bboxes: list[tuple[int, _BBox]] = []

    def _close() -> None:
        if seg_start is not None and seg_end is not None:
            segments.append(
                _Segment(seg_start, seg_end, anchor_enc, sample_bboxes=list(seg_bboxes))
            )

    for frame_idx, enc, bbox in frame_records:
        if (not extract_only and (enc is None or bbox is None)) or bbox is None:
            # Not a single-person frame — close any open segment.
            _close()
            seg_start = seg_end = anchor_enc = None
            seg_bboxes = []
        elif extract_only:
            # If extract_only, we just create segments for every detected face.
            # But since we don't have encodings, we can't group them.
            # To keep it simple, we treat each frame as its own segment? 
            # Or we group them if they are contiguous.
            if seg_start is None:
                seg_start = frame_idx
                seg_end = frame_idx + every_n - 1
                anchor_enc = None
                seg_bboxes = [(frame_idx, bbox)]
            else:
                # Extend current segment
                seg_end = frame_idx + every_n - 1
                seg_bboxes.append((frame_idx, bbox))
        elif anchor_enc is None:
            # Start a new segment.
            seg_start = frame_idx
            seg_end = frame_idx + every_n - 1
            anchor_enc = enc
            seg_bboxes = [(frame_idx, bbox)]
        else:
            dists = backend.face_distance([anchor_enc], enc)
            if dists[0] <= tolerance:
                # Same person — extend the current segment's window.
                seg_end = frame_idx + every_n - 1
                seg_bboxes.append((frame_idx, bbox))
            else:
                # Different person — close current segment, start a new one.
                _close()
                seg_start = frame_idx
                seg_end = frame_idx + every_n - 1
                anchor_enc = enc
                seg_bboxes = [(frame_idx, bbox)]

    _close()
    return segments


def _filter_and_split_segments(
    segments: list[_Segment],
    fps: float,
    total_frames: int,
    min_segment_length: float,
    max_segment_length: float,
) -> list[_Segment]:
    """Filter segments that are too short and split those that are too long."""
    min_frames = max(1, int(min_segment_length * fps))
    max_frames = max(1, int(max_segment_length * fps))

    result: list[_Segment] = []
    for seg in segments:
        start = seg.start_frame
        end = min(seg.end_frame, total_frames - 1)
        enc = seg.anchor_enc
        bboxes = seg.sample_bboxes

        while start <= end:
            chunk_end = min(start + max_frames - 1, end)
            if chunk_end - start + 1 >= min_frames:
                chunk_bboxes = [(fi, b) for fi, b in bboxes if start <= fi <= chunk_end]
                result.append(_Segment(start, chunk_end, enc, sample_bboxes=chunk_bboxes))
            start = chunk_end + 1

    return result


def _compute_crop_rect(
    sample_bboxes: list[tuple[int, "_BBox"]],
    margin_ratio: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Compute the crop rectangle that covers all face bboxes in a segment."""
    if not sample_bboxes:
        return (0, 0, frame_height, frame_width)

    bboxes = [b for _, b in sample_bboxes]
    top = min(b[0] for b in bboxes)
    right = max(b[1] for b in bboxes)
    bottom = max(b[2] for b in bboxes)
    left = min(b[3] for b in bboxes)

    face_h = max(1, bottom - top)
    face_w = max(1, right - left)
    margin_h = int(face_h * margin_ratio)
    margin_w = int(face_w * margin_ratio)

    crop_top = max(0, top - margin_h)
    crop_bottom = min(frame_height, bottom + margin_h)
    crop_left = max(0, left - margin_w)
    crop_right = min(frame_width, right + margin_w)

    return (crop_top, crop_left, crop_bottom, crop_right)


def _assign_person_ids(
    segments: list[_Segment],
    tolerance: float,
    backend: "FaceBackend",
) -> None:
    """Assign a ``person_id`` to each segment via greedy encoding clustering."""
    known_encs: list[np.ndarray] = []
    known_ids: list[int] = []

    for seg in segments:
        if seg.anchor_enc is None:
            # For extract_only segments
            seg.person_id = 0
            continue
            
        if not known_encs:
            seg.person_id = 1
            known_encs.append(seg.anchor_enc)
            known_ids.append(1)
        else:
            dists = backend.face_distance(known_encs, seg.anchor_enc)
            best = int(np.argmin(dists))
            if dists[best] <= tolerance:
                seg.person_id = known_ids[best]
            else:
                new_id = max(known_ids) + 1
                seg.person_id = new_id
                known_encs.append(seg.anchor_enc)
                known_ids.append(new_id)


def segment_video(
    video_path: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = 0.4,
    crop_size: int | None = None,
    crop_dim: tuple[int, int] | None = None,
    extract_only: bool = False,
    tolerance: float = 0.6,
    min_segment_length: float = DEFAULT_MIN_SEGMENT_LENGTH,
    max_segment_length: float = DEFAULT_MAX_SEGMENT_LENGTH,
    skip_existing: bool = True,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """Extract single-person video segments from *video_path*."""
    if backend is None:
        backend = _default_backend()

    output_dir = output_dir.resolve()
    video_stem_dir = output_dir / video_path.stem

    if skip_existing and video_stem_dir.exists() and any(video_stem_dir.rglob("*.mp4")):
        logger.info("Skipping (already processed): %s", video_path.name)
        return {"segments": 0, "persons": 0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return {"videos_processed": 0, "segments": 0, "persons": 0}

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "segment_video: %s  fps=%.2f  frames=%d  %dx%d",
        video_path.name, fps, total_frames, width, height,
    )

    frame_records: list[tuple[int, np.ndarray | None, _BBox | None]] = []
    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                locs = backend.detect_faces(frame_rgb)
                if len(locs) == 1:
                    if extract_only:
                        enc = None
                    else:
                        encs = backend.encode_faces(frame_rgb, locs)
                        enc = encs[0] if encs else None
                    bbox: _BBox | None = locs[0]
                else:
                    enc = None
                    bbox = None
                frame_records.append((frame_idx, enc, bbox))
            frame_idx += 1
    finally:
        cap.release()

    if total_frames <= 0:
        total_frames = frame_idx

    if not frame_records:
        logger.info("No frames sampled from %s", video_path.name)
        return {"videos_processed": 1, "segments": 0, "persons": 0}

    raw = _build_raw_segments(frame_records, every_n, tolerance, backend, extract_only=extract_only)
    segments = _filter_and_split_segments(
        raw, fps, total_frames, min_segment_length, max_segment_length
    )

    if not segments:
        logger.info("No qualifying segments found in %s", video_path.name)
        return {"videos_processed": 1, "segments": 0, "persons": 0}

    if not extract_only:
        _assign_person_ids(segments, tolerance, backend)
    else:
        # For extract_only, we don't assign real person IDs, just 0.
        for seg in segments:
            seg.person_id = 0

    video_stem_dir.mkdir(parents=True, exist_ok=True)
    seg_count_per_person: dict[int, int] = {}
    written = 0

    cap2 = cv2.VideoCapture(str(video_path))
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for seg in segments:
            pid = seg.person_id
            # If extract_only, we use a special folder or just the stem dir
            if extract_only:
                person_dir = video_stem_dir / "extracted"
            else:
                seg_count_per_person[pid] = seg_count_per_person.get(pid, 0) + 1
                seg_num = seg_count_per_person[pid]
                person_dir = video_stem_dir / f"person_{pid:02d}"
                seg_num_str = f"seg_{seg_num:03d}"
            
            if extract_only:
                seg_num_str = f"frame_{seg.start_frame:06d}"
            else:
                seg_num_str = f"seg_{seg_num:03d}"

            person_dir.mkdir(parents=True, exist_ok=True)
            out_path = person_dir / f"{seg_num_str}.mp4"

            crop_top, crop_left, crop_bottom, crop_right = _compute_crop_rect(
                seg.sample_bboxes, margin_ratio, width, height
            )
            
            if crop_dim:
                out_w, out_h = crop_dim
            elif crop_size:
                out_w, out_h = crop_size, crop_size
            else:
                out_w = max(1, crop_right - crop_left)
                out_h = max(1, crop_bottom - crop_top)

            cap2.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
            try:
                for _ in range(seg.end_frame - seg.start_frame + 1):
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    cropped = frame[crop_top:crop_bottom, crop_left:crop_right]
                    
                    # Handle resizing
                    if crop_dim:
                        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                    elif crop_size:
                        cropped = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
                    
                    # Ensure size matches exactly (in case of rounding)
                    if (cropped.shape[1], cropped.shape[0]) != (out_w, out_h):
                        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                    writer.write(cropped)
            finally:
                writer.release()

            written += 1

    finally:
        cap2.release()

    return {
        "videos_processed": 1,
        "frames_processed": frame_idx,
        "segments": written,
        "persons": len(seg_count_per_person) if not extract_only else 1,
    }


def segment_folder(
    input_dir: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = 0.4,
    crop_size: int | None = None,
    crop_dim: tuple[int, int] | None = None,
    extract_only: bool = False,
    tolerance: float = 0.6,
    min_segment_length: float = DEFAULT_MIN_SEGMENT_LENGTH,
    max_segment_length: float = DEFAULT_MAX_SEGMENT_LENGTH,
    skip_existing: bool = True,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """Scan a directory for videos and segment each one."""
    input_dir = input_dir.resolve()
    video_files = set()
    for ext in SUPPORTED_VIDEO_EXTS:
        for path in input_dir.rglob(f"*{ext}"):
            video_files.add(path.resolve())
        for path in input_dir.rglob(f"*{ext.upper()}"):
            video_files.add(path.resolve())

    stats = {
        "videos_processed": 0,
        "frames_processed": 0,
        "segments": 0,
        "persons": 0,
    }

    for video_path in video_files:
        video_stats = segment_video(
            video_path,
            output_dir,
            every_n=every_n,
            margin_ratio=margin_ratio,
            crop_size=crop_size,
            crop_dim=crop_dim,
            extract_only=extract_only,
            tolerance=tolerance,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            skip_existing=skip_existing,
            backend=backend,
        )
        stats["videos_processed"] += video_stats.get("videos_processed", 0)
        stats["frames_processed"] += video_stats.get("frames_processed", 0)
        stats["segments"] += video_stats.get("segments", 0)
        stats["persons"] += video_stats.get("persons", 0)

    return stats
