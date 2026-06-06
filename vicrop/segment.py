#!/usr/bin/env python3
"""
vicrop.segment

Extract single-person video segments from video files.

For each input video, scans frames at a configurable interval to detect
faces, identifies contiguous runs of frames that contain exactly one unique
person, and writes each qualifying run as a separate MP4 file under

    ``output_dir / <video_stem> / person_<NN> / seg_<NNN>.mp4``

Segments shorter than *min_segment_length* seconds are discarded.
Segments longer than *max_segment_length* seconds are split into
consecutive chunks at that boundary.
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


class _Segment:
    """Internal holder for a candidate video segment."""

    __slots__ = ("start_frame", "end_frame", "anchor_enc", "person_id")

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        anchor_enc: np.ndarray,
        person_id: int = 0,
    ) -> None:
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.anchor_enc = anchor_enc
        self.person_id = person_id


def _build_raw_segments(
    frame_records: list[tuple[int, np.ndarray | None]],
    every_n: int,
    tolerance: float,
    backend: "FaceBackend",
) -> list[_Segment]:
    """Convert per-sampled-frame records into contiguous single-person segments.

    Each entry in *frame_records* is ``(frame_idx, encoding_or_None)`` where
    *encoding_or_None* is ``None`` when the frame did not contain exactly one
    face.  The function groups consecutive single-face records whose face
    encodings match the segment's anchor within *tolerance* into a single
    :class:`_Segment`.

    The ``end_frame`` of each segment is set to
    ``last_good_sampled_frame + every_n - 1`` so that the unseen frames
    between the last good sample and the next sample are included.

    Args:
        frame_records: List of ``(frame_idx, encoding)`` pairs from the
                       analysis pass.
        every_n:       Frame sampling interval used during analysis.
        tolerance:     Face-distance threshold for same-person matching.
        backend:       :class:`FaceBackend` used for distance computation.

    Returns:
        List of raw (unfiltered, unsplit) :class:`_Segment` objects.
    """
    segments: list[_Segment] = []
    seg_start: int | None = None
    seg_end: int | None = None
    anchor_enc: np.ndarray | None = None

    def _close() -> None:
        if seg_start is not None and seg_end is not None and anchor_enc is not None:
            segments.append(_Segment(seg_start, seg_end, anchor_enc.copy()))

    for frame_idx, enc in frame_records:
        if enc is None:
            # Not a single-person frame — close any open segment.
            _close()
            seg_start = seg_end = anchor_enc = None
        elif anchor_enc is None:
            # Start a new segment.
            seg_start = frame_idx
            seg_end = frame_idx + every_n - 1
            anchor_enc = enc
        else:
            dists = backend.face_distance([anchor_enc], enc)
            if dists[0] <= tolerance:
                # Same person — extend the current segment's window.
                seg_end = frame_idx + every_n - 1
            else:
                # Different person — close current segment, start a new one.
                _close()
                seg_start = frame_idx
                seg_end = frame_idx + every_n - 1
                anchor_enc = enc

    _close()
    return segments


def _filter_and_split_segments(
    segments: list[_Segment],
    fps: float,
    total_frames: int,
    min_segment_length: float,
    max_segment_length: float,
) -> list[_Segment]:
    """Filter segments that are too short and split those that are too long.

    *end_frame* is clipped to ``total_frames - 1``.  Segments shorter than
    ``min_segment_length * fps`` frames (after clipping) are discarded.
    Segments longer than ``max_segment_length * fps`` frames are split into
    consecutive chunks; trailing chunks shorter than *min_frames* are also
    discarded.

    Args:
        segments:           Raw segments from :func:`_build_raw_segments`.
        fps:                Frames per second of the source video.
        total_frames:       Total frame count of the source video.
        min_segment_length: Minimum duration in seconds.
        max_segment_length: Maximum duration in seconds.

    Returns:
        Filtered and split list of :class:`_Segment` objects.
    """
    min_frames = max(1, int(min_segment_length * fps))
    max_frames = max(1, int(max_segment_length * fps))

    result: list[_Segment] = []
    for seg in segments:
        start = seg.start_frame
        end = min(seg.end_frame, total_frames - 1)
        enc = seg.anchor_enc

        while start <= end:
            chunk_end = min(start + max_frames - 1, end)
            if chunk_end - start + 1 >= min_frames:
                result.append(_Segment(start, chunk_end, enc))
            start = chunk_end + 1

    return result


def _assign_person_ids(
    segments: list[_Segment],
    tolerance: float,
    backend: "FaceBackend",
) -> None:
    """Assign a ``person_id`` to each segment via greedy encoding clustering.

    Segments sharing the same visual identity (within *tolerance*) receive the
    same ID.  IDs are assigned in order of first appearance starting from 1.
    The assignment is done **in-place**.

    Args:
        segments:  Segments to label.
        tolerance: Face-distance threshold for same-person matching.
        backend:   :class:`FaceBackend` used for distance computation.
    """
    known_encs: list[np.ndarray] = []
    known_ids: list[int] = []

    for seg in segments:
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
    tolerance: float = 0.6,
    min_segment_length: float = DEFAULT_MIN_SEGMENT_LENGTH,
    max_segment_length: float = DEFAULT_MAX_SEGMENT_LENGTH,
    skip_existing: bool = True,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """Extract single-person video segments from *video_path*.

    Scans the video at *every_n*-frame intervals, identifies contiguous runs
    of frames containing exactly one unique person, and writes each qualifying
    run as a separate MP4 file under
    ``output_dir / video_path.stem / person_NN / seg_NNN.mp4``.

    Args:
        video_path:          Path to the input video file.
        output_dir:          Root directory for output segments.
        every_n:             Frame sampling interval for face detection.
        tolerance:           Face-distance threshold for same-person matching.
        min_segment_length:  Minimum segment duration in seconds (default: 2).
        max_segment_length:  Maximum segment duration in seconds; longer
                             segments are split at this boundary (default: 30).
        skip_existing:       Skip the video when its output sub-directory
                             already contains MP4 files.
        backend:             :class:`FaceBackend` instance.  *None* creates a
                             default dlib backend.

    Returns:
        Summary dict with keys ``segments`` (number of MP4 files written)
        and ``persons`` (number of distinct person identities found).
    """
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
        return {"segments": 0, "persons": 0}

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "segment_video: %s  fps=%.2f  frames=%d  %dx%d",
        video_path.name, fps, total_frames, width, height,
    )

    # ------------------------------------------------------------------ #
    # Analysis pass — sample every_n frames to build a face-presence      #
    # timeline.  Each record is (frame_idx, encoding_or_None) where the   #
    # encoding is None unless exactly one face was detected.               #
    # ------------------------------------------------------------------ #
    frame_records: list[tuple[int, np.ndarray | None]] = []
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
                    encs = backend.encode_faces(frame_rgb, locs)
                    enc: np.ndarray | None = encs[0] if encs else None
                else:
                    enc = None
                frame_records.append((frame_idx, enc))
            frame_idx += 1
    finally:
        cap.release()

    # Fall back to the observed frame count if CAP_PROP_FRAME_COUNT was 0.
    if total_frames <= 0:
        total_frames = frame_idx

    if not frame_records:
        logger.info("No frames sampled from %s", video_path.name)
        return {"segments": 0, "persons": 0}

    # ------------------------------------------------------------------ #
    # Build, filter, and label segments.                                   #
    # ------------------------------------------------------------------ #
    raw = _build_raw_segments(frame_records, every_n, tolerance, backend)
    segments = _filter_and_split_segments(
        raw, fps, total_frames, min_segment_length, max_segment_length
    )

    if not segments:
        logger.info("No qualifying segments found in %s", video_path.name)
        return {"segments": 0, "persons": 0}

    _assign_person_ids(segments, tolerance, backend)

    # ------------------------------------------------------------------ #
    # Write pass — seek to each segment's start frame and copy frames.    #
    # ------------------------------------------------------------------ #
    video_stem_dir.mkdir(parents=True, exist_ok=True)
    seg_count_per_person: dict[int, int] = {}
    written = 0

    cap2 = cv2.VideoCapture(str(video_path))
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for seg in segments:
            pid = seg.person_id
            seg_count_per_person[pid] = seg_count_per_person.get(pid, 0) + 1
            seg_num = seg_count_per_person[pid]

            person_dir = video_stem_dir / f"person_{pid:02d}"
            person_dir.mkdir(parents=True, exist_ok=True)
            out_path = person_dir / f"seg_{seg_num:03d}.mp4"

            cap2.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            try:
                for _ in range(seg.end_frame - seg.start_frame + 1):
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    writer.write(frame)
            finally:
                writer.release()

            duration = (seg.end_frame - seg.start_frame + 1) / fps
            logger.info(
                "Wrote segment: %s  frames %d–%d  (%.1fs)  person %d",
                out_path.name, seg.start_frame, seg.end_frame, duration, pid,
            )
            written += 1
    finally:
        cap2.release()

    return {"segments": written, "persons": len(seg_count_per_person)}


def segment_folder(
    input_dir: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    tolerance: float = 0.6,
    min_segment_length: float = DEFAULT_MIN_SEGMENT_LENGTH,
    max_segment_length: float = DEFAULT_MAX_SEGMENT_LENGTH,
    skip_existing: bool = True,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """Process all videos in *input_dir*, extracting single-person segments.

    Args:
        input_dir:           Source directory (searched recursively).
        output_dir:          Destination directory for segments.
        every_n:             Frame sampling interval for face detection.
        tolerance:           Face-distance threshold for same-person matching.
        min_segment_length:  Minimum segment duration in seconds.
        max_segment_length:  Maximum segment duration in seconds.
        skip_existing:       Skip videos whose output sub-directory already
                             contains MP4 files.
        backend:             :class:`FaceBackend` instance.

    Returns:
        Aggregate summary dict with keys ``videos_processed``, ``segments``,
        and ``persons``.
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
        return {"videos_processed": 0, "segments": 0, "persons": 0}

    total: dict[str, int] = {"videos_processed": 0, "segments": 0, "persons": 0}

    for video_path in videos:
        logger.info("Processing video: %s", video_path.name)
        stats = segment_video(
            video_path,
            output_dir,
            every_n=every_n,
            tolerance=tolerance,
            min_segment_length=min_segment_length,
            max_segment_length=max_segment_length,
            skip_existing=skip_existing,
            backend=backend,
        )
        total["videos_processed"] += 1
        total["segments"] += stats["segments"]
        total["persons"] += stats["persons"]

    return total
