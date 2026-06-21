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


# (top, right, bottom, left) — same convention as face_ops
_BBox = tuple[int, int, int, int]


class _Segment:
    """Internal holder for a candidate video segment."""

    __slots__ = ("start_frame", "end_frame", "anchor_enc", "person_id", "sample_bboxes")

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        anchor_enc: np.ndarray,
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
) -> list[_Segment]:
    """Convert per-sampled-frame records into contiguous single-person segments.

    Debug logging enabled via ``logger.debug()`` for each frame record and
    segment boundary decisions.

    Each entry in *frame_records* is ``(frame_idx, encoding_or_None, bbox_or_None)``
    where *encoding_or_None* is ``None`` when the frame did not contain exactly
    one face.  The function groups consecutive single-face records whose face
    encodings match the segment's anchor within *tolerance* into a single
    :class:`_Segment`.

    The ``end_frame`` of each segment is set to
    ``last_good_sampled_frame + every_n - 1`` so that the unseen frames
    between the last good sample and the next sample are included.

    Args:
        frame_records: List of ``(frame_idx, encoding, bbox)`` triples from
                       the analysis pass.  *encoding* and *bbox* are ``None``
                       unless exactly one face was detected.
        every_n:       Frame sampling interval used during analysis.
        tolerance:     Face-distance threshold for same-person matching.
        backend:       :class:`FaceBackend` used for distance computation.

    Returns:
        List of raw (unfiltered, unsplit) :class:`_Segment` objects.
    """
    debug_logging = logger.isEnabledFor(logging.DEBUG)

    segments: list[_Segment] = []
    seg_start: int | None = None
    seg_end: int | None = None
    anchor_enc: np.ndarray | None = None
    seg_bboxes: list[tuple[int, _BBox]] = []

    def _close() -> None:
        if seg_start is not None and seg_end is not None and anchor_enc is not None:
            duration_frames = seg_end - seg_start + 1
            logger.debug(
                "_build_raw_segments: closing segment frames %d–%d (%d frames, %d samples)",
                seg_start, seg_end, duration_frames, len(seg_bboxes),
            )
            segments.append(
                _Segment(seg_start, seg_end, anchor_enc.copy(), sample_bboxes=list(seg_bboxes))
            )

    for frame_idx, enc, bbox in frame_records:
        if debug_logging:
            logger.debug(
                "_build_raw_segments: frame %d  enc=%s  bbox=%s",
                frame_idx, enc is not None, bbox,
            )

        if enc is None or bbox is None:
            # Not a single-person frame — close any open segment.
            logger.debug(
                "_build_raw_segments: frame %d no single-face → closing segment (if open)",
                frame_idx,
            )
            _close()
            seg_start = seg_end = anchor_enc = None
            seg_bboxes = []
        elif anchor_enc is None:
            # Start a new segment.
            logger.debug(
                "_build_raw_segments: frame %d starting new segment",
                frame_idx,
            )
            seg_start = frame_idx
            seg_end = frame_idx + every_n - 1
            anchor_enc = enc
            seg_bboxes = [(frame_idx, bbox)]
        else:
            dists = backend.face_distance([anchor_enc], enc)
            same_person = dists[0] <= tolerance
            logger.debug(
                "_build_raw_segments: frame %d face dist %.4f same_person=%s",
                frame_idx, dists[0], same_person,
            )
            if same_person:
                # Same person — extend the current segment's window.
                seg_end = frame_idx + every_n - 1
                seg_bboxes.append((frame_idx, bbox))
            else:
                # Different person — close current segment, start a new one.
                logger.debug(
                    "_build_raw_segments: frame %d different person → new segment",
                    frame_idx,
                )
                _close()
                seg_start = frame_idx
                seg_end = frame_idx + every_n - 1
                anchor_enc = enc
                seg_bboxes = [(frame_idx, bbox)]

    _close()
    logger.info(
        "_build_raw_segments: built %d raw segment(s) from %d sampled frames",
        len(segments), len(frame_records),
    )
    return segments


def _filter_and_split_segments(
    segments: list[_Segment],
    fps: float,
    total_frames: int,
    min_segment_length: float,
    max_segment_length: float,
) -> list[_Segment]:
    """Filter segments that are too short and split those that are too long.

    Debug logging reports each discarded (too-short) segment and each split
    chunk produced.

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
    debug_logging = logger.isEnabledFor(logging.DEBUG)

    logger.debug(
        "_filter_and_split_segments: min_frames=%d  max_frames=%d  input segments=%d",
        min_frames, max_frames, len(segments),
    )

    result: list[_Segment] = []
    for seg in segments:
        start = seg.start_frame
        end = min(seg.end_frame, total_frames - 1)

        if end < start:
            logger.debug(
                "_filter_and_split_segments: segment frames %d–%d clamped (end > total_frames), skipping",
                start, seg.end_frame,
            )
            continue

        seg_duration_s = (end - start + 1) / fps
        if seg_duration_s < min_segment_length:
            logger.debug(
                "_filter_and_split_segments: segment frames %d–%d (%.2fs) discarded as too short (min %.2fs)",
                start, end, seg_duration_s, min_segment_length,
            )
            continue

        enc = seg.anchor_enc
        bboxes = seg.sample_bboxes

        while start <= end:
            chunk_end = min(start + max_frames - 1, end)
            chunk_frames = chunk_end - start + 1
            if chunk_end - start + 1 >= min_frames:
                chunk_bboxes = [(fi, b) for fi, b in bboxes if start <= fi <= chunk_end]
                chunk_dur = chunk_frames / fps
                logger.debug(
                    "_filter_and_split_segments: chunk frames %d–%d (%.1fs) accepted",
                    start, chunk_end, chunk_dur,
                )
                result.append(_Segment(start, chunk_end, enc, sample_bboxes=chunk_bboxes))
            else:
                logger.debug(
                    "_filter_and_split_segments: trailing chunk frames %d–%d (%d frames) too short, discarding",
                    start, chunk_end, chunk_frames,
                )
            start = chunk_end + 1

    logger.info(
        "_filter_and_split_segments: %d input segments → %d output segments",
        len(segments), len(result),
    )
    return result


def _compute_crop_rect(
    sample_bboxes: list[tuple[int, "_BBox"]],
    margin_ratio: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Compute the crop rectangle that covers all face bboxes in a segment.

    Debug logging reports the union bbox and final crop rect.

    Takes the union of all sampled face bounding boxes, expands it by
    *margin_ratio* on each side, and clamps to the frame dimensions.

    Args:
        sample_bboxes:  List of ``(frame_idx, (top, right, bottom, left))``
                        pairs from the segment.
        margin_ratio:   Fractional padding to add around the union bbox.
        frame_width:    Source video frame width in pixels.
        frame_height:   Source video frame height in pixels.

    Returns:
        ``(crop_top, crop_left, crop_bottom, crop_right)`` — all clamped to
        the frame boundaries.
    """
    if not sample_bboxes:
        logger.debug("_compute_crop_rect: no bboxes, returning full frame")
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

    logger.debug(
        "_compute_crop_rect: union_bbox (%d,%d,%d,%d) margin=(%d,%d) crop (%d,%d,%d,%d)",
        top, right, bottom, left, margin_h, margin_w,
        crop_top, crop_left, crop_bottom, crop_right,
    )

    return (crop_top, crop_left, crop_bottom, crop_right)


def _assign_person_ids(
    segments: list[_Segment],
    tolerance: float,
    backend: "FaceBackend",
) -> None:
    """Assign a ``person_id`` to each segment via greedy encoding clustering.

    Debug logging reports each segment's best-match person ID or new ID assignment.

    Segments sharing the same visual identity (within *tolerance*) receive the
    same ID.  IDs are assigned in order of first appearance starting from 1.
    The assignment is done **in-place**.

    Args:
        segments:  Segments to label.
        tolerance: Face-distance threshold for same-person matching.
        backend:   :class:`FaceBackend` used for distance computation.
    """
    debug_logging = logger.isEnabledFor(logging.DEBUG)
    known_encs: list[np.ndarray] = []
    known_ids: list[int] = []

    for seg in segments:
        if not known_encs:
            seg.person_id = 1
            known_encs.append(seg.anchor_enc)
            known_ids.append(1)
            logger.debug(
                "_assign_person_ids: segment frames %d–%d → new person_id=1",
                seg.start_frame, seg.end_frame,
            )
        else:
            dists = backend.face_distance(known_encs, seg.anchor_enc)
            best = int(np.argmin(dists))
            if dists[best] <= tolerance:
                seg.person_id = known_ids[best]
                if debug_logging:
                    logger.debug(
                        "_assign_person_ids: segment frames %d–%d → person_id=%d (best_match dist=%.4f)",
                        seg.start_frame, seg.end_frame, seg.person_id, dists[best],
                    )
            else:
                new_id = max(known_ids) + 1
                seg.person_id = new_id
                known_encs.append(seg.anchor_enc)
                known_ids.append(new_id)
                if debug_logging:
                    logger.debug(
                        "_assign_person_ids: segment frames %d–%d → new person_id=%d (no match, best dist=%.4f)",
                        seg.start_frame, seg.end_frame, new_id, dists[best],
                    )

    logger.info(
        "_assign_person_ids: assigned %d distinct person IDs across %d segments",
        len(known_ids), len(segments),
    )


def segment_video(
    video_path: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = 0.4,
    crop_size: int | None = None,
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

    Each output frame is spatially cropped to the bounding box that covers
    the person's detected face positions across the whole segment (with
    *margin_ratio* padding), so the final video contains only that person.
    When *crop_size* is given the cropped region is resized to a square
    ``crop_size × crop_size`` frame.

    Args:
        video_path:          Path to the input video file.
        output_dir:          Root directory for output segments.
        every_n:             Frame sampling interval for face detection.
        margin_ratio:        Fractional padding added around the union of all
                             face bounding boxes when computing the crop rect
                             (default: 0.4).
        crop_size:           If given, each output frame is resized to this
                             square resolution in pixels (default: None, keep
                             the cropped rect dimensions).
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
        "segment_video: %s  fps=%.2f  frames=%d  %dx%d  every_n=%d  margin_ratio=%.2f  tolerance=%.2f",
        video_path.name, fps, total_frames, width, height, every_n, margin_ratio, tolerance,
    )

    do_resize = crop_size is not None
    logger.debug(
        "segment_video: crop_size=%s  resize_output=%s",
        crop_size, do_resize,
    )

    # ------------------------------------------------------------------ #
    # Analysis pass — sample every_n frames to build a face-presence      #
    # timeline.  Each record is (frame_idx, encoding_or_None, bbox_or_None)#
    # where both are None unless exactly one face was detected.            #
    # ------------------------------------------------------------------ #
    frame_records: list[tuple[int, np.ndarray | None, _BBox | None]] = []
    frame_idx = 0
    single_face_count = 0

    debug_logging = logger.isEnabledFor(logging.DEBUG)

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % 1000 == 0:
                logger.debug(
                    "segment_video: analysis pass reading frame %d", frame_idx,
                )

            if frame_idx % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                locs = backend.detect_faces(frame_rgb)

                if debug_logging and len(locs) != 1:
                    logger.debug(
                        "segment_video: frame %d detected %d face(s)",
                        frame_idx, len(locs),
                    )

                if len(locs) == 1:
                    encs = backend.encode_faces(frame_rgb, locs)
                    enc: np.ndarray | None = encs[0] if encs else None
                    bbox: _BBox | None = locs[0]
                    single_face_count += 1
                else:
                    enc = None
                    bbox = None

                frame_records.append((frame_idx, enc, bbox))
            frame_idx += 1
    finally:
        cap.release()

    logger.debug(
        "segment_video: analysis pass complete  %d frames read  %d single-face samples",
        frame_idx, single_face_count,
    )

    # Fall back to the observed frame count if CAP_PROP_FRAME_COUNT was 0.
    if total_frames <= 0:
        total_frames = frame_idx

    if not frame_records:
        logger.info("No frames sampled from %s", video_path.name)
        return {"segments": 0, "persons": 0}

    logger.debug(
        "segment_video: %d sampled frame records (%d with single-face)",
        len(frame_records), single_face_count,
    )

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
    # Write pass — seek to each segment's start frame, crop to the        #
    # person's bounding region, and write as a new MP4.                   #
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

            # Compute the crop rect that covers all detected face positions
            # across the segment, with margin padding.
            crop_top, crop_left, crop_bottom, crop_right = _compute_crop_rect(
                seg.sample_bboxes, margin_ratio, width, height
            )
            out_w = crop_size if crop_size else max(1, crop_right - crop_left)
            out_h = crop_size if crop_size else max(1, crop_bottom - crop_top)

            logger.debug(
                "segment_video: writing segment person=%d seg=%d frames %d–%d crop (%d,%d,%d,%d) size %dx%d",
                pid, seg_num, seg.start_frame, seg.end_frame,
                crop_top, crop_left, crop_bottom, crop_right,
                out_w, out_h,
            )

            cap2.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
            try:
                frame_count = seg.end_frame - seg.start_frame + 1
                written_frames = 0
                for _ in range(frame_count):
                    ret, frame = cap2.read()
                    if not ret:
                        break
                    cropped = frame[crop_top:crop_bottom, crop_left:crop_right]
                    if crop_size:
                        cropped = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
                    writer.write(cropped)
                    written_frames += 1
                logger.debug(
                    "segment_video: wrote %d/%d frames for segment %s",
                    written_frames, frame_count, out_path.name,
                )
            finally:
                writer.release()

            duration = (seg.end_frame - seg.start_frame + 1) / fps
            logger.info(
                "Wrote segment: %s  frames %d–%d  (%.1fs)  person %d  %dx%d",
                out_path.name, seg.start_frame, seg.end_frame, duration, pid, out_w, out_h,
            )
            written += 1
    finally:
        cap2.release()

    return {"segments": written, "persons": len(seg_count_per_person)}


def segment_folder(
    input_dir: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = 0.4,
    crop_size: int | None = None,
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
        margin_ratio:        Fractional padding around the crop bounding box.
        crop_size:           If given, each output frame is resized to this
                             square resolution in pixels.
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
            margin_ratio=margin_ratio,
            crop_size=crop_size,
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
