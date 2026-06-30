#!/usr/bin/env python3
"""
vicrop.extract – Extract raw video frames without face detection.

Provides an `extract_frames()` function that samples a video at the given interval
and writes each sampled frame as a PNG, optionally resizing to a specified width/height.
Used by the CLI's `--extract-only` mode.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PIL import Image

logger = logging.getLogger(__name__)
def extract_frames(video_path: Path, output_dir: Path, every_n: int = 30, crop_dim: tuple[int, int] | None = None):
    """
    Extract raw frames from a single video file.

    Frames are sampled every ``every_n`` frames and saved as PNG files in the output directory.
    The output filename pattern is ``frame{idx:06d}.png`` (one per extracted frame).
    If ``crop_dim`` is provided, each frame is resized to ``(width, height)``;
    otherwise a square size of 1024×1024 is used as fallback (matching --crop-size default).

    Args:
        video_path: Path to the input video file.
        output_dir: Root directory where extracted frames are stored.
        every_n: Sample every N-th frame (default 30).
        crop_dim: Optional tuple ``(width, height)`` for final frame size. If provided,
                  overrides the square ``--crop-size`` default of 1024.

    Returns:
        Dict with summary keys:
            "frames_extracted": total number of frames saved,
            "video_path": string representation (for logging).
    """
    if not video_path.exists():
        logger.error("Input video does not exist: %s", video_path)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video for extraction: %s", video_path)
        return {"frames_extracted": 0, "video_path": str(video_path)}

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Extract frames: %s (%d total) at %.1f fps, sampling every %d → ~%d frames",
        video_path.name,
        total_frames,
        fps,
        every_n,
        max(1, (total_frames + every_n - 1) // every_n),
    )

    # Determine final size: crop_dim overrides square fallback
    if crop_dim is None:
        crop_w = crop_h = 1024  # default --crop-size
    else:
        crop_w, crop_h = crop_dim

    extracted = 0
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            # convert BGR → RGB (optional for PNG)
            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            # resize to desired dimensions
            pil_img = pil_img.resize((crop_w, crop_h), Image.LANCZOS)
            out_name = f"frame{frame_idx:06d}.png"
            out_path = output_dir / out_name
            pil_img.save(out_path)
            logger.debug("Extracted frame: %s", out_path)
            extracted += 1
        frame_idx += 1
    cap.release()

    logger.info(
        "Finished extraction from video: %s  frames saved: %d",
        video_path.name,
        extracted,
    )
    return {"frames_extracted": extracted, "video_path": str(video_path)}
def extract_folder(input_dir: Path, output_dir: Path, every_n: int = 30, crop_dim: tuple[int, int] | None = None):
    """
    Process all video files under ``input_dir`` (searched recursively) and extract
    raw frames into subfolders named after the video stem.

    Args:
        input_dir: Source directory containing video files.
        output_dir: Root directory where extracted frames are stored.
        every_n: Sample interval for frames within each video.
        crop_dim: Optional width/height tuple (same meaning as in ``extract_frames``).

    Returns:
        Aggregated summary dict with keys:
            "videos_processed", "frames_extracted".
    """
    from vicrop.crop import SUPPORTED_VIDEO_EXTS  # reuse existing video filter

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ]
    if not videos:
        logger.warning("No video files found in %s", input_dir)
        return {"videos_processed": 0, "frames_extracted": 0}

    total_videos = len(videos)
    total_frames = 0

    for i, video_path in enumerate(videos):
        logger.info(
            "Processing (%d/%d) extract‑only: %s",
            i + 1,
            total_videos,
            video_path.name,
        )
        stats = extract_frames(video_path, output_dir, every_n=every_n, crop_dim=crop_dim)
        total_frames += stats["frames_extracted"]

    return {"videos_processed": total_videos, "frames_extracted": total_frames}