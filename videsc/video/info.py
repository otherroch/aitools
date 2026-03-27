import sys
import logging
import cv2
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> Dict[str, Any]:
    logger.debug("get_video_info: opening %s", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video file '%s'", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_seconds = total_frames / fps if fps > 0.0 else 0.0
    duration_minutes = duration_seconds / 60.0 if duration_seconds > 0 else 0.0

    cap.release()

    info = {
        "num_frames": total_frames,
        "FPS": fps,
        "width": width,
        "height": height,
        "tot_time": duration_seconds,
        "duration_minutes": duration_minutes,
    }
    logger.debug(
        "get_video_info: %s  frames=%d  fps=%.2f  size=%dx%d  duration=%.2fs",
        video_path, total_frames, fps, width, height, duration_seconds,
    )
    return info
