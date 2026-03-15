import sys
import cv2
from typing import Dict, Any


def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_seconds = total_frames / fps if fps > 0.0 else 0.0
    duration_minutes = duration_seconds / 60.0 if duration_seconds > 0 else 0.0

    cap.release()

    return {
        "num_frames": total_frames,
        "FPS": fps,
        "width": width,
        "height": height,
        "tot_time": duration_seconds,
        "duration_minutes": duration_minutes,
    }
