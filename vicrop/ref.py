#!/usr/bin/env python3
"""
vicrop.ref – Reference photo quality scoring for portrait training.

Scores face crops on multiple criteria to identify the best reference photos:

1. **Frontal pose** – low yaw/pitch deviation (landmark symmetry).
2. **Eyes open** – eye aspect ratio from landmarks.
3. **Face fill** – face-to-frame area ratio.
4. **Sharpness** – Laplacian variance of the face region.
5. **Good lighting** – luminance mean and contrast.
6. **Single face** – exactly one face detected in the frame.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_REF_THRESH: float = 0.8

# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------


def _eye_aspect_ratio(eye_points: list[tuple[int, int]]) -> float:
    """Eye Aspect Ratio from 6 landmark points.

    EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
    Higher values → more open eyes.
    """
    if len(eye_points) < 6:
        return 0.0

    pts = np.array(eye_points, dtype=np.float64)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def _frontality_score(landmarks: dict[str, list[tuple[int, int]]]) -> float:
    """Estimate how frontal the face is based on landmark symmetry.

    Returns 0 (profile) … 1 (perfectly frontal).
    """
    nose_tip_pts = landmarks.get("nose_tip", [])
    left_eye_pts = landmarks.get("left_eye", [])
    right_eye_pts = landmarks.get("right_eye", [])
    chin_pts = landmarks.get("chin", [])

    if not nose_tip_pts or not left_eye_pts or not right_eye_pts or not chin_pts:
        return 0.0

    nose_tip = np.array(
        nose_tip_pts[len(nose_tip_pts) // 2], dtype=np.float64,
    )
    left_eye = np.mean(np.array(left_eye_pts, dtype=np.float64), axis=0)
    right_eye = np.mean(np.array(right_eye_pts, dtype=np.float64), axis=0)

    # --- yaw: compare horizontal nose-to-eye distances ----
    dist_left = np.linalg.norm(nose_tip - left_eye)
    dist_right = np.linalg.norm(nose_tip - right_eye)
    max_dist = max(dist_left, dist_right)
    if max_dist < 1e-6:
        return 0.0
    yaw_ratio = min(dist_left, dist_right) / max_dist

    # --- pitch: nose position relative to eye-centre → chin ---
    eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
    chin_bottom = np.array(
        chin_pts[len(chin_pts) // 2], dtype=np.float64,
    )
    face_height = chin_bottom[1] - eye_center_y
    if face_height < 1e-6:
        return 0.0
    nose_relative = (nose_tip[1] - eye_center_y) / face_height
    # Ideal ratio ≈ 0.40; penalise deviations
    pitch_score = 1.0 - min(abs(nose_relative - 0.4) * 3.0, 1.0)

    return float(np.clip(yaw_ratio * 0.7 + max(pitch_score, 0) * 0.3, 0, 1))


def _sharpness_score(face_rgb: np.ndarray) -> float:
    """Score sharpness via Laplacian variance (0–1)."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(lap_var / 150.0, 0, 1))


def _lighting_score(face_rgb: np.ndarray) -> float:
    """Score lighting quality from luminance mean and contrast (0–1)."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mean_lum = gray.mean()
    std_lum = gray.std()

    if mean_lum < 40:
        brightness_score = mean_lum / 40.0
    elif mean_lum > 220:
        brightness_score = max(0.0, (255 - mean_lum) / 35.0)
    else:
        brightness_score = 1.0

    contrast_score = float(np.clip(std_lum / 50.0, 0, 1))
    return float(np.clip(brightness_score * 0.6 + contrast_score * 0.4, 0, 1))


def _single_face_score(face_count: int) -> float:
    """Return 1.0 if exactly one face is present in the frame, else 0.0.

    A frame with multiple faces risks including parts of another person,
    which can confuse downstream model training.
    """
    return 1.0 if face_count == 1 else 0.0


def _face_fill_score(
    face_bbox: tuple[int, int, int, int],
    frame_height: int,
    frame_width: int,
) -> float:
    """Score how much of the frame the face occupies (0–1).

    A face filling ≥ 15 % of the frame gets a perfect score.
    """
    top, right, bottom, left = face_bbox
    face_area = (bottom - top) * (right - left)
    frame_area = frame_height * frame_width
    if frame_area <= 0:
        return 0.0
    ratio = face_area / frame_area
    return float(np.clip(ratio / 0.15, 0, 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_reference_quality(
    frame_rgb: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    landmarks: dict[str, list[tuple[int, int]]] | None,
    face_region_rgb: np.ndarray,
    face_count: int = 1,
) -> float:
    """Composite reference-quality score for a single detected face.

    Args:
        frame_rgb:       Full frame (RGB numpy array).
        face_bbox:       ``(top, right, bottom, left)`` from *face_recognition*.
        landmarks:       Landmark dict from ``face_recognition.face_landmarks()``,
                         or *None* when landmarks are unavailable.
        face_region_rgb: Cropped (un-resized) face region as RGB array.
        face_count:      Total number of faces detected in the frame.  Frames
                         with more than one face receive a score of 0.0 to
                         avoid including another person's face in training data.

    Returns:
        Quality score in [0.0, 1.0].
    """
    if _single_face_score(face_count) == 0.0:
        return 0.0

    h_frame, w_frame = frame_rgb.shape[:2]

    fill = _face_fill_score(face_bbox, h_frame, w_frame)
    sharpness = _sharpness_score(face_region_rgb)
    lighting = _lighting_score(face_region_rgb)

    if landmarks is None:
        return float(np.clip(
            fill * 0.30 + sharpness * 0.35 + lighting * 0.35, 0, 1,
        ))

    frontality = _frontality_score(landmarks)

    left_ear = _eye_aspect_ratio(landmarks.get("left_eye", []))
    right_ear = _eye_aspect_ratio(landmarks.get("right_eye", []))
    avg_ear = (left_ear + right_ear) / 2.0
    eye_score = float(np.clip((avg_ear - 0.15) / 0.15, 0, 1))

    composite = (
        frontality * 0.30
        + eye_score * 0.20
        + fill * 0.15
        + sharpness * 0.20
        + lighting * 0.15
    )
    return float(np.clip(composite, 0, 1))


def collect_ref_photos(person_dir: Path, ref_paths: list[Path]) -> Path | None:
    """Move qualifying reference photos into a ``ref/`` sub-folder.

    Each file in *ref_paths* is moved from its current location (inside
    *person_dir*) into ``person_dir/ref/``.

    Returns:
        Path to the ``ref/`` directory, or *None* if *ref_paths* is empty.
    """
    if not ref_paths:
        return None

    ref_dir = person_dir / "ref"
    ref_dir.mkdir(exist_ok=True)

    for src in sorted(ref_paths):
        shutil.move(str(src), ref_dir / src.name)

    logger.info(
        "Moved %d reference photo(s) to %s", len(ref_paths), ref_dir,
    )
    return ref_dir
