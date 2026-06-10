"""Face detection and lightweight IoU-based tracking.

Uses a :class:`~face_ops.FaceBackend` obtained via
:func:`face_ops.get_backend` for face detection and encoding, wrapped
with a greedy IoU tracker that maintains temporally consistent track IDs.

The backend instance is exposed via the :attr:`backend` property so it
can be shared with ``FaceRecognizer`` (avoids loading duplicate ONNX
models into VRAM).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from face_ops import backend_for_model, _DLIB_MODELS
from face_ops.backend import FaceBackend

from .config import PipelineConfig
from .gpu_utils import get_onnx_providers

logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """A single tracked face across frames."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # (5, 2) – 5-point landmarks
    embedding: Optional[np.ndarray] = None
    age_since_seen: int = 0
    identity_label: Optional[str] = None
    identity_sim: float = 0.0  # confidence of the last identity match
    # The backend-specific face object for the swap engine
    face_obj: object = None


class FaceDetector:
    """Wraps a :class:`FaceBackend` with a simple IoU tracker.

    Each call to ``detect(frame)`` returns a list of TrackedFace objects
    with temporally-consistent ``track_id`` values.

    The :class:`FaceBackend` is exposed via :attr:`backend` so it can be
    shared with ``FaceRecognizer`` (avoids loading duplicate ONNX models
    into VRAM).
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._next_id = 0
        self._tracks: list[TrackedFace] = []

        if cfg.detection_model.lower() in _DLIB_MODELS:
            self._backend: FaceBackend = backend_for_model(cfg.detection_model)
            logger.info(
                "FaceDetector ready (backend=dlib, model=%s).",
                cfg.detection_model,
            )
        else:
            providers = get_onnx_providers(cfg.device_id)
            self._backend = backend_for_model(
                cfg.detection_model,
                ctx_id=cfg.device_id,
                det_size=cfg.detection_size,
                det_thresh=cfg.detection_threshold,
                providers=providers,
            )
            logger.info(
                "FaceDetector ready (backend=insightface, model=%s, det_size=%s).",
                cfg.detection_model,
                cfg.detection_size,
            )

    @property
    def backend(self) -> FaceBackend:
        """Shared :class:`FaceBackend` instance."""
        return self._backend

    def detect(self, frame: np.ndarray) -> list[TrackedFace]:
        """Detect and track faces in a single BGR frame."""
        detected = self._backend.detect(frame)
        detections: list[TrackedFace] = []
        for df in detected:
            top, right, bottom, left = df.bbox
            emb = None
            if df.embedding is not None:
                emb = np.array(df.embedding, dtype=np.float32)
            landmarks = (
                np.array(df.landmarks, dtype=np.float32)
                if df.landmarks is not None
                else np.zeros((5, 2), dtype=np.float32)
            )
            detections.append(
                TrackedFace(
                    track_id=-1,
                    bbox=np.array(
                        [left, top, right, bottom], dtype=np.float32
                    ),
                    landmarks=landmarks,
                    embedding=emb,
                    face_obj=df.raw,
                )
            )

        if not detections and detected:
            logger.debug("Faces detected but no valid embeddings found.")

        # Always run the tracker so that existing tracks age out correctly
        # even on frames where no faces are detected.
        matched = self._match_tracks(detections)
        if matched:
            logger.debug("Matched %d faces to existing tracks", len(matched))
        self._tracks = matched
        return matched

    def active_tracks(self) -> list[TrackedFace]:
        """Return only tracks that were seen in the most recent frame."""
        return [t for t in self._tracks if t.age_since_seen == 0]

    def reset_tracks(self) -> None:
        """Clear all tracked faces (e.g. after a scene cut)."""
        self._tracks.clear()

    # ── simple IoU tracker ───────────────────────────────────────────────

    def _match_tracks(
        self, detections: list[TrackedFace]
    ) -> list[TrackedFace]:
        """Greedy IoU matching between existing tracks and new detections.

        Also propagates the identity label from a matched track to its
        new detection so recurring faces keep their assigned identity
        without needing re-recognition every frame.
        """
        max_age = self._cfg.tracker_max_age
        iou_thresh = self._cfg.tracker_iou_threshold

        used_det: set[int] = set()
        updated: list[TrackedFace] = []

        # Try to match each existing track to the closest detection
        for track in self._tracks:
            best_iou = iou_thresh
            best_idx = -1
            for i, det in enumerate(detections):
                if i in used_det:
                    continue
                iou = _iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0:
                det = detections[best_idx]
                det.track_id = track.track_id
                # propagate identity from prior frame
                det.identity_label = track.identity_label
                det.identity_sim = track.identity_sim
                det.age_since_seen = 0
                updated.append(det)
                used_det.add(best_idx)
            else:
                track.age_since_seen += 1
                if track.age_since_seen <= max_age:
                    updated.append(track)

        # Assign new IDs to unmatched detections
        for i, det in enumerate(detections):
            if i not in used_det:
                det.track_id = self._next_id
                self._next_id += 1
                updated.append(det)

        return updated


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] bboxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = float(a[2] - a[0]) * float(a[3] - a[1])
    area_b = float(b[2] - b[0]) * float(b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
