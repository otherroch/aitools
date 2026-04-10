"""Face detection and lightweight IoU-based tracking.

Uses the shared :class:`face_ops.insightface_backend.InsightFaceBackend`
for face detection (RetinaFace) and encoding (ArcFace), wrapped with a
greedy IoU tracker that maintains temporally consistent track IDs.

The ``InsightFaceBackend`` instance is exposed via the :attr:`backend`
property so that it can be shared with ``FaceRecognizer`` (avoids
loading duplicate ONNX models into VRAM).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from face_ops.insightface_backend import InsightFaceBackend

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
    # The full InsightFace Face object for the swap engine
    face_obj: object = None


class FaceDetector:
    """Wraps InsightFace detection with a simple IoU tracker.

    Each call to ``detect(frame)`` returns a list of TrackedFace objects
    with temporally-consistent ``track_id`` values.

    The internal :class:`InsightFaceBackend` is exposed via the
    :attr:`backend` property so it can be shared with
    ``FaceRecognizer`` (avoids loading duplicate ONNX models into VRAM).
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._next_id = 0
        self._tracks: list[TrackedFace] = []

        providers = get_onnx_providers(cfg.device_id)
        self._backend = InsightFaceBackend(
            model_name=cfg.detection_model,
            ctx_id=cfg.device_id,
            det_size=cfg.detection_size,
            det_thresh=cfg.detection_threshold,
            providers=providers,
        )
        logger.info(
            "FaceDetector ready (model=%s, det_size=%s).",
            cfg.detection_model,
            cfg.detection_size,
        )

    @property
    def backend(self) -> InsightFaceBackend:
        """Shared :class:`InsightFaceBackend` instance."""
        return self._backend

    @property
    def app(self):
        """Shared FaceAnalysis instance (backward-compatible shortcut)."""
        return self._backend.app

    def detect(self, frame: np.ndarray) -> list[TrackedFace]:
        """Detect and track faces in a single BGR frame."""
        faces = self._backend.app.get(frame)
        detections: list[TrackedFace] = []
        if faces:
            for f in faces:
                emb = None
                if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                    emb = np.array(f.normed_embedding, dtype=np.float32)
                detections.append(
                    TrackedFace(
                        track_id=-1,
                        bbox=np.array(f.bbox, dtype=np.float32),
                        landmarks=np.array(f.kps, dtype=np.float32),
                        embedding=emb,
                        face_obj=f,
                    )
                )

        if not detections and faces:
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
