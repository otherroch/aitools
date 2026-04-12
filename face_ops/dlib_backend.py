"""
face_ops.dlib_backend

Concrete ``FaceBackend`` implementation backed by the
`face_recognition <https://github.com/ageitgey/face_recognition>`_ library
(dlib HOG / CNN).
"""

from __future__ import annotations
import logging

import numpy as np

from face_ops.mixin import FaceBackendMixin
from face_ops.types import DetectedFace, Encoding, FaceBBox

logger = logging.getLogger(__name__)

class DlibBackend(FaceBackendMixin):
    """Face detection and encoding via dlib / face_recognition."""

    def __init__(self, model: str = "hog") -> None:
        self._model = model
        try:
            import face_recognition  # noqa: F401

            self._fr = face_recognition
        except ImportError as exc:
            raise ImportError(
                "face_recognition is required for the dlib backend.\n"
                "Install it with: pip install face_recognition"
            ) from exc
        except Exception as exc:
            logger.error("Error initializing DlibBackend: %s", exc)
            raise
        logger.info("DlibBackend initialized with model: %s", model)    
    # ------------------------------------------------------------------

    # Detection
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        image: np.ndarray,
    ) -> list[FaceBBox]:
        return self._fr.face_locations(image, model=self._model)

    def detect(
        self,
        image: np.ndarray,
    ) -> list[DetectedFace]:
        locations = self._fr.face_locations(image, model=self._model)
        if not locations:
            return []
        encodings = self._fr.face_encodings(image, locations)
        results: list[DetectedFace] = []
        for i, loc in enumerate(locations):
            emb = encodings[i] if i < len(encodings) else None
            results.append(DetectedFace(bbox=loc, embedding=emb))
        return results

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_faces(
        self,
        image: np.ndarray,
        face_locations: list[FaceBBox],
    ) -> list[Encoding]:
        return self._fr.face_encodings(image, face_locations)

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def face_distance(
        self,
        known_encodings: list[Encoding],
        encoding: Encoding,
    ) -> np.ndarray:
        return self._fr.face_distance(known_encodings, encoding)

    # ------------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> np.ndarray:
        return self._fr.load_image_file(path)

    # ------------------------------------------------------------------
    # Landmarks
    # ------------------------------------------------------------------

    def face_landmarks(
        self,
        image: np.ndarray,
        face_locations: list[FaceBBox],
    ) -> list[dict[str, list[tuple[int, int]]] | None]:
        return self._fr.face_landmarks(image, face_locations)
