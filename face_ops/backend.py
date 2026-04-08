"""
face_ops.backend

Abstract ``FaceBackend`` protocol that concrete backends must implement.

A backend bundles face detection, encoding, landmark extraction, image
loading, and distance computation into a single cohesive object.  Higher
level helpers (``cluster_faces``, ``load_reference_encodings``) are
backend-agnostic — they only call methods on this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from face_ops.types import Encoding, FaceBBox


@runtime_checkable
class FaceBackend(Protocol):
    """Protocol every face-ops backend must satisfy."""

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        image: np.ndarray,
        *,
        model: str = "hog",
    ) -> list[FaceBBox]:
        """Return bounding boxes for all faces found in *image*.

        Args:
            image: RGB uint8 numpy array (H × W × 3).
            model: Backend-specific model hint (e.g. ``"hog"`` / ``"cnn"``
                   for dlib, ``"buffalo_l"`` for InsightFace).

        Returns:
            List of ``(top, right, bottom, left)`` tuples.
        """
        ...

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_faces(
        self,
        image: np.ndarray,
        face_locations: list[FaceBBox],
    ) -> list[Encoding]:
        """Compute an embedding vector for each detected face.

        Args:
            image:          RGB uint8 numpy array.
            face_locations: Bounding boxes from :meth:`detect_faces`.

        Returns:
            List of 1-D numpy arrays, one per face location.
        """
        ...

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def face_distance(
        self,
        known_encodings: list[Encoding],
        encoding: Encoding,
    ) -> np.ndarray:
        """Compute distances between *encoding* and each known encoding.

        Args:
            known_encodings: List of reference embeddings.
            encoding:        Query embedding.

        Returns:
            1-D numpy array of distances (lower = more similar).
        """
        ...

    # ------------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> np.ndarray:
        """Load an image file and return it as an RGB numpy array.

        Args:
            path: Filesystem path to the image.

        Returns:
            RGB uint8 numpy array (H × W × 3).
        """
        ...

    # ------------------------------------------------------------------
    # Landmarks (optional — quality scoring uses these)
    # ------------------------------------------------------------------

    def face_landmarks(
        self,
        image: np.ndarray,
        face_locations: list[FaceBBox],
    ) -> list[dict[str, list[tuple[int, int]]] | None]:
        """Return facial landmarks for each detected face.

        Backends that do not support landmarks should return a list of
        ``None`` values (one per face location).

        Args:
            image:          RGB uint8 numpy array.
            face_locations: Bounding boxes from :meth:`detect_faces`.

        Returns:
            One dict (or *None*) per face location.  The dict maps
            landmark group names (e.g. ``"left_eye"``, ``"nose_tip"``)
            to lists of ``(x, y)`` coordinate tuples.
        """
        ...
