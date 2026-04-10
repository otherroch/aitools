"""
face_ops.backend

Abstract ``FaceBackend`` protocol that concrete backends must implement.

A backend bundles face detection, encoding, landmark extraction, image
loading, distance computation, clustering, and reference loading into a
single cohesive object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from face_ops.types import DetectedFace, Encoding, FaceBBox


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

    def detect(
        self,
        image: np.ndarray,
        *,
        model: str = "hog",
    ) -> list[DetectedFace]:
        """Detect faces and return rich per-face metadata.

        A single-call alternative to :meth:`detect_faces` followed by
        :meth:`encode_faces` that avoids running detection twice.

        Each :class:`DetectedFace` bundles the bounding box, embedding,
        landmarks, and an opaque backend-specific *raw* object (e.g. an
        InsightFace ``Face``) for downstream consumers that need it.

        Args:
            image: RGB uint8 numpy array (H × W × 3).
            model: Backend-specific model hint.

        Returns:
            List of :class:`DetectedFace` instances, one per face.
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

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    def load_reference_encodings(
        self,
        classified_path: Path,
        *,
        model: str = "hog",
        max_per_identity: int = 0,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Load face encodings from a pre-classified reference directory.

        See :meth:`FaceBackendMixin.load_reference_encodings` for full docs.
        """
        ...

    def cluster_faces(
        self,
        all_results: list[tuple[Path, np.ndarray]],
        output_dir: Path,
        tolerance: float = 0.6,
        *,
        reference_encodings: list[np.ndarray] | None = None,
        reference_names: list[str] | None = None,
    ) -> dict[str, list[Path]]:
        """Group saved face crops by identity using greedy nearest-neighbour clustering.

        See :meth:`FaceBackendMixin.cluster_faces` for full docs.
        """
        ...
