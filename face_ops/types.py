"""
face_ops.types

Shared type aliases and data classes for face detection and recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

# (top, right, bottom, left) – matches the face_recognition convention.
FaceBBox: TypeAlias = tuple[int, int, int, int]

# A face embedding vector (128-d for dlib, 512-d for ArcFace, etc.).
Encoding: TypeAlias = np.ndarray


@dataclass
class DetectedFace:
    """Rich per-face result from :meth:`FaceBackend.detect`.

    Bundles the bounding box, embedding vector, landmark array, and an
    opaque backend-specific object into a single structure so that
    callers can obtain all face metadata in one detection pass.

    Attributes:
        bbox:       Bounding box as ``(top, right, bottom, left)``.
        embedding:  Face encoding vector, or *None* if unavailable.
        landmarks:  Landmark coordinates (backend-specific shape/format),
                    or *None*.
        raw:        Backend-specific face object (e.g. an InsightFace
                    ``Face``).  Opaque to backend-agnostic code but can
                    be passed through to specialised consumers such as
                    face-swap engines.
    """

    bbox: FaceBBox
    embedding: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    raw: object = None
