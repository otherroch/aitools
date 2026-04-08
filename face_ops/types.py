"""
face_ops.types

Shared type aliases for face detection and recognition.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

# (top, right, bottom, left) – matches the face_recognition convention.
FaceBBox: TypeAlias = tuple[int, int, int, int]

# A face embedding vector (128-d for dlib, 512-d for ArcFace, etc.).
Encoding: TypeAlias = np.ndarray
