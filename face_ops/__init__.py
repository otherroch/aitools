"""
face_ops – Pluggable face detection and recognition toolkit.

Provides a backend-agnostic interface for face detection, encoding,
distance computation, clustering, and reference loading.  Two concrete
backends are included:

* :class:`DlibBackend` — wraps the *face_recognition* (dlib) library.
* :class:`InsightFaceBackend` — wraps *insightface* / ArcFace models.

Quick start
-----------

>>> from face_ops import get_backend
>>> backend = get_backend("dlib")
>>> boxes = backend.detect_faces(image)
>>> encodings = backend.encode_faces(image, boxes)

Higher-level helpers (:func:`cluster_faces`, :func:`load_reference_encodings`)
accept any backend and are therefore encoding-dimension agnostic.
"""

from __future__ import annotations

__version__ = "0.1.0"

from face_ops.backend import FaceBackend
from face_ops.clustering import (
    SUPPORTED_IMAGE_EXTS,
    cluster_faces,
    load_reference_encodings,
)
from face_ops.testing import MockBackendShim, backend_from_fr
from face_ops.types import DetectedFace, Encoding, FaceBBox

__all__ = [
    # protocol
    "FaceBackend",
    # types
    "DetectedFace",
    "FaceBBox",
    "Encoding",
    # factory
    "get_backend",
    # high-level helpers
    "cluster_faces",
    "load_reference_encodings",
    "SUPPORTED_IMAGE_EXTS",
    # testing / compat
    "MockBackendShim",
    "backend_from_fr",
]


def get_backend(name: str = "dlib", **kwargs) -> FaceBackend:
    """Instantiate a face-ops backend by name.

    Args:
        name:   ``"dlib"`` or ``"insightface"``.
        kwargs: Forwarded to the backend constructor.

    Returns:
        A :class:`FaceBackend` instance.

    Raises:
        ValueError:  Unknown backend name.
        ImportError:  Backend dependency not installed.
    """
    name = name.lower()
    if name == "dlib":
        from face_ops.dlib_backend import DlibBackend

        return DlibBackend(**kwargs)
    if name in ("insightface", "arcface"):
        from face_ops.insightface_backend import InsightFaceBackend

        return InsightFaceBackend(**kwargs)
    raise ValueError(
        f"Unknown face_ops backend: {name!r}. "
        f"Choose 'dlib' or 'insightface'."
    )
