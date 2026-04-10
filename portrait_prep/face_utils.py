#!/usr/bin/env python3
"""
portrait_prep.face_utils

Shared face-recognition helpers used by both portrait_prep.crop and
vicrop.crop.  Centralises the lazy face_recognition import and the
greedy nearest-neighbour identity-clustering algorithm so that neither
package needs to duplicate them.

.. note::
    The heavy lifting is now provided by the :mod:`face_ops` package.
    This module re-exports the public API and keeps the legacy
    ``load_face_recognition`` / ``fr=`` calling convention for backward
    compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from face_ops import (
    SUPPORTED_IMAGE_EXTS,
    get_backend,
)
from face_ops.backend import FaceBackend
from face_ops.clustering import (
    cluster_faces as _cluster_faces_new,
    load_reference_encodings as _load_reference_encodings_new,
)

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_RATIO: float = 0.4
DEFAULT_CROP_SIZE: int = 1024

# Re-export so existing ``from portrait_prep.face_utils import
# SUPPORTED_IMAGE_EXTS`` keeps working.
from face_ops import SUPPORTED_IMAGE_EXTS as SUPPORTED_IMAGE_EXTS  # noqa: F811


def load_face_recognition():
    """Lazily import the face_recognition package, raising a clear error if absent."""
    try:
        import face_recognition

        return face_recognition
    except ImportError as exc:
        raise ImportError(
            "face_recognition is required for the crop task.\n"
            "Install it with: pip install face_recognition"
        ) from exc


def _backend_from_fr(fr=None) -> FaceBackend:
    """Return a :class:`FaceBackend` wrapping an ``fr`` module.

    When *fr* is a real ``face_recognition`` module (or ``None``), we
    return a :class:`DlibBackend`.  When *fr* is a mock object (unit
    tests), we wrap it in a lightweight shim that satisfies the
    :class:`FaceBackend` protocol.
    """
    if fr is None:
        fr = load_face_recognition()

    # Check whether ``fr`` is the real face_recognition module.
    module_name = getattr(fr, "__name__", "")
    if module_name == "face_recognition":
        from face_ops.dlib_backend import DlibBackend

        backend = DlibBackend.__new__(DlibBackend)
        backend._fr = fr
        return backend

    # ``fr`` is a mock — wrap it so cluster_faces/load_reference_encodings
    # can call the backend protocol methods.
    return _MockBackendShim(fr)


class _MockBackendShim:
    """Thin shim that adapts a mock ``face_recognition`` module to the
    :class:`FaceBackend` protocol.  Used only in tests.
    """

    def __init__(self, fr) -> None:
        self._fr = fr

    def detect_faces(self, image, *, model="hog"):
        return self._fr.face_locations(image, model=model)

    def detect(self, image, *, model="hog"):
        from face_ops.types import DetectedFace

        locations = self.detect_faces(image, model=model)
        encodings = self.encode_faces(image, locations)
        results = []
        for i, loc in enumerate(locations):
            emb = encodings[i] if i < len(encodings) else None
            results.append(DetectedFace(bbox=loc, embedding=emb))
        return results

    def encode_faces(self, image, face_locations):
        return self._fr.face_encodings(image, face_locations)

    def face_distance(self, known_encodings, encoding):
        return self._fr.face_distance(known_encodings, encoding)

    def load_image(self, path):
        return self._fr.load_image_file(path)

    def face_landmarks(self, image, face_locations):
        return self._fr.face_landmarks(image, face_locations)


def load_reference_encodings(
    classified_path: Path,
    model: str = "hog",
    *,
    fr=None,
    max_per_identity: int = 0,
) -> tuple[list[np.ndarray], list[str]]:
    """Load face encodings from a pre-classified reference directory.

    Backward-compatible wrapper around
    :func:`face_ops.clustering.load_reference_encodings`.
    """
    backend = _backend_from_fr(fr)
    return _load_reference_encodings_new(
        classified_path,
        backend,
        model=model,
        max_per_identity=max_per_identity,
    )


def cluster_faces(
    all_results: list[tuple[Path, np.ndarray]],
    output_dir: Path,
    tolerance: float = 0.6,
    *,
    fr=None,
    reference_encodings: list[np.ndarray] | None = None,
    reference_names: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Group saved face crops by identity.

    Backward-compatible wrapper around
    :func:`face_ops.clustering.cluster_faces`.
    """
    backend = _backend_from_fr(fr)
    return _cluster_faces_new(
        all_results,
        output_dir,
        backend,
        tolerance=tolerance,
        reference_encodings=reference_encodings,
        reference_names=reference_names,
    )
