"""
face_ops.testing

Test helpers for the ``face_ops`` package.

Provides :class:`MockBackendShim` — a lightweight adapter that wraps a
mock ``face_recognition`` module into the :class:`FaceBackend` protocol
so that tests can pass ``fr``-style mocks directly to
``backend.cluster_faces``, ``backend.load_reference_encodings``, etc.
"""

from __future__ import annotations

import numpy as np

from face_ops.mixin import FaceBackendMixin
from face_ops.types import DetectedFace


class MockBackendShim(FaceBackendMixin):
    """Adapt a mock ``face_recognition`` module to the :class:`FaceBackend` protocol.

    This is intentionally a concrete class (not a protocol implementor) so
    that test code can instantiate it with a MagicMock and have all
    backend-protocol methods forwarded to the mock's equivalents.

    Inherits :meth:`cluster_faces` and :meth:`load_reference_encodings`
    from :class:`FaceBackendMixin`.
    """

    def __init__(self, fr) -> None:
        self._fr = fr

    def detect_faces(self, image, *, model="hog"):
        return self._fr.face_locations(image, model=model)

    def detect(self, image, *, model="hog"):
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


def backend_from_fr(fr=None):
    """Return a :class:`FaceBackend` wrapping an ``fr`` module.

    When *fr* is the real ``face_recognition`` module (or ``None``), we
    return a :class:`DlibBackend`.  When *fr* is a mock object (unit
    tests), we wrap it in a :class:`MockBackendShim` that satisfies the
    :class:`FaceBackend` protocol.
    """
    if fr is None:
        from face_ops.dlib_backend import DlibBackend

        return DlibBackend()

    # Check whether ``fr`` is the real face_recognition module.
    module_name = getattr(fr, "__name__", "")
    if module_name == "face_recognition":
        from face_ops.dlib_backend import DlibBackend

        backend = DlibBackend.__new__(DlibBackend)
        backend._fr = fr
        return backend

    # ``fr`` is a mock — wrap it so cluster_faces/load_reference_encodings
    # can call the backend protocol methods.
    return MockBackendShim(fr)
