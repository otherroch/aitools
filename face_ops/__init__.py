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

Clustering and reference loading are methods on every backend:

>>> refs, names = backend.load_reference_encodings(classified_dir)
>>> person_dirs = backend.cluster_faces(results, output_dir)
"""

from __future__ import annotations
import logging

__version__ = "0.1.0"

from face_ops.backend import FaceBackend
from face_ops.mixin import SUPPORTED_IMAGE_EXTS
from face_ops.types import DetectedFace, Encoding, FaceBBox

  
logger = logging.getLogger(__name__)

__all__ = [
    # protocol
    "FaceBackend",
    # types
    "DetectedFace",
    "FaceBBox",
    "Encoding",
    # factory
    "get_backend",
    "backend_for_model",
    # backward-compat standalone helpers
    "cluster_faces",
    "load_reference_encodings",
    "SUPPORTED_IMAGE_EXTS",
]

# Model names that select the dlib backend.
_DLIB_MODELS: frozenset[str] = frozenset({"dlib", "hog", "cnn"})


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
    logger.debug("Instantiating face backend: %s", name)
    
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


def backend_for_model(detection_model: str, **kwargs) -> FaceBackend:
    """Create a :class:`FaceBackend` based on a ``--detection-model`` value.

    ``"dlib"``, ``"hog"``, and ``"cnn"`` select the dlib backend.
    Any other value (e.g. ``"buffalo_l"``) selects the InsightFace
    backend and is forwarded as the *model_name* parameter.

    Args:
        detection_model: Value from the CLI ``--detection-model`` argument.
        kwargs:          Extra keyword arguments forwarded to the backend
                         constructor (e.g. ``ctx_id``, ``providers``).
    """
    if detection_model.lower() in _DLIB_MODELS:
        # "dlib" defaults to "hog"; "hog" and "cnn" are passed through.
        dlib_model = "hog" if detection_model.lower() == "dlib" else detection_model.lower()
        return get_backend("dlib", model=dlib_model)
    return get_backend("insightface", model_name=detection_model, **kwargs)


# ------------------------------------------------------------------
# Backward-compatible standalone wrappers (delegate to backend)
# ------------------------------------------------------------------

def cluster_faces(all_results, output_dir, backend, tolerance=0.6, **kwargs):
    """Backward-compatible wrapper — delegates to ``backend.cluster_faces()``."""
    return backend.cluster_faces(all_results, output_dir, tolerance, **kwargs)


def load_reference_encodings(classified_path, backend, **kwargs):
    """Backward-compatible wrapper — delegates to ``backend.load_reference_encodings()``."""
    return backend.load_reference_encodings(classified_path, **kwargs)
