"""
face_ops.insightface_backend

Concrete ``FaceBackend`` implementation backed by
`insightface <https://github.com/deepinsight/insightface>`_ with ArcFace
recognition models.

Bounding boxes are normalised to the ``(top, right, bottom, left)``
convention used throughout the codebase so that downstream code
(clustering, cropping) works identically regardless of backend.
"""

from __future__ import annotations

import logging

import numpy as np

from face_ops.types import Encoding, FaceBBox

logger = logging.getLogger(__name__)


class InsightFaceBackend:
    """Face detection and ArcFace encoding via the insightface library.

    Args:
        model_name: InsightFace model pack (default ``"buffalo_l"``).
            Other useful choices: ``"buffalo_s"`` (smaller/faster),
            ``"buffalo_sc"`` (CPU-friendly).
        ctx_id: ONNX Runtime device id.  ``0`` = first GPU, ``-1`` = CPU.
        det_size: Detection input resolution as ``(width, height)``.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for the InsightFace backend.\n"
                "Install it with: pip install insightface onnxruntime"
            ) from exc

        self._app = FaceAnalysis(
            name=model_name,
            providers=self._providers(ctx_id),
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _providers(ctx_id: int) -> list[str]:
        if ctx_id >= 0:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        image: np.ndarray,
        *,
        model: str = "buffalo_l",
    ) -> list[FaceBBox]:
        faces = self._app.get(image)
        bboxes: list[FaceBBox] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            # Convert (x1, y1, x2, y2) → (top, right, bottom, left)
            bboxes.append((int(y1), int(x2), int(y2), int(x1)))
        return bboxes

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_faces(
        self,
        image: np.ndarray,
        face_locations: list[FaceBBox],
    ) -> list[Encoding]:
        faces = self._app.get(image)
        if not faces:
            return []

        # Match detected faces to the requested locations by IoU.
        encodings: list[Encoding] = []
        for top, right, bottom, left in face_locations:
            best_face = self._match_face(faces, top, right, bottom, left)
            if best_face is not None and best_face.embedding is not None:
                encodings.append(best_face.embedding)
            else:
                encodings.append(np.zeros(512, dtype=np.float32))
        return encodings

    @staticmethod
    def _match_face(faces, top: int, right: int, bottom: int, left: int):
        """Find the insightface Face object whose bbox best overlaps the query."""
        best, best_iou = None, 0.0
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            # Compute IoU
            ix1 = max(left, int(x1))
            iy1 = max(top, int(y1))
            ix2 = min(right, int(x2))
            iy2 = min(bottom, int(y2))
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (right - left) * (bottom - top)
            area_b = (int(x2) - int(x1)) * (int(y2) - int(y1))
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best, best_iou = f, iou
        return best

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def face_distance(
        self,
        known_encodings: list[Encoding],
        encoding: Encoding,
    ) -> np.ndarray:
        if not known_encodings:
            return np.array([], dtype=np.float64)
        known = np.array(known_encodings)
        # Cosine distance: 1 - cos_sim
        known_norm = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-10)
        enc_norm = encoding / (np.linalg.norm(encoding) + 1e-10)
        cos_sim = known_norm @ enc_norm
        return 1.0 - cos_sim

    # ------------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> np.ndarray:
        import cv2

        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Landmarks
    # ------------------------------------------------------------------

    def face_landmarks(
        self,
        image: np.ndarray,
        face_locations: list[FaceBBox],
    ) -> list[dict[str, list[tuple[int, int]]] | None]:
        # InsightFace provides a 5-point landmark set which does not map
        # cleanly to the 68-point dlib landmark dict.  Return None so
        # callers (e.g. ref scoring) fall back to the no-landmark path.
        return [None] * len(face_locations)
