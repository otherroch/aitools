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

from face_ops.mixin import FaceBackendMixin
from face_ops.types import DetectedFace, Encoding, FaceBBox

logger = logging.getLogger(__name__)


class InsightFaceBackend(FaceBackendMixin):
    """Face detection and ArcFace encoding via the insightface library.

    Args:
        model_name: InsightFace model pack (default ``"buffalo_l"``).
            Other useful choices: ``"buffalo_s"`` (smaller/faster),
            ``"buffalo_sc"`` (CPU-friendly).
        ctx_id: ONNX Runtime device id.  ``0`` = first GPU, ``-1`` = CPU.
        det_size: Detection input resolution as ``(width, height)``.
        det_thresh: Detection confidence threshold (default ``0.5``).
        providers: Optional explicit list of ONNX Runtime execution
            providers.  When *None* (the default), providers are chosen
            automatically via :meth:`_providers`.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        providers: list | None = None,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for the InsightFace backend.\n"
                "Install it with: pip install insightface onnxruntime"
            ) from exc

        if providers is not None:
            effective_providers = providers
            effective_ctx_id = ctx_id
        else:
            effective_providers, effective_ctx_id = self._providers(ctx_id)
        logger.info(
            "Initializing InsightFaceBackend with model=%s, ctx_id=%s, "
            "providers=%s, det_size=%s, det_thresh=%.2f",
            model_name,
            effective_ctx_id,
            effective_providers,
            det_size,
            det_thresh,
        )
        self._app = FaceAnalysis(
            name=model_name,
            providers=effective_providers,
        )
        logger.debug("Preparing InsightFace model on device %s...", effective_ctx_id)
        self._app.prepare(
            ctx_id=effective_ctx_id,
            det_size=det_size,
            det_thresh=det_thresh,
        )

    # ------------------------------------------------------------------
    # public access to underlying FaceAnalysis
    # ------------------------------------------------------------------

    @property
    def app(self):
        """The underlying InsightFace ``FaceAnalysis`` instance.

        Useful for callers (e.g. *chararep*) that need access to raw
        ``Face`` objects returned by ``app.get()`` — data that is richer
        than the backend-agnostic :class:`FaceBBox` / :class:`Encoding`
        types.
        """
        return self._app

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _providers(ctx_id: int) -> tuple[list[str], int]:
        try:
            import onnxruntime as ort

            available_providers = set(ort.get_available_providers())
        except Exception:
            available_providers = set()

        cpu_available = "CPUExecutionProvider" in available_providers

        if ctx_id >= 0 and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
            if cpu_available:
                providers.append("CPUExecutionProvider")
            return providers, ctx_id

        if ctx_id >= 0:
            logger.warning(
                "CUDAExecutionProvider requested (ctx_id=%s) but not available; "
                "falling back to CPUExecutionProvider.",
                ctx_id,
            )

        if cpu_available:
            return ["CPUExecutionProvider"], -1

        return ["CPUExecutionProvider"], -1
    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_faces(
        self,
        image: np.ndarray,
    ) -> list[FaceBBox]:
        # The model is set during __init__ — no per-call model hint needed.
        faces = self._app.get(image)
        bboxes: list[FaceBBox] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            # Convert (x1, y1, x2, y2) → (top, right, bottom, left)
            bboxes.append((int(y1), int(x2), int(y2), int(x1)))
        return bboxes

    def detect(
        self,
        image: np.ndarray,
    ) -> list[DetectedFace]:
        faces = self._app.get(image)
        results: list[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            bbox: FaceBBox = (y1, x2, y2, x1)

            emb = None
            if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                emb = np.array(f.normed_embedding, dtype=np.float32)
            elif hasattr(f, "embedding") and f.embedding is not None:
                emb = np.array(f.embedding, dtype=np.float32)

            landmarks = None
            if hasattr(f, "kps") and f.kps is not None:
                landmarks = np.array(f.kps, dtype=np.float32)

            results.append(
                DetectedFace(bbox=bbox, embedding=emb, landmarks=landmarks, raw=f)
            )
        return results

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

    @staticmethod
    def _cosine_distance(
        known_encodings: list[Encoding],
        encoding: Encoding,
    ) -> np.ndarray:
        """Compute cosine distances between *encoding* and each known encoding."""
        if not known_encodings:
            return np.array([], dtype=np.float64)
        known = np.array(known_encodings)
        # Cosine distance: 1 - cos_sim
        known_norm = known / (np.linalg.norm(known, axis=1, keepdims=True) + 1e-10)
        enc_norm = encoding / (np.linalg.norm(encoding) + 1e-10)
        cos_sim = known_norm @ enc_norm
        return 1.0 - cos_sim

    def face_distance(
        self,
        known_encodings: list[Encoding],
        encoding: Encoding,
    ) -> np.ndarray:
        return self._cosine_distance(known_encodings, encoding)

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
