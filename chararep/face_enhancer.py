"""Post-swap face enhancement (GFPGAN / ONNX CodeFormer).

Two enhancement backends are available:

- **gfpgan** – The original GFPGAN PyTorch restorer.  When used through
  ``enhance_faces()`` (the recommended path), faces are cropped using
  bounding boxes from the pipeline's existing detection so that GFPGAN's
  own internal face re-detection is skipped, significantly reducing
  per-frame latency.
- **codeformer_onnx** – A CodeFormer model exported to ONNX and executed
  via ONNX Runtime with ``CUDAExecutionProvider``.  This shares the same
  GPU inference path as the rest of the pipeline, avoids PyTorch overhead,
  and is typically the fastest option on CUDA-capable hardware.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Padding factor applied to face bounding boxes before cropping.
# 0.5 means 50 % of the face width/height is added on each side.
_CROP_PAD_FACTOR = 0.5

# Scale factor for normalising pixel values to [-1, 1].
_NORM_SCALE = 127.5


# ---------------------------------------------------------------------------
# Backend: GFPGAN (PyTorch)
# ---------------------------------------------------------------------------

class _GfpganBackend:
    """Thin wrapper around the GFPGAN restorer."""

    def __init__(self, cfg: PipelineConfig):
        self._restorer = None
        try:
            from gfpgan import GFPGANer
        except ImportError:
            logger.warning(
                "gfpgan not installed – face enhancement unavailable. "
                "Install with: pip install gfpgan"
            )
            return

        model_dir = Path.home() / ".gfpgan" / "weights"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = (
            cfg.enhance_model_path
            or str(model_dir / "GFPGANv1.4.pth")
        )

        device = torch.device(
            f"cuda:{cfg.device_id}"
            if torch.cuda.is_available()
            else "cpu"
        )
        try:
            self._restorer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=device,
            )
            logger.info("GFPGAN face enhancer ready.")
        except Exception as exc:
            logger.warning("GFPGAN init failed (%s) – enhancement disabled.", exc)

    @property
    def available(self) -> bool:
        return self._restorer is not None

    def enhance_crop(
        self,
        face_crop: np.ndarray,
        weight: float = 0.7,
    ) -> np.ndarray:
        """Enhance a single face crop with GFPGAN.

        The crop is passed with ``has_aligned=False`` and
        ``only_center_face=True`` so GFPGAN looks for at most one face
        in the (small) crop and skips scanning the full frame.
        """
        if self._restorer is None:
            return face_crop
        try:
            _, _, output = self._restorer.enhance(
                face_crop,
                has_aligned=False,
                only_center_face=True,
                paste_back=True,
                weight=weight,
            )
            return output
        except Exception:
            return face_crop

    def enhance_full_frame(
        self,
        frame: np.ndarray,
        weight: float = 0.7,
    ) -> np.ndarray:
        """Enhance all faces in a full frame (legacy path).

        This uses GFPGAN's internal face detection which is slower
        than the crop-based path but does not require external bounding
        boxes.
        """
        if self._restorer is None:
            return frame
        try:
            _, _, output = self._restorer.enhance(
                frame,
                has_aligned=False,
                only_center_face=True,
                paste_back=True,
                weight=weight,
            )
            return output
        except Exception:
            return frame


# ---------------------------------------------------------------------------
# Backend: ONNX CodeFormer
# ---------------------------------------------------------------------------

class _OnnxCodeFormerBackend:
    """CodeFormer model running via ONNX Runtime with CUDA support.

    The ONNX CodeFormer model expects a **512×512** normalised face crop
    (pixel values in ``[-1, 1]``, channel order RGB, NCHW layout).  The
    output is a restored 512×512 face in the same format.
    """

    _INPUT_SIZE = 512

    def __init__(self, cfg: PipelineConfig):
        self._session = None
        if not cfg.enhance_model_path:
            logger.warning(
                "codeformer_onnx selected but --enhance-model-path not set – "
                "enhancement disabled."
            )
            return

        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "onnxruntime not installed – ONNX CodeFormer unavailable."
            )
            return

        from .gpu_utils import get_onnx_providers

        model_path = cfg.enhance_model_path
        providers = get_onnx_providers(cfg.device_id)
        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            logger.info(
                "ONNX CodeFormer enhancer ready (model=%s, input=%s).",
                model_path,
                inp.shape,
            )
        except Exception as exc:
            logger.warning(
                "ONNX CodeFormer init failed (%s) – enhancement disabled.", exc
            )
            self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def enhance_crop(
        self,
        face_crop: np.ndarray,
        weight: float = 0.7,
    ) -> np.ndarray:
        """Enhance a single face crop via the ONNX CodeFormer model."""
        if self._session is None:
            return face_crop

        orig_h, orig_w = face_crop.shape[:2]
        sz = self._INPUT_SIZE

        try:
            # Pre-process: resize → RGB → float32 [-1, 1] → NCHW
            resized = cv2.resize(face_crop, (sz, sz), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            tensor = (rgb / _NORM_SCALE) - 1.0     # [-1, 1]
            tensor = tensor.transpose(2, 0, 1)     # HWC → CHW
            tensor = np.expand_dims(tensor, 0)     # → NCHW

            outputs = self._session.run(None, {self._input_name: tensor})
            out = np.squeeze(outputs[0])           # CHW
            out = np.clip(out, -1.0, 1.0)
            out = ((out + 1.0) * _NORM_SCALE).astype(np.uint8)
            out = out.transpose(1, 2, 0)           # → HWC
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            # Resize back to original crop dimensions
            if (orig_h, orig_w) != (sz, sz):
                out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # Blend with original according to weight
            if weight < 1.0:
                out = cv2.addWeighted(out, weight, face_crop, 1.0 - weight, 0)
            return out
        except Exception:
            return face_crop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FaceEnhancer:
    """Restores and super-resolves swapped faces.

    Delegates to either :class:`_GfpganBackend` or
    :class:`_OnnxCodeFormerBackend` based on the ``enhancement_model``
    config field.

    The recommended entry point is :meth:`enhance_faces`, which operates
    on individual face crops (extracted via bounding boxes already known
    to the pipeline) instead of processing the full frame.  This avoids
    redundant face re-detection and is significantly faster.
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._backend = None

        if not cfg.enable_face_enhancement:
            logger.info("Face enhancement disabled.")
            return

        model = cfg.enhancement_model.lower()
        if model == "codeformer_onnx":
            self._backend = _OnnxCodeFormerBackend(cfg)
        else:
            self._backend = _GfpganBackend(cfg)

    @property
    def available(self) -> bool:
        return self._backend is not None and self._backend.available

    # -- crop helpers ------------------------------------------------------

    @staticmethod
    def _padded_box(
        bbox: np.ndarray,
        frame_h: int,
        frame_w: int,
        pad: float = _CROP_PAD_FACTOR,
    ) -> tuple[int, int, int, int]:
        """Expand *bbox* by *pad* fraction and clip to frame bounds."""
        x1, y1, x2, y2 = bbox[:4].astype(float)
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad, bh * pad
        x1 = max(0, int(x1 - px))
        y1 = max(0, int(y1 - py))
        x2 = min(frame_w, int(x2 + px))
        y2 = min(frame_h, int(y2 + py))
        return x1, y1, x2, y2

    # -- public methods ----------------------------------------------------

    def enhance_faces(
        self,
        frame: np.ndarray,
        tracked_faces: list,
        frame_idx: int,
    ) -> np.ndarray:
        """Enhance every face in *frame* using bounding boxes from the tracker.

        Each face is cropped (with padding), enhanced through the active
        backend, and pasted back into *frame*.  This is the recommended
        path because it avoids re-detecting faces inside the enhancer.

        Args:
            frame: BGR image (uint8) – modified **in place** and returned.
            tracked_faces: ``TrackedFace`` objects from the detector.
            frame_idx: current frame number (used for logging).

        Returns:
            Frame with enhanced face regions.
        """
        if self._backend is None or not self._backend.available:
            return frame

        h, w = frame.shape[:2]
        weight = self._cfg.enhancement_weight

        for tf in tracked_faces:
            if tf.identity_label is None:
                continue
            x1, y1, x2, y2 = self._padded_box(tf.bbox, h, w)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            logger.debug(
                "Enhancing face (track %d) in frame %d, crop %dx%d",
                tf.track_id, frame_idx, crop.shape[1], crop.shape[0],
            )
            enhanced = self._backend.enhance_crop(crop, weight=weight)
            if enhanced.shape == crop.shape:
                frame[y1:y2, x1:x2] = enhanced
        return frame

    def enhance(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Enhance all faces detected in the frame (legacy full-frame path).

        Prefer :meth:`enhance_faces` when bounding boxes are available —
        it is significantly faster because the enhancer does not need to
        re-detect faces.

        Args:
            frame: BGR image (uint8).

        Returns:
            Enhanced frame.
        """
        if self._backend is None or not self._backend.available:
            return frame

        logger.debug("Enhancing faces in frame %d (full-frame path)...", frame_idx)
        try:
            # Full-frame path – only meaningful for GFPGAN which has its
            # own internal face detector.
            if isinstance(self._backend, _GfpganBackend):
                return self._backend.enhance_full_frame(
                    frame, weight=self._cfg.enhancement_weight
                )
            # For other backends fall back to returning the frame unchanged
            # (they require pre-cropped faces).
            return frame
        except Exception as exc:
            logger.debug("Enhancement failed for frame %d (%s) – returning original.", frame_idx, exc)
            return frame

    def enhance_crop(
        self,
        face_crop: np.ndarray,
        weight: Optional[float] = None,
    ) -> np.ndarray:
        """Enhance a single face crop.

        Args:
            face_crop: BGR face image (uint8).
            weight: blend weight (0 = original, 1 = fully restored).

        Returns:
            Enhanced face crop.
        """
        if self._backend is None or not self._backend.available:
            return face_crop

        w = weight if weight is not None else self._cfg.enhancement_weight
        return self._backend.enhance_crop(face_crop, weight=w)
