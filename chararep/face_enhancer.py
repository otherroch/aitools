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

# Track-to-track damping for enhancement crop boxes.
_BOX_HISTORY_WEIGHT = 0.65

# Max allowed center drift per frame as a fraction of the enhancement box.
_BOX_MAX_SHIFT_RATIO = 0.08

# Previous-frame weight for low-frequency enhancement residual damping.
_RESIDUAL_HISTORY_WEIGHT = 0.2

# Scale factor for normalising pixel values to [-1, 1].
_NORM_SCALE = 127.5

# Normalised FFHQ-style 5-point template used to align faces before
# enhancement when a backend supports canonical aligned crops.
_ENHANCEMENT_ALIGN_TEMPLATE = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Backend: GFPGAN (PyTorch)
# ---------------------------------------------------------------------------

class _GfpganBackend:
    """Thin wrapper around the GFPGAN restorer."""

    aligned_face_size = None

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
    aligned_face_size = _INPUT_SIZE

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
        self._track_boxes: dict[int, tuple[int, np.ndarray]] = {}
        self._track_residuals: dict[int, tuple[int, np.ndarray]] = {}
        self._aligned_mask_cache: dict[int, np.ndarray] = {}

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
    def _normalize_landmarks(landmarks) -> np.ndarray | None:
        """Return landmarks in finite ``(5, 2)`` layout when available."""
        if landmarks is None:
            return None

        pts = np.array(landmarks, dtype=np.float32, copy=True)
        if pts.shape == (2, 5):
            pts = pts.T
        if pts.shape != (5, 2) or not np.isfinite(pts).all():
            return None

        if pts[0, 0] > pts[1, 0]:
            pts[[0, 1]] = pts[[1, 0]]
        if pts[3, 0] > pts[4, 0]:
            pts[[3, 4]] = pts[[4, 3]]
        return pts

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

    @staticmethod
    def _clip_box(
        box: np.ndarray,
        frame_h: int,
        frame_w: int,
    ) -> tuple[int, int, int, int]:
        """Clip a float box to the frame while preserving at least 2 px."""
        x1, y1, x2, y2 = box.astype(np.float32)
        x1 = float(np.clip(x1, 0.0, max(frame_w - 2, 0)))
        y1 = float(np.clip(y1, 0.0, max(frame_h - 2, 0)))
        x2 = float(np.clip(x2, x1 + 2.0, frame_w))
        y2 = float(np.clip(y2, y1 + 2.0, frame_h))
        return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))

    @staticmethod
    def _aligned_template(size: int) -> np.ndarray:
        """Return the canonical enhancer template in pixel coordinates."""
        return _ENHANCEMENT_ALIGN_TEMPLATE * float(size)

    def _estimate_enhancement_affine(
        self,
        landmarks,
        size: int,
    ) -> np.ndarray | None:
        """Estimate a canonical face-alignment warp from stabilized landmarks."""
        pts = self._normalize_landmarks(landmarks)
        if pts is None:
            return None

        template = self._aligned_template(size)
        affine, _ = cv2.estimateAffinePartial2D(pts, template, method=cv2.RANSAC)
        if affine is None or not np.isfinite(affine).all():
            return None

        det = affine[0, 0] * affine[1, 1] - affine[0, 1] * affine[1, 0]
        if abs(det) < 1e-6:
            return None
        return affine.astype(np.float32)

    def _build_aligned_enhancement_mask(self, size: int) -> np.ndarray:
        """Return a soft canonical face mask for aligned enhancement crops."""
        cached = self._aligned_mask_cache.get(size)
        if cached is not None:
            return cached

        template = self._aligned_template(size)
        eye_mid = (template[0] + template[1]) * 0.5
        mouth_mid = (template[3] + template[4]) * 0.5
        eye_dist = max(float(np.linalg.norm(template[1] - template[0])), 1.0)
        mid_height = max(float(mouth_mid[1] - eye_mid[1]), eye_dist * 0.85)

        extra_pts = np.vstack(
            [
                np.array([eye_mid[0], eye_mid[1] - mid_height * 1.45]),
                np.array([template[0, 0] - eye_dist * 0.9, template[0, 1] - mid_height * 0.9]),
                np.array([template[1, 0] + eye_dist * 0.9, template[1, 1] - mid_height * 0.9]),
                np.array([template[3, 0] - eye_dist * 0.95, template[3, 1] + mid_height * 0.8]),
                np.array([template[4, 0] + eye_dist * 0.95, template[4, 1] + mid_height * 0.8]),
                np.array([mouth_mid[0], mouth_mid[1] + mid_height * 1.1]),
            ],
            dtype=np.float32,
        )

        hull_pts = np.vstack([template, extra_pts])
        hull_pts[:, 0] = np.clip(hull_pts[:, 0], 0.0, float(size - 1))
        hull_pts[:, 1] = np.clip(hull_pts[:, 1], 0.0, float(size - 1))

        mask = np.zeros((size, size), dtype=np.uint8)
        hull = cv2.convexHull(hull_pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        blur_k = max(17, int(size * 0.09)) | 1
        mask_f32 = cv2.GaussianBlur(mask, (blur_k, blur_k), 0).astype(np.float32) / 255.0
        mask_f32 = mask_f32 * mask_f32 * (3.0 - 2.0 * mask_f32)
        self._aligned_mask_cache[size] = mask_f32
        return mask_f32

    def _apply_aligned_enhancement(
        self,
        frame: np.ndarray,
        tracked_face,
        track_id: int,
        frame_idx: int,
        weight: float,
    ) -> bool:
        """Enhance a landmark-aligned canonical face crop and warp it back."""
        align_size = int(getattr(self._backend, "aligned_face_size", 0) or 0)
        if align_size < 16:
            return False

        affine = self._estimate_enhancement_affine(
            getattr(tracked_face, "landmarks", None),
            align_size,
        )
        if affine is None:
            return False

        aligned = cv2.warpAffine(
            frame,
            affine,
            (align_size, align_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        enhanced = self._backend.enhance_crop(aligned, weight=weight)
        if enhanced.shape != aligned.shape:
            return False

        residual = enhanced.astype(np.float32) - aligned.astype(np.float32)
        if track_id >= 0:
            residual = self._stabilize_enhancement_residual(track_id, frame_idx, residual)

        mask = self._build_aligned_enhancement_mask(align_size)[:, :, np.newaxis]
        residual *= mask

        affine_inv = cv2.invertAffineTransform(affine)
        warped_residual = cv2.warpAffine(
            residual,
            affine_inv,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        blended = frame.astype(np.float32) + warped_residual.astype(np.float32)
        frame[:] = np.clip(blended, 0, 255).astype(np.uint8)
        return True

    def _propose_enhancement_box(
        self,
        tracked_face,
        frame_h: int,
        frame_w: int,
    ) -> tuple[int, int, int, int]:
        """Build an enhancement ROI centered on stabilized facial landmarks."""
        x1, y1, x2, y2 = self._padded_box(tracked_face.bbox, frame_h, frame_w)
        pts = self._normalize_landmarks(getattr(tracked_face, "landmarks", None))
        if pts is None:
            return x1, y1, x2, y2

        box_w = max(float(x2 - x1), 2.0)
        box_h = max(float(y2 - y1), 2.0)
        bbox = np.array(tracked_face.bbox, dtype=np.float32)
        bbox_cx = float((bbox[0] + bbox[2]) * 0.5)
        bbox_cy = float((bbox[1] + bbox[3]) * 0.5)

        eye_mid = (pts[0] + pts[1]) * 0.5
        mouth_mid = (pts[3] + pts[4]) * 0.5
        landmark_cx = float(pts[:, 0].mean())
        landmark_cy = float((eye_mid[1] + mouth_mid[1]) * 0.5 + box_h * 0.08)

        center_x = 0.65 * landmark_cx + 0.35 * bbox_cx
        center_y = 0.65 * landmark_cy + 0.35 * bbox_cy

        recentered = np.array(
            [
                center_x - box_w * 0.5,
                center_y - box_h * 0.5,
                center_x + box_w * 0.5,
                center_y + box_h * 0.5,
            ],
            dtype=np.float32,
        )
        return self._clip_box(recentered, frame_h, frame_w)

    def _stabilize_enhancement_box(
        self,
        track_id: int,
        frame_idx: int,
        box: tuple[int, int, int, int],
        frame_h: int,
        frame_w: int,
    ) -> tuple[int, int, int, int]:
        """Reduce frame-to-frame ROI jitter for the enhancer crop."""
        current = np.array(box, dtype=np.float32)
        prev_state = self._track_boxes.get(track_id)
        if prev_state is None:
            self._track_boxes[track_id] = (frame_idx, current)
            return box

        prev_frame_idx, prev_box = prev_state
        if frame_idx != prev_frame_idx + 1:
            self._track_boxes[track_id] = (frame_idx, current)
            return box

        prev_center = np.array(
            [(prev_box[0] + prev_box[2]) * 0.5, (prev_box[1] + prev_box[3]) * 0.5],
            dtype=np.float32,
        )
        current_center = np.array(
            [(current[0] + current[2]) * 0.5, (current[1] + current[3]) * 0.5],
            dtype=np.float32,
        )

        prev_size = np.array(
            [prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]],
            dtype=np.float32,
        )
        current_size = np.array(
            [current[2] - current[0], current[3] - current[1]],
            dtype=np.float32,
        )

        max_shift = np.maximum(prev_size * _BOX_MAX_SHIFT_RATIO, 2.0)
        center_delta = np.clip(current_center - prev_center, -max_shift, max_shift)
        stabilized_center = prev_center + center_delta
        stabilized_size = (
            prev_size * _BOX_HISTORY_WEIGHT
            + current_size * (1.0 - _BOX_HISTORY_WEIGHT)
        )

        stabilized = np.array(
            [
                stabilized_center[0] - stabilized_size[0] * 0.5,
                stabilized_center[1] - stabilized_size[1] * 0.5,
                stabilized_center[0] + stabilized_size[0] * 0.5,
                stabilized_center[1] + stabilized_size[1] * 0.5,
            ],
            dtype=np.float32,
        )
        clipped = self._clip_box(stabilized, frame_h, frame_w)
        self._track_boxes[track_id] = (frame_idx, np.array(clipped, dtype=np.float32))
        return clipped

    def _build_enhancement_mask(
        self,
        tracked_face,
        crop_shape: tuple[int, int, int],
        box: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Return a soft spatial mask so enhancement fades out near crop edges."""
        crop_h, crop_w = crop_shape[:2]
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

        pts = self._normalize_landmarks(getattr(tracked_face, "landmarks", None))
        if pts is not None:
            x1, y1, _, _ = box
            local = pts - np.array([x1, y1], dtype=np.float32)
            eye_mid = (local[0] + local[1]) * 0.5
            mouth_mid = (local[3] + local[4]) * 0.5
            eye_dist = max(float(np.linalg.norm(local[1] - local[0])), 1.0)
            mid_height = max(float(mouth_mid[1] - eye_mid[1]), eye_dist * 0.85)

            extra_pts = np.vstack(
                [
                    np.array([eye_mid[0], eye_mid[1] - mid_height * 1.55]),
                    np.array([local[0, 0] - eye_dist * 0.85, local[0, 1] - mid_height * 0.95]),
                    np.array([local[1, 0] + eye_dist * 0.85, local[1, 1] - mid_height * 0.95]),
                    np.array([local[3, 0] - eye_dist * 0.95, local[3, 1] + mid_height * 0.75]),
                    np.array([local[4, 0] + eye_dist * 0.95, local[4, 1] + mid_height * 0.75]),
                    np.array([mouth_mid[0], mouth_mid[1] + mid_height * 1.15]),
                ],
                dtype=np.float32,
            )

            hull = cv2.convexHull(
                np.vstack([local, extra_pts]).astype(np.int32)
            )
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            cv2.ellipse(
                mask,
                (crop_w // 2, crop_h // 2),
                (max(1, int(crop_w * 0.34)), max(1, int(crop_h * 0.42))),
                0,
                0,
                360,
                255,
                -1,
            )

        k = max(9, int(min(crop_h, crop_w) * 0.14)) | 1
        return cv2.GaussianBlur(mask, (k, k), 0)

    def _prune_track_boxes(
        self,
        active_track_ids: set[int],
        frame_idx: int,
    ) -> None:
        """Drop stale per-track enhancement state."""
        max_age = max(1, int(self._cfg.tracker_max_age))
        stale_ids = [
            track_id
            for track_id, (last_frame_idx, _) in self._track_boxes.items()
            if track_id not in active_track_ids and frame_idx - last_frame_idx > max_age
        ]
        for track_id in stale_ids:
            self._track_boxes.pop(track_id, None)

        stale_residual_ids = [
            track_id
            for track_id, (last_frame_idx, _) in self._track_residuals.items()
            if track_id not in active_track_ids and frame_idx - last_frame_idx > max_age
        ]
        for track_id in stale_residual_ids:
            self._track_residuals.pop(track_id, None)

    def _stabilize_enhancement_residual(
        self,
        track_id: int,
        frame_idx: int,
        residual: np.ndarray,
    ) -> np.ndarray:
        """Damp low-frequency CodeFormer shimmer across consecutive frames."""
        current = residual.astype(np.float32, copy=True)
        prev_state = self._track_residuals.get(track_id)
        if prev_state is None:
            self._track_residuals[track_id] = (frame_idx, current)
            return current

        prev_frame_idx, prev_residual = prev_state
        if frame_idx != prev_frame_idx + 1:
            self._track_residuals[track_id] = (frame_idx, current)
            return current

        if prev_residual.shape != current.shape:
            prev_residual = cv2.resize(
                prev_residual,
                (current.shape[1], current.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        k = max(5, int(min(current.shape[:2]) * 0.06)) | 1
        current_low = cv2.GaussianBlur(current, (k, k), 0)
        prev_low = cv2.GaussianBlur(prev_residual, (k, k), 0)
        stabilized = current + (prev_low - current_low) * _RESIDUAL_HISTORY_WEIGHT
        self._track_residuals[track_id] = (frame_idx, stabilized)
        return stabilized

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
        active_track_ids: set[int] = set()

        for tf in tracked_faces:
            if tf.identity_label is None:
                continue

            track_id = int(getattr(tf, "track_id", -1))
            active_track_ids.add(track_id)
            if self._apply_aligned_enhancement(
                frame,
                tf,
                track_id,
                frame_idx,
                weight,
            ):
                continue

            box = self._propose_enhancement_box(tf, h, w)
            if track_id >= 0:
                box = self._stabilize_enhancement_box(track_id, frame_idx, box, h, w)

            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            logger.debug(
                "Enhancing face (track %d) in frame %d, crop %dx%d",
                tf.track_id, frame_idx, crop.shape[1], crop.shape[0],
            )
            enhanced = self._backend.enhance_crop(crop, weight=weight)
            if enhanced.shape == crop.shape:
                if track_id >= 0:
                    residual = enhanced.astype(np.float32) - crop.astype(np.float32)
                    residual = self._stabilize_enhancement_residual(
                        track_id, frame_idx, residual
                    )
                    enhanced = np.clip(
                        crop.astype(np.float32) + residual,
                        0,
                        255,
                    ).astype(np.uint8)

                mask = self._build_enhancement_mask(tf, crop.shape, box)
                alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
                blended = (
                    enhanced.astype(np.float32) * alpha
                    + crop.astype(np.float32) * (1.0 - alpha)
                )
                frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        self._prune_track_boxes(active_track_ids, frame_idx)
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
