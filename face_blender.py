"""Mask-based blending of swapped face regions back into the frame.

After the inswapper model composites the new face via ``paste_back=True``,
this module further refines the seam between the swapped region and the
original background.  Two strategies are available:

- **seamless** – OpenCV Poisson seamlessClone for lighting-matched blending.
- **alpha**    – Gaussian-blurred alpha mask for soft compositing.
"""

import logging

import cv2
import numpy as np

from config import PipelineConfig
from face_detector import TrackedFace

logger = logging.getLogger(__name__)


class FaceBlender:
    """Blends swapped face images smoothly into the original frame."""

    def __init__(self, cfg: PipelineConfig):
        self._mode = cfg.blend_mode
        self._blur_k = cfg.mask_blur_kernel
        self._erode_px = cfg.mask_erode_pixels

    def blend_all(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        faces: list[TrackedFace],
        frame_idx: int,
    ) -> np.ndarray:
        """Blend all swapped faces at once.

        Builds a combined soft mask from every face that has been swapped
        (i.e. ``identity_label is not None``) and applies a single blend pass
        so that overlapping regions are handled correctly.

        Args:
            original: unmodified BGR frame (uint8).
            swapped:  frame after all face swaps (BGR, uint8).
            faces:    tracked faces from this frame.

        Returns:
            Blended BGR frame.
        """
        h, w = original.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        swap_faces = [f for f in faces if f.identity_label is not None]
        if not swap_faces:
            return swapped

        logger.debug("Frame %d: Blending %d swapped faces with mode '%s'", frame_idx, len(swap_faces), self._mode)

        # Build and cache individual face masks so _poisson_blend_combined
        # can reuse them instead of rebuilding per face.
        face_masks: list[np.ndarray] = []
        for tf in swap_faces:
            face_mask = self._build_mask((h, w), tf.bbox, tf.landmarks)
            face_masks.append(face_mask)
            combined_mask = np.maximum(combined_mask, face_mask)

        if self._mode == "seamless":
            return self._poisson_blend_combined(
                original, swapped, combined_mask, swap_faces, face_masks
            )
        return self._alpha_blend(original, swapped, combined_mask)

    # ── mask construction ────────────────────────────────────────────────

    def _build_mask(
        self,
        frame_shape: tuple[int, int],
        bbox: np.ndarray,
        landmarks: np.ndarray | None,
    ) -> np.ndarray:
        """Create a soft binary mask for a single face region."""
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if landmarks is not None and len(landmarks) >= 5:
            pts = landmarks.astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
            # Dilate to include forehead and sides
            ksize = max(1, int((x2 - x1) * 0.15)) | 1
            mask = cv2.dilate(mask, np.ones((ksize, ksize), np.uint8))
            # Fade the mask downward so it does not bleed into beard/chin.
            # Use bbox proportion (stable per-frame) rather than mouth
            # landmark y-position, which shifts as the person talks and
            # causes Poisson-clone flickering when clip_y changes each frame.
            face_h = y2 - y1
            # Clip the mask at ~82% of the face bbox height, which
            # empirically lands around the upper chin / jaw line for
            # typical frontal and three-quarter views. This keeps the
            # blended region out of beards/chins while preserving cheeks.
            clip_y = min(h, y1 + int(face_h * 0.82))
            # Use ~20% of face height as the half-width of the vertical
            # fade band. This gives a soft transition instead of a hard
            # edge and scales with face size, while the max() ensures the
            # fade is at least as wide as the dilation kernel.
            fade_px = max(ksize // 2 + 1, int(face_h * 0.20))
            start_fade = max(0, clip_y - fade_px)
            end_fade = min(h, clip_y + fade_px)
            if start_fade < end_fade:
                fade_len = end_fade - start_fade
                ramp = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                mask[start_fade:end_fade, :] = (
                    mask[start_fade:end_fade, :].astype(np.float32)
                    * ramp[:, np.newaxis]
                ).astype(np.uint8)
            mask[end_fade:, :] = 0
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # Erode to pull the mask inward
        if self._erode_px > 0:
            kernel = np.ones(
                (self._erode_px * 2 + 1, self._erode_px * 2 + 1), np.uint8
            )
            mask = cv2.erode(mask, kernel)

        # Feather the edges
        if self._blur_k > 0:
            k = self._blur_k | 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return mask

    # ── blending strategies ──────────────────────────────────────────────

    def _poisson_blend_combined(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        mask: np.ndarray,
        faces: list[TrackedFace],
        face_masks: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Poisson seamless clone applied per-face.

        When there is only one face, a single seamlessClone call is used.
        For multiple faces the clone is applied independently for each face
        so that the centre point always lies inside the face mask, avoiding
        undefined behaviour when two faces are far apart.

        Args:
            original:   unmodified BGR frame.
            swapped:    frame after all face swaps.
            mask:       combined soft mask for all faces (used in single-face path).
            faces:      tracked faces that have been swapped.
            face_masks: pre-computed per-face masks from ``blend_all`` (reused
                        to avoid redundant ``_build_mask`` calls).
        """
        if len(faces) == 1:
            # Fast path: single face – use the combined mask centroid.
            coords = np.argwhere(mask > 0)
            if coords.size == 0:
                return swapped
            cy = int(coords[:, 0].mean())
            cx = int(coords[:, 1].mean())
            binary_mask = (mask > 128).astype(np.uint8) * 255
            try:
                return cv2.seamlessClone(
                    swapped, original, binary_mask, (cx, cy), cv2.NORMAL_CLONE
                )
            except cv2.error:
                return self._alpha_blend(original, swapped, mask)

        # Multiple faces: blend iteratively so each clone uses only its own
        # face mask, ensuring the centre is always inside the painted region.
        # Reuse cached masks from blend_all when available.
        h, w = original.shape[:2]
        result = original.copy()
        for i, tf in enumerate(faces):
            face_mask = (
                face_masks[i]
                if face_masks is not None and i < len(face_masks)
                else self._build_mask((h, w), tf.bbox, tf.landmarks)
            )
            coords = np.argwhere(face_mask > 0)
            if coords.size == 0:
                continue
            cy = int(coords[:, 0].mean())
            cx = int(coords[:, 1].mean())
            binary = (face_mask > 128).astype(np.uint8) * 255
            try:
                result = cv2.seamlessClone(
                    swapped, result, binary, (cx, cy), cv2.NORMAL_CLONE
                )
            except cv2.error:
                result = self._alpha_blend(result, swapped, face_mask)
        return result

    @staticmethod
    def _alpha_blend(
        original: np.ndarray,
        swapped: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Simple alpha compositing using the soft mask."""
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]
        blended = (
            swapped.astype(np.float32) * alpha
            + original.astype(np.float32) * (1.0 - alpha)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)
