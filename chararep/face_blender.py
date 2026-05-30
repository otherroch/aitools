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

from .config import PipelineConfig
from .face_detector import TrackedFace

logger = logging.getLogger(__name__)

_BLEND_TEMPLATE = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
)
_BLEND_MASK_SIZE = 256


class FaceBlender:
    """Blends swapped face images smoothly into the original frame."""

    def __init__(self, cfg: PipelineConfig):
        self._mode = cfg.blend_mode
        self._blur_k = cfg.mask_blur_kernel
        self._erode_px = cfg.mask_erode_pixels
        self._canonical_mask_cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _normalize_landmarks(landmarks: np.ndarray | None) -> np.ndarray | None:
        """Return finite landmarks in stable ``(5, 2)`` layout when available."""
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
    def _blend_template(size: int) -> np.ndarray:
        """Return the canonical 5-point template in pixel coordinates."""
        return _BLEND_TEMPLATE * float(size)

    def _canonical_face_mask(self, size: int) -> np.ndarray:
        """Build a stable face-shaped mask in canonical aligned space."""
        cached = self._canonical_mask_cache.get(size)
        if cached is not None:
            return cached

        template = self._blend_template(size)
        left_eye, right_eye, _, mouth_left, mouth_right = template
        eye_mid = (left_eye + right_eye) * 0.5
        mouth_mid = (mouth_left + mouth_right) * 0.5
        eye_dist = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
        mid_height = max(float(mouth_mid[1] - eye_mid[1]), eye_dist * 0.9)

        extra_pts = np.vstack(
            [
                np.array([eye_mid[0], eye_mid[1] - mid_height * 1.35]),
                np.array([left_eye[0] - eye_dist * 1.0, left_eye[1] - mid_height * 0.7]),
                np.array([right_eye[0] + eye_dist * 1.0, right_eye[1] - mid_height * 0.7]),
                np.array([left_eye[0] - eye_dist * 1.1, eye_mid[1] + mid_height * 0.1]),
                np.array([right_eye[0] + eye_dist * 1.1, eye_mid[1] + mid_height * 0.1]),
                np.array([mouth_left[0] - eye_dist * 0.95, mouth_left[1] + mid_height * 0.7]),
                np.array([mouth_right[0] + eye_dist * 0.95, mouth_right[1] + mid_height * 0.7]),
                np.array([mouth_mid[0], mouth_mid[1] + mid_height * 1.2]),
            ],
            dtype=np.float32,
        )

        hull_pts = np.vstack([template, extra_pts])
        hull_pts[:, 0] = np.clip(hull_pts[:, 0], 0.0, float(size - 1))
        hull_pts[:, 1] = np.clip(hull_pts[:, 1], 0.0, float(size - 1))

        mask = np.zeros((size, size), dtype=np.uint8)
        hull = cv2.convexHull(hull_pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        base_k = max(15, int(size * 0.09)) | 1
        mask_f32 = cv2.GaussianBlur(mask, (base_k, base_k), 0).astype(np.float32) / 255.0
        mask_f32 = mask_f32 * mask_f32 * (3.0 - 2.0 * mask_f32)

        y_grid = np.arange(size, dtype=np.float32).reshape(size, 1)
        brow_center = eye_mid[1] - mid_height * 0.05
        brow_sigma = max(8.0, mid_height * 0.65)
        brow_band = np.exp(-0.5 * ((y_grid - brow_center) / brow_sigma) ** 2)
        top_start = eye_mid[1] - mid_height * 1.05
        top_end = eye_mid[1] + mid_height * 0.2
        top_span = max(top_end - top_start, 1.0)
        top_ramp = np.clip((y_grid - top_start) / top_span, 0.0, 1.0)
        brow_weight = np.minimum(0.72 + 0.28 * top_ramp, 1.0 - brow_band * 0.22)
        mask_f32 *= brow_weight

        self._canonical_mask_cache[size] = mask_f32
        return mask_f32

    def _build_aligned_mask(
        self,
        frame_shape: tuple[int, int],
        bbox: np.ndarray,
        landmarks: np.ndarray | None,
    ) -> np.ndarray | None:
        """Warp a canonical face mask into frame space using facial landmarks."""
        pts = self._normalize_landmarks(landmarks)
        if pts is None:
            return None

        h, w = frame_shape
        template = self._blend_template(_BLEND_MASK_SIZE)
        affine, _ = cv2.estimateAffinePartial2D(pts, template, method=cv2.RANSAC)
        if affine is None or not np.isfinite(affine).all():
            return None

        det = affine[0, 0] * affine[1, 1] - affine[0, 1] * affine[1, 0]
        if abs(det) < 1e-6:
            return None

        mask_src = self._canonical_face_mask(_BLEND_MASK_SIZE)
        affine_inv = cv2.invertAffineTransform(affine)
        warped = cv2.warpAffine(
            mask_src,
            affine_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        x1, y1, x2, y2 = bbox.astype(int)
        face_w = max(x2 - x1, 2)
        face_h = max(y2 - y1, 2)
        pad_x = max(2, int(face_w * 0.55))
        pad_y = max(2, int(face_h * 0.7))
        clip_x1 = max(0, x1 - pad_x)
        clip_y1 = max(0, y1 - pad_y)
        clip_x2 = min(w, x2 + pad_x)
        clip_y2 = min(h, y2 + pad_y)
        clip_mask = np.zeros((h, w), dtype=np.float32)
        clip_mask[clip_y1:clip_y2, clip_x1:clip_x2] = 1.0
        warped *= clip_mask

        local_k = max(7, int(max(face_w, face_h) * 0.12)) | 1
        warped = cv2.GaussianBlur(warped, (local_k, local_k), 0)
        return np.clip(warped * 255.0, 0, 255).astype(np.uint8)

    def blend_all(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        faces: list[TrackedFace],
        frame_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend all swapped faces at once.

        Builds a combined soft mask from every face that has been swapped
        (i.e. ``identity_label is not None``) and applies a single blend pass
        so that overlapping regions are handled correctly.

        Args:
            original: unmodified BGR frame (uint8).
            swapped:  frame after all face swaps (BGR, uint8).
            faces:    tracked faces from this frame.

        Returns:
            Tuple of (Blended BGR frame, combined soft mask).
        """
        h, w = original.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        swap_faces = [f for f in faces if f.identity_label is not None]
        if not swap_faces:
            return swapped, combined_mask

        logger.debug("Frame %d: Blending %d swapped faces with mode '%s'", frame_idx, len(swap_faces), self._mode)

        # Build and cache individual face masks so _poisson_blend_combined
        # can reuse them instead of rebuilding per face.
        face_masks: list[np.ndarray] = []
        for tf in swap_faces:
            face_mask = self._build_mask((h, w), tf.bbox, tf.landmarks)
            face_masks.append(face_mask)
            combined_mask = np.maximum(combined_mask, face_mask)
        # Apply additional smoothing to reduce jitter in upper face regions
        # This helps with eyes/eyebrows which are particularly sensitive to 
        # edge transitions and can cause flickering
        if self._mode == "seamless":
            # Apply extra smoothing to the combined mask for seamless blending
            combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)
            
            # Apply even more aggressive smoothing specifically to the upper face region
            # to reduce jitter in eyes and eyebrows
            coords_y, coords_x = np.where(combined_mask > 0)
            if len(coords_y) > 0:
                # Calculate vertical extent of the face in the mask
                mask_height = coords_y.max() - coords_y.min()
                mask_center_y = (coords_y.min() + coords_y.max()) / 2
                
                # Apply stronger smoothing to upper 40% of the face where eyes/eyebrows are
                upper_region_end = int(coords_y.min() + mask_height * 0.3)
                if upper_region_end > 0:
                    # Extract upper region
                    upper_mask = combined_mask[:upper_region_end, :]
                    # Apply even stronger blur to upper region
                    upper_smoothed = cv2.GaussianBlur(upper_mask, (19, 19), 0)
                    combined_mask[:upper_region_end, :] = upper_smoothed

        if self._mode == "seamless":
            return self._poisson_blend_combined(
                original, swapped, combined_mask, swap_faces, face_masks
            ), combined_mask
        return self._alpha_blend(original, swapped, combined_mask), combined_mask

    # ── mask construction ────────────────────────────────────────────────

    def _build_mask(
        self,
        frame_shape: tuple[int, int],
        bbox: np.ndarray,
        landmarks: np.ndarray | None,
    ) -> np.ndarray:
        """Create a soft binary mask for a single face region.

        The 5-point landmark set from InsightFace (left eye, right eye,
        nose tip, mouth corners) is sparse and misses eyebrows/forehead.
        This method synthesises additional points by expanding the
        landmark hull outward so the convex hull covers the full face
        including eyebrows, hairline, and jawline.

        Edge vibration/flicker is reduced by applying a radial distance
        ramp from the face centroid so the alpha smoothly drops to near-
        zero at the boundary rather than an abrupt binary cut-off.
        """
        h, w = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        aligned_mask = self._build_aligned_mask(frame_shape, bbox, landmarks)
        if aligned_mask is not None:
            mask = aligned_mask
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rx, ry = (x2 - x1) // 2, (y2 - y1) // 2
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # Erode to pull the mask inward slightly (prevents border artefacts)
        # Use 0 erosion to maximize swap area - rely on blur for edge smoothness
        if self._erode_px > 0:
            kernel = np.ones(
                (self._erode_px * 2 + 1, self._erode_px * 2 + 1), np.uint8
            )
            mask = cv2.erode(mask, kernel)

        # Enhanced Gaussian feathering for ultra-smooth alpha blending
        # Larger kernel = wider transition zone = less edge flicker
        # Enhanced for upper face region to reduce jitter
        if self._blur_k > 0:
            k = self._blur_k | 1
            if k < 1:
                k = 1
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
        """Hybrid Poisson + alpha blend to reduce edge vibration.

        Strategy:
        1. Run seamlessClone with a *shrunk* binary mask so the hard
           poisson boundary sits well inside the visible face region.
        2. After Poisson, alpha-blend a soft transition ring between
           the poisson region and the original background. This ring
           uses the full (expanded) soft mask so the final composite
           has no abrupt edges.

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
            return self._hybrid_blend_one(
                original, swapped, mask
            )

        # Multiple faces: blend iteratively
        h, w = original.shape[:2]
        result = original.copy()
        for i, tf in enumerate(faces):
            face_mask = (
                face_masks[i]
                if face_masks is not None and i < len(face_masks)
                else self._build_mask((h, w), tf.bbox, tf.landmarks)
            )
            result = self._hybrid_blend_one(result, swapped, face_mask)
        return result

    @staticmethod
    def _hybrid_blend_one(
        original: np.ndarray,
        swapped: np.ndarray,
        soft_mask: np.ndarray,
    ) -> np.ndarray:
        """Blend one face region using Poisson core + alpha ring.

        The soft_mask (0-255) defines the full face area.  We shrink
        it to create a tighter binary mask for seamlessClone (pushing
        the hard poisson boundary inward), then alpha-composite a ring
        around that boundary to feather it smoothly.
        """
        coords = np.argwhere(soft_mask > 0)
        if coords.size == 0:
            return swapped

        cy = int(coords[:, 0].mean())
        cx = int(coords[:, 1].mean())

        # Shrink the mask to create a safe inner region for Poisson.
        # The inner region is where soft_mask > threshold (high values =
        # face centre).  Anything below becomes the transition zone.
        inner_thresh = 155  # increased from 140 to reduce jitter in eyebrow region
        inner_binary = (soft_mask > inner_thresh).astype(np.uint8) * 255

        # Check inner mask has enough pixels for seamlessClone
        if inner_binary.sum() < 200:
            # Mask too small — fall back to plain alpha blend
            return FaceBlender._alpha_blend(original, swapped, soft_mask)

        # Shrink inner mask slightly to guarantee the boundary is
        # safely inside the face (reduces artefacts).
        # One erosion pass instead of two keeps the Poisson region wider.
        erode_kernel = np.ones((3, 3), np.uint8)
        inner_binary = cv2.erode(inner_binary, erode_kernel)

        # ── Step 1: Poisson clone the inner region ──
        try:
            poisson_result = cv2.seamlessClone(
                swapped, original, inner_binary, (cx, cy), cv2.NORMAL_CLONE
            )
        except cv2.error:
            return FaceBlender._alpha_blend(original, swapped, soft_mask)

        # ── Step 2: Alpha-blend transition ring with distance-based fading ──
        # Build a smooth alpha mask using distance transform for edge smoothing.
        # This reduces jitter by creating a smooth gradient from center to edge.
        mask_f32 = soft_mask.astype(np.float32) / 255.0
        binary = (mask_f32 > 0.1).astype(np.uint8)
        
        # Use distance transform to create smooth edge fading
        if binary.any():
            # Distance transform: each pixel gets its distance to the mask boundary
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            # SIGNIFICANTLY INCREASED fade width for upper face region
            # to eliminate jitter in eyes/eyebrows area.
            coords_y, coords_x = np.where(binary > 0)
            if len(coords_y) > 0:
                face_h = coords_y.max() - coords_y.min()
                face_w = coords_x.max() - coords_x.min()
                face_dim = max(face_h, face_w)
                fade_width = max(14, int(face_dim * 0.18))  # ~18% (increased from 12%)
            else:
                fade_width = 16
            # Use SMOOTHSTEP curve for much gentler edge transition.
            # This dramatically reduces jitter since small boundary shifts
            # produce much smaller alpha changes near the edge.
            t = np.clip(dist / fade_width, 0.0, 1.0)
            edge_alpha = t * t * (3.0 - 2.0 * t)  # smoothstep: t^2*(3-2t)
            # Combine: inside the mask, use distance-based alpha
            mask_f32 = np.where(binary, edge_alpha, 0.0)
        
        # Apply additional smoothing specifically for the upper face region
        # to reduce jitter in eyes and eyebrows
        if len(coords_y) > 0:
            # Create a vertical weighting that emphasizes smoothing in the upper face
            # where jitter is most noticeable
            y_grid = np.arange(swapped.shape[0]).reshape(swapped.shape[0], 1)
            
            # Calculate vertical extent of the face
            mask_height = coords_y.max() - coords_y.min()
            mask_center_y = (coords_y.min() + coords_y.max()) / 2.0
            
            # Create vertical Gaussian weighting that peaks at the face center
            # and decreases toward the top (forehead) and bottom (chin)
            # This helps reduce jitter in the upper face region where eyes/eyebrows are
            vert_sigma = max(20, mask_height * 0.3)
            vert_weight = np.exp(-0.5 * ((y_grid - mask_center_y) / vert_sigma) ** 2)
            
            # Boost upper region smoothing for eyes/eyebrows
            vert_weight = np.clip(vert_weight, 0.0, 1.0)
            # Make upper region even more sensitive to smoothing
            vert_weight = 0.4 + 0.6 * vert_weight
            
            # Apply the vertical weighting to the mask
            mask_f32 = mask_f32 * vert_weight
            
        # Apply additional Gaussian blur for ultra-smooth transitions
        # Increased kernel size for more smoothing in upper face region
        mask_f32 = cv2.GaussianBlur(mask_f32, (11, 11), 0)
        mask_f32 = np.clip(mask_f32, 0, 1)
        
        # Set the inner region (Poisson result) to full alpha
        mask_f32[inner_binary > 0] = 1.0
        mask_alpha = mask_f32[:, :, np.newaxis]

        result = (
            poisson_result.astype(np.float32) * mask_alpha
            + original.astype(np.float32) * (1.0 - mask_alpha)
        )
        return np.clip(result, 0, 255).astype(np.uint8)

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