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
        # Apply additional smoothing to reduce jitter in upper face regions
        # This helps with eyes/eyebrows which are particularly sensitive to 
        # edge transitions and can cause flickering
        if self._mode == "seamless":
            # Apply extra smoothing to the face mask for seamless blending
            face_mask = cv2.GaussianBlur(face_mask, (11, 11), 0)
            
            # Apply even more aggressive smoothing specifically to the upper face region
            # to reduce jitter in eyes and eyebrows
            coords_y, coords_x = np.where(face_mask > 0)
            if len(coords_y) > 0:
                # Calculate vertical extent of the face in the mask
                mask_height = coords_y.max() - coords_y.min()
                mask_center_y = (coords_y.min() + coords_y.max()) / 2
                
                # Apply stronger smoothing to upper 30% of the face where eyes/eyebrows are
                upper_region_end = int(coords_y.min() + mask_height * 0.3)
                if upper_region_end > 0:
                    # Extract upper region
                    upper_mask = face_mask[:upper_region_end, :]
                    # Apply even stronger blur to upper region
                    upper_smoothed = cv2.GaussianBlur(upper_mask, (15, 15), 0)
                    face_mask[:upper_region_end, :] = upper_smoothed
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
        face_w = x2 - x1
        face_h = y2 - y1

        if landmarks is not None and len(landmarks) >= 5:
            pts = landmarks.astype(np.float32)
            # pts layout: [left_eye, right_eye, nose_tip, mouth_left, mouth_right]

            # ── Synthesise eyebrow points ──
            # The eyes are at pts[0] and pts[1]. Eyebrows sit roughly
            # 10-15% of face-height above each eye. We place two points
            # above and slightly outside the eyes to capture the full
            # brow arch, then two more points higher to reach the
            # forehead/hairline region.
            # Aggressive offsets to ensure full eyebrow + cheek coverage.
            # 0.42 pushes forehead points well into the hairline region
            # so eyebrows are fully included in the swapped area.
            brow_offset_y = face_h * 0.42
            brow_spread_x = face_w * 0.25  # significantly wider for full brow arch
            left_eye = pts[0]
            right_eye = pts[1]

            # Primary eyebrow points (directly above each eye)
            left_brow = np.array([left_eye[0] - brow_spread_x * 0.6,
                                  left_eye[1] - brow_offset_y])
            right_brow = np.array([right_eye[0] + brow_spread_x * 0.6,
                                   right_eye[1] - brow_offset_y])

            # Forehead points (higher, between eyes)
            eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
            eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
            forehead_center = np.array([eye_mid_x,
                                        left_eye[1] - brow_offset_y * 2.5])
            forehead_left = np.array([left_eye[0] - brow_spread_x * 1.2,
                                      left_eye[1] - brow_offset_y * 2.0])
            forehead_right = np.array([right_eye[0] + brow_spread_x * 1.2,
                                       right_eye[1] - brow_offset_y * 2.0])

            # ── Additional forehead/temple/ear points ──
            # 14 points per side (28 total) to cover the full forehead,
            # temples, sideburns, and upper-cheek region.  These fill
            # gaps between the 5-point landmarks and the synthesized
            # eyebrow/forehead points, ensuring the convex hull spans
            # from jaw to hairline on each side.
            mouth_left = pts[3]
            mouth_right = pts[4]
            nose = pts[2]
            temple_y = eye_mid_y - face_h * 0.15
            brow2_y = eye_mid_y - face_h * 0.35
            hairline_y = forehead_center[1] - face_h * 0.15
            sideburn_y = mouth_left[1] + face_h * 0.05

            # Left side (7 points): temple, brow-arch, forehead-edge,
            # hairline-left, sideburn, upper-cheek, mid-temple
            left_temple = np.array([left_eye[0] - face_w * 0.45, temple_y])
            left_brow_arch = np.array([left_eye[0] - face_w * 0.35, brow2_y])
            left_forehead_edge = np.array([
                left_eye[0] - face_w * 0.50,
                left_eye[1] - brow_offset_y * 1.5,
            ])
            left_hairline = np.array([
                eye_mid_x - face_w * 0.40,
                hairline_y,
            ])
            left_sideburn = np.array([
                left_eye[0] - face_w * 0.55,
                sideburn_y,
            ])
            left_upper_cheek = np.array([
                left_eye[0] - face_w * 0.30,
                left_eye[1] - face_h * 0.02,
            ])
            left_mid_temple = np.array([
                left_eye[0] - face_w * 0.40,
                (temple_y + brow2_y) / 2.0,
            ])

            # Right side (7 points): mirror of left side
            right_temple = np.array([right_eye[0] + face_w * 0.45, temple_y])
            right_brow_arch = np.array([right_eye[0] + face_w * 0.35, brow2_y])
            right_forehead_edge = np.array([
                right_eye[0] + face_w * 0.50,
                right_eye[1] - brow_offset_y * 1.5,
            ])
            right_hairline = np.array([
                eye_mid_x + face_w * 0.40,
                hairline_y,
            ])
            right_sideburn = np.array([
                right_eye[0] + face_w * 0.55,
                sideburn_y,
            ])
            right_upper_cheek = np.array([
                right_eye[0] + face_w * 0.30,
                right_eye[1] - face_h * 0.02,
            ])
            right_mid_temple = np.array([
                right_eye[0] + face_w * 0.40,
                (temple_y + brow2_y) / 2.0,
            ])

            # ── Synthesise jawline/cheek points ──
            # Extend the hull outward along the jaw to cover cheeks and
            # chin more fully.
            mouth_left = pts[3]
            mouth_right = pts[4]
            nose = pts[2]

            # Jaw points: extend mouth corners outward and downward
            jaw_extend_x = face_w * 0.28  # aggressively wider for full cheeks
            jaw_extend_y = face_h * 0.25  # aggressively deeper for full jawline
            jaw_left = np.array([mouth_left[0] - jaw_extend_x,
                                 mouth_left[1] + jaw_extend_y])
            jaw_right = np.array([mouth_right[0] + jaw_extend_x,
                                  mouth_right[1] + jaw_extend_y])
            # Chin point
            chin = np.array([(mouth_left[0] + mouth_right[0]) / 2.0,
                             mouth_left[1] + jaw_extend_y * 1.4])

            # ── Build expanded convex hull ──
            # 5 original + 5 brow/forehead + 28 temple/cheek/hairline + 3 jaw
            # = 41 total points for a tight convex hull
            expanded_pts = np.vstack([
                pts,            # original 5 landmarks
                left_brow,
                right_brow,
                forehead_center,
                forehead_left,
                forehead_right,
                left_temple,
                left_brow_arch,
                left_forehead_edge,
                left_hairline,
                left_sideburn,
                left_upper_cheek,
                left_mid_temple,
                right_temple,
                right_brow_arch,
                right_forehead_edge,
                right_hairline,
                right_sideburn,
                right_upper_cheek,
                right_mid_temple,
                jaw_left,
                jaw_right,
                chin,
            ]).astype(np.int32)

            hull = cv2.convexHull(expanded_pts)
            cv2.fillConvexPoly(mask, hull, 255)

            # Aggressive multi-directional dilation to eliminate gaps
            # and expand the mask well beyond the landmark hull
            ksize = max(5, int((x2 - x1) * 0.12)) | 1
            mask = cv2.dilate(mask, np.ones((ksize, ksize), np.uint8))
            # Second pass for even wider coverage
            ksize2 = max(7, int((x2 - x1) * 0.18)) | 1
            mask = cv2.dilate(mask, np.ones((ksize2, ksize2), np.uint8))
            # Third pass with elliptical kernel for smooth expansion
            ksize3 = max(9, int((x2 - x1) * 0.22)) | 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize3, ksize3)))

            # ── Enhanced distance-transform fade to reduce edge flicker ──
            # Enhanced approach for upper face region to reduce jitter in eyes/eyebrows
            mask_f32 = mask.astype(np.float32)
            binary_bool = mask_f32 > 128.0
            if binary_bool.any():
                # distanceTransform on the binary mask gives, for each
                # foreground pixel, its shortest distance to the boundary.
                # We use CV_DIST_L2 (Euclidean) for a smooth gradient.
                bin_u8 = binary_bool.astype(np.uint8)
                dist_map = cv2.distanceTransform(bin_u8, cv2.DIST_L2, 5)
                
                # Determine a fade width proportional to the face size
                # so small and large faces get the same relative smoothness.
                # SIGNIFICANTLY INCREASED fade width for upper face to
                # eliminate jitter in eyes/eyebrows region.
                face_dim = max(face_w, face_h)
                fade_width = max(12, int(face_dim * 0.15))  # ~15% of face size (increased from 9%)
                
                # Apply enhanced smoothing for the upper face region
                # Create a vertical gradient that applies more smoothing to the upper part
                # where jitter is most noticeable
                coords_y, coords_x = np.where(binary_bool)
                if len(coords_y) > 0:
                    # Calculate vertical position weights for upper face smoothing
                    mask_height = coords_y.max() - coords_y.min()
                    mask_center_y = (coords_y.min() + coords_y.max()) / 2
                    
                    # Create vertical weighting that increases smoothing in upper face
                    y_grid = np.arange(h).reshape(h, 1)
                    # Apply Gaussian weighting that peaks in the middle and decreases toward edges
                    # but with extra emphasis on the upper region where eyes are
                    upper_weight = np.exp(-0.5 * ((y_grid - mask_center_y) / (mask_height * 0.3)) ** 2)
                    # Boost upper region smoothing for eyes/eyebrows
                    upper_weight = np.clip(upper_weight, 0.0, 1.0)
                    # Make upper region even more sensitive to smoothing
                    upper_weight = 0.3 + 0.7 * upper_weight
                    
                    # Apply the vertical weighting to the distance map
                    dist_map_weighted = dist_map * upper_weight
                    
                    # Use a SMOOTHSTEP curve instead of linear ramp for
                    # much gentler transition at the edges. This dramatically
                    # reduces frame-to-frame jitter since small boundary
                    # shifts produce much smaller alpha changes.
                    # smoothstep(t) = t^2 * (3 - 2t) for t in [0, 1]
                    t = np.clip(dist_map_weighted / fade_width, 0.0, 1.0)
                    edge_alpha = t * t * (3.0 - 2.0 * t)  # smoothstep curve
                else:
                    # Use the original approach if no coordinates found
                    t = np.clip(dist_map / fade_width, 0.0, 1.0)
                    edge_alpha = t * t * (3.0 - 2.0 * t)  # smoothstep curve

                # Combine: inside the mask, modulate original mask values
                # with the edge alpha; outside, force to zero.
                mask_f32 = np.where(
                    binary_bool,
                    mask_f32 * edge_alpha,
                    0.0,
                )

            mask = np.clip(mask_f32, 0, 255).astype(np.uint8)

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
            # Apply additional smoothing in the upper face region for better edge blending
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            
            # Apply even more aggressive smoothing in the upper region where eyes/eyebrows are
            # This helps reduce the high-frequency jitter in these sensitive areas
            if len(coords_y) > 0:
                # Calculate upper region to apply extra smoothing
                upper_region_end = coords_y.min() + int((coords_y.max() - coords_y.min()) * 0.3)
                if upper_region_end > 0 and upper_region_end < h:
                    # Extract upper region
                    upper_mask = mask[:upper_region_end, :]
                    # Apply extra smoothing to upper region
                    upper_smoothed = cv2.GaussianBlur(upper_mask, (k*2, k*2), 0)
                    mask[:upper_region_end, :] = upper_smoothed

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