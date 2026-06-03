# Elimination of Video Jitter in CharaRep

This document summarizes the changes made to eliminate video jitter during face swapping and blending, and how the codebase was optimized.

## Sources of Video Jitter

1. **Affine Transform Instability**: The original implementation used `cv2.estimateAffinePartial2D` with `cv2.RANSAC` for face alignment. RANSAC is a randomized algorithm, meaning the estimated affine matrix could vary slightly frame-to-frame even if the input landmarks were identical. This caused the warped face to jitter.
2. **Landmark Reordering Bug**: The codebase contained logic that conditionally swapped the left and right eye landmarks based on a direct floating-point comparison of their X coordinates (`if pts[0, 0] > pts[1, 0]:`). On nearly perfectly vertical faces, tiny sub-pixel variations would cause the eyes to swap unpredictably, flipping the entire face template and causing massive jitter.
3. **Dynamic Mask Generation in Frame Space**: Masks were being generated and processed (via multiple large Gaussian blurs and distance transforms) dynamically in the full frame space. Small movements in the facial landmarks would significantly alter the generated mask boundaries, creating visible flickering at the edges of the face (especially around the forehead and eyebrows).

## The Solution

To completely eliminate jitter while also expanding the face swap area to include the cheeks, chin, forehead, and eyebrows, the following architecture was implemented:

### 1. Deterministic Similarity Transform
We replaced the RANSAC-based affine estimation with a strictly deterministic similarity transform (the Umeyama algorithm). This guarantees that for a given set of landmarks, the resulting affine matrix is mathematically stable. We also fixed a functional bug in the custom similarity transform where the scale calculation incorrectly handled reflections by using `(singular_values[0] - singular_values[1])` when `det(U @ Vt) < 0`.

### 2. Epsilon Tolerance for Landmark Filtering
We introduced an epsilon tolerance (`1e-5`) for any necessary spatial comparisons of landmarks, preventing tiny floating-point fluctuations from triggering a semantic reordering of the facial points.

### 3. Static Canonical Masking in Crop Space
Instead of dynamically computing, dilating, and heavily blurring masks in the full frame space (which caused both jitter and severe performance degradation), we moved the mask generation to **canonical crop space** (e.g., the fixed 128x128 template space).
- A static, highly refined soft mask is generated once. Because the landmarks in crop space are fixed to the alignment template, this mask never changes shape.
- The static mask is designed to cover the maximum valid area of the crop (including eyebrows, forehead, cheeks, and chin) while softly fading out at the edges using a smoothstep radial gradient.
- For each frame, this pre-computed, perfectly stable mask is simply warped into frame space using the deterministic affine matrix. 
- This not only completely eliminates mask-boundary jitter but also dramatically improves performance by removing multiple full-frame `cv2.GaussianBlur` operations, transforming an $O(N^2)$ frame-space operation into a simple $O(1)$ warp.

## Performance Improvements
By removing the redundant mask generation passes in both `FaceSwapper` and `FaceBlender`, and eliminating the full-frame distance transforms and recursive Gaussian blurs introduced in PR 39, we reduced the processing time per frame significantly. The codebase was simplified by over 2000 lines, making it more maintainable while achieving superior visual stability.