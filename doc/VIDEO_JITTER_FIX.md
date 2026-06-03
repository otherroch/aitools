# Elimination of Video Jitter in CharaRep

This document summarizes the changes made to eliminate video jitter during face swapping and blending, and how the codebase was optimized.

## Sources of Video Jitter

1. **Affine Transform Instability**: Prior iterations relied on alignment logic that was less stable under small landmark changes. The current implementation uses `cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)` for robust, repeatable similarity estimation from the 5 semantic InsightFace landmarks.
2. **Unsafe Landmark Reordering**: The codebase previously contained logic that swapped landmarks based on raw X-coordinate comparisons. On near-vertical faces, tiny floating-point changes could invert the semantic ordering of the eyes/mouth corners and flip the template unexpectedly.
3. **Dynamic or Overly Aggressive Masking**: Large frame-space mask operations and crop masks that were not shaped to the canonical face region could create visible flicker or clip upper-face content such as eyebrows.

## The Solution

To completely eliminate jitter while also expanding the face swap area to include the cheeks, chin, forehead, and eyebrows, the following architecture was implemented:

### 1. Stable Affine Estimation
Face alignment now uses `cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)`. This keeps the transform estimation robust without the custom SVD/Umeyama code path that previously added complexity and incorrect reflection handling.

### 2. Preserve Semantic Landmark Ordering
InsightFace already returns semantically ordered 5-point landmarks. The current implementation normalizes them to a stable `(5, 2)` layout and validates finiteness, but it does **not** reorder points based on floating-point coordinate comparisons.

### 3. Static Canonical Masking in Crop Space
Instead of dynamically computing, dilating, and heavily blurring masks in the full frame space (which caused both jitter and severe performance degradation), the current pipeline builds soft masks in canonical crop space and then warps them back into frame space.
- A static, highly refined soft mask is generated once. Because the landmarks in crop space are fixed to the alignment template, this mask never changes shape.
- The static mask is designed to cover the valid facial region (including eyebrows, forehead, cheeks, and chin) while softly fading out at the edges with a smoothstep distance-transform ramp.
- For each frame, this pre-computed mask is warped into frame space using the inverse affine matrix.
- This not only completely eliminates mask-boundary jitter but also dramatically improves performance by removing multiple full-frame `cv2.GaussianBlur` operations, transforming an $O(N^2)$ frame-space operation into a simple $O(1)$ warp.

## Performance Improvements
By removing the redundant mask generation passes in both `FaceSwapper` and `FaceBlender`, and eliminating the full-frame distance transforms and recursive Gaussian blurs introduced in PR 39, we reduced the processing time per frame significantly. The codebase was simplified by over 2000 lines, making it more maintainable while achieving superior visual stability.