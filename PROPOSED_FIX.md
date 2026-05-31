# Proposed Fix for Platform-Specific Video Output Differences

## Problem

The landmark reordering logic in `_filter_landmarks()` uses direct floating-point comparisons (`pts[0, 0] > pts[1, 0]`) which are sensitive to tiny numerical differences between platforms. When face detection produces landmarks that differ by fractions of a pixel between Windows and Ubuntu, the reordering logic may swap points on one platform but not the other, leading to completely different affine transforms and thus different video output.

## Solution Options

### Option 1: Epsilon-Based Comparison (Recommended for Quick Fix)

Add a tolerance threshold to avoid swapping landmarks that are nearly equal:

```python
def _filter_landmarks(
    self, kps: np.ndarray, sigma: float = 3.5,
    min_valid_points: int = 4,
) -> np.ndarray:
    """Normalize landmark layout and preserve semantic point ordering.

    Args:
        kps: Facial landmarks as either ``(5, 2)`` or legacy ``(2, 5)``.
        sigma: Unused legacy parameter retained for compatibility.
        min_valid_points: Unused legacy parameter retained for compatibility.

    Returns:
        Normalized landmark array in ``(5, 2)`` layout.
    """
    _ = sigma, min_valid_points

    if kps is None:
        return np.zeros((5, 2), dtype=np.float32)

    pts = np.array(kps, dtype=np.float32, copy=True)
    if pts.shape == (2, 5):
        pts = pts.T
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return pts

    # Use epsilon tolerance to avoid swapping nearly-equal landmarks
    # This prevents platform-specific floating-point differences from
    # causing different outputs
    EPSILON = 0.5  # 0.5 pixels - only swap if difference is meaningful

    # Ensure left eye is consistently to the left of right eye
    if pts[0, 0] > pts[1, 0] + EPSILON:
        pts[[0, 1]] = pts[[1, 0]]

    # Ensure left mouth corner is consistently to the left of right mouth corner
    if pts[3, 0] > pts[4, 0] + EPSILON:
        pts[[3, 4]] = pts[[4, 3]]

    return pts
```

**Pros:**
- Minimal code change
- Preserves the intent of the original code
- Makes behavior consistent across platforms
- 0.5 pixel threshold is small enough to catch truly misordered landmarks

**Cons:**
- Still relies on comparison logic that could theoretically differ near the boundary

### Option 2: Deterministic Sorting (Most Robust)

Use argsort for fully deterministic ordering:

```python
def _filter_landmarks(
    self, kps: np.ndarray, sigma: float = 3.5,
    min_valid_points: int = 4,
) -> np.ndarray:
    """Normalize landmark layout with deterministic sorting.

    Args:
        kps: Facial landmarks as either ``(5, 2)`` or legacy ``(2, 5)``.
        sigma: Unused legacy parameter retained for compatibility.
        min_valid_points: Unused legacy parameter retained for compatibility.

    Returns:
        Normalized landmark array in ``(5, 2)`` layout with consistent ordering.
    """
    _ = sigma, min_valid_points

    if kps is None:
        return np.zeros((5, 2), dtype=np.float32)

    pts = np.array(kps, dtype=np.float32, copy=True)
    if pts.shape == (2, 5):
        pts = pts.T
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return pts

    # Deterministically sort eye landmarks (indices 0, 1) by x-coordinate
    eyes = pts[0:2, :]
    eye_order = np.argsort(eyes[:, 0])  # Sort by x-coordinate
    pts[0:2, :] = eyes[eye_order]

    # Deterministically sort mouth corner landmarks (indices 3, 4) by x-coordinate
    mouth = pts[3:5, :]
    mouth_order = np.argsort(mouth[:, 0])  # Sort by x-coordinate
    pts[3:5, :] = mouth[mouth_order]

    return pts
```

**Pros:**
- Fully deterministic - argsort has well-defined behavior for equal values
- No conditional logic that could branch differently
- NumPy's argsort uses a stable sort (preserves original order for equal elements)

**Cons:**
- Slightly different logic from original (uses sorting instead of swapping)

### Option 3: Disable Landmark Filtering (Simplest for Testing)

If the landmark filtering is causing more problems than it solves:

```python
# In _warp_face() method, change:
use_landmark_filter=False  # Disable to ensure cross-platform consistency
```

**Pros:**
- Fastest way to test if this is the root cause
- Eliminates the problematic code path entirely

**Cons:**
- Loses the jitter reduction benefits
- May reintroduce the original video jitter problem

### Option 4: Use Fixed Random Seed for RANSAC (Revert to Original)

Return to the RANSAC-based approach but with a deterministic seed:

```python
def _warp_face(
    self, frame: np.ndarray, kps: np.ndarray, size: int,
    template_name: str = "arcface_112_v1",
    use_landmark_filter: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align and crop a face using RANSAC with fixed seed for reproducibility."""

    norm_template = _WARP_TEMPLATES.get(template_name)
    if norm_template is None:
        logger.warning(
            "_warp_face: unknown template '%s' — falling back to 'arcface_112_v1'.",
            template_name,
        )
        norm_template = _WARP_TEMPLATES["arcface_112_v1"]

    template = norm_template * size

    # Set deterministic seed based on landmark positions for reproducibility
    # This ensures RANSAC produces consistent results across platforms
    seed = int(abs(hash(tuple(kps.flatten().tolist()))) % (2**31))
    np.random.seed(seed)

    M, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.RANSAC)

    if M is None:
        raise RuntimeError(
            "Face alignment failed: could not estimate affine transform "
            "from the detected landmarks. The face may be too small, "
            "occluded, or at an extreme angle."
        )

    crop = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)
    return crop, M
```

**Pros:**
- May reduce jitter compared to completely random RANSAC
- Avoids the SVD numerical precision issues
- Deterministic given identical input landmarks

**Cons:**
- Reverts the custom similarity transform improvements
- RANSAC is still stochastic, just seeded
- Doesn't address the root cause of platform differences in face detection

## Recommended Implementation

**Use Option 1 (Epsilon-Based Comparison)** as an immediate fix, because:

1. It directly addresses the identified root cause
2. Minimal code change reduces risk
3. Preserves all the jitter-reduction benefits of the custom transform
4. Easy to test and validate

## Testing Strategy

1. **Add debug logging** to verify the fix:
```python
def _filter_landmarks(self, kps: np.ndarray, ...) -> np.ndarray:
    # ... existing code ...

    EPSILON = 0.5

    swapped_eyes = False
    if pts[0, 0] > pts[1, 0] + EPSILON:
        swapped_eyes = True
        pts[[0, 1]] = pts[[1, 0]]

    swapped_mouth = False
    if pts[3, 0] > pts[4, 0] + EPSILON:
        swapped_mouth = True
        pts[[3, 4]] = pts[[4, 3]]

    if swapped_eyes or swapped_mouth:
        logger.debug(
            "Landmark reordering: eyes=%s mouth=%s (diff: eyes=%.4f mouth=%.4f)",
            swapped_eyes, swapped_mouth,
            pts[0, 0] - pts[1, 0] if not swapped_eyes else pts[1, 0] - pts[0, 0],
            pts[3, 0] - pts[4, 0] if not swapped_mouth else pts[4, 0] - pts[3, 0],
        )

    return pts
```

2. **Run on both platforms** with the same input video and compare:
   - Log output to see if swap counts now match
   - Visual inspection of output videos
   - Frame-by-frame pixel difference analysis

3. **Verification command**:
```bash
# Run with debug logging enabled
chararep -i input.mp4 -o output_test.mp4 --char "Name" --timers --log-level DEBUG 2>&1 | grep "Landmark reordering" > reordering_log.txt

# Compare log files between platforms
diff windows_reordering_log.txt ubuntu_reordering_log.txt
```

If logs now match between platforms and videos are identical, the fix is confirmed.

## Files to Modify

- `chararep/face_swapper.py` - Update `_filter_landmarks()` method (around line 447-475 in video-jitter-best branch)

## Long-Term Improvements

1. Add regression tests that verify cross-platform consistency
2. Document the epsilon value choice and reasoning
3. Consider adding a config option to tune the epsilon
4. Add CI/CD tests that compare outputs on multiple platforms
5. Investigate using a fully deterministic face detection model
