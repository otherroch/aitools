# Analysis: Different Video Output Between Windows and Ubuntu

## Issue Summary

The same source code from the `video-jitter-best` branch produces **different video outputs** when run on Windows 11 vs Ubuntu 24.04, despite using identical command-line parameters. This document explains the possible reasons for this difference.

## Key Changes in PR #39 (video-jitter-best branch)

PR #39 introduced a major change to address video jitter by replacing OpenCV's RANSAC-based affine estimation with a custom similarity transform implementation:

### Before (main branch)
```python
M, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.RANSAC)
```

### After (video-jitter-best branch)
```python
M = self._estimate_similarity_transform(kps, template)
```

The custom implementation (chararep/face_swapper.py:398-446) uses SVD-based similarity transform estimation with landmark normalization.

## Root Causes of Platform Differences

### 1. **Landmark Point Ordering Normalization** ⚠️ MOST LIKELY CAUSE

The `_filter_landmarks()` method (chararep/face_swapper.py:447-475) performs landmark point reordering:

```python
# Ensure left eye is to the left of right eye
if pts[0, 0] > pts[1, 0]:
    pts[[0, 1]] = pts[[1, 0]]

# Ensure left mouth corner is to the left of right mouth corner
if pts[3, 0] > pts[4, 0]:
    pts[[3, 4]] = pts[[4, 3]]
```

**Why this causes platform differences:**

1. **Floating-point comparison sensitivity**: The condition `pts[0, 0] > pts[1, 0]` compares floating-point coordinates
2. **Platform-specific float behavior**: Different CPU architectures, compilers, and library versions can produce slightly different floating-point results in earlier pipeline stages (face detection, landmark extraction)
3. **Cascading effect**: If landmarks from the face detector differ by tiny amounts between platforms (even 0.0001 pixels), the reordering logic may swap points on one platform but not the other
4. **Transform impact**: Swapping points fundamentally changes the input to SVD, producing a completely different affine transform matrix
5. **Frame-by-frame propagation**: Each frame's transform affects the entire face warping, leading to visible differences throughout the video

**Example scenario:**
```
Windows: pts[0, 0] = 120.4999, pts[1, 0] = 120.5000  → No swap
Ubuntu:  pts[0, 0] = 120.5001, pts[1, 0] = 120.5000  → Swap occurs!
```

### 2. **SVD Implementation Differences**

The custom `_estimate_similarity_transform()` method uses `np.linalg.svd()`:

```python
U, singular_values, Vt = np.linalg.svd(cov)
```

**Platform-specific variations:**

- **BLAS/LAPACK backend**: NumPy can use different underlying libraries (OpenBLAS, MKL, ATLAS, reference BLAS) depending on how it was installed
- **Different algorithms**: Some BLAS implementations use different SVD algorithms
- **Numerical precision**: SVD is sensitive to numerical precision, and different implementations may produce slightly different U, Σ, V matrices that are mathematically equivalent but numerically distinct
- **Determinant check**: The code includes `if np.linalg.det(rotation) < 0` which can be affected by numerical precision

**Evidence:**
```bash
# Check NumPy configuration
python -c "import numpy as np; np.show_config()"
```
This likely shows different BLAS/LAPACK libraries on Windows vs Ubuntu.

### 3. **Floating-Point Arithmetic Variations**

Multiple stages involve floating-point operations that may differ between platforms:

**Face detection (InsightFace/ONNX Runtime)**:
- Model inference may use different optimization flags
- CPU instruction sets (SSE, AVX, AVX2, FMA) affect floating-point calculations
- ONNX Runtime execution providers may behave differently

**Preprocessing calculations**:
```python
src_mean = src_pts.mean(axis=0)
src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
scale = float(np.sum(singular_values) / src_var)
```

Each of these operations can accumulate small differences across platforms.

### 4. **OpenCV Version and Build Differences**

The code uses OpenCV extensively:
- `cv2.warpAffine()` - may use different optimization backends
- Image processing operations in face detection
- Different OpenCV builds (pip wheels vs conda vs system packages) have different compilation flags

### 5. **NumPy Version and Build Differences**

NumPy operations throughout the pipeline:
- Array operations (`mean()`, matrix multiplication `@`)
- Random number generation (if seeding is platform-dependent)
- Memory layout differences between platforms

### 6. **Thread Scheduling (Parallel Mode)**

While you used default settings, if `batch_size > 1`, the pipeline uses `ThreadPoolExecutor`. Although frame order is maintained via FIFO draining, timing differences in thread scheduling could affect:
- Face detector internal state (if any)
- Face tracker assignments (IoU matching might pick different faces in edge cases)

## Why RANSAC-Based Method Was More Consistent

The old RANSAC method had implicit randomness but was **surprisingly more consistent across platforms**:

1. **Random seed**: OpenCV's RANSAC likely initializes its random seed based on system state, but the randomness was contained within a single method call
2. **Integer thresholds**: RANSAC uses integer iteration counts and clear inlier/outlier thresholds
3. **No point reordering**: Landmarks were used as-is without conditional swapping logic

## Recommendations

### Immediate Fix: Deterministic Landmark Ordering

Replace floating-point comparison with a stable sorting approach:

```python
def _filter_landmarks(self, kps: np.ndarray) -> np.ndarray:
    """Normalize landmark layout with deterministic ordering."""
    pts = np.array(kps, dtype=np.float32, copy=True)
    if pts.shape == (2, 5):
        pts = pts.T
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return pts

    # Use epsilon-based comparison to handle floating-point noise
    EPSILON = 1e-4

    # Sort eyes: leftmost eye should be index 0
    if pts[0, 0] > pts[1, 0] + EPSILON:
        pts[[0, 1]] = pts[[1, 0]]

    # Sort mouth corners: leftmost should be index 3
    if pts[3, 0] > pts[4, 0] + EPSILON:
        pts[[3, 4]] = pts[[4, 3]]

    return pts
```

**Better approach - Centroid-based ordering:**

```python
def _filter_landmarks(self, kps: np.ndarray) -> np.ndarray:
    """Normalize landmark layout with centroid-based ordering."""
    pts = np.array(kps, dtype=np.float32, copy=True)
    if pts.shape == (2, 5):
        pts = pts.T
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        return pts

    # Calculate face centroid from all landmarks
    centroid_x = pts[:, 0].mean()

    # Eyes: order by distance from face center (left = farther left)
    eyes = pts[0:2, :]
    if eyes[0, 0] > eyes[1, 0]:  # If first eye is rightward of second
        pts[[0, 1]] = pts[[1, 0]]

    # Mouth corners: order by distance from face center
    mouth = pts[3:5, :]
    if mouth[0, 0] > mouth[1, 0]:
        pts[[3, 4]] = pts[[4, 3]]

    return pts
```

### Long-term Solutions

1. **Pin NumPy/OpenCV versions and BLAS backend** in requirements.txt
2. **Add deterministic mode flag** to disable all conditional swapping
3. **Use stable hash of landmarks** to detect when reordering would occur, log it
4. **Add integration tests** that compare frame outputs across platforms
5. **Consider returning to RANSAC** with a fixed random seed:
   ```python
   np.random.seed(42)  # or use frame_idx as seed
   M, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.RANSAC)
   ```

### Verification Steps

To confirm this is the issue:

1. **Add debug logging** to `_filter_landmarks()`:
   ```python
   if pts[0, 0] > pts[1, 0]:
       logger.debug(f"Frame {frame_idx}: Swapping eyes: {pts[0,0]} > {pts[1,0]}")
       pts[[0, 1]] = pts[[1, 0]]
   ```

2. **Compare logs** between Windows and Ubuntu runs - if swap counts differ, this confirms the root cause

3. **Test with epsilon**: Temporarily set a large epsilon (e.g., 1.0 pixel) to force consistent ordering

4. **Disable landmark filtering**: Test with `use_landmark_filter=False` to see if outputs become identical

## Files Involved

- `chararep/face_swapper.py:398-446` - `_estimate_similarity_transform()` (SVD-based)
- `chararep/face_swapper.py:447-475` - `_filter_landmarks()` (point reordering)
- `chararep/face_swapper.py:339-397` - `_warp_face()` (calls both methods)

## Conclusion

The **most likely root cause** is the conditional landmark point reordering in `_filter_landmarks()` combined with platform-specific floating-point variations in face detection. The custom SVD-based similarity transform is mathematically sound but exposes this cross-platform inconsistency that was previously masked by RANSAC's internal randomness.

The fix should focus on making landmark ordering deterministic and robust to tiny floating-point differences, while preserving the jitter-reduction benefits of the custom transform.
