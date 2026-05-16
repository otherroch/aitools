"""Face-swap engine supporting inswapper, simswap, uniface, hyperswap, and blendswap model families."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime
import insightface
from scipy.ndimage import gaussian_filter

from .config import PipelineConfig
from .gpu_utils import get_onnx_providers

logger = logging.getLogger(__name__)

# Default model to auto-detect when no path is specified
_DEFAULT_MODEL_NAME = "inswapper_128.onnx"

# 5-point landmark template for arcface_112_v1 alignment (in 112×112 space).
# Used to warp faces before feeding them to simswap models.
_ARCFACE_112_V1 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Normalised 5-point landmark templates (coordinates in [0, 1] space).
# Multiply by the desired crop size to obtain pixel coordinates.
# arcface_112_v1 is derived from the legacy constant above so that all
# template-based warps remain consistent.
_WARP_TEMPLATES: dict[str, np.ndarray] = {
    # Existing template (arcface_112_v1 ≈ InsightFace default, also matches
    # facefusion's arcface_112_v2 normalised values)
    "arcface_112_v1": _ARCFACE_112_V1 / 112.0,
    # arcface_128 — used by hyperswap models (facefusion)
    "arcface_128": np.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453],
        ],
        dtype=np.float32,
    ),
    # ffhq_512 — used by uniface and blendswap models (facefusion)
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ],
        dtype=np.float32,
    ),
}

# Per-model parameters (normalization, alignment template, source input type).
# Keys are lower-cased stem substrings matched against the model filename.
# ``template``     — key into _WARP_TEMPLATES for warping the target face crop.
# ``source_type``  — how the source (portrait) identity is provided:
#                    ``"embedding"``      raw ArcFace embedding + optional converter
#                    ``"embedding_norm"`` L2-normalised ArcFace embedding directly
#                    ``"image"``          pre-warped portrait face crop
# ``source_crop_attr`` (image models only) — attribute name on the portrait Face
#                    object that holds the pre-warped source crop (BGR uint8).
_SIMSWAP_MODEL_PARAMS: dict[str, dict] = {
    "simswap_256": {
        "size": 256,
        "template": "arcface_112_v1",
        "mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
        "std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
        "source_type": "embedding",
    },
    "simswap_unofficial_512": {
        "size": 512,
        "template": "arcface_112_v1",
        "mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "source_type": "embedding",
    },
    # ── New 256-class models ─────────────────────────────────────────────
    "uniface_256": {
        "size": 256,
        "template": "ffhq_512",
        "mean": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "std": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "source_type": "image",
        "source_crop_attr": "portrait_crop_ffhq",
    },
    "hyperswap_1a_256": {
        "size": 256,
        "template": "arcface_128",
        "mean": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "std": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "source_type": "embedding_norm",
    },
    "hyperswap_1b_256": {
        "size": 256,
        "template": "arcface_128",
        "mean": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "std": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "source_type": "embedding_norm",
    },
    "hyperswap_1c_256": {
        "size": 256,
        "template": "arcface_128",
        "mean": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "std": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "source_type": "embedding_norm",
    },
    "blendswap_256": {
        "size": 256,
        "template": "ffhq_512",
        "mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "source_type": "image",
        # blendswap source uses arcface_112_v2 (≡ the arcface_crop already
        # computed by FaceRecognizer, stored as portrait_crop_arcv2)
        "source_crop_attr": "portrait_crop_arcv2",
    },
}

# Fallback parameters for any unrecognised ONNX model variant
_SIMSWAP_DEFAULT_PARAMS: dict = {
    "size": 256,
    "template": "arcface_112_v1",
    "mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "source_type": "embedding",
}

# Model types that produce output in a normalised range and require
# de-normalisation (``x * std + mean``) before clipping to [0, 1].
_DENORMALIZE_OUTPUT_TYPES = frozenset({"hyperswap", "uniface"})


def _detect_model_type(path: str) -> str:
    """Return the model family based on the filename stem.

    Checks are performed in most-specific-first order so that a filename
    containing multiple keywords (e.g. a hypothetical
    ``uniface_simswap.onnx``) resolves to the most specific family.

    Returns one of ``'inswapper'``, ``'simswap'``, ``'uniface'``,
    ``'hyperswap'``, or ``'blendswap'``.
    """
    stem = Path(path).stem.lower()
    if "uniface" in stem:
        return "uniface"
    if "hyperswap" in stem:
        return "hyperswap"
    if "blendswap" in stem:
        return "blendswap"
    if "simswap" in stem:
        return "simswap"
    return "inswapper"


def _get_simswap_params(path: str) -> dict:
    """Return size/template/mean/std/source_type for an ONNX swap model."""
    stem = Path(path).stem.lower()
    for key, params in _SIMSWAP_MODEL_PARAMS.items():
        if key in stem:
            return params
    return _SIMSWAP_DEFAULT_PARAMS


class FaceSwapper:
    """Applies identity-transfer face swaps.

    Supports the following model families (auto-detected from the filename):

    * **inswapper** (e.g. ``inswapper_128.onnx``) — loaded through
      :mod:`insightface.model_zoo`; handles alignment and compositing
      internally via ``paste_back=True``.

    * **simswap** (e.g. ``simswap_256.onnx``, ``simswap_unofficial_512.onnx``)
      — loaded via ONNX Runtime; face alignment, preprocessing, and
      paste-back are performed explicitly.  An optional *embedding
      converter* (``crossface_simswap.onnx``) can be supplied via
      ``cfg.embedding_converter_path`` to improve identity fidelity.

    * **uniface** (e.g. ``uniface_256.onnx``) — ONNX Runtime; uses the
      ffhq_512 alignment template for both source and target crops.  The
      source portrait is passed as an image crop (no embedding).

    * **hyperswap** (e.g. ``hyperswap_1a_256.onnx``, ``hyperswap_1b_256.onnx``,
      ``hyperswap_1c_256.onnx``) — ONNX Runtime; uses the arcface_128
      template and a L2-normalised embedding as the source identity.

    * **blendswap** (e.g. ``blendswap_256.onnx``) — ONNX Runtime; uses the
      ffhq_512 template for the target crop and an arcface_112_v2 portrait
      crop as the source image.
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._model_path = self._resolve_model_path(cfg.swap_model_path)
        self._model_type = _detect_model_type(self._model_path)

        providers = get_onnx_providers(cfg.device_id)

        if self._model_type == "inswapper":
            self._model = insightface.model_zoo.get_model(
                self._model_path, providers=providers
            )
            self._ort_session = None
            self._embedding_converter = None
            self._simswap_params: Optional[dict] = None
        else:
            self._model = None
            self._ort_session = onnxruntime.InferenceSession(
                self._model_path, providers=providers
            )
            self._embedding_converter = self._load_embedding_converter(
                cfg.embedding_converter_path, providers
            )
            # Shallow copy so we can override individual values without
            # mutating the module-level _SIMSWAP_MODEL_PARAMS dict.
            self._simswap_params = dict(_get_simswap_params(self._model_path))
            # Override the crop size with the actual value encoded in the
            # model's image-input shape metadata.  This avoids a mismatch
            # when the filename doesn't match any known pattern (fallback
            # size = 256) but the model actually expects a larger crop.
            model_size = self._read_model_crop_size()
            if model_size is not None and model_size != self._simswap_params["size"]:
                logger.info(
                    "ONNX model crop size from metadata (%d) overrides "
                    "filename-derived size (%d) for model '%s'.",
                    model_size,
                    self._simswap_params["size"],
                    Path(self._model_path).name,
                )
                self._simswap_params["size"] = model_size
            self._converter_mode = self._detect_converter_mode()
            # Log actual input names so mismatches are easy to diagnose
            input_names = [inp.name for inp in self._ort_session.get_inputs()]
            logger.debug("ONNX model inputs: %s", input_names)

        logger.info(
            "FaceSwapper loaded: %s (type=%s)", self._model_path, self._model_type
        )

    # ── Model loading ────────────────────────────────────────────────────────

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve and validate the swap model file path.

        If *model_path* is given but the file does not exist, a
        :exc:`FileNotFoundError` is raised immediately so the user receives a
        clear error rather than a silent fall-through to the wrong model.
        """
        if model_path:
            if Path(model_path).is_file():
                return model_path
            raise FileNotFoundError(
                f"Swap model not found at the specified path: {model_path}"
            )

        # No path given — look for the default inswapper model
        home = Path.home() / ".insightface" / "models"
        default = home / _DEFAULT_MODEL_NAME
        if default.is_file():
            return str(default)

        buffalo_dir = home / "buffalo_l"
        candidate = buffalo_dir / _DEFAULT_MODEL_NAME
        if candidate.is_file():
            return str(candidate)

        raise FileNotFoundError(
            f"inswapper model not found. Please download "
            f"'{_DEFAULT_MODEL_NAME}' and place it at:\n"
            f"  {default}\n"
            f"or specify its path via --swap-model-path.\n"
            f"Download from: https://github.com/deepinsight/insightface/tree/master/examples/in_swapper"
        )

    def _load_embedding_converter(
        self, converter_path: Optional[str], providers: list
    ) -> Optional[onnxruntime.InferenceSession]:
        """Load the optional simswap embedding converter ONNX model."""
        if converter_path and Path(converter_path).is_file():
            session = onnxruntime.InferenceSession(converter_path, providers=providers)
            logger.info("Embedding converter loaded: %s", converter_path)
            return session
        if converter_path:
            logger.warning(
                "Embedding converter not found at '%s'; "
                "falling back to raw ArcFace embedding.",
                converter_path,
            )
        return None

    def _detect_converter_mode(self) -> str:
        """Determine how to use the embedding converter based on its input shape.

        Returns ``'image'`` when the converter is an ArcFace **image encoder**
        (e.g. ``simswap_arcface_model.onnx``) whose first input has shape
        ``(N, 3, 112, 112)``.  In this mode ``_prepare_source_embedding``
        passes the pre-warped portrait face crop rather than an embedding.

        Returns ``'embedding'`` for feature-space converters whose first input
        has a channel dimension other than 3 (e.g. the crossface converter with
        shape ``(N, 512, 1, 1)``), and also when no converter is loaded.
        """
        if self._embedding_converter is None:
            return "embedding"

        inputs = self._embedding_converter.get_inputs()
        if inputs:
            shape = getattr(inputs[0], "shape", None)
            if shape is not None and len(shape) == 4:
                def _to_int(d):
                    try:
                        return int(d)
                    except (TypeError, ValueError):
                        return None
                if _to_int(shape[1]) == 3 and _to_int(shape[2]) == 112 and _to_int(shape[3]) == 112:
                    logger.info(
                        "Embedding converter detected as ArcFace image encoder "
                        "(input shape %s); portrait face crops will be used.",
                        list(shape),
                    )
                    return "image"
        return "embedding"

    # ── SimSwap / ONNX helpers ───────────────────────────────────────────────

    def _warp_face(
        self, frame: np.ndarray, kps: np.ndarray, size: int,
        template_name: str = "arcface_112_v1",
        use_landmark_filter: bool = True,
        landmark_sigma: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align and crop a face to *size*×*size* using 5-point landmarks.

        The alignment template is selected by *template_name*, which must be
        a key in :data:`_WARP_TEMPLATES`.  The default ``"arcface_112_v1"``
        preserves the previous behaviour for simswap models.

        Args:
            frame: Source image to crop.
            kps: 2×5 array of detected facial landmarks.
            size: Output crop size.
            template_name: Template key from :data:`_WARP_TEMPLATES`.
            use_landmark_filter: If True, apply outlier filtering and smoothing
                to reduce jitter from noisy landmarks (especially helpful for
                eyes/eyebrows region).
            landmark_sigma: Standard deviation for 2D Gaussian smoothing of
                smoothed landmarks (applied when use_landmark_filter=True).

        Returns the cropped face and the 2×3 affine matrix used, so that the
        result can be pasted back with :meth:`_paste_back`.

        Raises:
            RuntimeError: If landmark-based affine estimation fails.
        """
        norm_template = _WARP_TEMPLATES.get(template_name)
        if norm_template is None:
            logger.warning(
                "_warp_face: unknown template '%s' — falling back to 'arcface_112_v1'.",
                template_name,
            )
            norm_template = _WARP_TEMPLATES["arcface_112_v1"]
        template = norm_template * size

        # Apply landmark filtering to reduce jitter from noisy detections
        if use_landmark_filter and kps is not None:
            kps = self._filter_landmarks(kps, landmark_sigma)

        M, _ = cv2.estimateAffinePartial2D(kps, template, method=cv2.RANSAC)
        if M is None:
            raise RuntimeError(
                "Face alignment failed: could not estimate affine transform "
                "from the detected landmarks. The face may be too small, "
                "occluded, or at an extreme angle."
            )
        crop = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)
        return crop, M

    def _filter_landmarks(
        self, kps: np.ndarray, sigma: float = 2.5,
        min_valid_points: int = 4,
    ) -> np.ndarray:
        """Filter and smooth facial landmarks to reduce jitter.

        Applies several techniques to improve landmark stability:

        1. **Distance-based validation**: Filters out landmark sets where any
           point is too far from its expected position (based on template).
           This catches grossly misaligned detections.

        2. **Outlier filtering using MAD**: For each landmark dimension,
           computes the Median Absolute Deviation and filters points that
           deviate significantly from the median. This removes spikes caused
           by tracking errors in the upper face (eyes, eyebrows).

        3. **Weighted Gaussian smoothing**: Applies stronger smoothing to the
           upper face landmarks (eyes at indices 0, 1) which are more prone
           to jitter and directly affect eyebrow appearance.

        4. **Geometric consistency check**: Ensures the distance between
           symmetric features (left/right eyes) is within expected range.

        Args:
            kps: 2×5 array of facial landmarks (x, y for each of 5 points).
            sigma: Standard deviation for 2D Gaussian smoothing.
            min_valid_points: Minimum number of valid points required after
                filtering (default 4, since we need at least 4 for affine).

        Returns:
            Filtered/smoothed landmark array. Returns original if filtering
            fails or insufficient valid points remain.
        """
        if kps is None or kps.shape != (2, 5):
            return kps if kps is not None else np.zeros((2, 5), dtype=np.float32)

        # Convert to float for processing
        kps_f = kps.astype(np.float64)
        original_kps = kps_f.copy()
        points = kps_f.T  # Shape: (5, 2) -> points[i] = [x, y] for landmark i

        # Landmark indices: 0=left_eye, 1=right_eye, 2=nose_tip, 3=mouth_left, 4=mouth_right
        # Eyes (0, 1) are most prone to jitter and affect eyebrow appearance

        # Get expected positions from template (in [0, 1] normalized space)
        template = _WARP_TEMPLATES.get("arcface_112_v1")
        if template is None:
            template = _WARP_TEMPLATES["arcface_112_v1"]

        # Estimate face size from actual landmark spread
        # Use inter-eye distance as a stable measure of face size
        left_eye = points[0]
        right_eye = points[1]
        inter_eye_dist = np.linalg.norm(right_eye - left_eye)
        # Typical inter-eye distance is about 25% of face width in the template
        estimated_size = inter_eye_dist / 0.32

        # 1. Distance-based validation: Check if any point is far from expected
        # Tighter threshold for eyes (0, 1) since they're more sensitive to jitter
        max_valid_dist = estimated_size * 0.12  # Reduced from 0.15
        eye_max_dist = estimated_size * 0.08     # Tighter for eyes
        valid_mask = np.ones(5, dtype=bool)
        for i in range(5):
            expected = template[:, i] * estimated_size
            actual = points[i]
            dist = np.linalg.norm(expected - actual)
            threshold = eye_max_dist if i in [0, 1] else max_valid_dist
            if dist > threshold:
                valid_mask[i] = False

        # Remove obviously bad points
        if np.sum(valid_mask) < min_valid_points:
            logger.debug("_filter_landmarks: too few valid points after distance check")
            return kps_f.T

        # 2. Geometric consistency: Check eye separation is reasonable
        # Eyes should be roughly symmetric and at similar y-levels
        eye_separation = np.linalg.norm(right_eye - left_eye)
        expected_separation = np.linalg.norm(template[1] - template[0]) * estimated_size
        if eye_separation > 0 and expected_separation > 0:
            ratio = eye_separation / expected_separation
            if ratio < 0.5 or ratio > 1.5:
                # Eyes are too far apart or too close - likely tracking error
                # Pull them toward expected positions
                eye_center = (left_eye + right_eye) / 2
                new_separation = expected_separation * 0.95  # Slightly tighter
                half_sep_x = new_separation / 2
                # Eyes should be roughly at the same height
                new_left_eye = np.array([eye_center[0] - half_sep_x, eye_center[1]])
                new_right_eye = np.array([eye_center[0] + half_sep_x, eye_center[1]])
                points[0] = new_left_eye
                points[1] = new_right_eye

        # 3. Outlier filtering using Median Absolute Deviation (MAD)
        # More aggressive for eyes to reduce jitter
        mad_threshold = 1.5  # Reduced from 2.0 for tighter filtering
        filtered_x = kps_f[0, :].copy()
        filtered_y = kps_f[1, :].copy()

        # Apply different thresholds for eyes vs other landmarks
        for dim_idx, dim_data in enumerate([filtered_x, filtered_y]):
            dim_name = "x" if dim_idx == 0 else "y"
            for i in range(5):
                is_eye = i in [0, 1]
                # Calculate MAD excluding this point for better outlier detection
                other_points = np.delete(dim_data, i)
                median = np.median(other_points)
                mad = np.median(np.abs(other_points - median))
                
                # Threshold is tighter for eyes (upper face jitter source)
                threshold = 1.2 if is_eye else mad_threshold
                if mad > 1e-8:
                    # How far is this point from the median?
                    deviation = np.abs(dim_data[i] - median)
                    if deviation > threshold * mad / 0.6745:
                        # Pull toward median but preserve some identity
                        blend = 0.3 if is_eye else 0.6
                        dim_data[i] = blend * median + (1 - blend) * dim_data[i]

        # 4. Apply weighted Gaussian smoothing
        # Higher sigma for eyes (0, 1) to reduce jitter in the upper face
        # which directly affects eyebrow appearance
        sigma_eyes = sigma * 1.5  # 50% stronger smoothing for eyes
        sigma_other = sigma

        x_smooth = filtered_x.copy()
        y_smooth = filtered_y.copy()

        # Smooth all points with base sigma
        x_smooth = gaussian_filter(x_smooth, sigma=sigma_other, mode='nearest')
        y_smooth = gaussian_filter(y_smooth, sigma=sigma_other, mode='nearest')

        # Additional smoothing pass for eyes (upper face region)
        # This significantly reduces jitter in the eyes which causes eyebrow flicker
        eye_x_smooth = gaussian_filter(filtered_x, sigma=sigma_eyes, mode='nearest')
        eye_y_smooth = gaussian_filter(filtered_y, sigma=sigma_eyes, mode='nearest')
        x_smooth[[0, 1]] = eye_x_smooth[[0, 1]]
        y_smooth[[0, 1]] = eye_y_smooth[[0, 1]]

        kps_smoothed = np.array([x_smooth, y_smooth], dtype=np.float32)

        return kps_smoothed

    # ── Model loading ────────────────────────────────────────────────────────

    def _prepare_crop_frame(self, crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop to a normalised ONNX input tensor ``[1,C,H,W]``."""
        mean = self._simswap_params["mean"]  # type: ignore[index]
        std = self._simswap_params["std"]  # type: ignore[index]
        # BGR → RGB, scale to [0, 1], normalise
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(x, 0)

    @staticmethod
    def _preprocess_arcface_crop(crop: np.ndarray) -> np.ndarray:
        """Preprocess a 112×112 BGR face crop for ArcFace inference.

        Converts BGR → RGB, scales pixels to ``[0, 1]``, applies the
        standard ArcFace normalisation to ``[−1, 1]``, and returns a
        ``(1, 3, 112, 112)`` float32 tensor.
        """
        x = crop[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB, [0, 1]
        x = (x - 0.5) / 0.5                               # normalise to [−1, 1]
        x = x.transpose(2, 0, 1)                           # HWC → CHW
        return np.expand_dims(x, 0)

    def _prepare_source_embedding(self, identity_face) -> Optional[np.ndarray]:
        """Build the source identity embedding for the simswap model.

        Two code paths depending on :attr:`_converter_mode`:

        * ``'image'`` — The converter is an ArcFace **image encoder**
          (e.g. ``simswap_arcface_model.onnx``).  The pre-warped 112×112
          portrait crop stored as ``identity_face.arcface_crop`` is
          preprocessed to ``(1, 3, 112, 112)`` (RGB, normalised to
          ``[−1, 1]``) and passed to the encoder to obtain the embedding.

        * ``'embedding'`` — Either no converter is loaded, or it is a
          feature-space converter (e.g. ``crossface_simswap.onnx``) that
          expects a ``(1, 512, 1, 1)`` embedding tensor.  The embedding
          stored on the face object by InsightFace is used, optionally
          transformed by the converter.

        In both paths the final embedding is L2-normalised and returned as
        shape ``(1, 512)``, or ``None`` if any step fails.
        """
        if self._converter_mode == "image":
            # ── ArcFace image-encoder path ────────────────────────────────
            arcface_crop = getattr(identity_face, "arcface_crop", None)
            if arcface_crop is None:
                logger.warning(
                    "SimSwap: ArcFace converter requires a portrait face crop "
                    "(arcface_crop) but none was stored on this face — "
                    "skipping this face."
                )
                return None
            # BGR → RGB, normalise to [−1, 1] (standard ArcFace preprocessing)
            converter_input = self._preprocess_arcface_crop(arcface_crop)
            try:
                embedding = self._embedding_converter.run(  # type: ignore[union-attr]
                    None, {"input": converter_input}
                )[0]
            except Exception as exc:
                logger.warning(
                    "SimSwap: embedding converter failed (%s: %s) "
                    "— skipping this face.",
                    type(exc).__name__,
                    exc,
                )
                return None
        else:
            # ── InsightFace embedding path (+ optional feature converter) ─
            raw = getattr(identity_face, "embedding", None)
            if raw is None:
                return None
            try:
                embedding = np.array(raw, dtype=np.float32).reshape(-1, 512)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "SimSwap: could not reshape embedding to (-1, 512): %s "
                    "— skipping this face.",
                    exc,
                )
                return None
            if self._embedding_converter is not None:
                # Feature converter expects (N, 512, 1, 1)
                try:
                    embedding = self._embedding_converter.run(
                        None, {"input": embedding.reshape(-1, 512, 1, 1)}
                    )[0]
                except Exception as exc:
                    logger.warning(
                        "SimSwap: embedding converter failed (%s: %s) "
                        "— skipping this face.",
                        type(exc).__name__,
                        exc,
                    )
                    return None

        embedding = embedding.ravel()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.reshape(1, -1)

    def _prepare_embedding_norm(self, identity_face) -> Optional[np.ndarray]:
        """Return a L2-normalised embedding for hyperswap models.

        Hyperswap expects the L2-normalised ArcFace embedding directly
        (no embedding converter).  ``face.normed_embedding`` is used when
        available (set by InsightFace); otherwise the raw ``face.embedding``
        is manually normalised.

        Returns shape ``(1, 512)`` or ``None`` when no embedding is found.
        """
        normed = getattr(identity_face, "normed_embedding", None)
        if normed is None:
            raw = getattr(identity_face, "embedding", None)
            if raw is None:
                logger.warning(
                    "Hyperswap: portrait face has no embedding — skipping this face."
                )
                return None
            raw = np.array(raw, dtype=np.float32).ravel()
            n = np.linalg.norm(raw)
            if n == 0:
                logger.warning(
                    "Hyperswap: portrait face has a zero-norm embedding — "
                    "identity may not transfer correctly."
                )
            normed = raw / n if n > 0 else raw
        return np.array(normed, dtype=np.float32).reshape(1, -1)

    def _prepare_source_frame(self, identity_face) -> Optional[np.ndarray]:
        """Return the pre-warped portrait crop as an ONNX image tensor.

        Used by ``uniface`` and ``blendswap`` models, which take a face
        *image* rather than an embedding as the source identity input.

        The portrait face object is expected to carry a pre-warped BGR
        ``uint8`` crop under the attribute named by
        ``self._simswap_params["source_crop_attr"]`` (set by
        :class:`~face_recognizer.FaceRecognizer` during gallery construction):

        * ``portrait_crop_ffhq``  — 256×256 (ffhq_512 template)  for uniface.
        * ``portrait_crop_arcv2`` — 112×112 (arcface_112_v2 template) for
          blendswap; falls back to ``arcface_crop`` (same template) if the
          dedicated attribute is absent.

        The crop is converted BGR → RGB, scaled to ``[0, 1]``, transposed to
        CHW, and expanded to ``(1, C, H, W) float32``.

        Returns ``None`` and logs a warning when the crop is unavailable.
        """
        attr = self._simswap_params.get("source_crop_attr")  # type: ignore[union-attr]
        crop = getattr(identity_face, attr, None) if attr else None
        # Blendswap fallback: portrait_crop_arcv2 uses the same template
        # as arcface_crop, so accept either attribute.
        if crop is None and self._model_type == "blendswap":
            crop = getattr(identity_face, "arcface_crop", None)
        if crop is None:
            logger.warning(
                "%s: portrait face has no source crop (expected attribute '%s') "
                "— skipping this face.",
                self._model_type,
                attr,
            )
            return None
        x = crop[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB, [0, 1]
        x = x.transpose(2, 0, 1)                          # HWC → CHW
        return np.expand_dims(x, 0)

    def _normalize_crop_frame(self, output: np.ndarray) -> np.ndarray:
        """Convert the model output tensor ``[C,H,W]`` to a BGR ``uint8`` image.

        For model families that produce output in normalised space
        (``hyperswap``, ``uniface``) the inverse normalisation
        ``x * std + mean`` is applied before clipping to ``[0, 1]``.
        """
        x = output.transpose(1, 2, 0)  # CHW → HWC (RGB)
        if self._model_type in _DENORMALIZE_OUTPUT_TYPES:
            mean = self._simswap_params["mean"]  # type: ignore[index]
            std = self._simswap_params["std"]  # type: ignore[index]
            x = x * std + mean
        x = x.clip(0, 1)
        x = x[:, :, ::-1] * 255  # RGB → BGR
        return x.astype(np.uint8)

    def _paste_back(
        self, frame: np.ndarray, crop: np.ndarray, affine_M: np.ndarray
    ) -> np.ndarray:
        """Composite *crop* back into *frame* using the inverse of *affine_M*.

        A soft alpha mask is built from the full crop extent, then dilated
        to expand the swapped region beyond the original crop boundaries so
        that cheeks, chin, forehead and eyebrows are included rather than
        leaving visible seams at the crop edge.

        To avoid black-border artefacts when the dilated mask extends past
        the warped crop, ``cv2.BORDER_REPLICATE`` is used so that pixels
        outside the crop are filled with edge pixels from the swapped face
        instead of black.

        Returns the original *frame* unchanged if the affine matrix is
        degenerate (determinant near zero).
        """
        h, w = frame.shape[:2]
        # Guard against a degenerate (singular) affine matrix
        det = affine_M[0, 0] * affine_M[1, 1] - affine_M[0, 1] * affine_M[1, 0]
        if abs(det) < 1e-6:
            logger.warning("_paste_back: degenerate affine matrix — skipping paste.")
            return frame
        M_inv = cv2.invertAffineTransform(affine_M)
        warped_back = cv2.warpAffine(
            crop, M_inv, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Build a soft mask covering the face region, then dilate to
        # expand the composite area so edges (cheeks, chin, forehead)
        # are fully replaced instead of leaving seams at crop boundary.
        crop_mask = np.ones(
            (crop.shape[0], crop.shape[1]), dtype=np.float32
        )
        mask = cv2.warpAffine(
            crop_mask, M_inv, (w, h), flags=cv2.INTER_LINEAR
        )
        # Dilate the mask in frame space to expand the swap region beyond
        # the exact crop boundary. Use a kernel proportional to crop size.
        crop_size = crop.shape[0]
        # Increased dilation for smoother blending in upper face region
        dilate_k = max(9, int(crop_size * 0.18) | 1)  # ~18% of crop size (increased for smoother edges)
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, np.ones((dilate_k, dilate_k), np.uint8))

        # Apply distance transform for smooth edge fading
        # This reduces jitter by creating a smooth gradient from center to edge
        mask_f32 = mask_uint8.astype(np.float32) / 255.0
        binary = (mask_f32 > 0.1).astype(np.uint8)
        if binary.any():
            # Distance transform: each pixel gets its distance to the mask boundary
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            # Fade width proportional to face size for consistent smoothing
            fade_width = max(14, int(crop_size * 0.22))  # Increased for smoother upper face blending

            # Create smooth alpha ramp: 1.0 at center, fading to 0 at edges
            alpha = np.clip(dist / fade_width, 0.0, 1.0)

            # Apply smoothstep curve for ultra-smooth falloff near edges.
            # smoothstep(t) = 3t^2 - 2t^3  (zero derivative at endpoints)
            # This creates a much gentler slope at the boundary than a linear
            # ramp, which significantly reduces high-frequency jitter at the
            # composite edge -- especially noticeable around eyes/eyebrows.
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)

            # Apply vertical position weighting: extra smoothing in upper face
            # The eyes/eyebrows region (upper 50% of the mask) gets an
            # additional gaussian-weighted mask so alpha drops off faster
            # near the forehead/temple boundary, further reducing jitter.
            coords_y, coords_x = np.where(binary > 0)
            if len(coords_y) > 0:
                mask_top = coords_y.min()
                mask_mid_y = (coords_y.min() + coords_y.max()) / 2.0
                # Build a vertical Gaussian that peaks at mask center and
                # drops toward the top (forehead) edge where jitter is worst.
                y_grid = np.arange(h).reshape(h, 1)
                # sigma proportional to mask height for consistent behavior
                mask_height = coords_y.max() - coords_y.min()
                vert_sigma = max(20, mask_height * 0.3)
                # Shift center slightly downward so the upper half gets
                # stronger attenuation.
                vert_center = mask_mid_y + mask_height * 0.15
                vert_weight = np.exp(-0.5 * ((y_grid - vert_center) / vert_sigma) ** 2)
                # Only attenuate (reduce alpha), never boost it.
                vert_weight = np.clip(vert_weight, 0.0, 1.0)
                # Scale the vertical weight so it has moderate effect (~0.6 to 1.0)
                vert_weight = 0.4 + 0.6 * vert_weight
                alpha = alpha * vert_weight

            mask = alpha
        else:
            mask = mask_f32

        # Very heavy Gaussian blur for ultra-smooth transitions
        # Larger kernel for smoother blending, especially in the upper face region
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        # Second blur pass for even wider smoothing
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        mask = np.clip(mask, 0, 1)[:, :, np.newaxis]
        result = (
            frame.astype(np.float32) * (1.0 - mask)
            + warped_back.astype(np.float32) * mask
        )
        return result.astype(np.uint8)

    def _read_model_crop_size(self) -> Optional[int]:
        """Read the required face-crop spatial size from the ONNX model.

        Scans the model's inputs for one classified as ``"image"`` by
        :meth:`_classify_input` and returns its height dimension
        (``shape[2]``).  SimSwap crops are always square, so height == width.

        Returns ``None`` when no ONNX session is loaded, when no image input
        is found, or when the shape metadata is absent or unreadable.
        """
        if self._ort_session is None:
            return None
        for inp in self._ort_session.get_inputs():
            if self._classify_input(inp) == "image":
                shape = getattr(inp, "shape", None)
                if shape is not None:
                    try:
                        return int(shape[2])
                    except (IndexError, TypeError, ValueError):
                        pass
        return None

    @staticmethod
    def _classify_input(inp_meta) -> str:
        """Return the semantic role of an ONNX input based on its shape.

        Returns
        -------
        ``"image"``
            The input is a face-image crop: rank-4 with ``shape[1] == 3``
            (RGB channels).
        ``"embedding"``
            The input carries the identity embedding: rank-2 (``(N, 512)``)
            or rank-4 with ``shape[1] != 3`` (e.g. ``(N, 512, 1, 1)``).
        ``"unknown"``
            The shape metadata is absent, unreadable, or doesn't fit either
            pattern; the caller should fall back to positional assignment.
        """
        shape = getattr(inp_meta, "shape", None)
        if shape is None:
            return "unknown"
        try:
            rank = len(shape)
        except TypeError:
            return "unknown"
        if rank == 2:
            # Flat (N, features) tensor — always the identity embedding
            return "embedding"
        if rank == 4:
            try:
                ch = shape[1]
            except (IndexError, TypeError):
                return "unknown"
            if ch == 3:
                return "image"
            return "embedding"
        return "unknown"

    @staticmethod
    def _fit_embedding_shape(
        embedding: np.ndarray, inp_meta
    ) -> np.ndarray:
        """Reshape *embedding* to match the rank expected by *inp_meta*.

        SimSwap-style models that pass the identity vector through
        convolutional blocks expect a rank-4 tensor ``(N, 512, 1, 1)``;
        feature-vector models expect the flat rank-2 form ``(N, 512)``.
        The correct rank is read from the ONNX input descriptor's
        ``shape`` attribute so no per-model hard-coding is required.

        The input *embedding* is expected to have shape ``(1, 512)`` — the
        canonical form returned by :meth:`_prepare_source_embedding`.  If
        the shape does not match this expectation, or if the shape metadata
        is absent or the rank is neither 2 nor 4, the embedding is returned
        unchanged rather than risking a silent misreshape.
        """
        shape = getattr(inp_meta, "shape", None)
        if shape is None:
            return embedding
        try:
            rank = len(shape)
        except TypeError:
            return embedding
        if embedding.shape != (1, 512):
            # Guard: only reshape when we have the canonical (1, 512) form
            return embedding
        if rank == 4:
            return embedding.reshape(1, -1, 1, 1)
        if rank == 2:
            return embedding.reshape(1, -1)
        return embedding

    def _build_feed_dict(
        self, source_embedding: np.ndarray, target_tensor: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Build the ONNX input dict for SimSwap inference.

        Tries to match inputs by the conventional names ``"source"`` and
        ``"target"``.  When neither name is present (e.g. unofficial model
        exports using ``"latent_id"`` / ``"face_image"`` or the official
        256-model with ``"input"`` / ``"latent_id"``), :meth:`_classify_input`
        inspects each input's shape metadata to determine which slot is the
        identity embedding and which is the image crop — regardless of their
        order in the model.  If shape metadata is absent or ambiguous, a final
        positional fallback assigns the first input to the embedding and the
        second to the crop.

        The identity embedding is automatically reshaped to match the
        model's expected input rank (rank-2 ``(N, 512)`` or rank-4
        ``(N, 512, 1, 1)``) using :meth:`_fit_embedding_shape`.
        """
        inputs = self._ort_session.get_inputs()  # type: ignore[union-attr]
        feed: dict[str, np.ndarray] = {}
        for inp in inputs:
            if inp.name == "source":
                feed["source"] = self._fit_embedding_shape(source_embedding, inp)
            elif inp.name == "target":
                feed["target"] = target_tensor

        if not feed:
            # Shape-based matching: identify which input is the embedding and
            # which is the image crop.  This handles both ordering conventions:
            #  • unofficial-512: embedding first, crop second
            #  • official-256:   crop first ("input"), embedding second ("latent_id")
            if len(inputs) >= 2:
                emb_inp = next(
                    (i for i in inputs if self._classify_input(i) == "embedding"),
                    None,
                )
                img_inp = next(
                    (i for i in inputs if self._classify_input(i) == "image"),
                    None,
                )
                if emb_inp is not None and img_inp is not None:
                    logger.debug(
                        "SimSwap model input names %s — shape-based mapping: "
                        "embedding→'%s', crop→'%s'.",
                        [i.name for i in inputs],
                        emb_inp.name,
                        img_inp.name,
                    )
                    feed = {
                        emb_inp.name: self._fit_embedding_shape(
                            source_embedding, emb_inp
                        ),
                        img_inp.name: target_tensor,
                    }
                else:
                    # Last-resort positional fallback (no usable shape metadata)
                    logger.debug(
                        "SimSwap model input names %s don't match expected "
                        "'source'/'target' and shapes are ambiguous — "
                        "using positional mapping.",
                        [i.name for i in inputs],
                    )
                    feed = {
                        inputs[0].name: self._fit_embedding_shape(
                            source_embedding, inputs[0]
                        ),
                        inputs[1].name: target_tensor,
                    }
            else:
                logger.warning(
                    "SimSwap model has fewer than 2 inputs (%s); "
                    "inference will likely fail.",
                    [inp.name for inp in inputs],
                )

        return feed

    def _build_image_feed_dict(
        self, source_tensor: np.ndarray, target_tensor: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Build the ONNX input dict for image-source models (uniface / blendswap).

        Both inputs are image tensors so they cannot be distinguished by shape.
        Matching strategy:

        1. If the model has inputs named exactly ``"source"`` and ``"target"``,
           use those names directly.
        2. Otherwise, use positional assignment: the first input receives the
           source portrait crop and the second receives the target face crop.
           This handles models exported with generic names such as
           ``"input_0"`` / ``"input_1"``.

        A DEBUG log is emitted to make the chosen mapping visible when
        diagnosing unexpected inference failures.
        """
        inputs = self._ort_session.get_inputs()  # type: ignore[union-attr]
        input_names = [inp.name for inp in inputs]

        if "source" in input_names and "target" in input_names:
            logger.debug(
                "%s model inputs %s — name-based mapping: "
                "source→'source', target→'target'.",
                self._model_type,
                input_names,
            )
            return {"source": source_tensor, "target": target_tensor}

        if len(inputs) >= 2:
            logger.debug(
                "%s model inputs %s don't include 'source'/'target' — "
                "using positional mapping: source→'%s', target→'%s'.",
                self._model_type,
                input_names,
                inputs[0].name,
                inputs[1].name,
            )
            return {inputs[0].name: source_tensor, inputs[1].name: target_tensor}

        logger.warning(
            "%s model has fewer than 2 inputs (%s); inference will likely fail.",
            self._model_type,
            input_names,
        )
        return {"source": source_tensor, "target": target_tensor}

    def _swap_simswap(
        self, frame: np.ndarray, frame_face, identity_face
    ) -> np.ndarray:
        """Run one ONNX-based face swap and return the updated frame.

        Handles all ONNX model families (simswap, uniface, hyperswap,
        blendswap).  The source identity is prepared differently depending
        on ``self._simswap_params["source_type"]``:

        * ``"embedding"``      — :meth:`_prepare_source_embedding` (simswap)
        * ``"embedding_norm"`` — :meth:`_prepare_embedding_norm` (hyperswap)
        * ``"image"``          — :meth:`_prepare_source_frame` (uniface/blendswap)

        All error conditions (missing keypoints, RANSAC failure, missing
        source, ONNX errors) are caught here and logged as warnings so
        that a single bad frame never crashes the pipeline.  The unmodified
        *frame* is returned when any step fails.
        """
        params = self._simswap_params  # type: ignore[union-attr]
        size = params["size"]
        template_name = params.get("template", "arcface_112_v1")
        source_type = params.get("source_type", "embedding")

        # 1. Guard: keypoints are required for face alignment
        kps = getattr(frame_face, "kps", None)
        if kps is None:
            logger.warning(
                "%s: face object has no keypoints — skipping this face.",
                self._model_type,
            )
            return frame

        # 2. Align and crop the target face from the frame
        # Enable landmark filtering to reduce jitter from noisy detections
        # This is especially helpful for the upper face (eyes/eyebrows)
        try:
            crop, affine_M = self._warp_face(
                frame, kps, size, template_name,
                use_landmark_filter=True,  # Enable jitter reduction
                landmark_sigma=2.5,        # Increased from 1.5 to reduce eyebrow jitter
            )
        except RuntimeError as exc:
            logger.warning(
                "%s alignment failed: %s — skipping this face.",
                self._model_type, exc,
            )
            return frame

        # 3. Prepare source identity and build the ONNX feed dict
        target_tensor = self._prepare_crop_frame(crop)

        if source_type == "image":
            # uniface / blendswap: source is a warped portrait crop image.
            # Use the model's actual input names (positional fallback handles
            # exports like input_0 / input_1 as well as source / target).
            source_tensor = self._prepare_source_frame(identity_face)
            if source_tensor is None:
                return frame
            feed_dict: dict[str, np.ndarray] = self._build_image_feed_dict(
                source_tensor, target_tensor
            )
        elif source_type == "embedding_norm":
            # hyperswap: source is the L2-normalised ArcFace embedding
            source_embedding = self._prepare_embedding_norm(identity_face)
            if source_embedding is None:
                return frame
            feed_dict = self._build_feed_dict(source_embedding, target_tensor)
        else:
            # simswap: raw embedding with optional converter
            source_embedding = self._prepare_source_embedding(identity_face)
            if source_embedding is None:
                logger.warning(
                    "SimSwap: portrait face has no embedding — skipping this face."
                )
                return frame
            feed_dict = self._build_feed_dict(source_embedding, target_tensor)

        # 4. Run ONNX inference
        try:
            outputs = self._ort_session.run(  # type: ignore[union-attr]
                None, feed_dict
            )
        except Exception as exc:
            # onnxruntime raises its own non-standard exception hierarchy;
            # catch broadly here so a bad frame never kills the pipeline.
            # The exception type is included in the log to make genuine bugs
            # (e.g. programming errors) clearly visible.
            logger.warning(
                "%s ONNX inference failed (%s: %s) — skipping this face.",
                self._model_type,
                type(exc).__name__,
                exc,
            )
            return frame

        # 5. Post-process and paste back
        result_crop = self._normalize_crop_frame(outputs[0][0])
        return self._paste_back(frame, result_crop, affine_M)

    # ── Public interface ─────────────────────────────────────────────────────

    def swap(
        self,
        frame: np.ndarray,
        source_face,
        target_face,
    ) -> np.ndarray:
        """Swap a single face in the frame.

        Args:
            frame: BGR image.
            source_face: InsightFace Face object detected in the frame
                (provides the pose/expression to preserve).
            target_face: InsightFace Face object from a portrait image
                (provides the identity to swap in).

        Returns:
            The frame with the source face replaced by the target identity.
        """
        if self._model_type == "inswapper":
            return self._model.get(frame, source_face, target_face, paste_back=True)
        return self._swap_simswap(frame, source_face, target_face)

    def swap_multiple(
        self,
        frame: np.ndarray,
        swap_pairs: list[tuple],
        frame_idx: int,
    ) -> np.ndarray:
        """Apply multiple face swaps on a single frame.

        Args:
            frame: BGR image.
            swap_pairs: list of (source_face, target_face) tuples.
            frame_idx: frame index used for debug logging only.

        Returns:
            The frame with all specified faces replaced.
        """
        result = frame.copy()
        for source_face, target_face in swap_pairs:
            result = self.swap(result, source_face, target_face)
        logger.debug("Frame %d: Applied %d face swaps", frame_idx, len(swap_pairs))
        return result
