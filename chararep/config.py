"""Configuration dataclasses for the character replacement pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_SUPPORTED_BACKENDS = frozenset({"classic", "scail2"})


@dataclass
class CharacterMapping:
    """Maps an original character in the video to a replacement identity.

    *   ``reference_paths`` — photos of the face to **find** in the video.
    *   ``portrait_paths``  — photos of the **new** face to swap in.
    """

    source_label: str  # descriptive label, e.g. "villain"
    reference_paths: list[str] = field(default_factory=list)  # face to FIND
    portrait_paths: list[str] = field(default_factory=list)   # face to REPLACE with
    similarity_threshold: float = 0.5  # ArcFace cosine-similarity cutoff


@dataclass
class PipelineConfig:
    """Top-level configuration for the face-replacement pipeline."""

    # ── Backend selection ────────────────────────────────────────────────
    backend: str = "classic"  # "classic" or "scail2"

    # ── I/O ──────────────────────────────────────────────────────────────
    input_video: str = ""
    output_video: str = ""

    # ── Character mappings (up to 3) ─────────────────────────────────────
    characters: list[CharacterMapping] = field(default_factory=list)

    # ── Detection / tracking ─────────────────────────────────────────────
    detection_model: str = "buffalo_l"  # "dlib"/"hog"/"cnn" → dlib backend; else InsightFace model pack
    detection_threshold: float = 0.5
    detection_size: tuple[int, int] = (640, 640)
    tracker_max_age: int = 30  # frames to keep a lost track
    tracker_iou_threshold: float = 0.3

    # ── Swap model ───────────────────────────────────────────────────────
    swap_model_path: Optional[str] = None  # path to swap .onnx (inswapper or simswap)
    # If None, the pipeline will attempt to auto-detect inswapper_128.onnx
    embedding_converter_path: Optional[str] = None  # optional crossface converter for simswap

    # ── Enhancement ──────────────────────────────────────────────────────
    enable_face_enhancement: bool = True
    enhancement_model: str = "gfpgan"  # "gfpgan" or "codeformer_onnx"
    enhancement_weight: float = 0.7  # blend weight for enhanced face
    enhance_model_path: Optional[str] = None  # custom path to enhancement model

    # ── GPU / Performance ────────────────────────────────────────────────
    device_id: int = 0  # CUDA device ordinal
    batch_size: int = 4  # frames to prefetch
    use_fp16: bool = True
    pin_memory: bool = True
    num_io_workers: int = 2

    # ── Output quality ───────────────────────────────────────────────────
    output_codec: str = "libx264"
    output_quality: int = 18  # CRF value (lower = better quality)
    copy_audio: bool = True  # mux original audio into output

    # ── Blending ─────────────────────────────────────────────────────────
    blend_mode: str = "alpha"  # "seamless" (hybrid Poisson+alpha) or "alpha"
    mask_blur_kernel: int = 15  # Gaussian blur kernel size for softening mask edges
    mask_erode_pixels: int = 2  # erosion pixels to avoid boundary artifacts

    # ── Scene-cut detection ──────────────────────────────────────────────
    # Mean per-pixel difference threshold between consecutive frames.
    # When exceeded, the pipeline resets all temporal state (landmark
    # smoothing history, tracker tracks, temporal blend buffers) so that
    # stale data from the previous scene does not contaminate the new one.
    # 0.0 disables scene-cut detection.
    scene_cut_threshold: float = 35.0

    # ── Temporal smoothing ────────────────────────────────────────────────
    # Previous-frame weight for overlap-only temporal smoothing.
    # 0.0 disables it; small values such as 0.1-0.2 can damp residual shimmer.
    temporal_smooth_alpha: float = 0.0

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ── Diagnostics ──────────────────────────────────────────────────────
    enable_timers: bool = False  # Collect and report per-stage timing distribution

    # ── SCAIL-2 prepared-assets mode ─────────────────────────────────────
    # These fields are used only when backend == "scail2".
    scail2_repo_path: Optional[str] = None
    scail2_ckpt_dir: Optional[str] = None
    scail2_model_path: Optional[str] = None
    scail2_model_name: str = "SCAIL-14B"
    scail2_reference_image: Optional[str] = None
    scail2_reference_mask: Optional[str] = None
    scail2_mask_video: Optional[str] = None
    scail2_prompt: Optional[str] = None
    scail2_prompt_file: Optional[str] = None
    scail2_target_width: int = 896
    scail2_target_height: int = 512
    scail2_sample_steps: int = 40
    scail2_sample_shift: float = 3.0
    scail2_sample_guide_scale: float = 5.0
    scail2_sample_solver: str = "unipc"
    scail2_offload_model: bool = True
    scail2_work_dir: Optional[str] = None
    scail2_keep_intermediates: bool = False

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = OK)."""
        errors: list[str] = []
        backend = str(self.backend).strip().lower()
        if backend not in _SUPPORTED_BACKENDS:
            errors.append(
                f"backend must be one of {sorted(_SUPPORTED_BACKENDS)}, got: {self.backend!r}"
            )
            return errors

        if not self.input_video:
            errors.append("input_video is required")
        elif not Path(self.input_video).is_file():
            errors.append(f"input_video not found: {self.input_video}")
        if not self.output_video:
            errors.append("output_video is required")

        if backend == "scail2":
            errors.extend(self._validate_scail2_backend())
            return errors

        errors.extend(self._validate_classic_backend())
        return errors

    def _validate_classic_backend(self) -> list[str]:
        """Validate the legacy face-swap pipeline configuration."""
        errors: list[str] = []
        if len(self.characters) == 0:
            errors.append("At least one character mapping is required")
        if len(self.characters) > 3:
            errors.append("Maximum of 3 character replacements supported")
        for i, ch in enumerate(self.characters):
            if not ch.reference_paths:
                errors.append(
                    f"Character '{ch.source_label}' has no reference images "
                    f"(needed to identify the face in the video)"
                )
            for p in ch.reference_paths:
                if not Path(p).is_file():
                    errors.append(
                        f"Reference image not found for '{ch.source_label}': {p}"
                    )
            if not ch.portrait_paths:
                errors.append(
                    f"Character '{ch.source_label}' has no portrait images "
                    f"(the replacement face)"
                )
            for p in ch.portrait_paths:
                if not Path(p).is_file():
                    errors.append(
                        f"Portrait not found for '{ch.source_label}': {p}"
                    )
        return errors

    def _validate_scail2_backend(self) -> list[str]:
        """Validate prepared-assets mode for the SCAIL-2 backend."""
        errors: list[str] = []

        if self.characters:
            errors.append(
                "SCAIL-2 backend does not yet support character mappings; provide prepared SCAIL-2 assets instead"
            )

        self._require_dir(
            errors,
            "scail2_repo_path",
            self.scail2_repo_path,
            must_contain="generate.py",
        )
        self._require_dir(errors, "scail2_ckpt_dir", self.scail2_ckpt_dir)
        self._require_file(errors, "scail2_model_path", self.scail2_model_path)
        self._require_file(
            errors,
            "scail2_reference_image",
            self.scail2_reference_image,
        )
        self._require_file(
            errors,
            "scail2_reference_mask",
            self.scail2_reference_mask,
        )
        self._require_file(errors, "scail2_mask_video", self.scail2_mask_video)

        prompt = (self.scail2_prompt or "").strip()
        if self.scail2_prompt_file:
            self._require_file(errors, "scail2_prompt_file", self.scail2_prompt_file)
        if not prompt and not self.scail2_prompt_file:
            errors.append("SCAIL-2 backend requires scail2_prompt or scail2_prompt_file")

        if self.scail2_target_width <= 0 or self.scail2_target_height <= 0:
            errors.append("SCAIL-2 target width and height must be positive")
        elif self.scail2_target_width % 32 != 0 or self.scail2_target_height % 32 != 0:
            errors.append(
                "SCAIL-2 target width and height must both be divisible by 32"
            )

        if self.scail2_sample_steps < 1:
            errors.append("SCAIL-2 sample steps must be >= 1")
        if self.scail2_sample_solver not in {"unipc", "dpm++"}:
            errors.append(
                "SCAIL-2 sample solver must be 'unipc' or 'dpm++'"
            )

        if self.scail2_work_dir:
            work_dir = Path(self.scail2_work_dir)
            if work_dir.exists() and not work_dir.is_dir():
                errors.append(
                    f"scail2_work_dir must be a directory when provided: {self.scail2_work_dir}"
                )

        return errors

    @staticmethod
    def _require_file(errors: list[str], field_name: str, value: Optional[str]) -> None:
        if not value:
            errors.append(f"{field_name} is required for SCAIL-2 backend")
            return
        if not Path(value).is_file():
            errors.append(f"{field_name} not found: {value}")

    @staticmethod
    def _require_dir(
        errors: list[str],
        field_name: str,
        value: Optional[str],
        *,
        must_contain: Optional[str] = None,
    ) -> None:
        if not value:
            errors.append(f"{field_name} is required for SCAIL-2 backend")
            return

        path = Path(value)
        if not path.is_dir():
            errors.append(f"{field_name} not found: {value}")
            return

        if must_contain and not (path / must_contain).is_file():
            errors.append(
                f"{field_name} must contain {must_contain}: {value}"
            )
