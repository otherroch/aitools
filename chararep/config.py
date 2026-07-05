"""Configuration dataclasses for the character replacement pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_SUPPORTED_BACKENDS = frozenset({"classic", "scail2"})
_SCAIL2_MEMORY_PRESETS = frozenset({"default", "low-vram"})
_SCAIL2_LOW_VRAM_TARGET_WIDTH = 672
_SCAIL2_LOW_VRAM_TARGET_HEIGHT = 384
_SCAIL2_LOW_VRAM_SAMPLE_STEPS = 28


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
    scail2_pose_repo_path: Optional[str] = None
    scail2_ckpt_dir: Optional[str] = None
    scail2_model_path: Optional[str] = None
    scail2_model_name: str = "SCAIL-14B"
    scail2_memory_preset: str = "default"
    scail2_reference_image: Optional[str] = None
    scail2_reference_mask: Optional[str] = None
    scail2_mask_video: Optional[str] = None
    scail2_additional_reference_images: list[str] = field(default_factory=list)
    scail2_additional_reference_masks: list[str] = field(default_factory=list)
    scail2_prompt: Optional[str] = None
    scail2_prompt_file: Optional[str] = None
    scail2_target_width: int = 896
    scail2_target_height: int = 512
    scail2_sample_steps: int = 40
    scail2_sample_shift: float = 3.0
    scail2_sample_guide_scale: float = 5.0
    scail2_sample_solver: str = "unipc"
    scail2_matchnearest: bool = False
    scail2_egocentric: bool = False
    scail2_sam_text: list[str] = field(default_factory=lambda: ["human", "character"])
    scail2_sam3_model: Optional[str] = None
    scail2_offload_model: bool = True
    scail2_extra_args: list[str] = field(default_factory=list)
    scail2_env: dict[str, str] = field(default_factory=dict)
    scail2_fail_on_vram_risk: bool = False
    scail2_work_dir: Optional[str] = None
    scail2_keep_intermediates: bool = False

    def apply_runtime_overrides(self) -> None:
        """Apply backend-specific runtime presets in-place after construction."""
        if str(self.backend).strip().lower() == "scail2":
            self._apply_scail2_memory_preset()

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = OK)."""
        errors: list[str] = []
        backend = str(self.backend).strip().lower()
        if backend not in _SUPPORTED_BACKENDS:
            errors.append(
                f"backend must be one of {sorted(_SUPPORTED_BACKENDS)}, got: {self.backend!r}"
            )
            return errors

        self.apply_runtime_overrides()

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
        """Validate prepared-assets and auto-prep SCAIL-2 modes."""
        errors: list[str] = []
        preset = self._normalized_scail2_memory_preset()
        if preset not in _SCAIL2_MEMORY_PRESETS:
            errors.append(
                "SCAIL-2 memory preset must be one of "
                f"{sorted(_SCAIL2_MEMORY_PRESETS)}, got: {self.scail2_memory_preset!r}"
            )
        else:
            self.scail2_memory_preset = preset

        self._require_dir(
            errors,
            "scail2_repo_path",
            self.scail2_repo_path,
            must_contain="generate.py",
        )
        self._require_dir(errors, "scail2_ckpt_dir", self.scail2_ckpt_dir)
        self._require_file(errors, "scail2_model_path", self.scail2_model_path)

        prompt = (self.scail2_prompt or "").strip()
        if self.scail2_prompt_file:
            self._require_file(errors, "scail2_prompt_file", self.scail2_prompt_file)
        if not prompt and not self.scail2_prompt_file:
            errors.append("SCAIL-2 backend requires scail2_prompt or scail2_prompt_file")

        if self.scail2_matchnearest and self.scail2_egocentric:
            errors.append("SCAIL-2 auto-prep cannot enable both scail2_matchnearest and scail2_egocentric")
        if self.scail2_sam3_model:
            self._require_file(errors, "scail2_sam3_model", self.scail2_sam3_model)
        if not self.scail2_sam_text:
            errors.append("SCAIL-2 auto-prep requires at least one SAM text prompt")
        errors.extend(self._validate_scail2_additional_references())

        prepared_assets = self.scail2_has_prepared_assets()
        any_prepared_asset = any(
            [
                self.scail2_reference_image,
                self.scail2_reference_mask,
                self.scail2_mask_video,
            ]
        )

        if prepared_assets:
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
        else:
            if any_prepared_asset:
                errors.append(
                    "SCAIL-2 prepared-assets mode requires scail2_reference_image, scail2_reference_mask, and scail2_mask_video together"
                )

            if self.scail2_reference_image:
                self._require_file(
                    errors,
                    "scail2_reference_image",
                    self.scail2_reference_image,
                )
            elif not self.characters:
                errors.append(
                    "SCAIL-2 auto-prep requires scail2_reference_image or one character mapping"
                )

            if len(self.characters) > 1:
                errors.append(
                    "SCAIL-2 auto-prep currently supports at most one character mapping"
                )
            elif self.characters:
                errors.extend(self._validate_character_mappings(max_characters=1))

            self._require_dir(
                errors,
                "scail2_pose_repo_path",
                self._resolved_scail2_pose_repo_path(),
                must_contain="NLFPoseExtract/process_replacement.py",
            )

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
        for arg in self.scail2_extra_args:
            if not str(arg).strip():
                errors.append("SCAIL-2 extra args must not contain empty values")
                break
        if not isinstance(self.scail2_env, dict):
            errors.append("SCAIL-2 env overrides must be a mapping of KEY to VALUE")
        else:
            for key in self.scail2_env:
                if not str(key).strip():
                    errors.append("SCAIL-2 env overrides require non-empty variable names")
                    break

        if self.scail2_work_dir:
            work_dir = Path(self.scail2_work_dir)
            if work_dir.exists() and not work_dir.is_dir():
                errors.append(
                    f"scail2_work_dir must be a directory when provided: {self.scail2_work_dir}"
                )

        return errors

    def _validate_scail2_additional_references(self) -> list[str]:
        """Validate optional multi-reference inputs for SCAIL-2."""
        errors: list[str] = []
        images = list(self.scail2_additional_reference_images)
        masks = list(self.scail2_additional_reference_masks)

        if not images and not masks:
            return errors

        if len(images) != len(masks):
            errors.append(
                "SCAIL-2 additional references require matching "
                "scail2_additional_reference_images and "
                "scail2_additional_reference_masks lists of equal length"
            )

        for index, path in enumerate(images):
            if not Path(path).is_file():
                errors.append(
                    "SCAIL-2 additional reference image not found "
                    f"at index {index}: {path}"
                )
        for index, path in enumerate(masks):
            if not Path(path).is_file():
                errors.append(
                    "SCAIL-2 additional reference mask not found "
                    f"at index {index}: {path}"
                )

        return errors

    def scail2_has_prepared_assets(self) -> bool:
        """Return True when explicit SCAIL-2 image/mask assets were provided."""
        return bool(
            self.scail2_reference_image
            and self.scail2_reference_mask
            and self.scail2_mask_video
        )

    def _apply_scail2_memory_preset(self) -> None:
        """Adjust SCAIL-2 defaults for named memory presets.

        This only rewrites fields that still match the class defaults so that
        explicit per-run overrides win over the preset.
        """
        preset = self._normalized_scail2_memory_preset()
        self.scail2_memory_preset = preset
        if preset != "low-vram":
            return

        if self.scail2_target_width == PipelineConfig.scail2_target_width:
            self.scail2_target_width = _SCAIL2_LOW_VRAM_TARGET_WIDTH
        if self.scail2_target_height == PipelineConfig.scail2_target_height:
            self.scail2_target_height = _SCAIL2_LOW_VRAM_TARGET_HEIGHT
        if self.scail2_sample_steps == PipelineConfig.scail2_sample_steps:
            self.scail2_sample_steps = _SCAIL2_LOW_VRAM_SAMPLE_STEPS

    def _normalized_scail2_memory_preset(self) -> str:
        """Return the lowercase preset name, defaulting empty values."""
        preset = str(self.scail2_memory_preset or "default").strip().lower()
        return preset or "default"

    def _resolved_scail2_pose_repo_path(self) -> Optional[str]:
        """Return the configured or default SCAIL-Pose checkout path."""
        if self.scail2_pose_repo_path:
            return self.scail2_pose_repo_path
        if not self.scail2_repo_path:
            return None
        return str(Path(self.scail2_repo_path) / "SCAIL-Pose")

    def _validate_character_mappings(self, *, max_characters: int) -> list[str]:
        """Validate character mappings with the existing chararep constraints."""
        errors: list[str] = []
        if len(self.characters) == 0:
            errors.append("At least one character mapping is required")
            return errors
        if len(self.characters) > max_characters:
            errors.append(
                f"Maximum of {max_characters} character replacements supported"
            )

        for ch in self.characters:
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
