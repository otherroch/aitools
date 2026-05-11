"""Configuration dataclasses for the character replacement pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    blend_mode: str = "seamless"  # "seamless" (hybrid Poisson+alpha) or "alpha"
    mask_blur_kernel: int = 31  # Gaussian blur for mask edges (wider for smoother transitions)
    mask_erode_pixels: int = 2  # erode mask to avoid boundary artifacts (reduced for wider swap area)

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ── Diagnostics ──────────────────────────────────────────────────────
    enable_timers: bool = False  # Collect and report per-stage timing distribution

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = OK)."""
        errors: list[str] = []
        if not self.input_video:
            errors.append("input_video is required")
        elif not Path(self.input_video).is_file():
            errors.append(f"input_video not found: {self.input_video}")
        if not self.output_video:
            errors.append("output_video is required")
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
