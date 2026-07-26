"""CLI entry point for the character replacement pipeline."""

import os

# Prevent OpenMP duplicate-library crash (conda + pip torch on Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path

from .config import CharacterMapping, PipelineConfig
from .pipeline import CharacterReplacementPipeline
from .scail2_runner import Scail2PreparedAssetsRunner


def _positive_int(value: str) -> int:
    """Argparse type that accepts only positive integers (>= 1)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer (>= 1), got {n}"
        )
    return n


def _unit_interval(value: str) -> float:
    """Argparse type for floats in the closed interval [0, 1]."""
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}")
    if math.isnan(x):
        raise argparse.ArgumentTypeError(
            f"expected a value in [0, 1], got {x}"
        )
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a value in [0, 1], got {x}"
        )
    return x


def _arg_get(args: argparse.Namespace, name: str, default):
    """Return an argparse value without triggering MagicMock fallback attrs."""
    values = getattr(args, "__dict__", None)
    if isinstance(values, dict) and name in values:
        return values[name]
    return default


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chararep",
        description=(
            "Replace characters (faces) in a video with different identities.\n"
            "Requires an NVIDIA GPU with CUDA support (optimised for RTX 5090)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Replace one character using a JSON config:
  chararep --config swap_config.json

  # Quick run — each --char takes a FIND folder and a REPLACE folder.
  # The FIND folder name becomes the character label.
  chararep -i input.mp4 -o output.mp4 \\
      --char originals/villain replacements/villain \\
      --char originals/hero replacements/hero

  Folder layout:
    originals/
      villain/             ← photos of the original "villain" face in the video
        screenshot1.jpg
        screenshot2.png
      hero/
        hero_frame.jpg
    replacements/
      villain/             ← photos of the new face to swap in for "villain"
        new_face1.jpg
      hero/
        new_hero.jpg

Config JSON format
------------------
  {
    "input_video": "input.mp4",
    "output_video": "output.mp4",
    "characters": [
      {
        "find": "originals/villain",
        "replace": "replacements/villain",
        "similarity_threshold": 0.5
      },
      {
        "find": "originals/hero",
        "replace": "replacements/hero"
      }
    ],
    "enable_face_enhancement": true,
    "device_id": 0
  }
""",
    )

    # ── I/O ──────────────────────────────────────────────────────────────
    p.add_argument("-i", "--input", dest="input_video", help="Input video path.")
    p.add_argument("-o", "--output", dest="output_video", help="Output video path.")
    p.add_argument(
        "--config",
        dest="config_file",
        help="JSON config file (overrides all other args).",
    )
    p.add_argument(
        "--backend",
        choices=["classic", "scail2"],
        default=PipelineConfig.backend,
        help="Execution backend (default: classic).",
    )

    # ── Characters ───────────────────────────────────────────────────────
    p.add_argument(
        "--char",
        dest="characters",
        action="append",
        default=[],
        nargs=2,
        metavar=("FIND_FOLDER", "REPLACE_FOLDER"),
        help=(
            "A pair of folders: the first contains images of the original "
            "face to find in the video, the second contains images of the "
            "new face to swap in.  The FIND folder name is used as the "
            "character label.  Repeat up to 3 times."
        ),
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Cosine-similarity threshold for identity matching (default: 0.5).",
    )

    # ── Model paths ──────────────────────────────────────────────────────
    p.add_argument(
        "--swap-model-path",
        default=None,
        help=(
            "Path to the face-swap ONNX model. "
            "Supports inswapper_128.onnx (default, auto-detected) and "
            "SimSwap models such as simswap_256.onnx or "
            "simswap_unofficial_512.onnx. "
            "The model family is detected automatically from the filename."
        ),
    )
    p.add_argument(
        "--embedding-converter-path",
        default=None,
        help=(
            "Optional path to the SimSwap embedding converter ONNX model "
            "(e.g. crossface_simswap.onnx). "
            "Only used with SimSwap models; improves identity fidelity. "
            "If omitted, the raw ArcFace embedding is used directly."
        ),
    )
    p.add_argument(
        "--detection-model",
        default="buffalo_l",
        help=(
            "Face detection backend and model. "
            "Use 'dlib', 'hog', or 'cnn' for the dlib backend; "
            "any other value (e.g. 'buffalo_l', 'buffalo_sc') selects "
            "an InsightFace model pack (default: buffalo_l)."
        ),
    )
    p.add_argument(
        "--detect-size",
        type=int,
        default=640,
        metavar="N",
        help="Detection resolution: frame is resized to NxN before RetinaFace runs (default: 640). Try 1024 for better landmark precision on HD video.",
    )

    # ── SCAIL-2 prepared-assets mode ────────────────────────────────────
    p.add_argument(
        "--scail2-repo-path",
        default=None,
        help="Path to an upstream SCAIL-2 checkout containing generate.py.",
    )
    p.add_argument(
        "--scail2-pose-repo-path",
        default=None,
        help="Optional path to an upstream SCAIL-Pose checkout. Defaults to <scail2_repo_path>/SCAIL-Pose.",
    )
    p.add_argument(
        "--scail2-ckpt-dir",
        default=None,
        help="Path to the SCAIL-2 checkpoint directory passed to --ckpt_dir.",
    )
    p.add_argument(
        "--scail2-model-path",
        default=None,
        help="Path to the converted SCAIL-2 .safetensors file passed to --scail_path.",
    )
    p.add_argument(
        "--scail2-model-name",
        default=PipelineConfig.scail2_model_name,
        help=f"SCAIL-2 model name argument for generate.py (default: {PipelineConfig.scail2_model_name}).",
    )
    p.add_argument(
        "--scail2-reference-image",
        default=None,
        help="SCAIL-2 reference image. When masks are omitted, this image is also used as the SCAIL-Pose auto-prep reference.",
    )
    p.add_argument(
        "--scail2-reference-mask",
        default=None,
        help="Prepared SCAIL-2 reference mask image.",
    )
    p.add_argument(
        "--scail2-mask-video",
        default=None,
        help="Prepared SCAIL-2 driving mask video.",
    )
    p.add_argument(
        "--scail2-additional-reference-image",
        dest="scail2_additional_reference_images",
        nargs="+",
        default=[],
        help=(
            "Optional extra SCAIL-2 reference images for multi-reference "
            "replacement. Pass a space-separated list."
        ),
    )
    p.add_argument(
        "--scail2-additional-reference-mask",
        dest="scail2_additional_reference_masks",
        nargs="+",
        default=[],
        help=(
            "Optional extra SCAIL-2 masks paired positionally with "
            "--scail2-additional-reference-image. Pass a space-separated list."
        ),
    )
    p.add_argument(
        "--scail2-prompt",
        default=None,
        help="Positive prompt describing the output video for SCAIL-2.",
    )
    p.add_argument(
        "--scail2-prompt-file",
        default=None,
        help="Path to a text file containing the SCAIL-2 positive prompt.",
    )
    p.add_argument(
        "--scail2-matchnearest",
        action="store_true",
        help="SCAIL-Pose auto-prep: allow two driving tracks and keep the one closest to the reference mask by IoU.",
    )
    p.add_argument(
        "--scail2-egocentric",
        action="store_true",
        help="SCAIL-Pose auto-prep: union multiple disconnected actor parts for egocentric or first-person footage.",
    )
    p.add_argument(
        "--scail2-sam-text",
        nargs="+",
        default=list(PipelineConfig().scail2_sam_text),
        help="SCAIL-Pose auto-prep SAM text prompts (default: human character).",
    )
    p.add_argument(
        "--scail2-sam3-model",
        default=None,
        help="Optional path to SAM3 weights for SCAIL-Pose auto-prep.",
    )
    p.add_argument(
        "--scail2-target-width",
        type=_positive_int,
        default=PipelineConfig.scail2_target_width,
        help=f"SCAIL-2 target width, divisible by 32 (default: {PipelineConfig.scail2_target_width}).",
    )
    p.add_argument(
        "--scail2-target-height",
        type=_positive_int,
        default=PipelineConfig.scail2_target_height,
        help=f"SCAIL-2 target height, divisible by 32 (default: {PipelineConfig.scail2_target_height}).",
    )
    p.add_argument(
        "--scail2-sample-steps",
        type=_positive_int,
        default=PipelineConfig.scail2_sample_steps,
        help=f"SCAIL-2 denoising steps (default: {PipelineConfig.scail2_sample_steps}).",
    )
    p.add_argument(
        "--scail2-sample-shift",
        type=float,
        default=PipelineConfig.scail2_sample_shift,
        help=f"SCAIL-2 sample shift (default: {PipelineConfig.scail2_sample_shift}).",
    )
    p.add_argument(
        "--scail2-sample-guide-scale",
        type=float,
        default=PipelineConfig.scail2_sample_guide_scale,
        help=f"SCAIL-2 guidance scale (default: {PipelineConfig.scail2_sample_guide_scale}).",
    )
    p.add_argument(
        "--scail2-sample-solver",
        choices=["unipc", "dpm++"],
        default=PipelineConfig.scail2_sample_solver,
        help=f"SCAIL-2 sampler (default: {PipelineConfig.scail2_sample_solver}).",
    )
    p.add_argument(
        "--scail2-no-offload-model",
        action="store_false",
        dest="scail2_offload_model",
        help="Disable SCAIL-2 model offload during generate.py execution.",
    )
    p.set_defaults(scail2_offload_model=PipelineConfig.scail2_offload_model)
    p.add_argument(
        "--scail2-work-dir",
        default=None,
        help="Optional parent directory for staged SCAIL-2 job files.",
    )
    p.add_argument(
        "--scail2-keep-intermediates",
        action="store_true",
        help="Keep the staged SCAIL-2 job directory after the run finishes.",
    )

    # ── Enhancement ──────────────────────────────────────────────────────
    p.add_argument(
        "--enhance",
        action="store_true",
        help="Enable GFPGAN face enhancement.",
    )
    p.add_argument(
        "--enhance-model",
        choices=["gfpgan", "codeformer_onnx"],
        default="gfpgan",
        help=(
            "Enhancement backend (default: gfpgan). "
            "'codeformer_onnx' runs a CodeFormer ONNX model via "
            "ONNX Runtime with CUDA, sharing the GPU path used by "
            "the rest of the pipeline and avoiding PyTorch overhead."
        ),
    )
    p.add_argument(
        "--enhance-model-path",
        default=None,
        help=(
            "Path to the enhancement model file. "
            "For gfpgan this defaults to ~/.gfpgan/weights/GFPGANv1.4.pth. "
            "For codeformer_onnx this is **required** (e.g. codeformer.onnx)."
        ),
    )
    p.add_argument(
        "--enhance-weight",
        type=float,
        default=0.7,
        help="Enhancement blend weight 0-1 (default: 0.7).",
    )

    # ── GPU / performance ────────────────────────────────────────────────
    p.add_argument(
        "--batch",
        type=_positive_int,
        default=4,
        dest="batch_size",
        help=(
            "Number of frames to process in parallel (default: 4). "
            "When greater than 1, the pipeline detects faces sequentially "
            "but runs swap/blend/enhance in a thread pool with this many "
            "workers.  Set to 1 for fully sequential processing."
        ),
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ID (default: 0).",
    )
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 (use FP32 everywhere).",
    )

    # ── Output quality ───────────────────────────────────────────────────
    p.add_argument(
        "--codec",
        default="libx264",
        help="Output video codec (default: libx264).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF quality value, lower = better (default: 18).",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Do not copy audio from the original video.",
    )

    # ── Blending ─────────────────────────────────────────────────────────
    p.add_argument(
        "--blend-mode",
        choices=["seamless", "alpha"],
        default=PipelineConfig.blend_mode,
        help=(
            f"Blending strategy (default: {PipelineConfig.blend_mode}). "
            "'alpha' uses a soft mask and is faster and more predictable, "
            "often good when colors/lighting already match reasonably well. "
            "'seamless' uses Poisson cloning to better match lighting and color "
            "with the background, which can look more natural but is slower and "
            "may produce artifacts on extreme lighting or high-contrast edges."
        ),
    )
    p.add_argument(
        "--blender-blur",
        type=int,
        default=PipelineConfig.mask_blur_kernel,
        dest="mask_blur_kernel",
        help=(
            "Gaussian blur kernel size for softening mask edges before blending. "
            "Higher values produce smoother transitions but may lose detail; "
            "lower values keep sharper edges but may leave visible seams. "
            f"0 disables blurring (default: {PipelineConfig.mask_blur_kernel})."
        ),
    )
    p.add_argument(
        "--blender-erode",
        type=int,
        default=PipelineConfig.mask_erode_pixels,
        dest="mask_erode_pixels",
        help=(
            "Pixels to erode from mask to avoid boundary artifacts "
            f"(default: {PipelineConfig.mask_erode_pixels})."
        ),
    )
    p.add_argument(
        "--temporal-smooth-alpha",
        type=_unit_interval,
        default=0.0,
        help=(
            "Previous-frame weight for overlap-only temporal smoothing. "
            "0 disables it; small values like 0.1-0.2 can reduce residual shimmer."
        ),
    )
    p.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=PipelineConfig.scene_cut_threshold,
        help=(
            "Mean per-pixel grayscale difference between consecutive frames "
            "above which a scene cut is declared and all temporal state "
            "(landmark smoothing, tracker tracks, blend buffers) is reset. "
            f"0 disables scene-cut detection (default: {PipelineConfig.scene_cut_threshold})."
        ),
    )

    # ── Logging ──────────────────────────────────────────────────────────
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Write log to a file in addition to stderr.",
    )

    # ── Diagnostics ──────────────────────────────────────────────────────
    p.add_argument(
        "--timers",
        action="store_true",
        help=(
            "Collect cumulative timing for each pipeline stage and report "
            "the percentage distribution when processing completes."
        ),
    )    
    p.add_argument(
        "--dump-config",
        action="store_true",
        help="Print the resolved pipeline configuration as JSON before running.",
    )
    return p.parse_args()


def _scan_image_dir(folder: str, kind: str) -> list[str]:
    """Return sorted image paths from a folder, or exit on error."""
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    p = Path(folder)
    if not p.is_dir():
        print(f"ERROR: {kind} path is not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        str(f) for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
    )
    if not paths:
        print(
            f"ERROR: no images found in {kind} folder: {folder}",
            file=sys.stderr,
        )
        sys.exit(1)

    return paths


def _build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Construct a PipelineConfig from CLI arguments."""

    backend = _arg_get(args, "backend", PipelineConfig.backend)

    temporal_smooth_alpha = getattr(
        args,
        "temporal_smooth_alpha",
        PipelineConfig.temporal_smooth_alpha,
    )
    if (
        not isinstance(temporal_smooth_alpha, (int, float))
        or isinstance(temporal_smooth_alpha, bool)
    ):
        temporal_smooth_alpha = PipelineConfig.temporal_smooth_alpha

    characters: list[CharacterMapping] = []
    for find_folder, replace_folder in args.characters:
        label = Path(find_folder).name
        ref_paths = _scan_image_dir(find_folder, "find")
        portrait_paths = _scan_image_dir(replace_folder, "replace")
        characters.append(
            CharacterMapping(
                source_label=label,
                reference_paths=ref_paths,
                portrait_paths=portrait_paths,
                similarity_threshold=args.similarity_threshold,
            )
        )

    return PipelineConfig(
        backend=backend,
        input_video=args.input_video or "",
        output_video=args.output_video or "",
        characters=characters,
        detection_model=args.detection_model,
        detection_size=(args.detect_size, args.detect_size),
        swap_model_path=args.swap_model_path,
        embedding_converter_path=args.embedding_converter_path,
        enable_face_enhancement=args.enhance,
        enhancement_model=args.enhance_model,
        enhance_model_path=args.enhance_model_path,
        enhancement_weight=args.enhance_weight,
        device_id=args.device,
        batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
        output_codec=args.codec,
        output_quality=args.crf,
        copy_audio=not args.no_audio,
        blend_mode=args.blend_mode,
        mask_blur_kernel=args.mask_blur_kernel,
        mask_erode_pixels=args.mask_erode_pixels,
        temporal_smooth_alpha=float(temporal_smooth_alpha),
        scene_cut_threshold=float(args.scene_cut_threshold),
        log_level="DEBUG" if args.verbose else "INFO",
        log_file=args.log_file,
        enable_timers=args.timers,
        scail2_repo_path=_arg_get(args, "scail2_repo_path", None),
        scail2_pose_repo_path=_arg_get(args, "scail2_pose_repo_path", None),
        scail2_ckpt_dir=_arg_get(args, "scail2_ckpt_dir", None),
        scail2_model_path=_arg_get(args, "scail2_model_path", None),
        scail2_model_name=_arg_get(
            args, "scail2_model_name", PipelineConfig.scail2_model_name
        ),
        scail2_reference_image=_arg_get(args, "scail2_reference_image", None),
        scail2_reference_mask=_arg_get(args, "scail2_reference_mask", None),
        scail2_mask_video=_arg_get(args, "scail2_mask_video", None),
        scail2_additional_reference_images=list(
            _arg_get(args, "scail2_additional_reference_images", []) or []
        ),
        scail2_additional_reference_masks=list(
            _arg_get(args, "scail2_additional_reference_masks", []) or []
        ),
        scail2_prompt=_arg_get(args, "scail2_prompt", None),
        scail2_prompt_file=_arg_get(args, "scail2_prompt_file", None),
        scail2_target_width=int(
            _arg_get(args, "scail2_target_width", PipelineConfig.scail2_target_width)
        ),
        scail2_target_height=int(
            _arg_get(args, "scail2_target_height", PipelineConfig.scail2_target_height)
        ),
        scail2_sample_steps=int(
            _arg_get(args, "scail2_sample_steps", PipelineConfig.scail2_sample_steps)
        ),
        scail2_sample_shift=float(
            _arg_get(args, "scail2_sample_shift", PipelineConfig.scail2_sample_shift)
        ),
        scail2_sample_guide_scale=float(
            _arg_get(
                args,
                "scail2_sample_guide_scale",
                PipelineConfig.scail2_sample_guide_scale,
            )
        ),
        scail2_sample_solver=_arg_get(
            args, "scail2_sample_solver", PipelineConfig.scail2_sample_solver
        ),
        scail2_matchnearest=bool(_arg_get(args, "scail2_matchnearest", False)),
        scail2_egocentric=bool(_arg_get(args, "scail2_egocentric", False)),
        scail2_sam_text=list(
            _arg_get(args, "scail2_sam_text", list(PipelineConfig().scail2_sam_text))
        ),
        scail2_sam3_model=_arg_get(args, "scail2_sam3_model", None),
        scail2_offload_model=bool(
            _arg_get(
                args,
                "scail2_offload_model",
                PipelineConfig.scail2_offload_model,
            )
        ),
        scail2_work_dir=_arg_get(args, "scail2_work_dir", None),
        scail2_keep_intermediates=bool(
            _arg_get(args, "scail2_keep_intermediates", False)
        ),
    )


def _build_config_from_json(path: str) -> PipelineConfig:
    """Load a PipelineConfig from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    characters: list[CharacterMapping] = []
    for ch in data.pop("characters", []):
        if "find" in ch and "replace" in ch:
            find_folder = ch.pop("find")
            replace_folder = ch.pop("replace")
            label = ch.pop("label", Path(find_folder).name)
            ref_paths = _scan_image_dir(find_folder, "find")
            portrait_paths = _scan_image_dir(replace_folder, "replace")
            threshold = ch.pop("similarity_threshold", 0.5)
            characters.append(
                CharacterMapping(
                    source_label=label,
                    reference_paths=ref_paths,
                    portrait_paths=portrait_paths,
                    similarity_threshold=threshold,
                )
            )
        else:
            # Explicit paths lists
            if "label" in ch:
                ch.setdefault("source_label", ch.pop("label"))
            characters.append(CharacterMapping(**ch))

    return PipelineConfig(characters=characters, **data)


def _setup_logging(cfg: PipelineConfig) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if cfg.log_file:
        handlers.append(logging.FileHandler(cfg.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def _build_runner(cfg: PipelineConfig):
    """Return the execution backend selected by configuration."""
    if str(cfg.backend).strip().lower() == "scail2":
        return Scail2PreparedAssetsRunner(cfg)
    return CharacterReplacementPipeline(cfg)


def main() -> None:
    args = _parse_args()

    # Build configuration
    if args.config_file:
        cfg = _build_config_from_json(args.config_file)
    else:
        cfg = _build_config_from_args(args)

    _setup_logging(cfg)
    logger = logging.getLogger("chararep")

    # Validate
    errors = cfg.validate()
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        sys.exit(1)

    logger.debug("Pipeline configuration: %s", cfg)

    if args.dump_config:
        print(json.dumps(dataclasses.asdict(cfg), indent=2))

    # Run pipeline
    runner = _build_runner(cfg)
    stats = runner.run()

    # Summary
    logger.info("=" * 60)
    logger.info("Done!  Output: %s", cfg.output_video)
    logger.info(
        "Processed %d frames in %.1fs (%.1f fps)",
        stats["frames_total"],
        stats["elapsed_s"],
        stats["fps"],
    )
    if cfg.backend == "classic":
        logger.info(
            "Frames with swaps: %d  |  Total faces swapped: %d",
            stats["frames_swapped"],
            stats["faces_swapped"],
        )
    else:
        logger.info("SCAIL-2 prepared-assets backend completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
