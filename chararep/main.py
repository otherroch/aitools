"""CLI entry point for the character replacement pipeline."""

import os

# Prevent OpenMP duplicate-library crash (conda + pip torch on Windows).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from .config import CharacterMapping, PipelineConfig
from .pipeline import CharacterReplacementPipeline


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
        help="InsightFace model pack (default: buffalo_l).",
    )
    p.add_argument(
        "--detect-size",
        type=int,
        default=640,
        metavar="N",
        help="Detection resolution: frame is resized to NxN before RetinaFace runs (default: 640). Try 1024 for better landmark precision on HD video.",
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
        default="alpha",
        help=(
            "Blending strategy (default: alpha). "
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
        default=15,
        dest="mask_blur_kernel",
        help=(
            "Number of pixels to erode from the mask before blending. "
            "Higher values shrink the mask more, which can hide halo/boundary artifacts "
            "but may cut into the swapped face; lower values preserve more detail but may "
            "leave visible seams. 0 disables erosion. For most HD footage, 1–3 works well "
            "(default: 2)."
        ),
    )
    p.add_argument(
        "--blender-erode",
        type=int,
        default=2,
        dest="mask_erode_pixels",
        help="Pixels to erode from mask to avoid boundary artifacts (default: 2).",
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
        use_fp16=not args.no_fp16,
        output_codec=args.codec,
        output_quality=args.crf,
        copy_audio=not args.no_audio,
        blend_mode=args.blend_mode,
        mask_blur_kernel=args.mask_blur_kernel,
        mask_erode_pixels=args.mask_erode_pixels,
        log_level="DEBUG" if args.verbose else "INFO",
        log_file=args.log_file,
        enable_timers=args.timers,
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
    pipeline = CharacterReplacementPipeline(cfg)
    stats = pipeline.run()

    # Summary
    logger.info("=" * 60)
    logger.info("Done!  Output: %s", cfg.output_video)
    logger.info(
        "Processed %d frames in %.1fs (%.1f fps)",
        stats["frames_total"],
        stats["elapsed_s"],
        stats["fps"],
    )
    logger.info(
        "Frames with swaps: %d  |  Total faces swapped: %d",
        stats["frames_swapped"],
        stats["faces_swapped"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
