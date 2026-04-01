#!/usr/bin/env python3
"""
portrait_prep.cli – Portrait Dataset Preparation CLI

Runs one or more dataset-preparation steps in sequence.

Steps (can be combined with --steps):
  convert   Convert HEIC/JPG images to PNG
  crop      Face-crop and classify persons into sub-folders
  caption   WD14 tagger captioning (generates .txt files)
  augment   Albumentations-based data augmentation
  cpcap     Copy captions to augmented images

Quick examples
--------------
# Full pipeline (all steps in order)
portrait-prep --input-dir ./raw --output-dir ./dataset --steps convert crop caption augment cpcap --prefix "ohwx man"

# Convert only
portrait-prep --input-dir ./raw_heic --output-dir ./png_out --steps convert

# Caption only (in-place .txt next to each image)
portrait-prep --input-dir ./cropped --steps caption --prefix "rocharch61"

# Augment then copy captions
portrait-prep --input-dir ./captioned --output-dir ./augmented --steps augment cpcap --per-image 8 --keep-originals
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("portrait_prep")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

ALL_STEPS = ["convert", "crop", "caption", "augment", "cpcap"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="portrait-prep",
        description="Portrait dataset preparation toolkit for diffusion model training.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ---- common ----
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Source directory containing original images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory for processed images.\n"
            "Required for: convert, crop, augment.\n"
            "Optional for caption (defaults to alongside source files).\n"
            "For cpcap use --source-dir and --aug-dir instead."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        metavar="STEP",
        help=(
            f"Which steps to run.  Choices: {', '.join(ALL_STEPS)}.\n"
            "Default: all steps in order."
        ),
    )

    # ---- convert ----
    convert_group = parser.add_argument_group("convert options")
    convert_group.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-convert even if the output PNG already exists.",
    )

    # ---- crop ----
    crop_group = parser.add_argument_group("crop options")
    crop_group.add_argument(
        "--margin-ratio",
        type=float,
        default=0.4,
        help="Fractional margin to add around each detected face bbox (default: 0.4).",
    )
    crop_group.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Output square resolution in pixels (default: 1024).",
    )
    crop_group.add_argument(
        "--no-classify",
        action="store_true",
        help="Do not cluster detected faces into person sub-folders.",
    )
    crop_group.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="Face distance tolerance for identity clustering (default: 0.6).",
    )
    crop_group.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="face_recognition detection model (default: hog).",
    )
    crop_group.add_argument(
        "--classified-path",
        type=Path,
        default=None,
        help=(
            "Path to a directory of pre-classified reference photos.\n"
            "Each sub-folder is treated as a known identity whose name is\n"
            "preserved in the output.  New faces that do not match any\n"
            "reference are placed in auto-generated person_NN folders."
        ),
    )

    # ---- caption ----
    caption_group = parser.add_argument_group("caption options")
    caption_group.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Token(s) prepended to every caption, e.g. 'ohwx man'.",
    )
    caption_group.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum WD14 tag confidence to include (default: 0.35).",
    )
    caption_group.add_argument(
        "--model-repo",
        type=str,
        default="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        help="HuggingFace repo ID for the WD14 ONNX model.",
    )
    caption_group.add_argument(
        "--include-ratings",
        action="store_true",
        help="Include WD14 rating tags (safe/questionable/explicit) in captions.",
    )
    caption_group.add_argument(
        "--caption-output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write .txt caption files.  Defaults to alongside each image.\n"
            "Useful when you want captions separate from source images."
        ),
    )

    # ---- augment ----
    aug_group = parser.add_argument_group("augment options")
    aug_group.add_argument(
        "--per-image",
        type=int,
        default=5,
        help="Number of augmented variants per source image (default: 5).",
    )
    aug_group.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[1024, 1024],
        help="Output image size as HEIGHT WIDTH (default: 1024 1024).",
    )
    aug_group.add_argument(
        "--keep-originals",
        action="store_true",
        help="Also copy a resized original (*_orig.png) into the augmented output.",
    )
    aug_group.add_argument(
        "--seed",
        type=int,
        default=4051888,
        help="Random seed for reproducible augmentation (default: 4051888).",
    )

    # ---- cpcap ----
    cpcap_group = parser.add_argument_group("cpcap options")
    cpcap_group.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Directory with original captions for the cpcap step.\n"
            "Defaults to --input-dir when run as part of a pipeline."
        ),
    )
    cpcap_group.add_argument(
        "--aug-dir",
        type=Path,
        default=None,
        help=(
            "Directory with augmented images for the cpcap step.\n"
            "Defaults to --output-dir when run as part of a pipeline."
        ),
    )
    cpcap_group.add_argument(
        "--caption-ext",
        type=str,
        default=".txt",
        help="Caption file extension (default: .txt).",
    )
    cpcap_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done for the cpcap step without writing files.",
    )

    # ---- logging ----
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def run_convert(args: argparse.Namespace) -> None:
    from portrait_prep.convert import convert_folder

    if args.output_dir is None:
        logger.error("--output-dir is required for the 'convert' step.")
        sys.exit(1)

    logger.info("=== STEP: convert ===")
    converted, skipped = convert_folder(
        args.input_dir,
        args.output_dir,
        skip_existing=not args.no_skip_existing,
    )
    logger.info("convert: %d converted, %d skipped", converted, skipped)


def run_crop(args: argparse.Namespace, input_dir: Path) -> None:
    from portrait_prep.crop import crop_folder

    if args.output_dir is None:
        logger.error("--output-dir is required for the 'crop' step.")
        sys.exit(1)

    logger.info("=== STEP: crop ===")
    stats = crop_folder(
        input_dir,
        args.output_dir,
        margin_ratio=args.margin_ratio,
        crop_size=args.crop_size,
        classify=not args.no_classify,
        tolerance=args.tolerance,
        model=args.detection_model,
        classified_path=args.classified_path,
    )
    logger.info(
        "crop: %d faces found in %d images, %d persons identified",
        stats["faces"],
        stats["images_processed"],
        stats["persons"],
    )


def run_caption(args: argparse.Namespace, input_dir: Path) -> None:
    from portrait_prep.caption import caption_folder

    logger.info("=== STEP: caption ===")
    stats = caption_folder(
        input_dir,
        output_dir=args.caption_output_dir,
        prefix=args.prefix,
        threshold=args.threshold,
        model_repo=args.model_repo,
        include_ratings=args.include_ratings,
        skip_existing=not args.no_skip_existing,
    )
    logger.info("caption: %d captioned, %d skipped", stats["captioned"], stats["skipped"])


def run_augment(args: argparse.Namespace, input_dir: Path) -> None:
    from portrait_prep.augment import augment_folder

    if args.output_dir is None:
        logger.error("--output-dir is required for the 'augment' step.")
        sys.exit(1)

    logger.info("=== STEP: augment ===")
    h, w = args.image_size
    stats = augment_folder(
        input_dir,
        args.output_dir,
        per_image=args.per_image,
        image_size=(h, w),
        keep_originals=args.keep_originals,
        seed=args.seed,
    )
    logger.info("augment: %d images generated, %d skipped", stats["augmented"], stats["skipped"])


def run_cpcap(args: argparse.Namespace, input_dir: Path) -> None:
    from portrait_prep.cpcap import copy_captions

    source = args.source_dir or input_dir
    aug = args.aug_dir or args.output_dir

    if aug is None:
        logger.error(
            "--aug-dir (or --output-dir) is required for the 'cpcap' step."
        )
        sys.exit(1)

    logger.info("=== STEP: cpcap ===")
    stats = copy_captions(
        source,
        aug,
        caption_ext=args.caption_ext,
        dry_run=args.dry_run,
    )
    logger.info(
        "cpcap: %d created, %d skipped (exists), %d skipped (no source)",
        stats["created"],
        stats["skipped_existing"],
        stats["skipped_no_source"],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, getattr(args, "log_level", "INFO")))
    logger.debug("portrait_prep starting with args: %s", args)

    # Maintain a running "current input" directory that flows through pipeline steps.
    # convert writes PNGs → crop reads them → caption captions them → augment augments →
    # cpcap copies captions.
    current_input = args.input_dir.resolve()

    # Preserve the original input for cpcap source when running a full pipeline.
    original_input = current_input

    steps = args.steps

    if "convert" in steps:
        run_convert(args)
        if args.output_dir:
            current_input = args.output_dir.resolve()

    if "crop" in steps:
        run_crop(args, current_input)
        if args.output_dir:
            current_input = args.output_dir.resolve()

    if "caption" in steps:
        run_caption(args, current_input)

    if "augment" in steps:
        run_augment(args, current_input)
        if args.output_dir:
            current_input = args.output_dir.resolve()

    if "cpcap" in steps:
        # When cpcap is combined with augment in a single invocation the source
        # captions are still in the pre-augment input directory.
        if "augment" in steps and args.source_dir is None:
            args.source_dir = original_input
        run_cpcap(args, original_input)

    logger.info("All steps complete.")


if __name__ == "__main__":
    main()
