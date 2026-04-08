#!/usr/bin/env python3
"""
vicrop.cli – Video Face-Crop CLI

Extract face-cropped PNG frames from video files.

Quick examples
--------------
# Process all videos in a directory (face-crop every 30th frame)
vicrop --input ./videos --output-dir ./frames

# Process a single video file
vicrop --input ./video.mp4 --output-dir ./frames

# Faster sampling, no identity clustering
vicrop --input ./videos --output-dir ./frames --every-n 15 --no-classify

# Use CNN model for higher-accuracy face detection
vicrop --input ./videos --output-dir ./frames --detection-model cnn
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vicrop")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vicrop",
        description="Extract face-cropped PNG frames from video files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "A single video file (e.g. .mp4, .mov) or a directory of video\n"
            "files (searched recursively)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for cropped face images.",
    )

    parser.add_argument(
        "--every-n",
        type=int,
        default=30,
        help="Process every N-th frame from each video (default: 30).",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.4,
        help="Fractional margin to add around each detected face bbox (default: 0.4).",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Output square resolution in pixels (default: 1024).",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Do not cluster detected faces into person sub-folders.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.7,
        help="Face distance tolerance for identity clustering (default: 0.7).",
    )
    parser.add_argument(
        "--detection-model",
        choices=["hog", "cnn"],
        default="hog",
        help="face_recognition detection model (default: hog).",
    )
    parser.add_argument(
        "--ref-thresh",
        type=float,
        default=0.65,
        help=(
            "Minimum quality score (0–1) for a face crop to be selected as a\n"
            "reference portrait photo.  Qualifying crops are moved into a\n"
            "ref/ sub-folder inside each person folder.  Set to 0 to disable\n"
            "reference-photo analysis entirely (default: 0.65)."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--classified-max",
        type=int,
        default=10,
        help=(
            "Maximum number of reference images to load per identity when\n"
            "using --classified-path.  0 means no limit (default: 10)."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process videos whose output directory already contains frames.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, getattr(args, "log_level", "INFO")))
    logger.info("vicrop starting with args: %s", args)

    from vicrop.crop import SUPPORTED_VIDEO_EXTS, crop_folder, crop_video

    if args.input.is_file():
        if args.input.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            logger.error(
                "Unsupported file type '%s' for file: %s. Supported extensions: %s",
                args.input.suffix,
                args.input,
                ", ".join(sorted(SUPPORTED_VIDEO_EXTS)),
            )
            raise SystemExit(1)
        logger.info("vicrop: processing single video %s", args.input)
        video_stats = crop_video(
            args.input,
            args.output_dir,
            every_n=args.every_n,
            margin_ratio=args.margin_ratio,
            crop_size=args.crop_size,
            model=args.detection_model,
            classify=not args.no_classify,
            tolerance=args.tolerance,
            skip_existing=not args.no_skip_existing,
            ref_thresh=args.ref_thresh,
            classified_path=args.classified_path,
            classified_max=args.classified_max,
        )
        stats = {**video_stats, "videos_processed": 1}
    else:
        logger.info("vicrop: processing videos in %s", args.input)
        stats = crop_folder(
            args.input,
            args.output_dir,
            every_n=args.every_n,
            margin_ratio=args.margin_ratio,
            crop_size=args.crop_size,
            model=args.detection_model,
            classify=not args.no_classify,
            tolerance=args.tolerance,
            skip_existing=not args.no_skip_existing,
            ref_thresh=args.ref_thresh,
            classified_path=args.classified_path,
            classified_max=args.classified_max,
        )
    logger.info(
        "vicrop: %d videos processed, %d frames sampled, %d faces saved, "
        "%d persons identified, %d reference photos selected",
        stats["videos_processed"],
        stats["frames_processed"],
        stats["faces"],
        stats["persons"],
        stats["ref_photos"],
    )


if __name__ == "__main__":
    main()
