#!/usr/bin/env python3
"""
videsc.cli – Video Description CLI

Generate AI-powered text descriptions for video files using the WD14 tagger.

Quick examples
--------------
# Describe all videos in a directory (captions written alongside each video)
videsc --input-dir ./videos

# Write captions to a separate directory
videsc --input-dir ./videos --output-dir ./captions

# Add a custom prefix token and lower the confidence threshold
videsc --input-dir ./videos --prefix "ohwx man" --threshold 0.25

# Sample more frames for a more thorough description
videsc --input-dir ./videos --max-frames 20 --every-n 15

# Describe a YouTube video (API key required)
videsc --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \\
       --youtube-api-key "YOUR_API_KEY" --output-dir ./captions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("videsc")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="videsc",
        description=(
            "Generate AI-powered text descriptions for video files.\n"
            "Uses the WD14 tagger to caption key frames and aggregates tags "
            "across the video."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Mutually exclusive input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing video files (searched recursively).",
    )
    input_group.add_argument(
        "--youtube-url",
        type=str,
        help="YouTube video URL to download and describe.",
    )

    parser.add_argument(
        "--youtube-api-key",
        type=str,
        default=None,
        help="YouTube Data API v3 key (required when using --youtube-url).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write .txt description files.\n"
            "Defaults to alongside each video file (--input-dir mode) or the\n"
            "current working directory (--youtube-url mode)."
        ),
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=30,
        help="Extract one frame every N frames (default: 30).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum key frames to process per video (default: 10).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Token(s) prepended to every description, e.g. 'ohwx man'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum WD14 tag confidence to include (default: 0.35).",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        help="HuggingFace repo ID for the WD14 ONNX model.",
    )
    parser.add_argument(
        "--include-ratings",
        action="store_true",
        help="Include WD14 rating tags (safe/questionable/explicit) in descriptions.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-describe videos whose .txt file already exists.",
    )

    args = parser.parse_args(argv)

    if args.youtube_url and not args.youtube_api_key:
        parser.error("--youtube-url requires --youtube-api-key")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.youtube_url:
        from videsc.describe import describe_youtube

        logger.info("videsc: processing YouTube video %s", args.youtube_url)
        stats = describe_youtube(
            args.youtube_url,
            args.youtube_api_key,
            output_dir=args.output_dir,
            every_n=args.every_n,
            max_frames=args.max_frames,
            prefix=args.prefix,
            threshold=args.threshold,
            model_repo=args.model_repo,
            include_ratings=args.include_ratings,
            skip_existing=not args.no_skip_existing,
        )
    else:
        from videsc.describe import describe_folder

        logger.info("videsc: processing videos in %s", args.input_dir)
        stats = describe_folder(
            args.input_dir,
            output_dir=args.output_dir,
            every_n=args.every_n,
            max_frames=args.max_frames,
            prefix=args.prefix,
            threshold=args.threshold,
            model_repo=args.model_repo,
            include_ratings=args.include_ratings,
            skip_existing=not args.no_skip_existing,
        )

    logger.info(
        "videsc: %d described, %d skipped",
        stats["described"],
        stats["skipped"],
    )


if __name__ == "__main__":
    main()
