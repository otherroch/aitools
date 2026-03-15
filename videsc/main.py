#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
videsc – Unified Video Description CLI

Supports two description modes:

  WD14 mode (default):
    Fast, tag-based descriptions using the WD14 ONNX tagger.
    Requires --input-dir or --youtube-url.

  VL mode (--vl):
    Rich, natural-language descriptions using a Qwen3-VL vision-language
    model (or its Qwen3-Omni multimodal variant).
    Requires --video, --videos, --indir, or --filelist.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("videsc")


def main(argv: list[str] | None = None) -> int:
    from videsc.cli.args import parse_args

    args = parse_args(argv)

    if args.vl:
        return _run_vl(args)
    return _run_wd14(args)


def _run_vl(args) -> int:
    """Run Qwen3-VL vision-language pipeline."""
    from videsc.model.loader import load_model_and_processor, load_omni_model_and_processor
    from videsc.pipeline.runner import run_batch, run_single_video

    print("args: ", str(args))
    is_batch = bool(args.videos or args.indir or getattr(args, "filelist", None))

    if is_batch:
        return run_batch(args)

    if args.omni:
        model, processor = load_omni_model_and_processor(args)
    else:
        model, processor = load_model_and_processor(args)
    return run_single_video(args, model, processor)


def _run_wd14(args) -> int:
    """Run WD14 tagger pipeline."""
    if not args.input_dir and not args.youtube_url:
        logger.error(
            "videsc: one of --input-dir or --youtube-url is required "
            "(or pass --vl to use the Qwen3-VL mode)"
        )
        return 1

    if args.youtube_url and not args.youtube_api_key:
        logger.error("videsc: --youtube-url requires --youtube-api-key")
        return 1

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
