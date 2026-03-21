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
    model (or its Qwen3-Omni multimodal variant, or a Qwen3.5 model).
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
    """Run Qwen3-VL / Qwen3-Omni / Qwen3.5 vision-language pipeline."""
    import tempfile
    import shutil
    from videsc.model.loader import load_model_and_processor, load_omni_model_and_processor, load_qwen35_model_and_processor
    from videsc.pipeline.runner import run_batch, run_single_video

    print("args: ", str(args))
    is_batch = bool(args.videos or args.indir or getattr(args, "filelist", None))

    if is_batch:
        return run_batch(args)

    # Handle YouTube URL: download to a temp dir and set args.video
    tmp_dir = None
    if args.youtube_url:
        from videsc.describe import _download_youtube_video

        if not args.youtube_api_key:
            logger.error("videsc: --youtube-url requires --youtube-api-key")
            return 1

        # In VL + YouTube mode, honour --output-dir as fallback for --outdir
        # so the description isn't written into (and deleted with) the temp dir.
        if not args.outdir and args.output_dir:
            args.outdir = str(args.output_dir)

        tmp_dir = tempfile.mkdtemp(prefix="videsc_yt_")
        logger.info("Downloading YouTube video %s …", args.youtube_url)
        video_path = _download_youtube_video(args.youtube_url, __import__("pathlib").Path(tmp_dir))
        if video_path is None:
            logger.error("Failed to download YouTube video: %s", args.youtube_url)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return 1
        args.video = video_path

    try:
        if args.omni:
            model, processor = load_omni_model_and_processor(args)
        elif args.qwen35:
            model, processor = load_qwen35_model_and_processor(args)
        else:
            model, processor = load_model_and_processor(args)
        return run_single_video(args, model, processor)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
            capture=args.capture,
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
            capture=args.capture,
        )

    logger.info(
        "videsc: %d described, %d skipped",
        stats["described"],
        stats["skipped"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
