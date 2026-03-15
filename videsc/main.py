#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-VL Video → Text (CLI)
- Fast, minimal, GPU-friendly.
- Uses: Qwen3VLForConditionalGeneration, AutoProcessor / Qwen3VLVideoProcessor, qwen_vl_utils.
- Works with long videos by sampling frames (fps/num_frames/stride) and limiting tokenized pixels.
- Now also uses Whisper-style audio chunks with timestamps, aligned approximately to frame ranges.
"""

import sys

from videsc.cli.args import parse_args
from videsc.model.loader import load_model_and_processor, load_omni_model_and_processor
from videsc.pipeline.runner import run_batch, run_single_video


def main() -> int:
    args = parse_args()
    print("args: ", str(args))
    is_batch = bool(args.videos or args.indir or getattr(args, "filelist", None))

    if is_batch:
        return run_batch(args)

    if args.omni:
        model, processor = load_omni_model_and_processor(args)
    else:
        model, processor = load_model_and_processor(args)
    return run_single_video(args, model, processor)


if __name__ == "__main__":
    sys.exit(main())
