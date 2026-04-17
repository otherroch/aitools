import argparse
from pathlib import Path
import platform
from typing import Optional


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="videsc",
        description=(
            "Generate AI-powered text descriptions for video files.\n\n"
            "By default uses the WD14 ONNX tagger for fast, tag-based captions.\n"
            "Pass --vl to use a Qwen3-VL vision-language model for rich,\n"
            "natural-language descriptions instead."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Mode selection ────────────────────────────────────────────────────────
    p.add_argument(
        "--vl",
        action="store_true",
        help="Use a vision-language model (Qwen3-VL, Qwen3-Omni, or Qwen3.5) instead of the WD14 tagger.",
    )

    # =========================================================================
    # WD14 mode arguments (active when --vl is NOT set)
    # =========================================================================
    wd14 = p.add_argument_group(
        "WD14 mode (default) — fast, tag-based descriptions",
        "Used when --vl is not set. Requires --input-dir or --youtube-url.",
    )
    wd14_input = wd14.add_mutually_exclusive_group()
    wd14_input.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing video files (searched recursively).",
    )
    wd14_input.add_argument(
        "--youtube-url",
        type=str,
        help="YouTube video URL to download and describe.",
    )
    wd14.add_argument(
        "--youtube-api-key",
        type=str,
        default=None,
        help="YouTube Data API v3 key (required when using --youtube-url).",
    )
    wd14.add_argument(
        "--save-video",
        type=Path,
        default=None,
        metavar="FILE",
        help="Download the YouTube video, save it to FILE (e.g. ./video.mp4), and exit without processing.",
    )
    wd14.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write .txt description files.\n"
            "Defaults to alongside each video file (--input-dir mode) or the\n"
            "current working directory (--youtube-url mode)."
        ),
    )
    wd14.add_argument(
        "--every-n",
        type=int,
        default=30,
        help="Extract one frame every N frames (default: 30).",
    )
    wd14.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum key frames to process per video (default: 10).",
    )
    wd14.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Token(s) prepended to every description, e.g. 'ohwx man'.",
    )
    wd14.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum WD14 tag confidence to include (default: 0.35).",
    )
    wd14.add_argument(
        "--model-repo",
        type=str,
        default="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        help="HuggingFace repo ID for the WD14 ONNX model.",
    )
    wd14.add_argument(
        "--include-ratings",
        action="store_true",
        help="Include WD14 rating tags (safe/questionable/explicit) in descriptions.",
    )
    wd14.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-describe videos whose .txt file already exists.",
    )

    # =========================================================================
    # VL mode arguments (active when --vl IS set)
    # =========================================================================
    vl = p.add_argument_group(
        "VL mode (--vl) — rich, natural-language descriptions",
        "Used when --vl is set. Requires --video, --videos, --indir, or --filelist.",
    )

    # Input / output
    vl.add_argument("--video", help="Path to input video (mp4, etc.)")
    vl_input = vl.add_mutually_exclusive_group()
    vl_input.add_argument("--videos", nargs="+", help="One or more glob patterns for videos")
    vl_input.add_argument("--indir", help="Directory to scan recursively for videos")
    vl_input.add_argument("--filelist", help="Text file with one video path per line")
    vl.add_argument("--ext", nargs="+", default=[], help="File extensions for --indir (e.g. .mp4 .mov)")
    vl.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for .txt results; defaults to <video_dir>/desc-<model>",
    )

    # Batch processing
    vl.add_argument("--workers", type=int, default=2, help="How many videos to process in parallel")
    vl.add_argument("--sleep", type=float, default=0.25, help="Polling interval (seconds) for job supervision")
    vl.add_argument("--dry-run", action="store_true", help="Print the resolved commands without running them")
    vl.add_argument(
        "--batch-mode",
        choices=["threads", "subprocess"],
        default="threads",
        help="Batch mode: 'threads' shares one model in a single process; 'subprocess' spawns one process per video.",
    )

    # Prompt
    vl.add_argument(
        "--prompt",
        default=(
            "First, list all distinct characters, actions, and any important visuals "
            "in several long, comprehensive paragraphs."
        ),
        help="User prompt.",
    )
    vl.add_argument(
        "--system",
        default="You are a helpful assistant that writes clear, concise video descriptions.",
        help="Optional system prompt.",
    )

    # Model / runtime
    model_group = vl.add_mutually_exclusive_group()
    model_group.add_argument("--omni", action="store_true", help="model is qwen3-omni")
    model_group.add_argument("--qwen35", action="store_true", help="model is Qwen3.5 (e.g. Qwen/Qwen3.5-4B)")
    model_group.add_argument("--gemma4", action="store_true", help="model is Gemma 4 (e.g. google/gemma-4-4eb-it)")
    vl.add_argument(
        "--gemma4-chunk-duration",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help=(
            "Maximum video duration (seconds) per chunk when using --gemma4.\n"
            "Gemma 4 can process at most 60 seconds at a time; longer videos are\n"
            "split into consecutive chunks and their descriptions are concatenated.\n"
            "(default: 30.0)"
        ),
    )
    vl.add_argument(
        "--gemma4-fps",
        type=float,
        default=1.0,
        metavar="FPS",
        help="Frames per second to sample from each Gemma 4 chunk (default: 1.0).",
    )
    vl.add_argument(
        "--consolidate",
        action="store_true",
        help=(
            "Enable the multi-stage segment consolidation pipeline (Gemma 4 only).\n"
            "When set, each chunk uses a structured segment prompt that extracts\n"
            "events, objects, actions and scene info as JSON.  After all chunks\n"
            "are described, segments are grouped into windows (see --window-size)\n"
            "and a final summary is produced with OVERVIEW, TIMELINE, ENTITIES,\n"
            "ACTIONS and THEMES sections.  The output file contains the final\n"
            "summary followed by raw per-segment descriptions.\n"
            "Only effective when --gemma4 is set and the video has more than one chunk."
        ),
    )
    vl.add_argument(
        "--consolidate-prompt",
        type=str,
        default=None,
        metavar="PROMPT",
        help=(
            "Custom prompt for the final consolidation step when --consolidate is\n"
            "set.  Overrides the built-in structured final-summary prompt.\n"
            "The per-segment / window descriptions are appended automatically."
        ),
    )
    vl.add_argument(
        "--segment-prompt",
        type=str,
        default=None,
        metavar="PROMPT",
        help=(
            "Custom prompt for per-segment analysis when --consolidate is set.\n"
            "Overrides the built-in structured segment prompt that requests\n"
            "JSON output (events, objects, actions, scene, summary).\n"
            "When provided, this prompt is used verbatim; include any desired\n"
            "segment timestamp or chunk-duration instructions yourself."
        ),
    )
    vl.add_argument(
        "--window-size",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Number of consecutive segments to group into a window for\n"
            "intermediate aggregation before the final summary.  Only used\n"
            "when --consolidate is set and chunk count > window-size.\n"
            "(default: 10)"
        ),
    )
    vl.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name or directory; used under model_dir unless --model_hf/--model_full.",
    )
    vl.add_argument(
        "--processor",
        default=None,
        help="Processor name or directory; used under model_dir unless --model_hf/--model_full. If not specified, defaults to the same value as --model.",
    )
    vl.add_argument(
        "--model_hf",
        action="store_true",
        help="Model is a HF model id (e.g. Qwen/Qwen3-VL-4B-Thinking) or local dir",
    )
    vl.add_argument("--model_full", action="store_true", help="Model is a full path")
    vl.add_argument(
        "--attn",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="sdpa",
        help="Attention implementation. 'flash_attention_2' may be unavailable or unsupported depending on your PyTorch/CUDA build; 'sdpa' is the efficient native PyTorch implementation; 'eager' is the unoptimized PyTorch implementation and may be very slow.",
    )
    vl.add_argument(
        "--quant",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Load quantized weights for lower VRAM and better speed.",
    )
    vl.add_argument("--max-new-tokens", type=int, default=8192)
    vl.add_argument("--optimize", action="store_true", help="Compile model with torch.compile for faster inference.")

    # Video decoding backend + sampling
    vl.add_argument(
        "--reader",
        choices=["auto", "torchvision", "decord", "torchcodec"],
        default="torchcodec" if platform.system() != "Windows" else "decord",
        help=(
            "Select the video reader.\n"
            "Previously, the default was 'auto', which tried to pick an available backend at runtime "
            "(typically preferring 'torchcodec' when installed, then falling back to 'decord' or "
            "'torchvision'). The default is now explicitly 'torchcodec' on non-Windows platforms to give "
            "more predictable performance and behavior when 'torchcodec' is available.\n"
            "On Windows, 'decord' remains the default because torchcodec with CUDA is not yet available. "
            "You can still pass '--reader auto' if you prefer the old automatic selection behavior."
        ),
    )
    vl.add_argument("--spf", type=float, default=4.0, help="Sampling seconds per frame or sampling interval.")
    vl.add_argument("--fps", type=float, default=1.0, help="Sampling FPS (approx).")
    vl.add_argument(
        "--num-frames",
        type=int,
        default=256,
        help="Max frames to sample (the loader may adapt).",
    )
    vl.add_argument("--stride", type=int, default=1, help="Take 1 of every N frames after initial sampling.")
    vl.add_argument("--clip-start", type=float, default=0.0, help="Start time (sec).")
    vl.add_argument("--clip-end", type=float, default=-1.0, help="End time (sec, -1 = full).")

    # Token/pixel limits (edge-style, converted internally)
    vl.add_argument(
        "--min-pixels",
        type=int,
        default=128,
        help="Per-frame min tokens as an edge multiplier (×28×28 or ×32×32).",
    )
    vl.add_argument(
        "--max-pixels",
        type=int,
        default=1280,
        help="Per-frame max tokens as an edge multiplier (×28×28 or ×32×32).",
    )
    vl.add_argument(
        "--total-pixels",
        type=int,
        default=24000,
        help="Total tokens budget across video as an edge multiplier (×28×28 or ×32×32).",
    )

    # Audio transcription
    vl.add_argument("--audio", action="store_true", help="Enable audio transcription.")
    vl.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="HF ASR model id for audio transcription (e.g. 'openai/whisper-small'). Empty disables.",
    )
    vl.add_argument(
        "--max-audio-seconds",
        type=float,
        default=0.0,
        help="If > 0, limit audio transcription to this many seconds from the start of the video.",
    )
    vl.add_argument(
        "--no-save-transcript",
        action="store_true",
        help="Do not save the raw audio transcript to a separate *.transcript.txt file.",
    )

    # Misc
    vl.add_argument("--no-think-trim", action="store_true", help="Do not trim '<think>...</think>' if present.")
    vl.add_argument("--half_cpu", action="store_true", help="Use half of the CPU cores")
    vl.add_argument("--dry", action="store_true", help="Dry run. Load model and processor but do not generate.")
    vl.add_argument("--cont_prompt", action="store_true", help="Add a continue prompt to generate more.")
    vl.add_argument("--no_meta", action="store_true", help="Do not use metadata from process_vision_info.")
    vl.add_argument("--seed", type=int, default=4051888)
    vl.add_argument("--rep_pen", type=float, default=1.05, help="repetition penalty. Default is 1.0")

    # ── Logging ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO).",
    )

    _VL_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

    args = p.parse_args(argv)

    # When --qwen35 is set and the user didn't explicitly change --model,
    # default to the Qwen3.5 HuggingFace model so the loader resolves correctly.
    if args.qwen35 and args.model == _VL_DEFAULT_MODEL:
        args.model = "Qwen/Qwen3.5-4B"
        args.model_hf = True

    # When --gemma4 is set and the user didn't explicitly change --model,
    # default to the Gemma 4 4B instruction-tuned model on HuggingFace.
    if args.gemma4 and args.model == _VL_DEFAULT_MODEL:
        args.model = "google/gemma-4-4eb-it"
        args.model_hf = True

    # Post-parse validation for Gemma 4 mode.
    if args.gemma4:
        if args.gemma4_chunk_duration <= 0:
            p.error("--gemma4-chunk-duration must be greater than 0 when using --gemma4")
        if args.gemma4_chunk_duration > 60:
            p.error("--gemma4-chunk-duration must be less than or equal to 60 when using --gemma4")


    # Post-parse validation for WD14 mode (--vl not set)
    if not args.vl:
        if args.input_dir is None and args.youtube_url is None:
            p.error("one of --input-dir or --youtube-url is required")
        if args.youtube_url is not None and args.youtube_api_key is None:
            p.error("--youtube-api-key is required when using --youtube-url")

    return args
