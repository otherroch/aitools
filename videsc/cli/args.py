import argparse
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate descriptive text for a video using Qwen3-VL."
    )
    # I/O
    p.add_argument("--video", help="Path to input video (mp4, etc.)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--videos", nargs="+", help="One or more glob patterns for videos")
    group.add_argument("--indir", help="Directory to scan recursively for videos")
    group.add_argument("--filelist", help="Text file with one video path per line")
    p.add_argument("--ext", nargs="+", default=[], help="File extensions for --indir (e.g. .mp4 .mov)")
    p.add_argument("--workers", type=int, default=2, help="How many videos to process in parallel")
    p.add_argument("--sleep", type=float, default=0.25, help="Polling interval (seconds) for job supervision")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved commands without running them")

    p.add_argument(
        "--batch-mode",
        choices=["threads", "subprocess"],
        default="threads",
        help="Batch mode: 'threads' shares one model in a single process; 'subprocess' spawns one process per video.",
    )

    p.add_argument(
        "--prompt",
        default=(
            "First, list all distinct characters, actions, and any important visuals "
            "in several long, comprehensive paragraphs."
        ),
        help="User prompt.",
    )

    p.add_argument(
        "--system",
        default="You are a helpful assistant that writes clear, concise video descriptions.",
        help="Optional system prompt.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for .txt results; defaults to <video_dir>/desc-<model>",
    )

    # Model / runtime
    p.add_argument("--omni", action="store_true", help="model is qwen3-omni")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name or directory; used under model_dir unless --model_hf/--model_full.",
    )
    p.add_argument(
        "--model_hf",
        action="store_true",
        help="Model is a HF model id (e.g. Qwen/Qwen3-VL-4B-Thinking) or local dir",
    )
    p.add_argument("--model_full", action="store_true", help="Model is a full path")
    p.add_argument(
        "--attn",
        choices=["flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help="Attention implementation.",
    )
    p.add_argument(
        "--quant",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Load quantized weights for lower VRAM and better speed.",
    )
    p.add_argument("--max-new-tokens", type=int, default=8192)

    # Video decoding backend + sampling
    p.add_argument(
        "--reader",
        choices=["auto", "torchvision", "decord", "torchcodec"],
        default="decord",
        help="Select the video reader. 'auto' keeps upstream default.",
    )
    p.add_argument("--spf", type=float, default=4.0, help="Sampling seconds per frame or sampling interval.")
    p.add_argument("--fps", type=float, default=1.0, help="Sampling FPS (approx).")
    p.add_argument(
        "--num-frames",
        type=int,
        default=256,
        help="Max frames to sample (the loader may adapt).",
    )
    p.add_argument("--stride", type=int, default=1, help="Take 1 of every N frames after initial sampling.")
    p.add_argument("--clip-start", type=float, default=0.0, help="Start time (sec).")
    p.add_argument("--clip-end", type=float, default=-1.0, help="End time (sec, -1 = full).")

    # Token/pixel limits (edge-style, converted internally)
    p.add_argument(
        "--min-pixels",
        type=int,
        default=128,
        help="Per-frame min tokens as an edge multiplier (×28×28 or ×32×32).",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=1280,
        help="Per-frame max tokens as an edge multiplier (×28×28 or ×32×32).",
    )
    p.add_argument(
        "--total-pixels",
        type=int,
        default=24000,
        help="Total tokens budget across video as an edge multiplier (×28×28 or ×32×32).",
    )

    # Audio transcription
    p.add_argument("--audio", action="store_true", help="Enable audio transcription.")
    p.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="HF ASR model id for audio transcription (e.g. 'openai/whisper-small'). Empty disables.",
    )
    p.add_argument(
        "--max-audio-seconds",
        type=float,
        default=0.0,
        help="If > 0, limit audio transcription to this many seconds from the start of the video.",
    )
    p.add_argument(
        "--no-save-transcript",
        action="store_true",
        help="Do not save the raw audio transcript to a separate *.transcript.txt file.",
    )

    # Misc
    p.add_argument("--no-think-trim", action="store_true", help="Do not trim '<think>...</think>' if present.")
    p.add_argument("--half_cpu", action="store_true", help="Use half of the CPU cores")
    p.add_argument("--dry", action="store_true", help="Dry run. Load model and processor but do not generate.")
    p.add_argument("--cont_prompt", action="store_true", help="Add a continue prompt to generate more.")
    p.add_argument("--no_meta", action="store_true", help="Do not use metadata from process_vision_info.")
    p.add_argument("--seed", type=int, default=4051888)
    p.add_argument("--rep_pen", type=float, default=1.05, help="repetition penalty. Default is 1.0")
    p.add_argument("--optimize", action="store_true", help="Compile model with torch.compile for faster inference.")

    return p.parse_args()
