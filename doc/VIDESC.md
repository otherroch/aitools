# videsc

Generate AI-powered text descriptions for video files.

`videsc` supports two modes selectable via the `--vl` flag:

| | WD14 mode (default) | VL mode (`--vl`) |
|---|---|---|
| **Model** | WD14 ONNX tagger | Qwen3-VL / Qwen3-Omni / Gemma 4 (LLM) |
| **Output style** | Comma-separated tag list | Fluent natural-language paragraphs |
| **GPU required** | No (CPU-capable) | Strongly recommended (8 GB+ VRAM) |
| **Audio support** | No | Yes (Whisper ASR integration, Qwen models only) |
| **Custom prompts** | No | Yes (`--prompt` / `--system`) |
| **Best for** | Fast tagging, LoRA caption files | Rich scene descriptions, storytelling |

Use the default WD14 mode when you need quick, reproducible tag-based captions that work on any hardware. Use `--vl` when you need detailed, human-readable descriptions — for example to create training captions that capture narrative context, character actions, or dialogue.

## Installation

```bash
# Standard (WD14 mode, CPU inference)
pip install -e .

# With Qwen3-VL support for --vl mode
pip install -e ".[vl]"
```

A CUDA-capable GPU is strongly recommended for VL mode. For lower VRAM use `--quant 4bit` or `--quant 8bit`.

## WD14 mode usage

Key frames are extracted from each video, tagged individually with the WD14 ONNX model, and the tags are aggregated across all frames (union of tags ranked by mean confidence). The result is written to a `.txt` file alongside the video (or in a specified output directory).

The first run downloads the WD14 ONNX model from HuggingFace (~350 MB) and caches it under `~/.cache/huggingface/`.

```bash
# Describe all videos in a directory (captions written alongside each video)
videsc --input-dir ./videos

# Write captions to a separate directory
videsc --input-dir ./videos --output-dir ./captions

# Add a custom prefix token and lower the confidence threshold
videsc --input-dir ./videos --prefix "ohwx man" --threshold 0.25

# Sample more frames for a more thorough description
videsc --input-dir ./videos --max-frames 20 --every-n 15

# Describe a YouTube video (YouTube Data API v3 key required)
videsc --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
       --youtube-api-key "YOUR_API_KEY" --output-dir ./captions
```

## VL mode usage (`--vl`)

```bash
# Describe a single video (output written to <video_dir>/desc-Qwen3-VL-8B-Instruct/)
videsc --vl --video ./interview.mp4

# Describe a single video with a custom prompt
videsc --vl --video ./interview.mp4 \
       --prompt "Describe the scene, characters, and key actions in detail."

# Describe all .mp4 and .mov files in a directory
videsc --vl --indir ./videos --ext .mp4 .mov

# Describe videos matching a glob pattern, writing results to a specific directory
videsc --vl --videos "./footage/**/*.mp4" --outdir ./captions

# Use a text file listing one video path per line
videsc --vl --filelist ./my_videos.txt --outdir ./captions

# Use 4-bit quantisation for lower VRAM consumption
videsc --vl --video ./clip.mp4 --quant 4bit

# Enable audio transcription (requires soundfile and a Whisper model)
videsc --vl --video ./clip.mp4 --audio

# Use a locally downloaded model
videsc --vl --video ./clip.mp4 --model /path/to/Qwen3-VL-8B-Instruct --model_full

# Use Qwen3-Omni for multimodal (audio + video) understanding
videsc --vl --video ./clip.mp4 --omni --model Qwen/Qwen3-Omni-8B --model_hf
```

## Gemma 4 mode (`--vl --gemma4`)

Gemma 4 is a multimodal model family from Google.  Unlike Qwen-VL models, Gemma 4
is limited to **60 seconds of video per inference call**; `videsc` handles this
automatically by splitting the video into consecutive chunks and concatenating the
descriptions.

```bash
# Describe a single video with the default Gemma 4 4B model
videsc --vl --gemma4 --video ./clip.mp4

# Use the 27B model with 4-bit quantisation (recommended for RTX 5090)
videsc --vl --gemma4 --video ./clip.mp4 \
       --model google/gemma-4-27b-it --model_hf --quant 4bit

# Adjust chunk duration and frame sampling rate
videsc --vl --gemma4 --video ./clip.mp4 \
       --gemma4-chunk-duration 45 --gemma4-fps 2.0

# Batch-describe a directory of videos
videsc --vl --gemma4 --indir ./videos --ext .mp4 --outdir ./captions
```

Key differences from Qwen-VL mode:
- Frames are extracted as still images via OpenCV (no `qwen-vl-utils` needed).
- Videos longer than `--gemma4-chunk-duration` (default 60 s) are automatically
  split into chunks and their descriptions are joined with blank lines.
- Audio transcription (`--audio`) is not supported in Gemma 4 mode.
- `--attn` defaults to `sdpa`; `flash_attention_2` may also work.



Output `.txt` files are placed alongside each video in a `desc-<model>` subdirectory by default, or in the directory specified by `--outdir`.

## CLI reference

### Mode

| Flag | Description |
|------|-------------|
| `--vl` | Use Qwen3-VL vision-language model instead of the WD14 tagger |

### WD14 mode arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(mutually exclusive with --youtube-url)* | Directory containing video files |
| `--youtube-url` | *(mutually exclusive with --input-dir)* | YouTube video URL to download and describe |
| `--youtube-api-key` | — | YouTube Data API v3 key (required with `--youtube-url`) |
| `--output-dir` | alongside videos / cwd | Where to write `.txt` description files |
| `--every-n` | `30` | Extract one frame every N frames |
| `--max-frames` | `10` | Maximum key frames to process per video |
| `--prefix` | `""` | Token(s) prepended to every description |
| `--threshold` | `0.35` | Minimum WD14 tag confidence to include |
| `--model-repo` | `SmilingWolf/wd-v1-4-convnextv2-tagger-v2` | HuggingFace model repo |
| `--include-ratings` | — | Include rating tags (safe/questionable/explicit) |
| `--no-skip-existing` | — | Re-describe videos whose `.txt` already exists |

### VL mode arguments (`--vl`)

#### Input / output

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Path to a single input video file |
| `--videos` | — | One or more glob patterns matching video files *(mutually exclusive with --indir / --filelist)* |
| `--indir` | — | Directory to scan recursively for video files *(mutually exclusive with --videos / --filelist)* |
| `--filelist` | — | Text file containing one video path per line *(mutually exclusive with --videos / --indir)* |
| `--ext` | *(all)* | File extensions to match when using `--indir` (e.g. `.mp4 .mov`) |
| `--outdir` | `<video_dir>/desc-<model>` | Directory for output `.txt` description files |

#### Batch processing

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | `2` | Number of videos to process in parallel |
| `--batch-mode` | `threads` | `threads` – one shared model process; `subprocess` – one process per video |
| `--sleep` | `0.25` | Polling interval in seconds for job supervision |
| `--dry-run` | — | Print resolved commands without running them |

#### Prompt

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `"First, list all distinct characters, actions, and any important visuals in several long, comprehensive paragraphs."` | User prompt sent to the model |
| `--system` | `"You are a helpful assistant that writes clear, concise video descriptions."` | System prompt |

#### Model / runtime

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | Model name (looked up under model_dir) or HF id / local path |
| `--model_hf` | — | Treat `--model` as a HuggingFace model id or local directory |
| `--model_full` | — | Treat `--model` as a full filesystem path |
| `--omni` | — | Load as Qwen3-Omni (multimodal audio + video model) |
| `--qwen35` | — | Load as Qwen3.5; defaults `--model` to `Qwen/Qwen3.5-4B` |
| `--gemma4` | — | Load as Gemma 4; defaults `--model` to `google/gemma-4-4b-it` |
| `--gemma4-chunk-duration` | `60.0` | Max seconds per video chunk when using `--gemma4` |
| `--gemma4-fps` | `1.0` | Frames per second to sample from each Gemma 4 chunk |
| `--attn` | `flash_attention_2` | Attention implementation: `flash_attention_2`, `sdpa`, or `eager` |
| `--quant` | `none` | Weight quantisation: `none`, `8bit`, or `4bit` |
| `--max-new-tokens` | `8192` | Maximum tokens to generate |
| `--optimize` | — | Compile the model with `torch.compile` for faster inference |

#### Video decoding & frame sampling

| Flag | Default | Description |
|------|---------|-------------|
| `--reader` | `decord` | Video reader backend: `auto`, `torchvision`, `decord`, or `torchcodec` |
| `--spf` | `4.0` | Sampling interval in seconds per frame |
| `--fps` | `1.0` | Approximate sampling frame rate |
| `--num-frames` | `256` | Maximum frames to sample (the loader may adapt) |
| `--stride` | `1` | Take 1 of every N frames after initial sampling |
| `--clip-start` | `0.0` | Start time in seconds |
| `--clip-end` | `-1.0` | End time in seconds (`-1` = full video) |

#### Pixel / token budget

These values are *edge multipliers*: the actual pixel count per frame is `value × 28 × 28` (e.g. `--max-pixels 1280` → 1 003 520 pixels per frame). Adjust them to trade off detail against VRAM usage and throughput.

| Flag | Default | Description |
|------|---------|-------------|
| `--min-pixels` | `128` | Per-frame minimum token budget as an edge multiplier (128 × 28² ≈ 100 K pixels) |
| `--max-pixels` | `1280` | Per-frame maximum token budget as an edge multiplier (1280 × 28² ≈ 1 M pixels) |
| `--total-pixels` | `24000` | Total token budget across the whole video as an edge multiplier (24000 × 28² ≈ 19 M pixels) |

#### Audio transcription

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | — | Enable Whisper-based audio transcription |
| `--asr-model` | `openai/whisper-large-v3-turbo` | HuggingFace ASR model id |
| `--max-audio-seconds` | `0.0` | Limit transcription to this many seconds from the start (`0` = no limit) |
| `--no-save-transcript` | — | Do not write a separate `*.transcript.txt` file |

#### Miscellaneous

| Flag | Default | Description |
|------|---------|-------------|
| `--no-think-trim` | — | Keep `<think>…</think>` reasoning tokens in the output |
| `--cont_prompt` | — | Append a continuation prompt to generate longer output |
| `--no_meta` | — | Skip metadata from `process_vision_info` |
| `--rep_pen` | `1.05` | Repetition penalty |
| `--seed` | `4051888` | Random seed |
| `--half_cpu` | — | Limit PyTorch to half the available CPU cores |
| `--dry` | — | Load the model but skip generation (for testing) |
