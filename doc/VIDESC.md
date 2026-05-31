# videsc

Generate AI-powered text descriptions for video files.

`videsc` supports two modes selectable via the `--vl` flag:

| | WD14 mode (default) | VL mode (`--vl`) | vLLM mode (`--vllm`) |
|---|---|---|---|
| **Model** | WD14 ONNX tagger | Qwen3-VL / Qwen3-Omni / Qwen3.5 / Gemma 4 (LLM) | Any vision-language model served by a remote vLLM server |
| **Output style** | Comma-separated tag list | Fluent natural-language paragraphs | Fluent natural-language paragraphs |
| **GPU required** | No (CPU-capable) | Strongly recommended (8 GB+ VRAM) | No (GPU runs on the remote vLLM server) |
| **Audio support** | No | Yes (Whisper ASR integration, Qwen models only) | No |
| **Custom prompts** | No | Yes (`--prompt` / `--system`) | Yes (`--prompt` / `--system`) |
| **Best for** | Fast tagging, LoRA caption files | Rich scene descriptions, storytelling | Multi-machine / shared-server inference, large models that don't fit locally |

Use the default WD14 mode when you need quick, reproducible tag-based captions that work on any hardware. Use `--vl` when you need detailed, human-readable descriptions — for example to create training captions that capture narrative context, character actions, or dialogue. Use `--vllm` when the model you want to use is already served by a remote vLLM instance — this offloads GPU inference to the server and allows the client machine to be CPU-only.

## Installation

```bash
# Standard (WD14 mode, CPU inference)
pip install -e .

# With Qwen3-VL support for --vl mode
pip install -e ".[vl]"

# With vLLM remote server support for --vllm mode
pip install -e ".[vllm]"
```

A CUDA-capable GPU is strongly recommended for VL mode. For lower VRAM use `--quant 4bit` or `--quant 8bit`.

vLLM mode does not require a local GPU — all inference runs on the remote vLLM server. The client only needs the `openai` and `requests` Python packages (installed by the `vllm` extra) plus OpenCV for frame extraction.

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
       --model google/gemma-4-31b-it --model_hf --quant 4bit

# Adjust chunk duration and frame sampling rate
videsc --vl --gemma4 --video ./clip.mp4 \
       --gemma4-chunk-duration 30 --gemma4-fps 2.0

# Batch-describe a directory of videos
videsc --vl --gemma4 --indir ./videos --ext .mp4 --outdir ./captions
```

### Segment consolidation (`--consolidate`)

When processing longer videos, Gemma 4 produces one description per chunk.  By
default these are simply joined with blank lines.  With `--consolidate`, the
pipeline switches to a **multi-stage consolidation** approach adapted from the
[design document](https://github.com/otherroch/aitools/issues/33):

#### Stage 1 — Structured segment analysis

Each chunk is analysed with a structured prompt that requests JSON output
containing `events`, `objects`, `actions`, `scene` and `summary` fields.
This produces low-entropy, consistently formatted output that makes downstream
merging reliable.  You can override the built-in prompt with `--segment-prompt`.

#### Stage 2 — Window aggregation

When the video has many chunks (more than `--window-size`, default 10),
consecutive segments are grouped into windows and each window is summarised
independently.  This removes redundancy, merges duplicate events and preserves
chronological order before the final stage.

#### Stage 3 — Final summary

All window summaries (or segment summaries for shorter videos) are fed to a
final-summary prompt that produces a structured result with:

1. **OVERVIEW** — what the video is about
2. **TIMELINE OF KEY EVENTS** — ordered bullet points
3. **MAIN ENTITIES** — people, objects, recurring elements
4. **IMPORTANT ACTIONS** — core actions driving the video
5. **THEMES OR PATTERNS** — only if clearly supported by the data

You can override the final-summary prompt with `--consolidate-prompt`.

The output file contains:
1. The **consolidated summary** (under a `=== Consolidated Summary ===` header).
2. The **raw per-segment descriptions** (under a `=== Per-Segment Descriptions ===` header).

Consolidation is only performed when the video is split into more than one chunk.

```bash
# Consolidate segment descriptions into a structured summary
videsc --vl --gemma4 --consolidate --video ./long_clip.mp4

# Use a custom final-summary prompt
videsc --vl --gemma4 --consolidate \
       --consolidate-prompt "Merge the following video segment descriptions into one paragraph." \
       --video ./long_clip.mp4

# Use a custom segment-level prompt
videsc --vl --gemma4 --consolidate \
       --segment-prompt "Describe the key events in this video segment." \
       --video ./long_clip.mp4

# Control window grouping size (e.g. group every 5 segments)
videsc --vl --gemma4 --consolidate --window-size 5 --video ./long_clip.mp4
```

## vLLM mode usage (`--vllm`)

vLLM mode offloads video description inference to a remote [vLLM](https://docs.vllm.ai/) server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint. Any vision-language model served by vLLM can be used (e.g. Qwen3-VL, LLaVA, InternVL). The client extracts frames locally with OpenCV, encodes them as base64 JPEG images, and sends them alongside a text prompt to the server.

### Prerequisites

1. **A running vLLM server** serving a vision-language model. For example:

   ```bash
   # Install vLLM on the server machine
   pip install vllm

   # Start vLLM with a vision-language model
   vllm serve Qwen/Qwen3-VL-8B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 32768
   ```

2. **Client dependencies** installed on the machine running `videsc`:

   ```bash
   pip install -e ".[vllm]"
   # Also requires OpenCV (pip install opencv-python) for frame extraction
   ```

### Connection & verification

On initialisation the client sends a `GET` request to the `/v1/models` endpoint of the vLLM server. If the server is unreachable or returns an error, a `RuntimeError` is raised immediately — this "fail fast" behaviour prevents long waits or silent failures when the server URL is misconfigured.

### Basic usage

```bash
# Describe a single video using a vLLM server on localhost:8000
videsc --vllm --video ./interview.mp4

# Specify the model name (must match what vLLM is serving)
videsc --vllm --video ./interview.mp4 --vllm-model Qwen/Qwen3-VL-8B-Instruct

# Connect to a remote server
videsc --vllm --video ./interview.mp4 \
       --vllm-host gpu-server.local --vllm-port 8000

# Use a full base URL (for reverse proxies, TLS, or non-standard paths)
videsc --vllm --video ./interview.mp4 \
       --vllm-base-url https://proxy.example.com/vllm/v1

# Use an API key (if the vLLM server requires authentication)
videsc --vllm --video ./interview.mp4 --vllm-api-key "my-secret-key"

# Adjust sampling temperature and top-p
videsc --vllm --video ./interview.mp4 \
       --vllm-temperature 0.5 --vllm-top-p 0.9

# Custom prompt and system message
videsc --vllm --video ./interview.mp4 \
       --prompt "Describe the scene, characters, and key actions in detail." \
       --system "You are an expert video analyst."

# Dry run — load config and extract frames but skip generation
videsc --vllm --video ./interview.mp4 --dry
```

### Video chunking

For long videos, you can split the video into chunks of a fixed duration. Each chunk is processed independently and the resulting descriptions are concatenated. This is useful when the model has a limited context window or when processing very long videos.

```bash
# Split into 30-second chunks
videsc --vllm --video ./long_lecture.mp4 --vllm-chunk-duration 30

# Combine chunking with segment consolidation (structured summary)
videsc --vllm --video ./long_lecture.mp4 \
       --vllm-chunk-duration 30 --consolidate
```

When `--consolidate` is enabled and the video produces more than one chunk, the pipeline generates a structured summary by sending all per-segment descriptions back to the vLLM server in a text-only consolidation call. The output file contains:

1. A **Consolidated Summary** section (overview, timeline, entities, actions, themes).
2. The **Per-Segment Descriptions** section with the raw output from each chunk.

You can customise the consolidation prompt with `--consolidate-prompt`.

### Frame sampling

Frames are extracted using OpenCV at a configurable FPS rate. You can control the number of frames sent to the model and the maximum image size:

```bash
# Sample at 2 frames per second instead of the default 1
videsc --vllm --video ./clip.mp4 --vllm-fps 2.0

# Reduce image size for faster upload / lower VRAM on the server
videsc --vllm --video ./clip.mp4 --vllm-max-image-size 640

# Process only a segment of the video
videsc --vllm --video ./clip.mp4 --clip-start 10.0 --clip-end 60.0
```

### Batch processing

vLLM mode supports processing multiple videos in parallel using threads:

```bash
# Describe all videos in a directory
videsc --vllm --indir ./videos --ext .mp4 .mov \
       --vllm-host gpu-server.local

# Use glob patterns
videsc --vllm --videos "./footage/**/*.mp4" --outdir ./captions

# Use a filelist
videsc --vllm --filelist ./my_videos.txt --outdir ./captions

# Process 4 videos in parallel
videsc --vllm --indir ./videos --workers 4

# Dry run — list which videos would be processed
videsc --vllm --indir ./videos --dry-run
```

### YouTube support

vLLM mode also supports YouTube URLs. The video is downloaded to a temporary directory, processed, and then cleaned up:

```bash
videsc --vllm --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
       --youtube-api-key "YOUR_API_KEY" --outdir ./captions
```

### Output

Output `.txt` files are placed in a `desc-vllm-<model>` subdirectory alongside each video by default (where `<model>` is the `--vllm-model` name with `/` replaced by `_`), or in the directory specified by `--outdir`.

### Architecture

```
CLI (args.py)  →  main.py._run_vllm()  →  vllm_runner.py  →  VLLMClient (model/vllm_client.py)
                                            ├─ extract_frames_as_pil() (cv2 → PIL)
                                            ├─ run_single_video_vllm() (chunking, consolidation)
                                            └─ run_batch_vllm() (ThreadPoolExecutor)
```

The `VLLMClient` wraps the OpenAI Python SDK, pointing it at the vLLM server's base URL. Frames are encoded as base64 JPEG images and sent as `image_url` content parts in the chat messages. The client supports configurable `temperature`, `top_p`, and `max_tokens` parameters.

Key differences from Qwen-VL mode:
- Frames are extracted as still images via OpenCV (no `qwen-vl-utils` needed).
- Videos longer than `--gemma4-chunk-duration` (default 30 s) are automatically
  split into chunks and their descriptions are joined with blank lines.
  Values up to 60 seconds are supported, but we observed that values over
  30 seconds increase runtime dramatically.
- Audio transcription (`--audio`) is not supported in Gemma 4 mode.
- `--attn` defaults to `sdpa`; `flash_attention_2` may also work.



Output `.txt` files are placed alongside each video in a `desc-<model>` subdirectory by default, or in the directory specified by `--outdir`.

## CLI reference

### Mode

| Flag | Description |
|------|-------------|
| `--vl` | Use Qwen3-VL vision-language model instead of the WD14 tagger |
| `--vllm` | Use a remote vLLM server for video description instead of loading a local model |

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
| `--gemma4-chunk-duration` | `30.0` | Max seconds per video chunk when using `--gemma4` |
| `--gemma4-fps` | `1.0` | Frames per second to sample from each Gemma 4 chunk |
| `--consolidate` | — | Enable multi-stage segment consolidation with structured prompts (Gemma 4 only) |
| `--consolidate-prompt` | *(built-in)* | Custom prompt for the final consolidation step |
| `--segment-prompt` | *(built-in)* | Custom prompt for per-segment structured analysis |
| `--window-size` | `10` | Number of segments per window for intermediate aggregation |
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

### vLLM mode arguments (`--vllm`)

#### Connection

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-host` | `localhost` | Hostname of the vLLM server |
| `--vllm-port` | `8000` | Port of the vLLM server |
| `--vllm-base-url` | — | Full base URL for the vLLM server (e.g. `http://host:8000/v1`). Overrides `--vllm-host` and `--vllm-port` when set |
| `--vllm-model` | `default` | Model name served by the vLLM server (as reported by `/v1/models`) |
| `--vllm-api-key` | `EMPTY` | API key for the vLLM server |

#### Generation

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-temperature` | `0.7` | Sampling temperature for vLLM generation |
| `--vllm-top-p` | `0.95` | Nucleus sampling (top-p) parameter |
| `--max-new-tokens` | `8192` | Maximum tokens to generate (shared with VL mode) |

#### Frame extraction

| Flag | Default | Description |
|------|---------|-------------|
| `--vllm-fps` | `1.0` | Frames per second to sample from the video |
| `--vllm-max-image-size` | `1280` | Maximum image edge size in pixels when encoding frames |
| `--vllm-chunk-duration` | `0.0` | Split video into chunks of this duration (seconds); `0` = process entire video at once |
| `--clip-start` | `0.0` | Start time in seconds (shared with VL mode) |
| `--clip-end` | `-1.0` | End time in seconds; `-1` = full video (shared with VL mode) |

#### Consolidation

When `--consolidate` is set and the video produces multiple chunks, the per-segment descriptions are sent to the vLLM server for a final structured summary. The consolidation flags (`--consolidate`, `--consolidate-prompt`, `--window-size`) behave identically to Gemma 4 mode — see the [Segment consolidation](#segment-consolidation---consolidate) section above.

#### Input / output / batch

vLLM mode reuses the same input, output, and batch flags as VL mode: `--video`, `--videos`, `--indir`, `--filelist`, `--ext`, `--outdir`, `--workers`, `--batch-mode`, `--sleep`, `--dry-run`. See the [VL mode arguments](#vl-mode-arguments---vl) section for details.
