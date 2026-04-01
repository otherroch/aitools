# aitools

**AI dataset preparation toolkit for diffusion model LoRA training.**

`aitools` provides three command-line tools and Python APIs for preparing image and video datasets:

| Tool | Command | Description |
|------|---------|-------------|
| Portrait Prep | `portrait-prep` | End-to-end portrait image preparation (convert → crop → caption → augment) |
| Video Crop | `vicrop` | Extract face-cropped PNG frames from video files |
| Video Description | `videsc` | Generate text descriptions for video files — fast WD14 tag-based captions (default) or rich natural-language descriptions via Qwen3-VL (`--vl`) |

---

## Table of Contents

- [Installation](#installation)
- [Project structure](#project-structure)
- [portrait-prep](#portrait-prep)
- [vicrop](#vicrop)
- [videsc](#videsc)
- [Python API](#python-api)
- [Running tests](#running-tests)

---

## Installation

```bash
# Standard (CPU inference)
pip install -e .

# With GPU support for WD14 captioning (replaces onnxruntime with onnxruntime-gpu)
pip install -e ".[gpu]"

# With Qwen3-VL support for videsc --vl (adds PyTorch, transformers, and related dependencies)
pip install -e ".[vl]"

# With YouTube download support (adds yt-dlp)
pip install -e ".[youtube]"

# Including dev / test dependencies
pip install -e ".[dev]"
```

> **Note – HEIC support:** `pillow-heif` ships with pre-built wheels on PyPI for
> Windows, macOS, and Linux — no extra system libraries required in most cases.

> **Note – face_recognition:** Requires `cmake` and `dlib` to be present before
> installation. On Windows the easiest path is:
> ```bash
> pip install cmake
> pip install dlib
> pip install face_recognition
> ```

> **Note – GPU inference:** Install the `[gpu]` extra (see above) to use
> `onnxruntime-gpu` for significantly faster WD14 captioning on CUDA devices.

> **Note – YouTube support:** The `--youtube-url` flag requires `yt-dlp` and a
> YouTube Data API v3 key. Install `yt-dlp` with `pip install -e ".[youtube]"` or
> `pip install yt-dlp`.

> **Note – videsc VL mode:** The `--vl` flag requires PyTorch, the
> Transformers library, and related dependencies. Install them with
> `pip install -e ".[vl]"`. A CUDA-capable GPU with sufficient VRAM is strongly
> recommended (8 GB+ for the default 8B model; use `--quant 4bit` or `--quant 8bit`
> to reduce VRAM requirements).

---

## Project structure

```
aitools/
├── portrait_prep/
│   ├── __init__.py
│   ├── convert.py        # Step 1 – format conversion
│   ├── crop.py           # Step 2 – face crop + classification
│   ├── caption.py        # Step 3 – WD14 captioning
│   ├── augment.py        # Step 4 – data augmentation
│   ├── cpcap.py          # Step 5 – caption propagation
│   └── cli.py            # portrait-prep entry point
├── vicrop/
│   ├── __init__.py
│   ├── crop.py           # Video face-crop logic
│   └── cli.py            # vicrop entry point
├── videsc/
│   ├── __init__.py
│   ├── describe.py       # WD14-based video description logic
│   ├── wd_cli.py         # Legacy WD14 CLI module (superseded by main.py)
│   ├── main.py           # Unified videsc entry point (WD14 + VL modes)
│   ├── config.py         # Model directory configuration
│   ├── cli/
│   │   └── args.py       # Unified CLI argument parsing
│   ├── audio/
│   │   └── transcription.py  # Whisper-based audio transcription
│   ├── model/
│   │   └── loader.py     # Qwen3-VL / Qwen3-Omni model loading
│   ├── pipeline/
│   │   └── runner.py     # Batch & single-video runner for VL mode
│   ├── video/
│   │   ├── info.py       # Video metadata extraction
│   │   ├── messages.py   # LLM message construction
│   │   └── sampling.py   # Frame sampling logic
│   └── utils/
│       └── helpers.py    # Shared utility functions
├── tests/
│   ├── test_convert.py
│   ├── test_crop.py
│   ├── test_caption.py
│   ├── test_augment.py
│   ├── test_cpcap.py
│   ├── test_vicrop.py
│   ├── test_videsc.py
│   └── test_videsc_main.py
├── main.py               # Thin shim for portrait-prep
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## portrait-prep

Portrait dataset preparation toolkit for diffusion model LoRA training.

### Features

| Step | Description |
|------|-------------|
| `convert` | Convert HEIC / JPG (and other formats) to PNG |
| `crop` | Face-detect, crop, and classify persons into sub-folders |
| `caption` | WD14 tagger auto-captioning with a custom token prefix |
| `augment` | Identity-preserving Albumentations augmentations |
| `cpcap` | Replicate captions from originals to augmented images |

Steps can be run individually or chained as a full pipeline in a single command.

### Usage

**Full pipeline**

```bash
portrait-prep \
  --input-dir ./raw_photos \
  --output-dir ./dataset \
  --steps convert crop caption augment cpcap \
  --prefix "ohwx man" \
  --per-image 8 \
  --keep-originals
```

**Step 1 – Convert HEIC/JPG to PNG**
```bash
portrait-prep \
  --input-dir ./raw_heic \
  --output-dir ./png_out \
  --steps convert
```

**Step 2 – Face-crop and classify persons**
```bash
portrait-prep \
  --input-dir ./png_out \
  --output-dir ./cropped \
  --steps crop \
  --margin-ratio 0.4 \
  --crop-size 1024
```

Each detected person is placed in a `person_NN` sub-folder (use `--no-classify` to skip clustering).

**Step 3 – WD14 captioning**
```bash
portrait-prep \
  --input-dir ./cropped \
  --steps caption \
  --prefix "rocharch61" \
  --threshold 0.35
```

Captions are written as `.txt` files alongside each image (or in `--caption-output-dir`).
The first run downloads the WD14 ONNX model from HuggingFace (~350 MB) and caches it.

**Step 4 – Augment images**
```bash
portrait-prep \
  --input-dir ./cropped \
  --output-dir ./augmented \
  --steps augment \
  --per-image 10 \
  --image-size 1024 1024 \
  --keep-originals \
  --seed 4051888
```

**Step 5 – Copy captions to augmented images**
```bash
portrait-prep \
  --source-dir ./cropped \
  --aug-dir ./augmented \
  --steps cpcap
```

Or combined with augment (source captions are automatically inferred):
```bash
portrait-prep \
  --input-dir ./cropped \
  --output-dir ./augmented \
  --steps augment cpcap
```

### CLI reference

#### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required)* | Source directory |
| `--output-dir` | — | Destination directory (required for convert, crop, augment) |
| `--steps` | all | Steps to run: `convert crop caption augment cpcap` |
| `--no-skip-existing` | — | Re-process files whose output already exists |

#### crop options

| Flag | Default | Description |
|------|---------|-------------|
| `--margin-ratio` | `0.4` | Fractional padding around each face bbox |
| `--crop-size` | `1024` | Output square resolution (pixels) |
| `--no-classify` | — | Disable identity clustering |
| `--tolerance` | `0.6` | Face-distance threshold for clustering |
| `--detection-model` | `hog` | `hog` (fast) or `cnn` (accurate) |

#### caption options

| Flag | Default | Description |
|------|---------|-------------|
| `--prefix` | `""` | Token prepended to every caption |
| `--threshold` | `0.35` | Minimum WD14 confidence to include a tag |
| `--model-repo` | `SmilingWolf/wd-v1-4-convnextv2-tagger-v2` | HuggingFace model repo |
| `--include-ratings` | — | Include rating tags (safe/questionable/explicit) |
| `--caption-output-dir` | alongside images | Separate dir for `.txt` files |

#### augment options

| Flag | Default | Description |
|------|---------|-------------|
| `--per-image` | `5` | Augmented variants per source image |
| `--image-size` | `1024 1024` | Output `HEIGHT WIDTH` |
| `--keep-originals` | — | Also copy a resized `*_orig.png` |
| `--seed` | `4051888` | Random seed |

#### cpcap options

| Flag | Default | Description |
|------|---------|-------------|
| `--source-dir` | `--input-dir` | Directory containing original captions |
| `--aug-dir` | `--output-dir` | Directory with augmented images |
| `--caption-ext` | `.txt` | Caption file extension |
| `--dry-run` | — | Report without writing files |

### Typical end-to-end workflow

```
raw HEIC/JPG photos
       │
       ▼ convert
  PNG images
       │
       ▼ crop
  person_01/  person_02/  …
       │
       ▼ caption  (generates .txt alongside each .png)
  captioned PNGs
       │
       ▼ augment
  augmented PNGs (×N per original)
       │
       ▼ cpcap
  each augmented image has a matching .txt caption
       │
       ▼
  ready for musubi-tuner / sd-scripts LoRA training
```

---

## vicrop

Extract face-cropped PNG frames from video files.

Reads video files using OpenCV, samples frames at a configurable interval, detects faces in each frame, crops them with padding, and saves them as PNG files. Optionally clusters face crops by identity into `person_NN` sub-folders (same greedy nearest-neighbour algorithm as `portrait-prep crop`). Optionally scores each crop for reference-photo quality and writes a `reflist.txt` per identity.

### Usage

```bash
# Process all videos in a directory (face-crop every 30th frame)
vicrop --input-dir ./videos --output-dir ./frames

# Faster sampling, no identity clustering
vicrop --input-dir ./videos --output-dir ./frames --every-n 15 --no-classify

# Higher-accuracy face detection
vicrop --input-dir ./videos --output-dir ./frames --detection-model cnn

# Tighter margin around the face (less background context)
vicrop --input-dir ./videos --output-dir ./frames --margin-ratio 0.2

# Select reference photos scoring 0.75 or higher (more permissive than default)
vicrop --input-dir ./videos --output-dir ./frames --ref-thresh 0.75

# Disable reference-photo selection entirely
vicrop --input-dir ./videos --output-dir ./frames --ref-thresh 0
```

Output is organised as:
```
frames/
└── <video_stem>/
    ├── person_01/
    │   ├── frame000000_face1.png
    │   ├── frame000030_face1.png
    │   └── reflist.txt          ← reference photos for this person (if any pass --ref-thresh)
    └── person_02/
        ├── frame000060_face1.png
        └── reflist.txt
```

### Reference photo selection

Training a portrait LoRA requires a small set of high-quality *reference photos* — images where the subject is looking directly at the camera, eyes fully open, face well-lit, sharp, and occupying a significant area of the frame. `vicrop` can automatically identify those images from all the face crops it produces.

When `--ref-thresh` is greater than zero (the default is `0.8`), every saved face crop is scored on five criteria:

| Criterion | Weight | What is measured |
|-----------|--------|-----------------|
| Frontal pose | 30 % | Landmark symmetry — how evenly the nose sits between both eyes (yaw), and how far down the face the nose tip sits (pitch). Scores drop as the face turns away from the camera. |
| Eyes open | 20 % | Eye Aspect Ratio (EAR) from six landmark points per eye. Closed or partially closed eyes score lower. |
| Sharpness | 20 % | Laplacian variance of the face-crop region. Blurry or motion-smeared crops score lower. |
| Face fill | 15 % | Ratio of face bounding-box area to total frame area. A face occupying ≥ 15 % of the frame earns a full score; smaller faces score proportionally lower. |
| Lighting | 15 % | Luminance mean and contrast. Very dark (< 40/255) or severely overexposed (> 220/255) crops score lower; well-exposed crops with natural contrast score higher. |

Crops whose composite score meets or exceeds `--ref-thresh` have their filename recorded. At the end of processing, each `person_NN/` folder that contains at least one qualifying crop receives a `reflist.txt` listing those filenames (one per line, alphabetically sorted).

**Choosing a threshold:**

| `--ref-thresh` | Effect |
|----------------|--------|
| `1.0` | Only near-perfect frontal shots with fully open eyes and excellent exposure |
| `0.8` *(default)* | Good frontal poses; minor angle deviations and slight blur accepted |
| `0.6` | More permissive; useful when footage quality is variable |
| `0` | Disables the analysis entirely — no scoring, no `reflist.txt` |

Lower values cast a wider net and produce a larger reference set; higher values are more selective. The goal is to feed the LoRA trainer images that anchor the subject's likeness without injecting off-angle or blurry samples that can reduce identity coherence.

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required)* | Directory containing video files |
| `--output-dir` | *(required)* | Destination directory for cropped frames |
| `--every-n` | `30` | Process every N-th frame |
| `--margin-ratio` | `0.4` | Fractional padding around each detected face bbox (see below) |
| `--crop-size` | `1024` | Output square resolution (pixels) |
| `--no-classify` | — | Disable identity clustering |
| `--tolerance` | `0.6` | Face-distance threshold for clustering (see below) |
| `--detection-model` | `hog` | `hog` (fast) or `cnn` (more accurate) |
| `--ref-thresh` | `0.8` | Minimum quality score (0–1) for reference-photo selection; `0` disables |
| `--no-skip-existing` | — | Re-process videos whose output already contains frames |

#### `--margin-ratio` — controlling how much context surrounds the face

`--margin-ratio` is a multiplier applied to the height and width of the raw face bounding box returned by the detector. The padded region is then cropped out of the frame and resized to `--crop-size` × `--crop-size`.

| Value | Effect |
|-------|--------|
| `0.1–0.2` | Tight crop — face fills most of the image, very little neck or hair visible. Useful if you want maximum facial detail at a given resolution. |
| `0.4` *(default)* | Balanced — includes forehead, chin, ears, and a sliver of neck/shoulders. Recommended for most portrait LoRA use cases. |
| `0.6–0.8` | Wide crop — substantial background and shoulders included. Helpful for full-head or upper-body training examples, but effective facial resolution is lower. |

> **Clipping:** margins are clamped to the frame edges, so very large values on faces near the border simply include as much of the frame as available rather than creating black padding.

#### `--tolerance` — controlling how strictly faces are grouped into identities

After all face crops from a video are collected, `vicrop` groups them by identity using a greedy nearest-neighbour algorithm on 128-dimensional face encodings. `--tolerance` is the maximum *face distance* (Euclidean distance in encoding space) allowed before two crops are considered different people. Lower distance → higher similarity must be met to join an existing cluster.

| Value | Effect |
|-------|--------|
| `0.4–0.5` | Strict — only very similar encodings map to the same person. Reduces cross-person contamination in a scene with multiple look-alike subjects, but can split a single person across two `person_NN` folders when lighting or angle changes significantly. |
| `0.6` *(default)* | Balanced — works well for most footage with a dominant subject. |
| `0.7–0.8` | Permissive — merges more crops into each cluster. Good for footage where the subject's appearance varies widely (different lighting, head angles, partial occlusion), but risks merging distinct people who look somewhat similar. |

> **Tip:** if you find one person split across `person_01` and `person_03`, increase tolerance slightly. If two distinct people are being merged into the same folder, decrease it.

---

## videsc

Generate AI-powered text descriptions for video files.

`videsc` supports two modes selectable via the `--vl` flag:

| | WD14 mode (default) | VL mode (`--vl`) |
|---|---|---|
| **Model** | WD14 ONNX tagger | Qwen3-VL / Qwen3-Omni (LLM) |
| **Output style** | Comma-separated tag list | Fluent natural-language paragraphs |
| **GPU required** | No (CPU-capable) | Strongly recommended (8 GB+ VRAM) |
| **Audio support** | No | Yes (Whisper ASR integration) |
| **Custom prompts** | No | Yes (`--prompt` / `--system`) |
| **Best for** | Fast tagging, LoRA caption files | Rich scene descriptions, storytelling |

Use the default WD14 mode when you need quick, reproducible tag-based captions that work on any hardware. Use `--vl` when you need detailed, human-readable descriptions — for example to create training captions that capture narrative context, character actions, or dialogue.

### Installation

```bash
# Standard (WD14 mode, CPU inference)
pip install -e .

# With Qwen3-VL support for --vl mode
pip install -e ".[vl]"
```

A CUDA-capable GPU is strongly recommended for VL mode. For lower VRAM use `--quant 4bit` or `--quant 8bit`.

### WD14 mode usage

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

### VL mode usage (`--vl`)

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

Output `.txt` files are placed alongside each video in a `desc-<model>` subdirectory by default, or in the directory specified by `--outdir`.

### CLI reference

#### Mode

| Flag | Description |
|------|-------------|
| `--vl` | Use Qwen3-VL vision-language model instead of the WD14 tagger |

#### WD14 mode arguments

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

#### VL mode arguments (`--vl`)

##### Input / output

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Path to a single input video file |
| `--videos` | — | One or more glob patterns matching video files *(mutually exclusive with --indir / --filelist)* |
| `--indir` | — | Directory to scan recursively for video files *(mutually exclusive with --videos / --filelist)* |
| `--filelist` | — | Text file containing one video path per line *(mutually exclusive with --videos / --indir)* |
| `--ext` | *(all)* | File extensions to match when using `--indir` (e.g. `.mp4 .mov`) |
| `--outdir` | `<video_dir>/desc-<model>` | Directory for output `.txt` description files |

##### Batch processing

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | `2` | Number of videos to process in parallel |
| `--batch-mode` | `threads` | `threads` – one shared model process; `subprocess` – one process per video |
| `--sleep` | `0.25` | Polling interval in seconds for job supervision |
| `--dry-run` | — | Print resolved commands without running them |

##### Prompt

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `"First, list all distinct characters, actions, and any important visuals in several long, comprehensive paragraphs."` | User prompt sent to the model |
| `--system` | `"You are a helpful assistant that writes clear, concise video descriptions."` | System prompt |

##### Model / runtime

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | Model name (looked up under model_dir) or HF id / local path |
| `--model_hf` | — | Treat `--model` as a HuggingFace model id or local directory |
| `--model_full` | — | Treat `--model` as a full filesystem path |
| `--omni` | — | Load as Qwen3-Omni (multimodal audio + video model) |
| `--attn` | `flash_attention_2` | Attention implementation: `flash_attention_2`, `sdpa`, or `eager` |
| `--quant` | `none` | Weight quantisation: `none`, `8bit`, or `4bit` |
| `--max-new-tokens` | `8192` | Maximum tokens to generate |
| `--optimize` | — | Compile the model with `torch.compile` for faster inference |

##### Video decoding & frame sampling

| Flag | Default | Description |
|------|---------|-------------|
| `--reader` | `decord` | Video reader backend: `auto`, `torchvision`, `decord`, or `torchcodec` |
| `--spf` | `4.0` | Sampling interval in seconds per frame |
| `--fps` | `1.0` | Approximate sampling frame rate |
| `--num-frames` | `256` | Maximum frames to sample (the loader may adapt) |
| `--stride` | `1` | Take 1 of every N frames after initial sampling |
| `--clip-start` | `0.0` | Start time in seconds |
| `--clip-end` | `-1.0` | End time in seconds (`-1` = full video) |

##### Pixel / token budget

These values are *edge multipliers*: the actual pixel count per frame is `value × 28 × 28` (e.g. `--max-pixels 1280` → 1 003 520 pixels per frame). Adjust them to trade off detail against VRAM usage and throughput.

| Flag | Default | Description |
|------|---------|-------------|
| `--min-pixels` | `128` | Per-frame minimum token budget as an edge multiplier (128 × 28² ≈ 100 K pixels) |
| `--max-pixels` | `1280` | Per-frame maximum token budget as an edge multiplier (1280 × 28² ≈ 1 M pixels) |
| `--total-pixels` | `24000` | Total token budget across the whole video as an edge multiplier (24000 × 28² ≈ 19 M pixels) |

##### Audio transcription

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | — | Enable Whisper-based audio transcription |
| `--asr-model` | `openai/whisper-large-v3-turbo` | HuggingFace ASR model id |
| `--max-audio-seconds` | `0.0` | Limit transcription to this many seconds from the start (`0` = no limit) |
| `--no-save-transcript` | — | Do not write a separate `*.transcript.txt` file |

##### Miscellaneous

| Flag | Default | Description |
|------|---------|-------------|
| `--no-think-trim` | — | Keep `<think>…</think>` reasoning tokens in the output |
| `--cont_prompt` | — | Append a continuation prompt to generate longer output |
| `--no_meta` | — | Skip metadata from `process_vision_info` |
| `--rep_pen` | `1.05` | Repetition penalty |
| `--seed` | `4051888` | Random seed |
| `--half_cpu` | — | Limit PyTorch to half the available CPU cores |
| `--dry` | — | Load the model but skip generation (for testing) |

---

## Python API

### portrait-prep

```python
from pathlib import Path
from portrait_prep.convert import convert_folder
from portrait_prep.crop import crop_folder
from portrait_prep.caption import caption_folder
from portrait_prep.augment import augment_folder
from portrait_prep.cpcap import copy_captions

# 1. Convert
convert_folder(Path("raw"), Path("png_out"))

# 2. Crop
crop_folder(Path("png_out"), Path("cropped"), classify=True)

# 3. Caption
caption_folder(Path("cropped"), prefix="ohwx man", threshold=0.35)

# 4. Augment
augment_folder(Path("cropped"), Path("augmented"), per_image=8, keep_originals=True)

# 5. Copy captions
copy_captions(Path("cropped"), Path("augmented"))
```

### vicrop

```python
from pathlib import Path
from vicrop.crop import crop_folder, crop_video

# Process a single video
stats = crop_video(
    Path("interview.mp4"),
    Path("frames"),
    every_n=30,
    crop_size=1024,
    classify=True,
)
print(stats)  # {'frames_processed': 20, 'faces': 5, 'persons': 1}

# Process all videos in a directory
stats = crop_folder(Path("videos"), Path("frames"))
```

### videsc

```python
from pathlib import Path
from videsc.describe import describe_folder, describe_video, describe_youtube

# Describe a single video
stats = describe_video(
    Path("interview.mp4"),
    prefix="ohwx man",
    threshold=0.35,
)
print(stats)  # {'described': 1, 'skipped': 0}

# Describe all videos in a directory
stats = describe_folder(
    Path("videos"),
    output_dir=Path("captions"),
    prefix="ohwx man",
)

# Describe a YouTube video (YouTube Data API v3 key required; yt-dlp must be installed)
stats = describe_youtube(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    youtube_api_key="YOUR_API_KEY",
    output_dir=Path("captions"),
    prefix="ohwx man",
)
print(stats)  # {'described': 1, 'skipped': 0}
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The `[dev]` extra pulls in `pytest` and `pytest-cov`. Test paths and verbosity
are configured in `pyproject.toml` so no extra flags are needed.

Heavy dependencies (`onnxruntime`, `face_recognition`) are mocked in the test
suite so the full suite runs without a GPU or dlib installation.

To run tests for a specific tool:

```bash
# portrait-prep
pytest tests/test_convert.py
pytest tests/test_cpcap.py
pytest tests/test_augment.py

# vicrop
pytest tests/test_vicrop.py

# videsc (WD14 and VL modes)
pytest tests/test_videsc.py
pytest tests/test_videsc_main.py
```

Generate a coverage report:

```bash
pytest --cov=portrait_prep --cov=vicrop --cov=videsc --cov-report=term-missing
```
