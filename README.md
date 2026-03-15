# aitools

**AI dataset preparation toolkit for diffusion model LoRA training.**

`aitools` provides three command-line tools and Python APIs for preparing image and video datasets:

| Tool | Command | Description |
|------|---------|-------------|
| Portrait Prep | `portrait-prep` | End-to-end portrait image preparation (convert → crop → caption → augment) |
| Video Crop | `vicrop` | Extract face-cropped PNG frames from video files |
| Video Description | `videsc` | Generate AI-powered text descriptions for video files |

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
│   ├── describe.py       # Video description logic
│   └── cli.py            # videsc entry point
├── tests/
│   ├── test_convert.py
│   ├── test_crop.py
│   ├── test_caption.py
│   ├── test_augment.py
│   ├── test_cpcap.py
│   ├── test_vicrop.py
│   └── test_videsc.py
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

Reads video files using OpenCV, samples frames at a configurable interval, detects faces in each frame, crops them with padding, and saves them as PNG files. Optionally clusters face crops by identity into `person_NN` sub-folders (same greedy nearest-neighbour algorithm as `portrait-prep crop`).

### Usage

```bash
# Process all videos in a directory (face-crop every 30th frame)
vicrop --input-dir ./videos --output-dir ./frames

# Faster sampling, no identity clustering
vicrop --input-dir ./videos --output-dir ./frames --every-n 15 --no-classify

# Higher-accuracy face detection
vicrop --input-dir ./videos --output-dir ./frames --detection-model cnn
```

Output is organised as:
```
frames/
└── <video_stem>/
    ├── person_01/
    │   ├── frame000000_face1.png
    │   └── frame000030_face1.png
    └── person_02/
        └── frame000060_face1.png
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required)* | Directory containing video files |
| `--output-dir` | *(required)* | Destination directory for cropped frames |
| `--every-n` | `30` | Process every N-th frame |
| `--margin-ratio` | `0.4` | Fractional padding around each detected face bbox |
| `--crop-size` | `1024` | Output square resolution (pixels) |
| `--no-classify` | — | Disable identity clustering |
| `--tolerance` | `0.6` | Face-distance threshold for clustering |
| `--detection-model` | `hog` | `hog` (fast) or `cnn` (accurate) |
| `--no-skip-existing` | — | Re-process videos whose output already contains frames |

---

## videsc

Generate AI-powered text descriptions for video files using the WD14 tagger.

Key frames are extracted from each video, tagged individually with WD14, and the tags are aggregated across all frames (union of tags ranked by mean confidence). The result is written to a `.txt` file alongside the video (or in a specified output directory).

The first run downloads the WD14 ONNX model from HuggingFace (~350 MB) and caches it under `~/.cache/huggingface/`.

### Usage

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

### CLI reference

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

# videsc
pytest tests/test_videsc.py
```

Generate a coverage report:

```bash
pytest --cov=portrait_prep --cov=vicrop --cov=videsc --cov-report=term-missing
```
