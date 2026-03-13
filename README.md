# portrait-prep

**Portrait dataset preparation toolkit for diffusion model LoRA training.**

`portrait-prep` provides a single command-line interface and a clean Python API for the full pipeline needed to prepare a photo portrait dataset—covering every step from raw camera files all the way to augmented, captioned training images.

---

## Features

| Step | Description |
|------|-------------|
| `convert` | Convert HEIC / JPG (and other formats) to PNG |
| `crop` | Face-detect, crop, and classify persons into sub-folders |
| `caption` | WD14 tagger auto-captioning with a custom token prefix |
| `augment` | Identity-preserving Albumentations augmentations |
| `cpcap` | Replicate captions from originals to augmented images |

Steps can be run individually or chained as a full pipeline in a single command.

---

## Project structure

```
augment/
├── portrait_prep/
│   ├── __init__.py
│   ├── convert.py        # Step 1 – format conversion
│   ├── crop.py           # Step 2 – face crop + classification
│   ├── caption.py        # Step 3 – WD14 captioning
│   ├── augment.py        # Step 4 – data augmentation
│   └── cpcap.py          # Step 5 – caption propagation
├── tests/
│   ├── conftest.py
│   ├── test_convert.py
│   ├── test_crop.py
│   ├── test_caption.py
│   ├── test_augment.py
│   └── test_cpcap.py
├── main.py               # CLI entry point
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Standard (CPU inference)
pip install -e .

# With GPU support for WD14 captioning (replaces onnxruntime with onnxruntime-gpu)
pip install -e ".[gpu]"

# Including dev / test dependencies
pip install -e ".[dev]"
```

Or install directly without a local clone:

```bash
pip install portrait-prep
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

---

## Usage

### Full pipeline

```bash
python main.py \
  --input-dir ./raw_photos \
  --output-dir ./dataset \
  --steps convert crop caption augment cpcap \
  --prefix "ohwx man" \
  --per-image 8 \
  --keep-originals
```

### Individual steps

**Step 1 – Convert HEIC/JPG to PNG**
```bash
python main.py \
  --input-dir ./raw_heic \
  --output-dir ./png_out \
  --steps convert
```

**Step 2 – Face-crop and classify persons**
```bash
python main.py \
  --input-dir ./png_out \
  --output-dir ./cropped \
  --steps crop \
  --margin-ratio 0.4 \
  --crop-size 1024
```

Each detected person is placed in a `person_NN` sub-folder (use `--no-classify` to skip clustering).

**Step 3 – WD14 captioning**
```bash
python main.py \
  --input-dir ./cropped \
  --steps caption \
  --prefix "rocharch61" \
  --threshold 0.35
```

Captions are written as `.txt` files alongside each image (or in `--caption-output-dir`).  
The first run downloads the WD14 ONNX model from HuggingFace (~350 MB) and caches it.

**Step 4 – Augment images**
```bash
python main.py \
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
python main.py \
  --source-dir ./cropped \
  --aug-dir ./augmented \
  --steps cpcap
```

Or combined with augment (source captions are automatically inferred):
```bash
python main.py \
  --input-dir ./cropped \
  --output-dir ./augmented \
  --steps augment cpcap
```

---

## CLI reference

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | *(required)* | Source directory |
| `--output-dir` | — | Destination directory (required for convert, crop, augment) |
| `--steps` | all | Steps to run: `convert crop caption augment cpcap` |
| `--no-skip-existing` | — | Re-process files whose output already exists |

### crop options

| Flag | Default | Description |
|------|---------|-------------|
| `--margin-ratio` | `0.4` | Fractional padding around each face bbox |
| `--crop-size` | `1024` | Output square resolution (pixels) |
| `--no-classify` | — | Disable identity clustering |
| `--tolerance` | `0.6` | Face-distance threshold for clustering |
| `--detection-model` | `hog` | `hog` (fast) or `cnn` (accurate) |

### caption options

| Flag | Default | Description |
|------|---------|-------------|
| `--prefix` | `""` | Token prepended to every caption |
| `--threshold` | `0.35` | Minimum WD14 confidence to include a tag |
| `--model-repo` | `SmilingWolf/wd-v1-4-convnextv2-tagger-v2` | HuggingFace model repo |
| `--include-ratings` | — | Include rating tags (safe/questionable/explicit) |
| `--caption-output-dir` | alongside images | Separate dir for `.txt` files |

### augment options

| Flag | Default | Description |
|------|---------|-------------|
| `--per-image` | `5` | Augmented variants per source image |
| `--image-size` | `1024 1024` | Output `HEIGHT WIDTH` |
| `--keep-originals` | — | Also copy a resized `*_orig.png` |
| `--seed` | `4051888` | Random seed |

### cpcap options

| Flag | Default | Description |
|------|---------|-------------|
| `--source-dir` | `--input-dir` | Directory containing original captions |
| `--aug-dir` | `--output-dir` | Directory with augmented images |
| `--caption-ext` | `.txt` | Caption file extension |
| `--dry-run` | — | Report without writing files |

---

## Python API

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

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The `[dev]` extra pulls in `pytest` and `pytest-cov`. Test paths and verbosity
are configured in `pyproject.toml` so no extra flags are needed.

Tests for the `caption` and `crop` steps mock out their heavy dependencies
(`onnxruntime`, `face_recognition`) so the full test suite runs without a GPU
or dlib installation.

To run a specific step's tests:

```bash
pytest tests/test_convert.py
pytest tests/test_cpcap.py
pytest tests/test_augment.py
```

Generate a coverage report:

```bash
pytest --cov=portrait_prep --cov-report=term-missing
```

---

## Typical end-to-end workflow

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
