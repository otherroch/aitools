# portrait-prep

Portrait dataset preparation toolkit for diffusion model LoRA training.

## Features

| Step | Description |
|------|-------------|
| `convert` | Convert HEIC / JPG (and other formats) to PNG |
| `crop` | Face-detect, crop, and classify persons into sub-folders |
| `caption` | WD14 tagger auto-captioning with a custom token prefix |
| `augment` | Identity-preserving Albumentations augmentations |
| `cpcap` | Replicate captions from originals to augmented images |

Steps can be run individually or chained as a full pipeline in a single command.

## Usage

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
