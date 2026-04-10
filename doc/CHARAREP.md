# CharaRep – Video Character Replacement Pipeline

A GPU-accelerated Python pipeline that replaces up to **3 characters** (faces) in a video with different identities, using portrait photos as the source of the new face. Optimised for **NVIDIA RTX 5090** (CUDA, FP16 Tensor Core inference).

## Architecture

```
Video File
  │
  ▼
┌──────────────────┐
│  Frame Decoding   │  (video_io.py – threaded OpenCV reader)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Face Detection   │  (face_detector.py – via face_ops InsightFaceBackend)
│  & IoU Tracking   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Face Recognition │  (face_recognizer.py – ArcFace via face_ops)
│  (ID matching)    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Face Swap        │  (face_swapper.py – inswapper_128 ONNX model)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Face Enhancement │  (face_enhancer.py – GFPGAN restoration)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Seam Blending    │  (face_blender.py – Poisson / alpha blend)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Video Encoding   │  (video_io.py – ffmpeg subprocess writer)
│  + Audio Muxing   │
└──────────────────┘
```

> **Shared code:** Face detection and recognition use the
> :mod:`face_ops` package (``InsightFaceBackend``), the same backend
> shared by *vicrop* and *portrait_prep*.  This avoids duplicating
> InsightFace / ArcFace initialisation code across applications.

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| CUDA Toolkit | 12.x+ |
| ffmpeg | 4.4+ (on PATH) |
| NVIDIA GPU | RTX 5090 recommended; any CUDA GPU works |

### Model files

1. **InsightFace buffalo_l** – auto-downloaded on first run to `~/.insightface/models/`.
2. **inswapper_128.onnx** – download manually and place at `~/.insightface/models/inswapper_128.onnx`.
   - Source: <https://github.com/deepinsight/insightface/tree/master/examples/in_swapper>
3. **GFPGAN v1.4** – auto-downloaded by the `gfpgan` package on first run.

## Installation

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install stable PyTorch with CUDA support first
# where XXX is the CUDA version that matches your needs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
#or
# install CPU version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install patched basicsr (required on Python 3.13+)
python scripts/install_basicsr.py

# Install chararep with its dependencies
pip install -e ".[chararep]"
```

## Usage

### CLI (direct arguments)

```bash
chararep \
    -i input_video.mp4 \
    -o output_video.mp4 \
    --char originals/villain replacements/villain \
    --char originals/hero replacements/hero \
    --device 0 \
    -v
```

Each `--char` takes **two folders**:
1. **FIND folder** — images of the **original** face to locate in the video.  The folder name becomes the character label.
2. **REPLACE folder** — images of the **new** face to swap in.

Example folder layout:
```
originals/
  villain/              ← label = "villain"
    screenshot1.jpg
    screenshot2.png
  hero/                 ← label = "hero"
    hero_frame.jpg
replacements/
  villain/
    new_face1.jpg
  hero/
    new_hero.jpg
```

### CLI (JSON config)

```bash
chararep --config swap_config.json
```

Example `swap_config.json`:

```json
{
    "input_video": "input_video.mp4",
    "output_video": "output_video.mp4",
    "characters": [
        {
            "find": "originals/villain",
            "replace": "replacements/villain",
            "similarity_threshold": 0.5
        },
        {
            "find": "originals/hero",
            "replace": "replacements/hero"
        }
    ],
    "enable_face_enhancement": true,
    "device_id": 0,
    "output_quality": 18,
    "blend_mode": "seamless"
}
```

### Key CLI options

| Flag | Description | Default |
|---|---|---|
| `-i` / `--input` | Input video path | required |
| `-o` / `--output` | Output video path | required |
| `--char` | Character mapping (repeat up to 3×) | required |
| `--config` | JSON config file | – |
| `--similarity-threshold` | Cosine-similarity threshold for identity matching | `0.5` |
| `--swap-model-path` | Path to face-swap ONNX model (`inswapper`/`simswap`) | auto-detect |
| `--embedding-converter-path` | Optional SimSwap embedding converter ONNX path | none |
| `--detection-model` | InsightFace model pack name | `buffalo_l` |
| `--detect-size` | Detection resolution (try 1024 for HD video) | `640` |
| `--enhance` | Enable GFPGAN face enhancement | false |
| `--enhance-model` | Enhancement backend: `gfpgan` or `codeformer_onnx` | `gfpgan` |
| `--enhance-model-path` | Path to enhancement model file | – |
| `--enhance-weight` | Enhancement blend weight (`0-1`) | `0.7` |
| `--device` | CUDA device ID | 0 |
| `--no-fp16` | Disable FP16 and use FP32 | false |
| `--codec` | Output video codec | `libx264` |
| `--blend-mode` | `seamless` or `alpha` | seamless |
| `--blender-blur` | Gaussian blur kernel size for mask edges | `15` |
| `--blender-erode` | Pixels to erode from mask edge | `5` |
| `--crf` | Output quality (lower=better) | 18 |
| `--no-audio` | Don't copy original audio | false |
| `-v` / `--verbose` | Debug logging | false |
| `--log-file` | Write logs to file in addition to stderr | none |
| `--timers` | Print cumulative per-stage pipeline timing report | false |
| `--dump-config` | Print resolved pipeline config as JSON before run | false |

### Common command recipes

Quick run (single character, default settings):

```bash
chararep -i input.mp4 -o output.mp4 \
  --char originals/villain replacements/villain
```

Higher quality output (slower, better visual quality):

```bash
chararep -i input.mp4 -o output_hq.mp4 \
  --char originals/villain replacements/villain \
  --crf 14 --blend-mode seamless --enhance --enhance-weight 0.8
```

Faster run (no enhancement, alpha blending):

```bash
chararep -i input.mp4 -o output_fast.mp4 \
  --char originals/villain replacements/villain \
  --blend-mode alpha
```

Multi-character replacement:

```bash
chararep -i input.mp4 -o output.mp4 \
  --char originals/villain replacements/villain \
  --char originals/hero replacements/hero
```

SimSwap model with optional embedding converter:

```bash
chararep -i input.mp4 -o output_simswap.mp4 \
  --char originals/villain replacements/villain \
  --swap-model-path models/simswap_256.onnx \
  --embedding-converter-path models/crossface_simswap.onnx
```

Run from JSON config:

```bash
chararep --config swap_config.json
```

Debug and diagnostics (verbose logs, timing report, config dump):

```bash
chararep -i input.mp4 -o output.mp4 \
  --char originals/villain replacements/villain \
  -v --timers --dump-config --log-file run.log
```

## Python API

```python
from chararep.config import CharacterMapping, PipelineConfig
from chararep.pipeline import CharacterReplacementPipeline

cfg = PipelineConfig(
    input_video="input.mp4",
    output_video="output.mp4",
    characters=[
        CharacterMapping(
            source_label="villain",
            reference_paths=["originals/villain/frame1.jpg"],
            portrait_paths=["replacements/villain/new_face.jpg"],
            similarity_threshold=0.5,
        ),
    ],
    enable_face_enhancement=True,
    device_id=0,
)

pipeline = CharacterReplacementPipeline(cfg)
stats = pipeline.run()
print(stats)  # {'frames_total': 1200, 'frames_swapped': 450, ...}
```

## Input requirements and restrictions

### Input video

| Property | Requirement | Notes |
|---|---|---|
| **Container format** | Any container supported by OpenCV/ffmpeg | `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm` all work |
| **Video codec** | Any codec decodable by OpenCV | H.264, H.265, VP9, ProRes, etc. |
| **Colour space** | BGR / YUV420 | OpenCV decodes to BGR uint8 automatically |
| **Frame size (width × height)** | Any resolution ≥ ~160 × 120 | Faces must be large enough to detect (see below) |
| **Maximum resolution** | No hard limit; 4K is practical on ≥ 16 GB VRAM | Higher resolutions increase VRAM and processing time |
| **Frame rate** | Any; float values supported | Stored in output unchanged |
| **Duration / frame count** | No hard limit | Tested up to ~10 000 frames; longer videos work |
| **Audio track** | Optional; any codec | Re-encoded to AAC in the output when `--no-audio` is not set |
| **Variable frame rate (VFR)** | ⚠ Not recommended | OpenCV may miscount frames; convert to CFR first with `ffmpeg -vsync cfr` |

#### Minimum detectable face size

The InsightFace RetinaFace detector processes every frame resized to **640 × 640** pixels internally.  A face must span at least **~32 × 32 pixels in that 640 × 640 space** to be reliably detected.

To estimate the minimum face size for your video:

```
min_face_width_px  = video_width  / 640 × 32  ≈  video_width  / 20
min_face_height_px = video_height / 640 × 32  ≈  video_height / 20
```

For example, in a **1920 × 1080** video a face must be at least **≈96 × 54 pixels**.

> **Tip**: If faces are consistently missed, use footage where faces are larger in frame (zoom/crop/upscale before processing).

#### Face size required for a high-quality swap

The `inswapper_128` model operates on **128 × 128** pixel crops extracted by the affine-alignment step.  Small faces (< ~64 px wide in the original frame) produce blurry or distorted swaps.  For best results aim for faces that are at least **128 × 128 pixels** in the source video.

---

### Reference images (FIND folder)

These are photos of the **original face to locate** in the video.

| Property | Requirement | Notes |
|---|---|---|
| **File formats** | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif` | Case-insensitive extensions |
| **Minimum number** | 1 image | More images improve recognition robustness |
| **Recommended number** | 3–10 images | Diverse angles/lighting improve ArcFace mean embedding quality |
| **Maximum number** | No hard limit | Each image is encoded once at startup |
| **Image dimensions** | Any size that allows face detection | The same 640 × 640 det_size applies; faces < 32 px in the scaled image will not be encoded |
| **Recommended minimum face size** | ≥ 128 × 128 px in the reference image | Ensures a high-quality ArcFace embedding |
| **Colour space** | BGR / RGB (standard photos) | OpenCV decodes all common formats automatically |
| **Number of faces** | Exactly 1 clearly visible face per image | The **largest** detected face by bounding-box area is used; extra faces are ignored |
| **Face occlusion** | Minimal | Sunglasses, heavy makeup, profile angles > ~60° degrade embedding quality and matching accuracy |
| **Lighting** | Varied lighting recommended | Homogeneous lighting in all reference images biases the mean embedding |

---

### Portrait images (REPLACE folder)

These are photos of the **new face to swap in**.

| Property | Requirement | Notes |
|---|---|---|
| **File formats** | Same as reference images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`, `.tif` |
| **Minimum number** | 1 image | Only the first portrait is used by the swap engine at runtime |
| **Recommended number** | 1–3 frontal-face images | Portraits are ranked by detection confidence; the highest-scoring one is passed to `inswapper` |
| **Image dimensions** | Any size; face must be detectable | Same 640 × 640 detection constraint applies |
| **Recommended face size** | ≥ 128 × 128 px | Larger faces yield sharper inswapper results |
| **Face orientation** | Near-frontal preferred (< 30° yaw/pitch) | `inswapper_128` performs best with frontal faces; profiles may produce artefacts |
| **Colour space** | Standard BGR / RGB photos | |
| **Number of faces** | Exactly 1 clearly visible face | Largest-by-area face is selected automatically |
| **Background** | Any | The swap engine only uses the aligned face crop, not the background |

---

### Maximum characters (simultaneous face replacements)

The pipeline supports **up to 3 character replacements** in a single run.  This limit exists because:

- The ArcFace gallery comparison is O(faces × characters) per frame.
- `inswapper_128` swaps are applied sequentially; more swaps increases per-frame latency.
- Memory footprint of three independent ONNX inference sessions.

To replace more than 3 characters, run the pipeline in multiple passes.

---

### VRAM requirements (guidelines)

| Configuration | Estimated VRAM |
|---|---|
| Detection + swap only (no GFPGAN) | ~3–4 GB |
| Detection + swap + GFPGAN v1.4 | ~6–8 GB |
| 4K video, 3 characters, GFPGAN | ~12–16 GB |

## How it works

1. **Detection & Tracking**: Each frame is processed by the shared `face_ops.InsightFaceBackend` (RetinaFace detector). A lightweight IoU tracker maintains consistent face IDs across frames.

2. **Identity Matching**: ArcFace embeddings are computed for each detected face and compared (cosine similarity) against the **recognition gallery** built from the *find* folder images. Faces matching above the similarity threshold are queued for swapping with the corresponding *replace* identity.

3. **Face Swap**: The `inswapper_128` model transfers the target identity onto the source face while preserving the source's expression and head pose.

4. **Enhancement**: GFPGAN restores fine details and removes swap artefacts.

5. **Blending**: Poisson seamless cloning (or alpha blending) smooths the seam between the swapped face and the original frame.

6. **Output**: Frames are encoded via ffmpeg with the original audio muxed back in.

## Performance notes

- **Threaded I/O**: Video decode and encode run in separate threads so the GPU is never idle waiting on disk.
- **FP16 inference**: All ONNX models run with CUDAExecutionProvider in FP16 precision (Tensor Cores on RTX 5090).
- **Pinned memory**: Frame buffers use page-locked memory for faster GPU transfers.
- **5-minute video** (~9,000 frames at 30 fps): processing depends on the number of faces per frame and enhancement settings.

## Module overview

| File | Responsibility |
|---|---|
| `config.py` | Dataclass-based configuration |
| `gpu_utils.py` | CUDA/ONNX provider setup, GPU diagnostics |
| `video_io.py` | Threaded video reader & ffmpeg-based writer |
| `face_detector.py` | Face detection via `face_ops.InsightFaceBackend` + IoU tracker |
| `face_recognizer.py` | ArcFace gallery + identity matching (via shared `face_ops` backend) |
| `face_swapper.py` | inswapper_128 face transfer |
| `face_enhancer.py` | GFPGAN face restoration |
| `face_blender.py` | Poisson / alpha mask blending |
| `pipeline.py` | End-to-end pipeline orchestrator |
| `main.py` | CLI argument parsing & entry point |

## License

This pipeline integrates several open-source components. Ensure you comply with their respective licenses:
- InsightFace / inswapper: check the InsightFace license terms
- GFPGAN: Apache 2.0
- OpenCV: Apache 2.0
