# vicrop

Extract face-cropped PNG frames or per-person video segments from video files.

Reads video files using OpenCV, samples frames at a configurable interval (`--every-n`), detects faces in each sampled frame, crops them with padding, and saves them as PNG files. Optionally clusters face crops by identity into `person_NN` sub-folders (same greedy nearest-neighbour algorithm as `portrait-prep crop`). Optionally scores each crop for reference-photo quality and writes a `reflist.txt` per identity.

When `--output-type video` is used, vicrop instead extracts MP4 segments based on face detection performed on sampled frames (every `--every-n` frames). Contiguous runs of sampled frames that contain exactly one person are grouped into segments; the frames in between sampled frames are included in the segment without individual detection, so multi-person frames may appear in the output if they fall between sampled frames. Each segment is cropped and resized to a square around the detected face. This is useful for preparing single-subject training clips.

## Usage

### Photo mode (default)

```bash
# Process all videos in a directory (face-crop every 30th frame)
vicrop --input ./videos --output-dir ./frames

# Process a single video file (must use a supported video extension, e.g. .mp4 or .mov;
# unsupported file types will cause the tool to exit with an error)
vicrop --input ./video.mp4 --output-dir ./frames

# Faster sampling, no identity clustering
vicrop --input ./videos --output-dir ./frames --every-n 15 --no-classify

# Higher-accuracy face detection
vicrop --input ./videos --output-dir ./frames --detection-model cnn

# Tighter margin around the face (less background context)
vicrop --input ./videos --output-dir ./frames --margin-ratio 0.2

# Select reference photos scoring 0.75 or higher (more permissive than default)
vicrop --input ./videos --output-dir ./frames --ref-thresh 0.75

# Disable reference-photo selection entirely
vicrop --input ./videos --output-dir ./frames --ref-thresh 0

# Extract frames without face detection (raw frame extraction)
vicrop --input ./videos --output-dir ./frames --extract-only --every-n 30

# Extract frames with custom output dimensions (640×480)
vicrop --input ./videos --output-dir ./frames --extract-only --crop-dim 640 480
```

Photo output is organised as:
```
frames/
└── <video_stem>/
    ├── person_01/
    │   ├── frame000000_face1.png
    │   ├── frame000030_face1.png
    │   └── ref/                         ← reference photos for this person (if any pass --ref-thresh)
    │       └── frame000000_face1.png
    └── person_02/
        ├── frame000060_face1.png
        └── ref/
            └── frame000060_face1.png
```

### Video segment mode

```bash
# Extract single-person video segments (max 30 s each, discard clips < 2 s)
vicrop --input ./videos --output-dir ./segments --output-type video

# Custom segment length limits
vicrop --input ./videos --output-dir ./segments --output-type video \
    --max-segment-length 15 --min-segment-length 5

# Process a single file
vicrop --input ./interview.mp4 --output-dir ./segments --output-type video
```

Video output is organised as:
```
segments/
└── <video_stem>/
    ├── person_01/
    │   ├── seg_001.mp4
    │   └── seg_002.mp4
    └── person_02/
        └── seg_001.mp4
```

## Reference photo selection

Training a portrait LoRA requires a small set of high-quality *reference photos* — images where the subject is looking directly at the camera, eyes fully open, face well-lit, sharp, and occupying a significant area of the frame. `vicrop` can automatically identify those images from all the face crops it produces.

When `--ref-thresh` is greater than zero (the default is `0.65`), every saved face crop is scored on five criteria:

| Criterion | Weight | What is measured |
|-----------|--------|-----------------|
| Single face | hard gate | If more than one face is detected in the frame the crop is immediately disqualified (score → 0.0), regardless of all other criteria. This prevents another person's face from leaking into training data. |
| Frontal pose | 30 % | Landmark symmetry — how evenly the nose sits between both eyes (yaw), and how far down the face the nose tip sits (pitch). Scores drop as the face turns away from the camera. |
| Eyes open | 20 % | Eye Aspect Ratio (EAR) from six landmark points per eye. Closed or partially closed eyes score lower. |
| Sharpness | 20 % | Laplacian variance of the face-crop region. Blurry or motion-smeared crops score lower. |
| Face fill | 15 % | Ratio of face bounding-box area to total frame area. A face occupying ≥ 15 % of the frame earns a full score; smaller faces score proportionally lower. |
| Lighting | 15 % | Luminance mean and contrast. Very dark (< 40/255) or severely overexposed (> 220/255) crops score lower; well-exposed crops with natural contrast score higher. |

Crops whose composite score meets or exceeds `--ref-thresh` are **moved** into a `ref/` sub-folder inside their identity directory. At the end of processing, each `person_NN/` folder that contains at least one qualifying crop will have a `ref/` sub-folder holding only those images.

**Choosing a threshold:**

| `--ref-thresh` | Effect |
|----------------|--------|
| `1.0` | Only near-perfect frontal shots with fully open eyes and excellent exposure |
| `0.65` *(default)* | Good frontal poses; minor angle deviations and slight blur accepted |
| `0.4` | More permissive; useful when footage quality is variable |
| `0` | Disables the analysis entirely — no scoring, no `reflist.txt` |

Lower values cast a wider net and produce a larger reference set; higher values are more selective. The goal is to feed the LoRA trainer images that anchor the subject's likeness without injecting off-angle or blurry samples that can reduce identity coherence.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Video file (e.g. `.mp4`) or directory containing video files |
| `--output-dir` | *(required)* | Destination directory for cropped frames |
| `--every-n` | `30` | Process every N-th frame |
| `--margin-ratio` | `0.4` | Fractional padding around each detected face bbox (see below) |
| `--crop-size` | `1024` | Output square resolution (pixels) |
| `--crop-dim W H` | `None` | Output rectangular resolution in pixels (width W, height H). Overrides `--crop-size` |
| `--extract-only` | `false` | Extract frames without face detection. Saves every N-th frame as-is (see below) |
| `--no-classify` | — | Disable identity clustering |
| `--tolerance` | `0.7` | Face-distance threshold for clustering (see below) |
| `--detection-model` | `hog` | `hog` (fast) or `cnn` (more accurate) |
| `--ref-thresh` | `0.65` | Minimum quality score (0–1) for reference-photo selection; `0` disables |
| `--no-skip-existing` | — | Re-process videos whose output already contains frames |
| `--output-type` | `photo` | `photo` (face-cropped PNGs) or `video` (single-person MP4 segments) |
| `--max-segment-length` | `30` | Maximum segment duration in seconds; longer segments are split (video mode only) |
| `--min-segment-length` | `2` | Minimum segment duration in seconds; shorter segments are discarded (video mode only) |

### `--margin-ratio` — controlling how much context surrounds the face

`--margin-ratio` is a multiplier applied to the height and width of the raw face bounding box returned by the detector. The padded region is then cropped out of the frame and resized to `--crop-size` × `--crop-size`.

| Value | Effect |
|-------|--------|
| `0.1–0.2` | Tight crop — face fills most of the image, very little neck or hair visible. Useful if you want maximum facial detail at a given resolution. |
| `0.4` *(default)* | Balanced — includes forehead, chin, ears, and a sliver of neck/shoulders. Recommended for most portrait LoRA use cases. |
| `0.6–0.8` | Wide crop — substantial background and shoulders included. Helpful for full-head or upper-body training examples, but effective facial resolution is lower. |

> **Clipping:** margins are clamped to the frame edges, so very large values on faces near the border simply include as much of the frame as available rather than creating black padding.

### `--tolerance` — controlling how strictly faces are grouped into identities

After all face crops from a video are collected, `vicrop` groups them by identity using a greedy nearest-neighbour algorithm on 128-dimensional face encodings. `--tolerance` is the maximum *face distance* (Euclidean distance in encoding space) allowed before two crops are considered different people. Lower distance → higher similarity must be met to join an existing cluster.

| Value | Effect |
|-------|--------|
| `0.4–0.6` | Strict — only very similar encodings map to the same person. Reduces cross-person contamination in a scene with multiple look-alike subjects, but can split a single person across two `person_NN` folders when lighting or angle changes significantly. |
| `0.7` *(default)* | Balanced — works well for most footage with a dominant subject. |
| `0.8–0.9` | Permissive — merges more crops into each cluster. Good for footage where the subject's appearance varies widely (different lighting, head angles, partial occlusion), but risks merging distinct people who look somewhat similar. |

> **Tip:** if you find one person split across `person_01` and `person_03`, increase tolerance slightly. If two distinct people are being merged into the same folder, decrease it.

## `--extract-only` — raw frame extraction without face detection

When `--extract-only` is specified, `vicrop` skips face detection entirely and simply copies every N-th frame from the input video to the output directory as a raw PNG. This is useful for:

- Creating frame-level indexes of videos
- Extracting keyframes for thumbnails or previews
- Quickly sampling video content without the overhead of face detection and clustering
- Preparing frames for downstream processing by other tools

```bash
# Extract every 30th frame from a video
vicrop --input ./video.mp4 --output-dir ./frames --extract-only

# Extract every 10th frame (higher resolution sampling)
vicrop --input ./videos --output-dir ./frames --extract-only --every-n 10

# Extract every 60th frame (lower resolution sampling)
vicrop --input ./videos --output-dir ./frames --extract-only --every-n 60
```

The extracted frames are saved with the naming pattern `frame<N>.png` inside the `<video_stem>` sub-directory under `--output-dir`.

## `--crop-dim W H` — custom output dimensions

By default, `vicrop` resizes face crops to a square of `--crop-size` × `--crop-size` pixels (default 1024×1024). The `--crop-dim` flag allows specifying a rectangular output resolution with separate width and height values:

```bash
# Output 640×480 portraits (4:3 aspect ratio)
vicrop --input ./videos --output-dir ./frames --crop-dim 640 480

# Output 1920×1080 portraits (16:9 widescreen)
vicrop --input ./videos --output-dir ./frames --crop-dim 1920 1080

# Extract-only with custom dimensions
vicrop --input ./video.mp4 --output-dir ./frames --extract-only --crop-dim 800 600
```

The output directory for `--crop-dim` uses the same structure as `--crop-size` (with a `.dim_<W>x<H>` suffix appended to the video stem directory):

```
frames/
└── <video_stem>.dim_640x480/
    └── frame000000_face1.png
```

When used together with `--extract-only`, the frames are saved at the specified dimensions without any face detection overhead.

> **Note:** `--crop-dim` overrides `--crop-size` when both are specified. The width (W) must come first, followed by the height (H), separated by a space. Exactly two values are required.
