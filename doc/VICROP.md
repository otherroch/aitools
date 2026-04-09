# vicrop

Extract face-cropped PNG frames from video files.

Reads video files using OpenCV, samples frames at a configurable interval, detects faces in each frame, crops them with padding, and saves them as PNG files. Optionally clusters face crops by identity into `person_NN` sub-folders (same greedy nearest-neighbour algorithm as `portrait-prep crop`). Optionally scores each crop for reference-photo quality and writes a `reflist.txt` per identity.

## Usage

```bash
# Process all videos in a directory (face-crop every 30th frame)
vicrop --input ./videos --output-dir ./frames

# Process a single video file (must use a supported video extension, e.g. .mp4 or .mov;
# unsupported file types will cause the tool to exit with an error)
# The file must use a supported video extension (for example: .mp4, .mov);
# otherwise, vicrop exits with an error.
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
```

Output is organised as:
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
| `--no-classify` | — | Disable identity clustering |
| `--tolerance` | `0.6` | Face-distance threshold for clustering (see below) |
| `--detection-model` | `hog` | `hog` (fast) or `cnn` (more accurate) |
| `--ref-thresh` | `0.8` | Minimum quality score (0–1) for reference-photo selection; `0` disables |
| `--no-skip-existing` | — | Re-process videos whose output already contains frames |

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
