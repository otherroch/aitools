# Python API and Testing

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

### chararep

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

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The `[dev]` extra pulls in `pytest` and `pytest-cov`. Test paths and verbosity
are configured in `pyproject.toml` so no extra flags are needed.

Heavy dependencies (`onnxruntime`, `face_recognition`, `insightface`, `torch`,
`gfpgan`) are mocked in the test suite so the full suite runs without a GPU or
dlib installation.

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

# chararep
pytest tests/test_chararep_config.py
pytest tests/test_chararep_face_detector.py
pytest tests/test_chararep_face_swapper.py

# face_ops
pytest tests/test_face_ops.py
```

Generate a coverage report:

```bash
pytest --cov=portrait_prep --cov=vicrop --cov=videsc --cov=chararep --cov=face_ops --cov-report=term-missing
```
