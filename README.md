# aitools

**AI dataset preparation toolkit for diffusion model LoRA training.**

`aitools` provides four command-line tools and Python APIs for preparing image and video datasets:

| Tool | Command | Description |
|------|---------|-------------|
| Portrait Prep | `portrait-prep` | End-to-end portrait image preparation (convert → crop → caption → augment) |
| Video Crop | `vicrop` | Extract face-cropped PNG frames from video files |
| Video Description | `videsc` | Generate text descriptions for video files — fast WD14 tag-based captions (default) or rich natural-language descriptions via Qwen3-VL (`--vl`) |
| Character Replace | `chararep` | Replace character faces in a video using deep face-swapping models (inswapper, SimSwap, uniface, hyperswap, blendswap) |

---

## Quick start

```bash
git clone https://github.com/otherroch/aitools.git
cd aitools

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# upgrade pip to get --group support
python -m pip install -U pip

# install base dependencies (CPU only support)
pip install --group base

# install aitools
pip install -e .
```

For individual feature installs, GPU support, chararep dependencies (including basicsr), or system prerequisites, see [installation](doc/INSTALLATION.md) .

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
├── chararep/
│   ├── __init__.py
│   ├── main.py           # chararep CLI entry point
│   ├── pipeline.py       # End-to-end face-replacement pipeline
│   ├── config.py         # PipelineConfig and CharacterMapping dataclasses
│   ├── face_detector.py  # InsightFace detection + IoU-based tracking
│   ├── face_recognizer.py# ArcFace-based identity matching
│   ├── face_swapper.py   # ONNX model swap (inswapper / SimSwap / uniface / hyperswap / blendswap)
│   ├── face_enhancer.py  # GFPGAN and CodeFormer ONNX enhancement
│   ├── face_blender.py   # Poisson seamless-clone and alpha blending
│   ├── video_io.py       # OpenCV video read / FFmpeg video write
│   └── gpu_utils.py      # CUDA / ONNX Runtime provider helpers
├── face_ops/
│   ├── __init__.py       # get_backend(), FaceBackend protocol
│   ├── backend.py        # DlibBackend and InsightFaceBackend
│   └── clustering.py     # Backend-agnostic cluster_faces() and load_reference_encodings()
├── scripts/
│   └── install_basicsr.py # Download, patch, and install basicsr for Python 3.13+
├── doc/
│   ├── INSTALLATION.md
│   ├── PORTRAIT_PREP.md
│   ├── VICROP.md
│   ├── VIDESC.md
│   ├── CHARAREP.md
│   └── API_AND_TESTING.md
├── tests/
│   ├── conftest.py       # Stubs for insightface, onnxruntime, torch, gfpgan
│   ├── test_crop.py
│   ├── test_vicrop.py
│   ├── test_videsc.py
│   ├── test_videsc_main.py
│   ├── test_face_ops.py
│   ├── test_chararep_config.py
│   ├── test_chararep_face_detector.py
│   ├── test_chararep_face_recognizer.py
│   ├── test_chararep_face_swapper.py
│   ├── test_chararep_face_enhancer.py
│   ├── test_chararep_face_blender.py
│   └── test_chararep_gpu_utils.py
├── main.py               # Thin shim for portrait-prep
├── pyproject.toml
├── Dockerfile
├── LICENSE
└── README.md
```

---

## Documentation

For a complete reference of each tool's command line options, usage examples, and Python API:

- [Installation](doc/INSTALLATION.md) — all install extras, GPU support, basicsr workaround, and system prerequisites
- [portrait-prep](doc/PORTRAIT_PREP.md) — portrait dataset preparation pipeline (convert, crop, caption, augment, cpcap)
- [vicrop](doc/VICROP.md) — video face-crop extraction with identity clustering and reference photo selection
- [videsc](doc/VIDESC.md) — video description generator (WD14 tags and Qwen3-VL natural language)
- [chararep](doc/CHARAREP.md) — video character face-replacement pipeline (architecture, CLI, config, input requirements, VRAM guidelines)
- [Python API and Testing](doc/API_AND_TESTING.md) — Python API examples for each tool, test commands, and coverage

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
