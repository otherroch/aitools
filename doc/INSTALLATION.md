# Installation

```bash

# 1. upgrade pip to get --group support
python -m pip install -U pip

# 2. Install CUDA if you have a GPU that supports it. XXX is the CUDA version, for example 130
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cuXXX
#    OR you can install the stable version  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX

# 3. With GPU support for WD14 captioning (replaces onnxruntime with onnxruntime-gpu)
pip install --group gpu

# 4. install DLIB with CUDA support 
#    see instructions below

# 5. With chararep face-replacement pipeline (adds insightface, torch, gfpgan, onnxruntime-gpu)
# Note: install basicsr BEFORE this step (see below)
python scripts/install_basicsr.py
pip install --group chararep

# 6. command packages for CPU and GPU 
pip install --group base

# 7. With Qwen3-VL support for videsc --vl (adds PyTorch, transformers, and related dependencies)
pip install --group vl

# 8. With YouTube download support (adds yt-dlp)
pip install --group youtube

# 9. With vLLM remote server support for videsc --vllm (adds openai, requests)
pip install --group vllm

# 10. Including dev / test dependencies
pip install --group dev

# 11. install the aitools
pip install -e .  
```

## Notes

### Docker images 

- Built with [Dockerfile](../Dockerfile)
- `docker pull otherroch/aitools`
- `docker run -it --rm --gpus all otherroch/aitools bash`  

### HEIC support

`pillow-heif` ships with pre-built wheels on PyPI for
Windows, macOS, and Linux — no extra system libraries required in most cases.


### CNN face detection on GPU (`--detection-model cnn`)

By default `dlib` is built for CPU only. To run the CNN detector on a GPU you must install
a CUDA-enabled build of `dlib`. First ensure the **CUDA Toolkit** and **cuDNN**
are installed and visible on your system path, then build `dlib` with the CUDA
flag **before** installing `face_recognition`:

*Linux*
```bash
pip install cmake
DLIB_USE_CUDA=1 pip install -v dlib
pip install face_recognition
```

*Windows (PowerShell)*
```powershell
pip install cmake
$env:DLIB_USE_CUDA=1; pip install -v dlib
pip install face_recognition
```

If CUDA is correctly detected, dlib's build output will include a line such as
`"Enabling CUDA support"`. Without this, `--detection-model cnn` will still
work but will run on CPU and be significantly slower.



### GPU inference

Install the `[gpu]` extra (see above) to use
`onnxruntime-gpu` for significantly faster WD14 captioning on CUDA devices.

### YouTube support

The `--youtube-url` flag requires `yt-dlp` and a
YouTube Data API v3 key. Install `yt-dlp` with `pip install -e ".[youtube]"` or
`pip install yt-dlp`.

### videsc VL mode

The `--vl` flag requires PyTorch, the
Transformers library, and related dependencies. Install them with
`pip install -e ".[vl]"`. A CUDA-capable GPU with sufficient VRAM is strongly
recommended (8 GB+ for the default 8B model; use `--quant 4bit` or `--quant 8bit`
to reduce VRAM requirements).

### videsc vLLM mode

The `--vllm` flag connects to a remote vLLM server for inference. Install the
client dependencies with `pip install -e ".[vllm]"` (adds `openai>=1.0.0` and
`requests>=2.31.0`). OpenCV (`opencv-python`) is also required for local frame
extraction. No local GPU is needed — all model inference runs on the remote
server. See the [videsc documentation](VIDESC.md#vllm-mode-usage---vllm) for
server setup instructions and usage examples.

### chararep

Requires an NVIDIA GPU with CUDA support. The `[chararep]`
extra installs InsightFace, PyTorch, GFPGAN, and ONNX Runtime GPU. The `basicsr`
dependency (needed by GFPGAN) may require a manual patch on Python 3.13+; run
`python scripts/install_basicsr.py` **before** `pip install -e ".[chararep]"` to
download, patch, and install it automatically.


# Project structure

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
│   │   ├── loader.py     # Qwen3-VL / Qwen3-Omni / Gemma4 model loading
│   │   └── vllm_client.py # OpenAI-compatible client for remote vLLM servers
│   ├── pipeline/
│   │   ├── runner.py     # Batch & single-video runner for VL mode
│   │   ├── vllm_runner.py # Batch & single-video runner for vLLM mode
│   │   └── prompt.py     # Summary prompt generation for Gemma4 mode
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
│   ├── test_augment.py
│   ├── test_caption.py
│   ├── test_chararep_config.py
│   ├── test_chararep_face_detector.py
│   ├── test_chararep_face_recognizer.py
│   ├── test_chararep_face_swapper.py
│   ├── test_chararep_face_enhancer.py
│   ├── test_chararep_face_blender.py
│   ├── test_chararep_gpu_utils.py
│   ├── test_convert.py
│   ├── test_coverage_targeted.py
│   ├── test_cpcap.py
│   ├── test_crop.py
│   ├── test_face_ops.py
│   ├── test_vicrop.py
│   ├── test_videsc.py
│   ├── test_videsc_main.py
├── main.py               # Thin shim for portrait-prep
├── pyproject.toml
├── Dockerfile
├── LICENSE
└── README.md
```
