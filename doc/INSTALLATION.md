# Installation

```bash
# Standard (CPU inference)
pip install -e .

# With GPU support for WD14 captioning (replaces onnxruntime with onnxruntime-gpu)
pip install -e ".[gpu]"

# With Qwen3-VL support for videsc --vl (adds PyTorch, transformers, and related dependencies)
pip install -e ".[vl]"

# With YouTube download support (adds yt-dlp)
pip install -e ".[youtube]"

# With chararep face-replacement pipeline (adds insightface, torch, gfpgan, onnxruntime-gpu)
# Note: install basicsr BEFORE this step (see below)
python scripts/install_basicsr.py
pip install -e ".[chararep]"

# Including dev / test dependencies
pip install -e ".[dev]"
```

## Notes

### HEIC support

`pillow-heif` ships with pre-built wheels on PyPI for
Windows, macOS, and Linux — no extra system libraries required in most cases.

### face_recognition

Requires `cmake` and `dlib` to be present before
installation. On Windows the easiest path is:
```bash
pip install cmake
pip install dlib
pip install face_recognition
```

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

### chararep

Requires an NVIDIA GPU with CUDA support. The `[chararep]`
extra installs InsightFace, PyTorch, GFPGAN, and ONNX Runtime GPU. The `basicsr`
dependency (needed by GFPGAN) may require a manual patch on Python 3.13+; run
`python scripts/install_basicsr.py` **before** `pip install -e ".[chararep]"` to
download, patch, and install it automatically.
