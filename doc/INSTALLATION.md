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

# 9. Including dev / test dependencies
pip install --group dev

# 10. install the aitools
pip install -e .  
```

## Notes

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

### chararep

Requires an NVIDIA GPU with CUDA support. The `[chararep]`
extra installs InsightFace, PyTorch, GFPGAN, and ONNX Runtime GPU. The `basicsr`
dependency (needed by GFPGAN) may require a manual patch on Python 3.13+; run
`python scripts/install_basicsr.py` **before** `pip install -e ".[chararep]"` to
download, patch, and install it automatically.
