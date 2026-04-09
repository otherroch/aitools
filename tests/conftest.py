"""Pytest configuration and stubs for heavy GPU dependencies.

Stubs ``insightface``, ``onnxruntime``, ``torch``, and ``gfpgan`` so the
chararep test suite can run on any machine without a GPU or those
libraries installed.

These stubs are only installed when the real packages are not available.
"""

import sys
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# insightface stub
# ---------------------------------------------------------------------------

def _make_insightface_stub():
    if "insightface" in sys.modules:
        return
    insightface = types.ModuleType("insightface")
    insightface_app = types.ModuleType("insightface.app")
    insightface_model_zoo = types.ModuleType("insightface.model_zoo")

    class FaceAnalysis:
        def __init__(self, name="buffalo_l", providers=None, **kw):
            self.name = name
            self.providers = providers or []

        def prepare(self, ctx_id=0, det_size=(640, 640), det_thresh=0.5):
            pass

        def get(self, img):
            return []

    insightface_app.FaceAnalysis = FaceAnalysis

    def get_model(path, providers=None):
        class _Model:
            def get(self, frame, src_face, tgt_face, paste_back=True):
                return frame.copy()
        return _Model()

    insightface_model_zoo.get_model = get_model

    insightface.app = insightface_app
    insightface.model_zoo = insightface_model_zoo

    sys.modules["insightface"] = insightface
    sys.modules["insightface.app"] = insightface_app
    sys.modules["insightface.model_zoo"] = insightface_model_zoo


# ---------------------------------------------------------------------------
# onnxruntime stub
# ---------------------------------------------------------------------------

def _make_onnxruntime_stub():
    if "onnxruntime" in sys.modules:
        return
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]

    class InferenceSession:
        def __init__(self, path, providers=None, **kw):
            self._path = path

        def _image_size(self):
            """Return the spatial size implied by the model filename."""
            stem = Path(self._path).stem.lower()
            if any(
                m in stem
                for m in ["simswap_256", "uniface", "hyperswap", "blendswap"]
            ):
                return 256
            return 512

        def _uses_generic_input_names(self):
            """Return True for models known to export with input_0/input_1."""
            stem = Path(self._path).stem.lower()
            return "uniface" in stem or "blendswap" in stem

        def get_inputs(self):
            sz = self._image_size()
            if self._uses_generic_input_names():
                return [
                    types.SimpleNamespace(name="input_0", shape=[1, 3, sz, sz]),
                    types.SimpleNamespace(name="input_1", shape=[1, 3, sz, sz]),
                ]
            return [
                types.SimpleNamespace(name="source", shape=[1, 512]),
                types.SimpleNamespace(name="target", shape=[1, 3, sz, sz]),
            ]

        def run(self, output_names, input_feed):
            for v in input_feed.values():
                if isinstance(v, np.ndarray) and v.ndim == 4 and v.shape[1] == 3:
                    return [v]
            source = input_feed.get("input", np.zeros((1, 512), dtype=np.float32))
            return [source]

    ort.InferenceSession = InferenceSession
    sys.modules["onnxruntime"] = ort


# ---------------------------------------------------------------------------
# torch stub
# ---------------------------------------------------------------------------

def _make_torch_stub():
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda i: "FakeCUDA",
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=8 * 1024**3),
        memory_allocated=lambda i: 0,
        set_device=lambda i: None,
        synchronize=lambda i: None,
    )
    torch.version = types.SimpleNamespace(cuda="12.0")
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False, enabled=False)
    )
    torch.zeros = lambda *a, **kw: np.zeros(a)

    class _TorchDevice:
        def __init__(self, s: str):
            self._repr = s
            self.type = s.split(":")[0]
            self.index = 0

        def __str__(self):
            return self._repr

    torch.device = lambda s: _TorchDevice(s)

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    torch.no_grad = _NoGrad
    torch.from_numpy = lambda arr: types.SimpleNamespace(
        pin_memory=lambda: types.SimpleNamespace(numpy=lambda: arr)
    )

    class _Autocast:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    torch.amp = types.SimpleNamespace(autocast=_Autocast)
    sys.modules["torch"] = torch


# ---------------------------------------------------------------------------
# gfpgan stub
# ---------------------------------------------------------------------------

def _make_gfpgan_stub():
    if "gfpgan" in sys.modules:
        return
    gfpgan = types.ModuleType("gfpgan")

    class GFPGANer:
        def __init__(self, **kw):
            pass

        def enhance(self, img, **kw):
            return None, None, img.copy()

    gfpgan.GFPGANer = GFPGANer
    sys.modules["gfpgan"] = gfpgan


# ---------------------------------------------------------------------------
# Register all stubs before any chararep module is imported
# ---------------------------------------------------------------------------

_make_insightface_stub()
_make_onnxruntime_stub()
_make_torch_stub()
_make_gfpgan_stub()
