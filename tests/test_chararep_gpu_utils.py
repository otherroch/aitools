"""Tests for gpu_utils.py."""

import types
import sys

import numpy as np
import pytest

import chararep.gpu_utils as gpu_utils


# ---------------------------------------------------------------------------
# check_gpu
# ---------------------------------------------------------------------------

class TestCheckGpu:
    def test_returns_dict(self):
        info = gpu_utils.check_gpu()
        assert isinstance(info, dict)
        assert "cuda_available" in info

    def test_no_cuda_with_cpu_only_ort(self):
        """Stub only exposes CPUExecutionProvider → cuda_available False."""
        info = gpu_utils.check_gpu()
        # The ort stub only provides CPUExecutionProvider
        assert info["cuda_available"] is False

    def test_onnx_providers_populated(self):
        info = gpu_utils.check_gpu()
        assert "onnx_providers" in info
        assert "CPUExecutionProvider" in info["onnx_providers"]


# ---------------------------------------------------------------------------
# log_gpu_info (smoke test – should not raise)
# ---------------------------------------------------------------------------

class TestLogGpuInfo:
    def test_no_raise(self):
        gpu_utils.log_gpu_info()


# ---------------------------------------------------------------------------
# warmup_cuda (stub torch has no real CUDA)
# ---------------------------------------------------------------------------

class TestWarmupCuda:
    def test_no_raise_without_cuda(self):
        gpu_utils.warmup_cuda(device_id=0)


# ---------------------------------------------------------------------------
# get_onnx_providers
# ---------------------------------------------------------------------------

class TestGetOnnxProviders:
    def test_returns_list(self):
        providers = gpu_utils.get_onnx_providers(0)
        assert isinstance(providers, list)
        assert len(providers) >= 1

    def test_cpu_provider_always_present(self):
        providers = gpu_utils.get_onnx_providers(0)
        provider_names = [
            p[0] if isinstance(p, tuple) else p for p in providers
        ]
        assert "CPUExecutionProvider" in provider_names

    def test_no_cuda_provider_when_not_available(self):
        """With the CPU-only ort stub, CUDA provider should not appear."""
        providers = gpu_utils.get_onnx_providers(0)
        provider_names = [
            p[0] if isinstance(p, tuple) else p for p in providers
        ]
        assert "CUDAExecutionProvider" not in provider_names

    def test_import_error_falls_back_to_cpu(self, monkeypatch):
        """If onnxruntime is not importable, fall back to CPU."""
        original = sys.modules.pop("onnxruntime", None)
        try:
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "onnxruntime":
                    raise ImportError
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", mock_import)
            providers = gpu_utils.get_onnx_providers(0)
            assert providers == [("CPUExecutionProvider", {})]
        finally:
            if original is not None:
                sys.modules["onnxruntime"] = original


# ---------------------------------------------------------------------------
# gpu_mem_info
# ---------------------------------------------------------------------------

class TestGpuMemInfo:
    def test_returns_tuple_of_floats(self):
        used, total = gpu_utils.gpu_mem_info(0)
        assert isinstance(used, float)
        assert isinstance(total, float)

    def test_returns_zeros_without_cuda(self):
        used, total = gpu_utils.gpu_mem_info(0)
        assert used == 0.0


# ---------------------------------------------------------------------------
# pin_numpy_array
# ---------------------------------------------------------------------------

class TestPinNumpyArray:
    def test_returns_array(self):
        arr = np.zeros((10, 10), dtype=np.float32)
        result = gpu_utils.pin_numpy_array(arr)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# torch_inference_context (smoke test)
# ---------------------------------------------------------------------------

class TestTorchInferenceContext:
    def test_context_manager_no_raise(self):
        with gpu_utils.torch_inference_context(use_fp16=False):
            pass

    def test_context_manager_fp16_no_raise(self):
        with gpu_utils.torch_inference_context(use_fp16=True):
            pass
