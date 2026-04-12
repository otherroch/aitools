"""GPU / CUDA utilities for the character replacement pipeline.

Provides GPU diagnostics, ONNX Runtime provider configuration,
CUDA warm-up routines, and FP16/AMP helpers optimised for RTX 5090.
"""

import logging
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)


def check_gpu() -> dict:
    """Return information about the available GPU and providers."""
    info: dict = {"cuda_available": False, "gpu_name": None, "vram_gb": None}

    # ── ONNX Runtime GPU ─────────────────────────────────────────────────
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        info["onnx_providers"] = providers
        info["cuda_available"] = "CUDAExecutionProvider" in providers
    except ImportError:
        info["onnx_providers"] = []

    # ── PyTorch GPU (optional, used by GFPGAN) ───────────────────────────
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(vram_bytes / (1024**3), 1)
            info["torch_cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return info


def log_gpu_info() -> None:
    """Log GPU diagnostics at startup."""
    info = check_gpu()
    if info["cuda_available"]:
        logger.info(
            "GPU detected: %s  |  VRAM: %s GB",
            info.get("gpu_name", "unknown"),
            info.get("vram_gb", "?"),
        )
        logger.info("ONNX providers: %s", info.get("onnx_providers", []))
    else:
        logger.warning(
            "No CUDA GPU detected – pipeline will fall back to CPU "
            "(expect very slow performance)."
        )


def warmup_cuda(device_id: int = 0) -> None:
    """Pre-warm PyTorch CUDA context, cuDNN and caching allocator.

    Calling this once at startup avoids first-frame latency spikes.
    cuDNN benchmark mode is enabled so the fastest convolution algorithm
    is selected automatically (especially beneficial on Tensor Core GPUs
    like the RTX 5090).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return

        torch.cuda.set_device(device_id)

        # Enable cuDNN auto-tuner (recommended by the research doc)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Force-initialise the CUDA context with a small allocation
        _ = torch.zeros(1, device=f"cuda:{device_id}")
        torch.cuda.synchronize(device_id)
        logger.info("CUDA context warm-up complete (device %d).", device_id)
    except ImportError:
        pass
    except RuntimeError as exc:
        logger.warning("CUDA warm-up failed: %s", exc)


def get_onnx_providers(device_id: int = 0) -> list:
    """Return the ordered list of ONNX Runtime execution providers.

    For CUDAExecutionProvider, uses exhaustive cuDNN search to find the
    fastest convolution algorithm on first run (cached for subsequent calls).
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except ImportError:
        return [("CPUExecutionProvider", {})]

    logger.debug("Available ONNX providers: %s", available)
    providers: list = []
    if "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                },
            )
        )
    providers.append(("CPUExecutionProvider", {}))
    logger.debug("Selected ONNX providers: %s", providers)
    return providers


@contextmanager
def torch_inference_context(use_fp16: bool = True):
    """Context manager for PyTorch inference with optional AMP."""
    import torch

    with torch.no_grad():
        if use_fp16 and torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                yield
        else:
            yield


def pin_numpy_array(arr: np.ndarray) -> np.ndarray:
    """Pin a NumPy array in page-locked memory for faster H2D transfer.

    Returns the original array unchanged if PyTorch is not available.
    """
    try:
        import torch

        t = torch.from_numpy(arr)
        t = t.pin_memory()
        return t.numpy()
    except (ImportError, RuntimeError):
        return arr


def gpu_mem_info(device_id: int = 0) -> tuple[float, float]:
    """Return (used_gb, total_gb) for the given CUDA device."""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            return round(allocated / (1024**3), 2), round(total / (1024**3), 2)
    except ImportError:
        pass
    return 0.0, 0.0
