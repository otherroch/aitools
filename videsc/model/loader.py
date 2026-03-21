import os
from datetime import datetime
from typing import Optional

import torch
from torchsummary import summary  # unused but kept if you rely on it elsewhere

from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Qwen3_5ForConditionalGeneration,
)

from videsc.config import model_dir


# Shared model / processor for threaded batch mode
_SHARED_MODEL = None
_SHARED_PROCESSOR = None


def _quant_config(quant: str) -> Optional[BitsAndBytesConfig]:
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True)
    return None


def _maybe_set_reader(reader: str):
    # Qwen video loader honors this env var
    if reader != "auto":
        os.environ["FORCE_QWENVL_VIDEO_READER"] = reader


def load_model_and_processor(args):
    """
    Load Qwen3-VL model and processor once and reuse across videos.
    In threaded batch mode, this lets all threads share the same CUDA weights.
    """
    global _SHARED_MODEL, _SHARED_PROCESSOR

    if _SHARED_MODEL is not None and _SHARED_PROCESSOR is not None:
        return _SHARED_MODEL, _SHARED_PROCESSOR

    # Resolve model path
    if getattr(args, "model_hf", False):
        model_path_local = args.model
    elif getattr(args, "model_full", False):
        model_path_local = args.model
    else:
        model_path_local = model_dir + args.model

    print("model_path=", model_path_local)

    # Optional CPU thread limiting
    if getattr(args, "half_cpu", False):
        cpu_count = os.cpu_count()
        print(f"Number of CPU cores in the system: {cpu_count}")
        half_cpu_count = cpu_count // 2
        os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
        torch.set_num_threads(half_cpu_count)

    # Log start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"✅ start time (model load): {current_time}")

    # Quantization + reader
    quant_cfg = _quant_config(args.quant)
    _maybe_set_reader(args.reader)

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path_local,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation=args.attn,
        quantization_config=quant_cfg,
    )

    if args.optimize:
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            backend="inductor",
        )

    # Load processor with pixel limits
    processor = AutoProcessor.from_pretrained(
        model_path_local,
        min_pixels=args.min_pixels * 32 * 32,
        max_pixels=args.max_pixels * 32 * 32,
    )

    print("model loaded")

    _SHARED_MODEL = model
    _SHARED_PROCESSOR = processor
    return model, processor


def load_omni_model_and_processor(args):
    """
    Load Qwen3-Omni model and processor once and reuse across videos.
    In threaded batch mode, this lets all threads share the same CUDA weights.
    """
    global _SHARED_MODEL, _SHARED_PROCESSOR

    if _SHARED_MODEL is not None and _SHARED_PROCESSOR is not None:
        return _SHARED_MODEL, _SHARED_PROCESSOR

    # Resolve model path
    if getattr(args, "model_hf", False):
        model_path_local = args.model
    elif getattr(args, "model_full", False):
        model_path_local = args.model
    else:
        model_path_local = model_dir + args.model

    print("model_path=", model_path_local)

    # Optional CPU thread limiting
    if getattr(args, "half_cpu", False):
        cpu_count = os.cpu_count()
        print(f"Number of CPU cores in the system: {cpu_count}")
        half_cpu_count = cpu_count // 2
        os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
        torch.set_num_threads(half_cpu_count)

    # Log start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"✅ start time (model load): {current_time}")

    # Quantization + reader
    quant_cfg = _quant_config(args.quant)
    _maybe_set_reader(args.reader)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path_local,
        device_map="auto",
        dtype="auto",
    )

    if args.optimize:
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            backend="inductor",
        )

    model.disable_talker()

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path_local)

    print("model loaded")

    _SHARED_MODEL = model
    _SHARED_PROCESSOR = processor
    return model, processor


def load_qwen35_model_and_processor(args):
    """
    Load Qwen3.5 model and processor once and reuse across videos.
    In threaded batch mode, this lets all threads share the same CUDA weights.
    """
    global _SHARED_MODEL, _SHARED_PROCESSOR

    if _SHARED_MODEL is not None and _SHARED_PROCESSOR is not None:
        return _SHARED_MODEL, _SHARED_PROCESSOR

    # Resolve model path
    if getattr(args, "model_hf", False):
        model_path_local = args.model
    elif getattr(args, "model_full", False):
        model_path_local = args.model
    else:
        model_path_local = model_dir + args.model

    print("model_path=", model_path_local)

    # Optional CPU thread limiting
    if getattr(args, "half_cpu", False):
        cpu_count = os.cpu_count()
        print(f"Number of CPU cores in the system: {cpu_count}")
        half_cpu_count = cpu_count // 2
        os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
        torch.set_num_threads(half_cpu_count)

    # Log start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"✅ start time (model load): {current_time}")

    # Quantization + reader
    quant_cfg = _quant_config(args.quant)
    _maybe_set_reader(args.reader)

    # Load model
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path_local,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation=args.attn,
        quantization_config=quant_cfg,
    )

    if args.optimize:
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            backend="inductor",
        )

    # Load processor with pixel limits
    processor = AutoProcessor.from_pretrained(
        model_path_local,
        min_pixels=args.min_pixels * 32 * 32,
        max_pixels=args.max_pixels * 32 * 32,
    )

    print("model loaded")

    _SHARED_MODEL = model
    _SHARED_PROCESSOR = processor
    return model, processor
