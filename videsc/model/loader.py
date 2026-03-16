import os
from datetime import datetime
from typing import Optional

import torch
from torchsummary import summary  # unused but kept if you rely on it elsewhere

from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from videsc.config import model_dir
from videsc.utils.helpers import _is_qwen35_model


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
    Load Qwen3-VL or Qwen3.5 model and processor once and reuse across videos.
    Automatically detects Qwen3.5 models (Qwen3_5ForConditionalGeneration) vs
    Qwen3-VL models (Qwen3VLForConditionalGeneration) from the model path.
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

    # Select model class: Qwen3.5 uses Qwen3_5ForConditionalGeneration,
    # which extends Qwen3VLForConditionalGeneration with a hybrid
    # (Gated Delta Network + full-attention) text backbone.
    is_qwen35 = _is_qwen35_model(model_path_local)
    if is_qwen35:
        print("Detected Qwen3.5 model: using Qwen3_5ForConditionalGeneration")
        model_cls = Qwen3_5ForConditionalGeneration
    else:
        model_cls = Qwen3VLForConditionalGeneration

    # Load model
    model = model_cls.from_pretrained(
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

    # Pixel limits: Qwen3.5 uses patch_size=16 so pixel counts are scaled
    # by 16×16; Qwen3-VL uses 32×32.
    pixel_factor = 16 * 16 if is_qwen35 else 32 * 32
    processor = AutoProcessor.from_pretrained(
        model_path_local,
        min_pixels=args.min_pixels * pixel_factor,
        max_pixels=args.max_pixels * pixel_factor,
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
