import os
import logging
from datetime import datetime
from typing import Optional

import torch
from torchsummary import summary  # unused but kept if you rely on it elsewhere

from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    AutoModelForMultimodalLM,
    BitsAndBytesConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Qwen3_5ForConditionalGeneration,
)

from videsc.config import model_dir
from videsc.utils.helpers import _patch_size_for_model

logger = logging.getLogger(__name__)

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
        logger.debug("load_model_and_processor: reusing cached model/processor")
        return _SHARED_MODEL, _SHARED_PROCESSOR

    # Resolve model path
    if getattr(args, "model_hf", False):
        model_path_local = args.model
    elif getattr(args, "model_full", False):
        model_path_local = args.model
    else:
        model_path_local = model_dir + args.model



        
    logger.debug("load_model_and_processor: model_path=%s  quant=%s  attn=%s",
                 model_path_local, getattr(args, "quant", None), getattr(args, "attn", None))
    print("model_path=", model_path_local)

    # Optional CPU thread limiting
    if getattr(args, "half_cpu", False):
        cpu_count = os.cpu_count()
        print(f"Number of CPU cores in the system: {cpu_count}")
        half_cpu_count = cpu_count // 2
        logger.debug("load_model_and_processor: limiting to %d CPU threads", half_cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
        torch.set_num_threads(half_cpu_count)

    # Log start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.debug("load_model_and_processor: start time %s", current_time)
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
        logger.debug("load_model_and_processor: compiling model with torch.compile")
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            backend="inductor",
        )

    # Load processor with pixel limits (use model-specific patch size)
    patch = _patch_size_for_model(model_path_local)
    logger.debug(
        "load_model_and_processor: patch=%d  min_pixels=%d  max_pixels=%d",
        patch, args.min_pixels * patch * patch, args.max_pixels * patch * patch,
    )
    processor = AutoProcessor.from_pretrained(
        model_path_local,
        min_pixels=args.min_pixels * patch * patch,
        max_pixels=args.max_pixels * patch * patch,
    )

    logger.debug("load_model_and_processor: model loaded")
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
        logger.debug("load_omni_model_and_processor: reusing cached model/processor")
        return _SHARED_MODEL, _SHARED_PROCESSOR

    # Resolve model path
    if getattr(args, "model_hf", False):
        model_path_local = args.model
    elif getattr(args, "model_full", False):
        model_path_local = args.model
    else:
        model_path_local = model_dir + args.model

    logger.debug("load_omni_model_and_processor: model_path=%s  quant=%s",
                 model_path_local, getattr(args, "quant", None))
    print("model_path=", model_path_local)

    # Optional CPU thread limiting
    if getattr(args, "half_cpu", False):
        cpu_count = os.cpu_count()
        print(f"Number of CPU cores in the system: {cpu_count}")
        half_cpu_count = cpu_count // 2
        logger.debug("load_omni_model_and_processor: limiting to %d CPU threads", half_cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
        torch.set_num_threads(half_cpu_count)

    # Log start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.debug("load_omni_model_and_processor: start time %s", current_time)
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
        logger.debug("load_omni_model_and_processor: compiling model with torch.compile")
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            backend="inductor",
        )

    model.disable_talker()

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path_local)

    logger.debug("load_omni_model_and_processor: model loaded")
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
        logger.debug("load_qwen35_model_and_processor: reusing cached model/processor")
        return _SHARED_MODEL, _SHARED_PROCESSOR

    # Resolve model path
    if getattr(args, "model_hf", False):
        model_path_local = args.model
    elif getattr(args, "model_full", False):
        model_path_local = args.model
    else:
        model_path_local = model_dir + args.model

    logger.debug("load_qwen35_model_and_processor: model_path=%s  quant=%s  attn=%s",
                 model_path_local, getattr(args, "quant", None), getattr(args, "attn", None))
    print("model_path=", model_path_local)

    # Optional CPU thread limiting
    if getattr(args, "half_cpu", False):
        cpu_count = os.cpu_count()
        print(f"Number of CPU cores in the system: {cpu_count}")
        half_cpu_count = cpu_count // 2
        logger.debug("load_qwen35_model_and_processor: limiting to %d CPU threads", half_cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
        torch.set_num_threads(half_cpu_count)

    # Log start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.debug("load_qwen35_model_and_processor: start time %s", current_time)
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
        logger.debug("load_qwen35_model_and_processor: compiling model with torch.compile")
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            backend="inductor",
        )

    # Load processor with pixel limits (use model-specific patch size)
    patch = _patch_size_for_model(model_path_local)
    logger.debug(
        "load_qwen35_model_and_processor: patch=%d  min_pixels=%d  max_pixels=%d",
        patch, args.min_pixels * patch * patch, args.max_pixels * patch * patch,
    )
    processor = AutoProcessor.from_pretrained(
        model_path_local,
        min_pixels=args.min_pixels * patch * patch,
        max_pixels=args.max_pixels * patch * patch,
    )

    logger.debug("load_qwen35_model_and_processor: model loaded")
    print("model loaded")

    _SHARED_MODEL = model
    _SHARED_PROCESSOR = processor
    return model, processor


def load_gemma4_model_and_processor(args):
    """
    Load Gemma 4 model and processor once and reuse across videos.
    Uses AutoModelForImageTextToText which is the standard class for Gemma 4.
    In threaded batch mode, this lets all threads share the same CUDA weights.
    """
    global _SHARED_MODEL, _SHARED_PROCESSOR

    if _SHARED_MODEL is not None and _SHARED_PROCESSOR is not None:
        logger.debug("load_gemma4_model_and_processor: reusing cached model/processor")
        return _SHARED_MODEL, _SHARED_PROCESSOR

    # Resolve model path
    if getattr(args, "model_hf", False):
        model_path_local = args.model
    elif getattr(args, "model_full", False):
        model_path_local = args.model
    else:
        model_path_local = model_dir + args.model

    logger.debug("load_gemma4_model_and_processor: model_path=%s  quant=%s  attn=%s",
                 model_path_local, getattr(args, "quant", None), getattr(args, "attn", None))

    processor_path_local = model_path_local  # For now, processor is in same location as model  
    if args.processor is not None:
        processor_path_local = args.processor
        logger.debug("load_gemma4_model_and_processor: using custom processor path %s", processor_path_local)
        
    # Optional CPU thread limiting
    if getattr(args, "half_cpu", False):
        cpu_count = os.cpu_count() or 2
        print(f"Number of CPU cores in the system: {cpu_count}")
        half_cpu_count = max(1, cpu_count // 2)
        logger.debug("load_gemma4_model_and_processor: limiting to %d CPU threads", half_cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
        os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)
        torch.set_num_threads(half_cpu_count)

    # Log start time
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.debug("load_gemma4_model_and_processor: start time %s", current_time)
    print(f"✅ start time (model load): {current_time}")

    quant_cfg = _quant_config(args.quant)

    torch_dtype = getattr(args, "torch_dtype", "auto")
    if isinstance(torch_dtype, str):
        torch_dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(torch_dtype.lower(), torch_dtype)

    # Load model using AutoModelForMultimodalLM (standard for Gemma 4)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path_local,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation=args.attn,
        quantization_config=quant_cfg,
    )

    if args.optimize:
        logger.debug("load_gemma4_model_and_processor: compiling model with torch.compile")
        model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
            backend="inductor",
        )

    # Gemma 4 processor requires padding_side="left" for batched generation
    processor = AutoProcessor.from_pretrained(
        processor_path_local, 
        padding_side="left",
    )

    logger.debug("load_gemma4_model_and_processor: model loaded")
    print("model loaded")

    _SHARED_MODEL = model
    _SHARED_PROCESSOR = processor
    return model, processor
