#!/usr/bin/env python3
"""
caption.py

Task 3 – WD14 tagger captioning for portrait images.

Uses the wd14-tagger (SmilingWolf/wd-v1-4-*) ONNX models via the
`wd14-tagger` / `onnxruntime` stack.  For each image a companion ``.txt``
file is written with:

    <prefix>, <tag1>, <tag2>, ...

The tags are sorted by confidence, filtered by a threshold, and the
character / rating tags can optionally be suppressed.

Dependencies:
    pip install onnxruntime huggingface_hub pillow numpy tqdm

The first run will download the model from HuggingFace (~350 MB) and cache it
under ``~/.cache/huggingface/``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Default WD14 model hosted on HuggingFace
DEFAULT_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
DEFAULT_THRESHOLD = 0.35
MODEL_INPUT_SIZE = 448  # WD14 models expect 448×448


def _download_model(model_repo: str) -> tuple[Path, Path]:
    """Return (model_path, tags_csv_path) downloading from HF if needed."""
    from huggingface_hub import hf_hub_download

    model_path = Path(
        hf_hub_download(repo_id=model_repo, filename="model.onnx")
    )
    tags_path = Path(
        hf_hub_download(repo_id=model_repo, filename="selected_tags.csv")
    )
    return model_path, tags_path


def _load_labels(tags_csv: Path) -> tuple[list[str], list[int], list[int]]:
    """Parse WD14 tags CSV.

    Returns:
        (tag_names, rating_indices, general_indices)
    """
    tag_names: list[str] = []
    rating_indices: list[int] = []
    general_indices: list[int] = []

    with open(tags_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tag_names.append(row["name"])
            cat = int(row.get("category", 0))
            if cat == 9:
                rating_indices.append(i)
            elif cat == 0:
                general_indices.append(i)

    return tag_names, rating_indices, general_indices


def _preprocess(image_path: Path, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Load and prepare an image as a (1, size, size, 3) float32 array."""
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        # Paste onto white background (removes alpha transparency)
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background.convert("RGB")

        # Pad to square
        w, h = img.size
        max_dim = max(w, h)
        pad_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        pad_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

        img_resized = pad_img.resize((size, size), Image.LANCZOS)

    arr = np.array(img_resized, dtype=np.float32)
    # WD14 expects BGR
    arr = arr[:, :, ::-1]
    arr = np.expand_dims(arr, axis=0)
    return arr


def caption_image(
    image_path: Path,
    session,  # onnxruntime.InferenceSession
    tag_names: list[str],
    general_indices: list[int],
    threshold: float = DEFAULT_THRESHOLD,
    prefix: str = "",
    include_ratings: bool = False,
    rating_indices: list[int] | None = None,
) -> str:
    """Run WD14 inference on a single image and return a caption string."""
    logger.debug("caption_image: running inference on %s  threshold=%.2f", image_path.name, threshold)
    input_arr = _preprocess(image_path)
    input_name = session.get_inputs()[0].name
    probs: np.ndarray = session.run(None, {input_name: input_arr})[0][0]

    tags: list[tuple[float, str]] = []
    for idx in general_indices:
        if probs[idx] >= threshold:
            tags.append((float(probs[idx]), tag_names[idx].replace("_", " ")))

    if include_ratings and rating_indices:
        for idx in rating_indices:
            if probs[idx] >= threshold:
                tags.append((float(probs[idx]), tag_names[idx].replace("_", " ")))

    # Sort by confidence descending
    tags.sort(key=lambda t: t[0], reverse=True)
    tag_strs = [t[1] for t in tags]

    logger.debug("caption_image: %s  found %d tag(s) above threshold", image_path.name, len(tags))
    parts = ([prefix.strip()] if prefix.strip() else []) + tag_strs
    return ", ".join(parts)


def caption_folder(
    input_dir: Path,
    output_dir: Path | None = None,
    prefix: str = "",
    threshold: float = DEFAULT_THRESHOLD,
    model_repo: str = DEFAULT_MODEL_REPO,
    include_ratings: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate WD14 caption ``.txt`` files for every image in *input_dir*.

    Args:
        input_dir:       Directory with images (searched recursively).
        output_dir:      Where to write ``.txt`` files; defaults to same folder
                         as each image (i.e. alongside the source files).
        prefix:          Token(s) prepended to every caption, e.g. ``"ohwx man"``.
        threshold:       Minimum tag confidence to include.
        model_repo:      HuggingFace repo ID for the WD14 ONNX model.
        include_ratings: Include rating tags (safe / questionable / explicit).
        skip_existing:   Skip images whose caption ``.txt`` already exists.

    Returns:
        Summary dict with keys ``captioned`` and ``skipped``.
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for captioning.\n"
            "Install with: pip install onnxruntime"
        ) from exc

    return _caption_folder_impl(
        ort=ort,
        input_dir=input_dir,
        output_dir=output_dir,
        prefix=prefix,
        threshold=threshold,
        model_repo=model_repo,
        include_ratings=include_ratings,
        skip_existing=skip_existing,
    )


def _caption_folder_impl(
    ort,
    input_dir: Path,
    output_dir: Path | None,
    prefix: str,
    threshold: float,
    model_repo: str,
    include_ratings: bool,
    skip_existing: bool,
) -> dict[str, int]:
    """Internal implementation of caption_folder (accepts injected ort)."""
    input_dir = input_dir.resolve()
    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(
        "_caption_folder_impl: input=%s output=%s prefix=%r threshold=%.2f skip_existing=%s",
        input_dir, output_dir, prefix, threshold, skip_existing,
    )

    images = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]

    if not images:
        logger.warning("No images found in %s", input_dir)
        return {"captioned": 0, "skipped": 0}

    logger.debug("caption_folder: found %d image(s) in %s", len(images), input_dir)
    logger.info("Downloading / loading WD14 model from %s …", model_repo)
    model_path, tags_csv = _download_model(model_repo)
    tag_names, rating_indices, general_indices = _load_labels(tags_csv)

    logger.debug(
        "caption: model loaded  tags=%d  general=%d  ratings=%d",
        len(tag_names), len(general_indices), len(rating_indices),
    )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    captioned = 0
    skipped = 0

    for img_path in images:
        # Determine where to write the txt file
        if output_dir is not None:
            rel = img_path.relative_to(input_dir)
            txt_dir = output_dir / rel.parent
            txt_dir.mkdir(parents=True, exist_ok=True)
        else:
            txt_dir = img_path.parent

        txt_path = txt_dir / (img_path.stem + ".txt")

        if skip_existing and txt_path.exists():
            logger.debug("caption: skipping (exists): %s", txt_path)
            skipped += 1
            continue

        try:
            caption = caption_image(
                img_path,
                session,
                tag_names,
                general_indices,
                threshold=threshold,
                prefix=prefix,
                include_ratings=include_ratings,
                rating_indices=rating_indices,
            )
            txt_path.write_text(caption, encoding="utf-8")
            logger.info("Captioned: %s", img_path.name)
            logger.debug("caption: wrote %d char(s) to %s", len(caption), txt_path)
            captioned += 1
        except Exception as exc:
            logger.error("Failed to caption %s: %s", img_path, exc)
            skipped += 1

    return {"captioned": captioned, "skipped": skipped}
