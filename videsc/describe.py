#!/usr/bin/env python3
"""
videsc.describe

Generate AI-powered text descriptions for video files.

Key frames are extracted from each video using OpenCV.  Each frame is run
through the WD14 ONNX tagger; the resulting tag probabilities are aggregated
across all frames (union of tags, ranked by mean confidence) and written to a
companion ``.txt`` file next to the video (or in a specified output directory).

Dependencies:
    pip install onnxruntime huggingface_hub opencv-python pillow numpy

The first run will download the WD14 model from HuggingFace (~350 MB) and
cache it under ``~/.cache/huggingface/``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS: set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv",
}

DEFAULT_EVERY_N_FRAMES: int = 30
DEFAULT_MAX_FRAMES: int = 10
DEFAULT_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
DEFAULT_THRESHOLD = 0.35
MODEL_INPUT_SIZE = 448  # WD14 models expect 448×448


def extract_keyframes(
    video_path: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[np.ndarray]:
    """Extract up to *max_frames* evenly-spaced frames from *video_path*.

    Args:
        video_path: Path to the video file.
        every_n:    Sample one frame every N frames.
        max_frames: Maximum number of frames to return.

    Returns:
        List of RGB numpy arrays shaped ``(H, W, 3)``.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return []

    frames: list[np.ndarray] = []
    frame_idx = 0

    try:
        while len(frames) < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_idx += 1
    finally:
        cap.release()

    return frames


def _download_model(model_repo: str) -> tuple[Path, Path]:
    """Return ``(model_path, tags_csv_path)``, downloading from HF if needed."""
    from huggingface_hub import hf_hub_download

    model_path = Path(hf_hub_download(repo_id=model_repo, filename="model.onnx"))
    tags_path = Path(hf_hub_download(repo_id=model_repo, filename="selected_tags.csv"))
    return model_path, tags_path


def _load_labels(tags_csv: Path) -> tuple[list[str], list[int], list[int]]:
    """Parse WD14 tags CSV.

    Returns:
        ``(tag_names, rating_indices, general_indices)``
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


def _preprocess_frame(frame_rgb: np.ndarray, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Prepare an RGB frame array as a ``(1, size, size, 3)`` float32 WD14 input."""
    pil_img = Image.fromarray(frame_rgb).convert("RGBA")
    background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    background.paste(pil_img, mask=pil_img.split()[3])
    pil_img = background.convert("RGB")

    w, h = pil_img.size
    max_dim = max(w, h)
    pad_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    pad_img.paste(pil_img, ((max_dim - w) // 2, (max_dim - h) // 2))
    img_resized = pad_img.resize((size, size), Image.LANCZOS)

    arr = np.array(img_resized, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB → BGR (WD14 convention)
    arr = np.expand_dims(arr, axis=0)
    return arr


def describe_video(
    video_path: Path,
    output_dir: Path | None = None,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    prefix: str = "",
    threshold: float = DEFAULT_THRESHOLD,
    model_repo: str = DEFAULT_MODEL_REPO,
    include_ratings: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate a WD14-based text description for a single video file.

    Key frames are extracted and tagged individually; tags are aggregated
    (union across all frames, ranked by mean confidence) and written to a
    ``.txt`` file alongside the video (or in *output_dir* if given).

    Args:
        video_path:      Path to the input video.
        output_dir:      Where to write the ``.txt`` file.  Defaults to the
                         same directory as the video.
        every_n:         Extract one frame every N frames.
        max_frames:      Maximum key frames to process per video.
        prefix:          Token(s) prepended to the description, e.g. ``"ohwx man"``.
        threshold:       Minimum WD14 tag confidence to include (default: 0.35).
        model_repo:      HuggingFace repo ID for the WD14 ONNX model.
        include_ratings: Include rating tags (safe / questionable / explicit).
        skip_existing:   Skip the video if its ``.txt`` file already exists.

    Returns:
        Dict with keys ``described`` (0 or 1) and ``skipped`` (0 or 1).
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for videsc.\n"
            "Install with: pip install onnxruntime"
        ) from exc

    return _describe_video_impl(
        ort=ort,
        video_path=video_path,
        output_dir=output_dir,
        every_n=every_n,
        max_frames=max_frames,
        prefix=prefix,
        threshold=threshold,
        model_repo=model_repo,
        include_ratings=include_ratings,
        skip_existing=skip_existing,
    )


def _describe_video_impl(
    ort,
    video_path: Path,
    output_dir: Path | None,
    every_n: int,
    max_frames: int,
    prefix: str,
    threshold: float,
    model_repo: str,
    include_ratings: bool,
    skip_existing: bool,
) -> dict[str, int]:
    """Internal implementation of :func:`describe_video` (accepts injected ort)."""
    txt_dir = output_dir.resolve() if output_dir is not None else video_path.parent
    txt_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / (video_path.stem + ".txt")

    if skip_existing and txt_path.exists():
        logger.debug("Skipping (exists): %s", txt_path)
        return {"described": 0, "skipped": 1}

    frames = extract_keyframes(video_path, every_n=every_n, max_frames=max_frames)
    if not frames:
        logger.warning("No frames extracted from %s", video_path.name)
        return {"described": 0, "skipped": 1}

    logger.info("Downloading / loading WD14 model from %s …", model_repo)
    model_path, tags_csv = _download_model(model_repo)
    tag_names, rating_indices, general_indices = _load_labels(tags_csv)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    caption = _build_caption(
        session=session,
        frames=frames,
        tag_names=tag_names,
        general_indices=general_indices,
        rating_indices=rating_indices,
        threshold=threshold,
        prefix=prefix,
        include_ratings=include_ratings,
    )

    txt_path.write_text(caption, encoding="utf-8")
    logger.info("Described: %s → %s", video_path.name, txt_path)
    return {"described": 1, "skipped": 0}


def _build_caption(
    session,
    frames: list[np.ndarray],
    tag_names: list[str],
    general_indices: list[int],
    rating_indices: list[int],
    threshold: float,
    prefix: str,
    include_ratings: bool,
) -> str:
    """Run WD14 on each frame and return an aggregated caption string."""
    input_name = session.get_inputs()[0].name
    tag_scores: dict[str, list[float]] = {}

    for frame_rgb in frames:
        arr = _preprocess_frame(frame_rgb)
        probs: np.ndarray = session.run(None, {input_name: arr})[0][0]

        for idx in general_indices:
            if probs[idx] >= threshold:
                tag = tag_names[idx].replace("_", " ")
                tag_scores.setdefault(tag, []).append(float(probs[idx]))

        if include_ratings:
            for idx in rating_indices:
                if probs[idx] >= threshold:
                    tag = tag_names[idx].replace("_", " ")
                    tag_scores.setdefault(tag, []).append(float(probs[idx]))

    ranked = sorted(
        ((sum(scores) / len(scores), tag) for tag, scores in tag_scores.items()),
        reverse=True,
    )
    tag_strs = [tag for _, tag in ranked]
    parts = ([prefix.strip()] if prefix.strip() else []) + tag_strs
    return ", ".join(parts)


def describe_folder(
    input_dir: Path,
    output_dir: Path | None = None,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    prefix: str = "",
    threshold: float = DEFAULT_THRESHOLD,
    model_repo: str = DEFAULT_MODEL_REPO,
    include_ratings: bool = False,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate descriptions for all video files in *input_dir*.

    The WD14 model is loaded once and reused across all videos for efficiency.

    Args:
        input_dir:       Source directory (searched recursively for videos).
        output_dir:      Where to write ``.txt`` files.  Defaults to alongside
                         each video file.
        every_n:         Extract one frame every N frames.
        max_frames:      Maximum key frames to process per video.
        prefix:          Token(s) prepended to each description.
        threshold:       Minimum WD14 tag confidence to include.
        model_repo:      HuggingFace repo ID for the WD14 ONNX model.
        include_ratings: Include rating tags in output.
        skip_existing:   Skip videos whose ``.txt`` file already exists.

    Returns:
        Summary dict with keys ``described`` and ``skipped``.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for videsc.\n"
            "Install with: pip install onnxruntime"
        ) from exc

    return _describe_folder_impl(
        ort=ort,
        input_dir=input_dir,
        output_dir=output_dir,
        every_n=every_n,
        max_frames=max_frames,
        prefix=prefix,
        threshold=threshold,
        model_repo=model_repo,
        include_ratings=include_ratings,
        skip_existing=skip_existing,
    )


def _describe_folder_impl(
    ort,
    input_dir: Path,
    output_dir: Path | None,
    every_n: int,
    max_frames: int,
    prefix: str,
    threshold: float,
    model_repo: str,
    include_ratings: bool,
    skip_existing: bool,
) -> dict[str, int]:
    """Internal implementation of :func:`describe_folder` (accepts injected ort)."""
    input_dir = input_dir.resolve()
    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ]

    if not videos:
        logger.warning("No video files found in %s", input_dir)
        return {"described": 0, "skipped": 0}

    logger.info("Downloading / loading WD14 model from %s …", model_repo)
    model_path, tags_csv = _download_model(model_repo)
    tag_names, rating_indices, general_indices = _load_labels(tags_csv)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    described = 0
    skipped = 0

    for video_path in videos:
        if output_dir is not None:
            rel = video_path.relative_to(input_dir)
            txt_dir = output_dir / rel.parent
            txt_dir.mkdir(parents=True, exist_ok=True)
        else:
            txt_dir = video_path.parent

        txt_path = txt_dir / (video_path.stem + ".txt")

        if skip_existing and txt_path.exists():
            skipped += 1
            continue

        frames = extract_keyframes(video_path, every_n=every_n, max_frames=max_frames)
        if not frames:
            logger.warning("No frames extracted from %s", video_path.name)
            skipped += 1
            continue

        caption = _build_caption(
            session=session,
            frames=frames,
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=rating_indices,
            threshold=threshold,
            prefix=prefix,
            include_ratings=include_ratings,
        )

        txt_path.write_text(caption, encoding="utf-8")
        logger.info("Described: %s", video_path.name)
        described += 1

    return {"described": described, "skipped": skipped}
