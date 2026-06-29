#!/usr/bin/env python3
"""
vicrop.crop

Extract face-cropped PNG frames from video files.
"""

from __future__ import annotations

import cv2
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from face_ops.backend import FaceBackend


logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
DEFAULT_EVERY_N_FRAMES = 30
DEFAULT_MARGIN_RATIO = 0.4
DEFAULT_CROP_SIZE = 1024


def score_reference_quality(face_crop: np.ndarray) -> float:
    """
    A simple heuristic to score the quality of a face crop.
    In a real implementation, this might use a facial landmark detector
    or a dedicated quality model.
    """
    if face_crop.size == 0:
        return 0.0
    # For now, just return a dummy score.
    return 0.8


def crop_video(
    video_path: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    crop_dim: tuple[int, int] | None = None,
    extract_only: bool = False,
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = False,
    ref_thresh: float = 0.65,
    classified_path: Path | None = None,
    classified_max: int = 10,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """
    Extract face-cropped PNG frames from a video file.
    """
    if backend is None:
        from face_ops import backend_for_model
        backend = backend_for_model("hog")

    output_dir = output_dir.resolve()
    stem_dir = output_dir / video_path.stem

    if skip_existing and stem_dir.exists() and any(stem_dir.rglob("*.png")):
        logger.info("Skipping existing: %s", video_path.name)
        return {"videos_processed": 0, "frames_processed": 0, "faces": 0, "persons": 0, "ref_photos": 0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return {"videos_processed": 0, "frames_processed": 0, "faces": 0, "persons": 0, "ref_photos": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "crop_video: %s (fps=%.2f, frames=%d, %dx%d)",
        video_path.name, fps, total_frames, width, height,
    )

    stem_dir.mkdir(parents=True, exist_ok=True)

    faces_data: list[tuple[int, np.ndarray, tuple[int, int, int, int], np.ndarray]] = []
    
    frame_idx = 0
    sampled_frames_count = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            if frame_idx % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                locs = backend.detect_faces(frame_rgb)
                if len(locs) == 1:
                    encs = backend.encode_faces(frame_rgb, locs)
                    if encs:
                        bbox = locs[0]
                        encoding = encs[0]
                        faces_data.append((frame_idx, frame_rgb, bbox, encoding))
                sampled_frames_count += 1
            frame_idx += 1
    finally:
        cap.release()

    if not faces_data:
        return {"videos_processed": 1, "frames_processed": sampled_frames_count, "faces": 0, "persons": 0, "ref_photos": 0}

    # 1. Group faces into identities
    # We'll use a simple approach: process faces one by one and cluster.
    
    # For simplicity, let's use the backend's cluster_faces if available, 
    # but since we don't know the exact signature of backend.cluster_faces, 
    # let's implement a basic greedy clustering here or assume the backend can do it.
    
    # Looking at tests, backend.cluster_faces(list_of_tuples, output_dir, ...)
    # where tuples are (path, encoding)
    
    # First, we need to save the raw face crops to a temporary staging area
    staging_dir = stem_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    face_files: list[tuple[Path, np.ndarray]] = []
    
    for idx, (f_idx, f_rgb, bbox, enc) in enumerate(faces_data):
        # Compute crop rect
        top, right, bottom, left = bbox
        # Add margin
        face_h = max(1, bottom - top)
        face_w = max(1, right - left)
        margin_h = int(face_h * margin_ratio)
        margin_w = int(face_w * margin_ratio)
        
        crop_top = max(0, top - margin_h)
        crop_bottom = min(height, bottom + margin_h)
        crop_left = max(0, left - margin_w)
        crop_right = min(width, right + margin_w)
        
        cropped = f_rgb[crop_top:crop_bottom, crop_left:crop_right]
        
        # Resize to crop_size if needed
        if crop_dim:
            out_w, out_h = crop_dim
        else:
            out_w, out_h = crop_size, crop_size
            
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            cropped_resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to PIL to save as PNG
            img = Image.fromarray(cropped_resized)
            face_path = staging_dir / f"face_{idx:06d}.png"
            img.save(face_path)
            face_files.append((face_path, enc))

    # 2. Cluster faces
    if classify and not extract_only:
        # Get reference encodings if any
        ref_encs = []
        ref_names = []
        if classified_path and classified_path.exists():
            # Very simplified: load first N images from each subfolder
            for person_dir in sorted(classified_path.iterdir()):
                if person_dir.is_dir():
                    images = list(person_dir.glob("*.png"))[:classified_max]
                    for img_path in images:
                        img_bgr = cv2.imread(str(img_path))
                        if img_bgr is not None:
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            encs = backend.encode_faces(img_rgb, [(0, 0, img_bgr.shape[0], img_bgr.shape[1])])
                            if encs:
                                ref_encs.append(encs[0])
                                ref_names.append(person_dir.name)

        # Cluster
        person_map = backend.cluster_faces(
            face_files, 
            stem_dir, 
            tolerance=tolerance,
            reference_encodings=ref_encs,
            reference_names=ref_names
        )
        # person_map is a dict: {face_path: person_name}
        
        # Move faces to person folders
        for person_name, face_paths in person_map.items():
            person_dir = stem_dir / person_name
            person_dir.mkdir(parents=True, exist_ok=True)
            for face_path in face_paths:
                target_path = person_dir / face_path.name
                face_path.replace(target_path)

        # Handle reference photos for clustered faces
        if ref_thresh > 0:
            for person_name, face_paths in person_map.items():
                person_dir = stem_dir / person_name
                for face_path in face_paths:
                    target_path = person_dir / face_path.name
                    if target_path.exists():
                        img_bgr = cv2.imread(str(target_path))
                        if img_bgr is not None:
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            score = score_reference_quality(img_rgb)
                            if score >= ref_thresh:
                                ref_dir = person_dir / "ref"
                                ref_dir.mkdir(parents=True, exist_ok=True)
                                target_path.replace(ref_dir / face_path.name)

    else:
        # extract_only or no classify
        for face_path, _ in face_files:
            person_name = "extracted" if extract_only else "person_01"
            person_dir = stem_dir / person_name
            person_dir.mkdir(parents=True, exist_ok=True)
            target_path = person_dir / face_path.name
            face_path.replace(target_path)

        # Handle reference photos for non-clustered faces
        if ref_thresh > 0:
            for face_path, _ in face_files:
                # We need to find where it was moved to. 
                # In the 'else' block, it's moved to person_dir / face_path.name
                person_name = "extracted" if extract_only else "person_01"
                person_dir = stem_dir / person_name
                target_path = person_dir / face_path.name
                
                if target_path.exists():
                    img_bgr = cv2.imread(str(target_path))
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        score = score_reference_quality(img_rgb)
                        if score >= ref_thresh:
                            ref_dir = person_dir / "ref"
                            ref_dir.mkdir(parents=True, exist_ok=True)
                            target_path.replace(ref_dir / target_path.name)

    # Cleanup staging
    for f, _ in face_files:
        if f.exists():
            f.unlink()
    if staging_dir.exists():
        staging_dir.rmdir()

    # Final stats
    # This is a bit simplified. In a real implementation we'd count properly.
    # Let's try to match the expected stats from tests and cli.py.
    # We need: videos_processed, frames_processed, faces, persons, ref_photos
    
    # For now, let's do a quick scan of the output to get accurate numbers.
    processed_faces = len(face_files)
    persons = len([d for d in stem_dir.iterdir() if d.is_dir() and d.name != "staging"])
    ref_photos = len(list(stem_dir.rglob("ref/*.png")))

    return {
        "videos_processed": 1,
        "frames_processed": sampled_frames_count,
        "faces": processed_faces,
        "persons": persons,
        "ref_photos": ref_photos,
    }


def crop_folder(
    input_dir: Path,
    output_dir: Path,
    every_n: int = DEFAULT_EVERY_N_FRAMES,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    crop_size: int = DEFAULT_CROP_SIZE,
    crop_dim: tuple[int, int] | None = None,
    extract_only: bool = False,
    classify: bool = True,
    tolerance: float = 0.6,
    skip_existing: bool = False,
    ref_thresh: float = 0.65,
    classified_path: Path | None = None,
    classified_max: int = 10,
    backend: "FaceBackend | None" = None,
) -> dict[str, int]:
    """
    Scan a directory for videos and crop faces from each.
    """
    input_dir = input_dir.resolve()
    video_files = set()
    for ext in SUPPORTED_VIDEO_EXTS:
        for path in input_dir.rglob(f"*{ext}"):
            video_files.add(path.resolve())
        for path in input_dir.rglob(f"*{ext.upper()}"):
            video_files.add(path.resolve())

    stats = {
        "videos_processed": 0,
        "frames_processed": 0,
        "faces": 0,
        "persons": 0,
        "ref_photos": 0,
    }

    for video_path in video_files:
        video_stats = crop_video(
            video_path,
            output_dir,
            every_n=every_n,
            margin_ratio=margin_ratio,
            crop_size=crop_size,
            crop_dim=crop_dim,
            extract_only=extract_only,
            classify=classify,
            tolerance=tolerance,
            skip_existing=skip_existing,
            ref_thresh=ref_thresh,
            classified_path=classified_path,
            classified_max=classified_max,
            backend=backend,
        )
        stats["videos_processed"] += video_stats["videos_processed"]
        stats["frames_processed"] += video_stats["frames_processed"]
        stats["faces"] += video_stats["faces"]
        stats["persons"] += video_stats["persons"]
        stats["ref_photos"] += video_stats["ref_photos"]

    return stats