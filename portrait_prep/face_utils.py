#!/usr/bin/env python3
"""
portrait_prep.face_utils

Shared face-recognition helpers used by both portrait_prep.crop and
vicrop.crop.  Centralises the lazy face_recognition import and the
greedy nearest-neighbour identity-clustering algorithm so that neither
package needs to duplicate them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_RATIO: float = 0.4
DEFAULT_CROP_SIZE: int = 1024


def load_face_recognition():
    """Lazily import the face_recognition package, raising a clear error if absent."""
    try:
        import face_recognition

        return face_recognition
    except ImportError as exc:
        raise ImportError(
            "face_recognition is required for the crop task.\n"
            "Install it with: pip install face_recognition"
        ) from exc


def cluster_faces(
    all_results: list[tuple[Path, np.ndarray]],
    output_dir: Path,
    tolerance: float = 0.6,
    *,
    fr=None,
) -> dict[int, list[Path]]:
    """Group saved face crops by identity using greedy nearest-neighbour clustering.

    Each unique identity is assigned an integer label starting from 1 and its
    crops are moved into ``output_dir/person_NN/``.

    Args:
        all_results: List of ``(face_image_path, face_encoding)`` tuples.
        output_dir:  Root directory; ``person_NN`` sub-folders are created here.
        tolerance:   Maximum face-distance for two faces to be the same identity.
        fr:          Pre-loaded face_recognition module.  Loaded automatically
                     when *None* (useful for callers that already have it loaded,
                     and makes unit-test patching easier).

    Returns:
        Mapping ``{person_id: [list of face image paths]}``.
    """
    if fr is None:
        fr = load_face_recognition()

    known_encodings: list[np.ndarray] = []
    known_labels: list[int] = []
    next_label = 1
    label_map: dict[Path, int] = {}

    for face_path, encoding in all_results:
        if not known_encodings:
            known_encodings.append(encoding)
            known_labels.append(next_label)
            label_map[face_path] = next_label
            next_label += 1
            continue

        distances = fr.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= tolerance:
            label_map[face_path] = known_labels[best_idx]
        else:
            known_encodings.append(encoding)
            known_labels.append(next_label)
            label_map[face_path] = next_label
            next_label += 1

    # Re-organise: move each crop into output_dir/person_N/
    person_dirs: dict[int, list[Path]] = {}
    for face_path, label in label_map.items():
        person_dir = output_dir / f"person_{label:02d}"
        person_dir.mkdir(parents=True, exist_ok=True)
        dest = person_dir / face_path.name
        face_path.rename(dest)
        person_dirs.setdefault(label, []).append(dest)

    return person_dirs
