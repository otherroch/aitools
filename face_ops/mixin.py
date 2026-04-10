"""
face_ops.mixin

Default implementations for high-level ``FaceBackend`` operations.

:class:`FaceBackendMixin` provides :meth:`cluster_faces` and
:meth:`load_reference_encodings` as concrete methods that call only the
abstract primitives (``detect_faces``, ``encode_faces``, ``face_distance``,
``load_image``).  Both :class:`DlibBackend` and :class:`InsightFaceBackend`
inherit from this mixin.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from face_ops.types import Encoding

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS: set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif",
}


class FaceBackendMixin:
    """Default implementations for high-level face operations.

    Concrete backends inherit from this mixin to gain
    :meth:`load_reference_encodings` and :meth:`cluster_faces` without
    having to implement them from scratch.
    """

    def load_reference_encodings(
        self,
        classified_path: Path,
        *,
        model: str = "hog",
        max_per_identity: int = 0,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Load face encodings from a pre-classified reference directory.

        The directory is expected to contain identity sub-folders (e.g.
        ``alice/``, ``bob/``, or ``person_01/``), each holding one or more
        reference face images.

        Args:
            classified_path:  Root directory containing identity sub-folders.
            model:            Backend-specific detection model hint.
            max_per_identity: Maximum number of reference encodings to load per
                              identity folder.  ``0`` (the default) means no limit.

        Returns:
            ``(encodings, names)`` — parallel lists where *names[i]* is the
            sub-folder name that encoding *i* belongs to.
        """
        encodings: list[np.ndarray] = []
        names: list[str] = []

        identity_dirs = sorted(
            p for p in classified_path.iterdir() if p.is_dir()
        )

        for identity_dir in identity_dirs:
            ref_images = sorted(
                p for p in identity_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
            )

            loaded_for_identity = 0
            for img_path in ref_images:
                if max_per_identity > 0 and loaded_for_identity >= max_per_identity:
                    break
                image = self.load_image(str(img_path))  # type: ignore[attr-defined]
                locs = self.detect_faces(image, model=model)  # type: ignore[attr-defined]
                face_encs = self.encode_faces(image, locs)  # type: ignore[attr-defined]
                if face_encs:
                    encodings.append(face_encs[0])
                    names.append(identity_dir.name)
                    loaded_for_identity += 1
                    logger.debug(
                        "Loaded reference encoding from %s (%s)",
                        img_path.name, identity_dir.name,
                    )

        logger.info(
            "Loaded %d reference encoding(s) for %d identity(ies) from %s",
            len(encodings), len(identity_dirs), classified_path,
        )
        return encodings, names

    def cluster_faces(
        self,
        all_results: list[tuple[Path, np.ndarray]],
        output_dir: Path,
        tolerance: float = 0.6,
        *,
        reference_encodings: list[np.ndarray] | None = None,
        reference_names: list[str] | None = None,
    ) -> dict[str, list[Path]]:
        """Group saved face crops by identity using greedy nearest-neighbour clustering.

        Each unique identity is assigned a string label and its crops are moved
        into ``output_dir/<label>/``.

        When *reference_encodings* and *reference_names* are provided (from a
        pre-classified directory), new faces are first compared against these
        known identities.  Matching faces are placed in a folder with the
        original reference name.  Faces that do not match any reference get
        auto-generated ``person_NN`` folder names.

        Args:
            all_results:         List of ``(face_image_path, face_encoding)`` tuples.
            output_dir:          Root directory; identity sub-folders are created here.
            tolerance:           Maximum face-distance for two faces to be the same
                                 identity.
            reference_encodings: Encodings pre-loaded from a classified directory.
            reference_names:     Folder name for each reference encoding.

        Returns:
            Mapping ``{identity_label: [list of face image paths]}``.
        """
        known_encodings: list[np.ndarray] = list(reference_encodings or [])
        known_labels: list[str] = list(reference_names or [])

        # Determine starting number for auto-generated person_NN labels.
        next_person_num = 1
        for name in known_labels:
            if name.startswith("person_"):
                try:
                    num = int(name.split("_", 1)[1])
                    next_person_num = max(next_person_num, num + 1)
                except (IndexError, ValueError):
                    pass

        label_map: dict[Path, str] = {}

        for face_path, encoding in all_results:
            if not known_encodings:
                label = f"person_{next_person_num:02d}"
                known_encodings.append(encoding)
                known_labels.append(label)
                label_map[face_path] = label
                next_person_num += 1
                continue

            distances = self.face_distance(known_encodings, encoding)  # type: ignore[attr-defined]
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= tolerance:
                label_map[face_path] = known_labels[best_idx]
            else:
                label = f"person_{next_person_num:02d}"
                known_encodings.append(encoding)
                known_labels.append(label)
                label_map[face_path] = label
                next_person_num += 1

        # Re-organise: move each crop into output_dir/<label>/
        person_dirs: dict[str, list[Path]] = {}
        for face_path, label in label_map.items():
            person_dir = output_dir / label
            person_dir.mkdir(parents=True, exist_ok=True)
            dest = person_dir / face_path.name
            face_path.rename(dest)
            person_dirs.setdefault(label, []).append(dest)

        return person_dirs
