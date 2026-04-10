"""Face identity recognition and matching against portrait galleries.

Uses a :class:`~face_ops.FaceBackend` (loaded once by ``FaceDetector``
and shared via its :attr:`backend` property) to build two galleries per
character:

*   **Recognition gallery** (from ``reference_paths``):
    embeddings of the original character in the video, used to
    determine which detected face should be replaced.

*   **Swap gallery** (from ``portrait_paths``):
    backend-specific face objects of the replacement identity, fed to
    the swap engine so it knows what the new face looks like.
"""

import logging

import cv2
import numpy as np

from face_ops.backend import FaceBackend

from .config import CharacterMapping, PipelineConfig
from .face_detector import TrackedFace

# 5-point landmark template for arcface_112_v1 alignment (in 112×112 space).
# Mirrors the constant in face_swapper.py — both must stay in sync.
_ARCFACE_112_V1 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# 5-point ffhq_512 template scaled to 256×256 space.
# Used to pre-warp portrait faces for uniface_256 source input.
_FFHQ_512_256 = np.array(
    [
        [0.37691676, 0.46864664],
        [0.62285697, 0.46912813],
        [0.50123859, 0.61331904],
        [0.39308822, 0.72541100],
        [0.61150205, 0.72490465],
    ],
    dtype=np.float32,
) * 256.0

logger = logging.getLogger(__name__)


class TargetIdentity:
    """Pre-computed identity representation for one character.

    Attributes
    ----------
    label : str
        Descriptive name (e.g. ``"villain"``).
    recognition_embedding : np.ndarray
        Mean ArcFace embedding of the *find* images (the original face
        in the video), used for cosine-similarity matching.
    reference_faces : list
        Backend-specific face objects from the *replace* images.
        The swap engine picks from these at runtime.
    """

    def __init__(
        self,
        label: str,
        recognition_embeddings: list[np.ndarray],
        swap_faces: list,
    ):
        self.label = label
        self.recognition_embedding: np.ndarray = np.mean(
            recognition_embeddings, axis=0
        )
        norm = np.linalg.norm(self.recognition_embedding)
        if norm > 0:
            self.recognition_embedding /= norm
        self.reference_faces = swap_faces


def _bbox_area(bbox) -> float:
    """Compute the area of a (top, right, bottom, left) bounding box."""
    top, right, bottom, left = bbox
    return float(right - left) * float(bottom - top)


class FaceRecognizer:
    """Computes face embeddings and matches detected faces to targets.

    Accepts an existing :class:`FaceBackend` instance (from
    ``FaceDetector``) so that the heavy models are loaded only once.

    On initialisation it encodes reference images (find folder) into
    the recognition gallery and portrait images (replace folder) into
    the swap gallery.
    At runtime, ``identify_faces`` compares video faces against the
    gallery and assigns identity labels.
    """

    def __init__(self, cfg: PipelineConfig, backend: FaceBackend):
        self._cfg = cfg
        self._backend = backend  # shared – DO NOT re-prepare / reload
        self._targets: list[TargetIdentity] = []
        self._build_gallery(cfg.characters)

    # ── gallery construction ─────────────────────────────────────────────

    def _encode_images(
        self, paths: list[str], kind: str, label: str
    ) -> tuple[list[np.ndarray], list]:
        """Encode a list of images and return (embeddings, face_objects)."""
        embeddings: list[np.ndarray] = []
        faces: list = []
        logger.info("looking at %s images for character %s", kind, label)
        if paths is None:
            return [], []

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                logger.warning("Could not read %s image: %s", kind, p)
                continue
            logger.debug("looking at %s image: %s", kind, p)
            detected = self._backend.detect(img)
            if not detected:
                logger.warning("No face detected in %s image: %s", kind, p)
                continue
            best = max(detected, key=lambda d: _bbox_area(d.bbox))
            if best.embedding is not None:
                embeddings.append(
                    np.array(best.embedding, dtype=np.float32)
                )
                # Pre-warp the portrait to 112×112 so the SimSwap ArcFace
                # image encoder can use it directly without needing the
                # original image again at swap time.
                raw_face = best.raw
                if raw_face is not None:
                    if best.landmarks is not None:
                        M, _ = cv2.estimateAffinePartial2D(
                            best.landmarks, _ARCFACE_112_V1, method=cv2.RANSAC
                        )
                        if M is not None:
                            raw_face.arcface_crop = cv2.warpAffine(
                                img, M, (112, 112), flags=cv2.INTER_LINEAR
                            )
                            # blendswap uses the same template (arcface_112_v2 ≡
                            # _ARCFACE_112_V1); alias intentionally shares the
                            # same array since both attributes are read-only at
                            # swap time.
                            raw_face.portrait_crop_arcv2 = raw_face.arcface_crop

                        # Pre-warp to 256×256 (ffhq_512 template) for uniface source.
                        M_ffhq, _ = cv2.estimateAffinePartial2D(
                            best.landmarks, _FFHQ_512_256, method=cv2.RANSAC
                        )
                        if M_ffhq is not None:
                            raw_face.portrait_crop_ffhq = cv2.warpAffine(
                                img, M_ffhq, (256, 256), flags=cv2.INTER_LINEAR
                            )
                    faces.append(raw_face)
                logger.debug("appended face image: %s", p)
            else:
                logger.warning("No embedding for %s image: %s", kind, p)
        return embeddings, faces

    def _build_gallery(self, characters: list[CharacterMapping]) -> None:
        """Encode reference and portrait images for each character."""
        for ch in characters:
            # Recognition gallery – who to FIND in the video
            rec_embeddings, _ = self._encode_images(
                ch.reference_paths, "reference", ch.source_label
            )
            # Swap gallery – who to REPLACE with
            _, swap_faces = self._encode_images(
                ch.portrait_paths, "portrait", ch.source_label
            )

            if not rec_embeddings:
                logger.error(
                    "No usable reference images for '%s' – "
                    "cannot identify this character in the video.",
                    ch.source_label,
                )
                continue
            if not swap_faces:
                logger.error(
                    "No usable portrait images for '%s' – "
                    "cannot swap this character.",
                    ch.source_label,
                )
                continue

            # Sort portraits by detection confidence so reference_faces[0]
            # is always the highest-quality face.
            swap_faces.sort(
                key=lambda f: getattr(f, "det_score", 0.0), reverse=True
            )

            tid = TargetIdentity(ch.source_label, rec_embeddings, swap_faces)
            self._targets.append(tid)
            best_score = getattr(swap_faces[0], "det_score", None)
            logger.info(
                "Character '%s': %d reference(s) for recognition, "
                "%d portrait(s) for swapping (best det_score=%.3f).",
                ch.source_label,
                len(rec_embeddings),
                len(swap_faces),
                best_score if best_score is not None else 0.0,
            )

    # ── runtime matching ─────────────────────────────────────────────────

    @property
    def targets(self) -> list[TargetIdentity]:
        return self._targets

    def match(self, tracked_face: TrackedFace) -> tuple[str | None, float]:
        """Match a tracked face against the reference gallery.

        Uses :meth:`FaceBackend.face_distance` to compare the face's
        embedding against the mean embedding of each character's
        reference images.

        Returns ``(label, similarity)`` or ``(None, 0.0)`` if no match
        exceeds the per-character similarity threshold.
        """
        if tracked_face.embedding is None:
            return None, 0.0
        
        track_id = tracked_face.track_id
        
        if track_id == -1:
            return None, 0.0
        
        logger.debug("Matching track %d with embedding norm %.3f", track_id, np.linalg.norm(tracked_face.embedding))
        emb = tracked_face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        best_label: str | None = None 
        best_sim = 0.0

        for target in self._targets:
            distances = self._backend.face_distance(
                [target.recognition_embedding], emb
            )
            sim = 1.0 - float(distances[0])
            logger.debug("Track %d → '%s' (sim=%.3f)", track_id, target.label, sim)    
            if sim > best_sim:
                best_sim = sim
                best_label = target.label

        # Per-character threshold
        threshold = self._cfg.detection_threshold
        for ch in self._cfg.characters:
            if ch.source_label == best_label:
                threshold = ch.similarity_threshold
                break

        logger.debug("Best match for track %d: '%s' (sim=%.3f, threshold=%.3f)", track_id, best_label, best_sim, threshold)    
        
        if best_sim < threshold:
            return None, best_sim

        logger.debug("Track %d → '%s' (sim=%.3f)", track_id, best_label, best_sim)
        
        return best_label, best_sim

    def get_target(self, label: str) -> TargetIdentity | None:
        """Look up a TargetIdentity by its label."""
        for t in self._targets:
            if t.label == label:
                return t
        return None

    def identify_faces(self, faces: list[TrackedFace]) -> list[TrackedFace]:
        """Assign identity labels to faces that don't already have one.

        Faces that the tracker has already labelled (propagated from
        prior frames) are left untouched.
        """
        for face in faces:
            if face.identity_label is None:
                label, sim = self.match(face)
                face.identity_label = label
                face.identity_sim = sim
                if label:
                    logger.debug(
                        "Track %d → '%s' (sim=%.3f)",
                        face.track_id,
                        label,
                        sim,
                    )

        return faces
