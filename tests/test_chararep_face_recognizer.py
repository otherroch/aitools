"""Tests for face_recognizer.py."""

import types

import numpy as np
import pytest

from chararep.config import CharacterMapping, PipelineConfig
from chararep.face_detector import TrackedFace
from chararep.face_recognizer import FaceRecognizer, TargetIdentity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _make_cfg(characters=None, detection_threshold=0.5) -> PipelineConfig:
    return PipelineConfig(
        input_video="",
        output_video="",
        characters=characters or [],
        detection_threshold=detection_threshold,
    )


def _make_tracked(emb=None, track_id=1) -> TrackedFace:
    tf = TrackedFace(
        track_id=track_id,
        bbox=np.array([0, 0, 50, 50], dtype=np.float32),
        landmarks=np.zeros((5, 2), dtype=np.float32),
        embedding=emb,
    )
    return tf


def _make_stub_backend(faces_by_path=None):
    """Return a stub FaceBackend-like object for tests.

    The returned object provides a ``detect()`` method that returns an
    empty list, matching the :class:`FaceBackend` protocol.
    """
    faces_by_path = faces_by_path or {}

    class _StubBackend:
        def detect_faces(self, image):
            return []

        def detect(self, image):
            return []

        def encode_faces(self, image, face_locations):
            return []

        def face_distance(self, known_encodings, encoding):
            from face_ops.insightface_backend import InsightFaceBackend
            return InsightFaceBackend._cosine_distance(known_encodings, encoding)

        def load_image(self, path):
            return np.zeros((100, 100, 3), dtype=np.uint8)

        def face_landmarks(self, image, face_locations):
            return [None] * len(face_locations)

    return _StubBackend()


# ---------------------------------------------------------------------------
# TargetIdentity
# ---------------------------------------------------------------------------

class TestTargetIdentity:
    def test_mean_embedding_normalised(self):
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        ti = TargetIdentity("hero", [emb1, emb2], [])
        norm = np.linalg.norm(ti.recognition_embedding)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_zero_embedding_not_normalised(self):
        """Zero vector stays zero (no division by zero)."""
        emb = np.zeros(3)
        ti = TargetIdentity("hero", [emb], [])
        np.testing.assert_array_equal(ti.recognition_embedding, np.zeros(3))


# ---------------------------------------------------------------------------
# FaceRecognizer.match
# ---------------------------------------------------------------------------

class TestFaceRecognizerMatch:
    def _recognizer_with_target(self, label, emb):
        """Build a recognizer with a manually injected target."""
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        ti = TargetIdentity(label, [emb], [object()])
        rec._targets.append(ti)
        return rec

    def test_no_embedding_returns_none(self):
        rec = self._recognizer_with_target("hero", np.ones(512))
        tf = _make_tracked(emb=None, track_id=1)
        label, sim = rec.match(tf)
        assert label is None
        assert sim == 0.0

    def test_track_id_minus_one_returns_none(self):
        rec = self._recognizer_with_target("hero", np.ones(512))
        tf = _make_tracked(emb=np.ones(512, dtype=np.float32), track_id=-1)
        label, sim = rec.match(tf)
        assert label is None
        assert sim == 0.0

    def test_high_similarity_matches(self):
        emb = _unit(np.ones(512, dtype=np.float32))
        rec = self._recognizer_with_target("hero", emb.copy())
        tf = _make_tracked(emb=emb.copy(), track_id=1)
        label, sim = rec.match(tf)
        assert label == "hero"
        assert sim > 0.5

    def test_low_similarity_no_match(self):
        target_emb = _unit(np.ones(512, dtype=np.float32))
        rec = self._recognizer_with_target("hero", target_emb)
        # Orthogonal embedding → cosine similarity near 0
        query_emb = np.zeros(512, dtype=np.float32)
        query_emb[0] = 1.0
        query_emb[1] = -1.0
        query_emb = _unit(query_emb)
        tf = _make_tracked(emb=query_emb, track_id=1)
        label, sim = rec.match(tf)
        # With a 0.5 threshold the low-similarity result should not match
        assert label is None

    def test_no_targets_returns_none(self):
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        tf = _make_tracked(emb=np.ones(512, dtype=np.float32), track_id=1)
        label, sim = rec.match(tf)
        assert label is None
        assert sim == 0.0

    def test_per_character_threshold(self):
        """Per-character threshold of 0.9 rejects a mediocre match."""
        high_threshold_char = CharacterMapping(
            source_label="hero",
            similarity_threshold=0.9,
        )
        cfg = _make_cfg(characters=[high_threshold_char])
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        emb = _unit(np.ones(512, dtype=np.float32))
        rec._targets.append(TargetIdentity("hero", [emb], [object()]))

        # Query that gives ~1.0 similarity
        tf = _make_tracked(emb=emb.copy(), track_id=1)
        label, sim = rec.match(tf)
        assert label == "hero"

        # Query that gives low similarity → should not match
        low_emb = np.zeros(512, dtype=np.float32)
        low_emb[0] = 1.0
        low_emb = _unit(low_emb)
        tf2 = _make_tracked(emb=low_emb, track_id=2)
        label2, _ = rec.match(tf2)
        assert label2 is None


# ---------------------------------------------------------------------------
# FaceRecognizer.identify_faces
# ---------------------------------------------------------------------------

class TestFaceRecognizerIdentifyFaces:
    def _recognizer_with_target(self, label, emb):
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        ti = TargetIdentity(label, [emb], [object()])
        rec._targets.append(ti)
        return rec

    def test_labels_unidentified_faces(self):
        emb = _unit(np.ones(512, dtype=np.float32))
        rec = self._recognizer_with_target("hero", emb.copy())
        tf = _make_tracked(emb=emb.copy(), track_id=1)
        result = rec.identify_faces([tf])
        assert result[0].identity_label == "hero"

    def test_does_not_overwrite_existing_label(self):
        emb = _unit(np.ones(512, dtype=np.float32))
        rec = self._recognizer_with_target("hero", emb.copy())
        tf = _make_tracked(emb=emb.copy(), track_id=1)
        tf.identity_label = "villain"  # already labelled
        rec.identify_faces([tf])
        assert tf.identity_label == "villain"

    def test_empty_list(self):
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        result = rec.identify_faces([])
        assert result == []


# ---------------------------------------------------------------------------
# FaceRecognizer.get_target
# ---------------------------------------------------------------------------

class TestFaceRecognizerGetTarget:
    def test_existing_label(self):
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        ti = TargetIdentity("hero", [np.ones(512)], [])
        rec._targets.append(ti)
        result = rec.get_target("hero")
        assert result is ti

    def test_missing_label_returns_none(self):
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        assert rec.get_target("nobody") is None


# ---------------------------------------------------------------------------
# FaceRecognizer._encode_images (None paths edge case)
# ---------------------------------------------------------------------------

class TestEncodeImages:
    def test_none_paths_returns_empty(self):
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []
        embs, faces = rec._encode_images(None, "reference", "hero")
        assert embs == []
        assert faces == []

    def test_unreadable_image_skipped(self, monkeypatch, tmp_path):
        """A path that cv2.imread can't read is silently skipped."""
        cfg = _make_cfg()
        backend = _make_stub_backend()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []

        import cv2
        monkeypatch.setattr(cv2, "imread", lambda p: None)

        embs, faces = rec._encode_images(["bad_path.jpg"], "reference", "hero")
        assert embs == []
        assert faces == []

    def test_arcface_crop_stored_on_face(self, monkeypatch, tmp_path):
        """Portrait face objects get a 112×112 arcface_crop attached for SimSwap."""
        import cv2
        from face_ops.types import DetectedFace

        cfg = _make_cfg()

        # Build a fake face with realistic kps so RANSAC succeeds
        kps = np.array(
            [[200, 250], [350, 250], [275, 350], [210, 450], [340, 450]],
            dtype=np.float32,
        )
        fake_raw = types.SimpleNamespace(
            kps=kps,
            normed_embedding=np.random.randn(512).astype(np.float32),
            det_score=0.95,
            bbox=np.array([100, 150, 450, 550], dtype=np.float32),
        )

        df = DetectedFace(
            bbox=(150, 450, 550, 100),
            embedding=np.random.randn(512).astype(np.float32),
            landmarks=kps,
            raw=fake_raw,
        )

        class _BackendWithFace:
            def detect_faces(self, image):
                return []
            def detect(self, image):
                return [df]
            def encode_faces(self, image, face_locations):
                return []
            def face_distance(self, known_encodings, encoding):
                return np.array([])
            def load_image(self, path):
                return np.zeros((100, 100, 3), dtype=np.uint8)
            def face_landmarks(self, image, face_locations):
                return []

        backend = _BackendWithFace()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []

        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        img_path = str(tmp_path / "portrait.jpg")
        monkeypatch.setattr(cv2, "imread", lambda p: img)

        _, faces = rec._encode_images([img_path], "portrait", "hero")

        assert len(faces) == 1
        assert hasattr(faces[0], "arcface_crop"), "arcface_crop should be attached"
        assert faces[0].arcface_crop.shape == (112, 112, 3)
        assert faces[0].arcface_crop.dtype == np.uint8

    def test_portrait_crop_ffhq_stored_on_face(self, monkeypatch, tmp_path):
        """Portrait face objects get a 256×256 portrait_crop_ffhq for uniface."""
        import cv2
        from face_ops.types import DetectedFace

        cfg = _make_cfg()
        kps = np.array(
            [[200, 250], [350, 250], [275, 350], [210, 450], [340, 450]],
            dtype=np.float32,
        )
        fake_raw = types.SimpleNamespace(
            kps=kps,
            normed_embedding=np.random.randn(512).astype(np.float32),
            det_score=0.95,
            bbox=np.array([100, 150, 450, 550], dtype=np.float32),
        )

        df = DetectedFace(
            bbox=(150, 450, 550, 100),
            embedding=np.random.randn(512).astype(np.float32),
            landmarks=kps,
            raw=fake_raw,
        )

        class _BackendWithFace:
            def detect_faces(self, image):
                return []
            def detect(self, image):
                return [df]
            def encode_faces(self, image, face_locations):
                return []
            def face_distance(self, known_encodings, encoding):
                return np.array([])
            def load_image(self, path):
                return np.zeros((100, 100, 3), dtype=np.uint8)
            def face_landmarks(self, image, face_locations):
                return []

        backend = _BackendWithFace()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []

        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        monkeypatch.setattr(cv2, "imread", lambda p: img)

        _, faces = rec._encode_images([str(tmp_path / "portrait.jpg")], "portrait", "hero")

        assert len(faces) == 1
        assert hasattr(faces[0], "portrait_crop_ffhq"), "portrait_crop_ffhq should be attached"
        assert faces[0].portrait_crop_ffhq.shape == (256, 256, 3)
        assert faces[0].portrait_crop_ffhq.dtype == np.uint8

    def test_portrait_crop_arcv2_stored_on_face(self, monkeypatch, tmp_path):
        """Portrait face objects get portrait_crop_arcv2 (same as arcface_crop) for blendswap."""
        import cv2
        from face_ops.types import DetectedFace

        cfg = _make_cfg()
        kps = np.array(
            [[200, 250], [350, 250], [275, 350], [210, 450], [340, 450]],
            dtype=np.float32,
        )
        fake_raw = types.SimpleNamespace(
            kps=kps,
            normed_embedding=np.random.randn(512).astype(np.float32),
            det_score=0.95,
            bbox=np.array([100, 150, 450, 550], dtype=np.float32),
        )

        df = DetectedFace(
            bbox=(150, 450, 550, 100),
            embedding=np.random.randn(512).astype(np.float32),
            landmarks=kps,
            raw=fake_raw,
        )

        class _BackendWithFace:
            def detect_faces(self, image):
                return []
            def detect(self, image):
                return [df]
            def encode_faces(self, image, face_locations):
                return []
            def face_distance(self, known_encodings, encoding):
                return np.array([])
            def load_image(self, path):
                return np.zeros((100, 100, 3), dtype=np.uint8)
            def face_landmarks(self, image, face_locations):
                return []

        backend = _BackendWithFace()
        rec = FaceRecognizer.__new__(FaceRecognizer)
        rec._cfg = cfg
        rec._backend = backend
        rec._targets = []

        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        monkeypatch.setattr(cv2, "imread", lambda p: img)

        _, faces = rec._encode_images([str(tmp_path / "portrait.jpg")], "portrait", "hero")

        assert len(faces) == 1
        assert hasattr(faces[0], "portrait_crop_arcv2"), "portrait_crop_arcv2 should be attached"
        assert faces[0].portrait_crop_arcv2.shape == (112, 112, 3)
        assert faces[0].portrait_crop_arcv2.dtype == np.uint8
        # arcv2 and arcface_crop use the same template; they should be identical arrays
        np.testing.assert_array_equal(faces[0].portrait_crop_arcv2, faces[0].arcface_crop)
