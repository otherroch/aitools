"""
tests/test_face_ops.py

Unit tests for the face_ops shared package:
  - FaceBackend protocol conformance
  - get_backend factory
  - cluster_faces (backend-agnostic)
  - load_reference_encodings (backend-agnostic)
  - InsightFaceBackend distance computation
  - DlibBackend (via mock)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from face_ops import (
    SUPPORTED_IMAGE_EXTS,
    FaceBackend,
    cluster_faces,
    get_backend,
    load_reference_encodings,
)
from face_ops.backend import FaceBackend as FaceBackendProtocol
from face_ops.insightface_backend import InsightFaceBackend
from face_ops.mixin import FaceBackendMixin
from face_ops.types import Encoding, FaceBBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(path: Path, size: tuple = (100, 100)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(path)
    return path


class _StubBackend(FaceBackendMixin):
    """Minimal backend for testing clustering/ref-loading in isolation.
    
    Inherits ``cluster_faces`` and ``load_reference_encodings`` from
    :class:`FaceBackendMixin`.
    """

    def __init__(
        self,
        *,
        face_locations: list[FaceBBox] | None = None,
        face_encs: list[Encoding] | None = None,
        distance_fn=None,
    ):
        self._locs = face_locations or []
        self._encs = face_encs or []
        self._dist_fn = distance_fn

    def detect_faces(self, image, *, model="hog"):
        return self._locs

    def detect(self, image, *, model="hog"):
        from face_ops.types import DetectedFace
        boxes = self.detect_faces(image, model=model)
        encs = self.encode_faces(image, boxes)
        results = []
        for i, b in enumerate(boxes):
            emb = encs[i] if i < len(encs) else None
            results.append(DetectedFace(bbox=b, embedding=emb))
        return results

    def encode_faces(self, image, face_locations):
        return self._encs

    def face_distance(self, known_encodings, encoding):
        if self._dist_fn:
            return self._dist_fn(known_encodings, encoding)
        dists = [np.linalg.norm(k - encoding) for k in known_encodings]
        return np.array(dists, dtype=np.float64)

    def load_image(self, path):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def face_landmarks(self, image, face_locations):
        return [None] * len(face_locations)


# ---------------------------------------------------------------------------
# FaceBackend protocol
# ---------------------------------------------------------------------------


class TestFaceBackendProtocol:
    def test_stub_satisfies_protocol(self):
        stub = _StubBackend()
        assert isinstance(stub, FaceBackendProtocol)

    def test_non_backend_does_not_satisfy_protocol(self):
        assert not isinstance(42, FaceBackendProtocol)


# ---------------------------------------------------------------------------
# get_backend factory
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown face_ops backend"):
            get_backend("nonexistent_backend")

    def test_dlib_import_error_is_clear(self):
        with patch.dict("sys.modules", {"face_recognition": None}):
            with pytest.raises(ImportError, match="face_recognition"):
                get_backend("dlib")

    def test_insightface_import_error_is_clear(self):
        with patch.dict("sys.modules", {"insightface": None, "insightface.app": None}):
            with pytest.raises(ImportError, match="insightface"):
                get_backend("insightface")

    def test_arcface_alias_works(self):
        with patch.dict("sys.modules", {"insightface": None, "insightface.app": None}):
            with pytest.raises(ImportError, match="insightface"):
                get_backend("arcface")


# ---------------------------------------------------------------------------
# SUPPORTED_IMAGE_EXTS
# ---------------------------------------------------------------------------


class TestSupportedExts:
    def test_common_extensions_present(self):
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            assert ext in SUPPORTED_IMAGE_EXTS


# ---------------------------------------------------------------------------
# cluster_faces (as method on backend)
# ---------------------------------------------------------------------------


class TestClusterFaces:
    def test_single_face_creates_person_01(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc = np.zeros(128, dtype=np.float64)
        backend = _StubBackend()

        result = backend.cluster_faces(
            [(staging / "face1.png", enc)],
            tmp_path,
        )
        assert "person_01" in result
        assert (tmp_path / "person_01" / "face1.png").exists()

    def test_two_similar_faces_same_person(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "f1.png")
        _make_png(staging / "f2.png")

        enc = np.zeros(128, dtype=np.float64)
        backend = _StubBackend()

        result = backend.cluster_faces(
            [(staging / "f1.png", enc), (staging / "f2.png", enc)],
            tmp_path,
            tolerance=0.6,
        )
        assert len(result) == 1

    def test_two_different_faces_different_persons(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "f1.png")
        _make_png(staging / "f2.png")

        enc_a = np.zeros(128, dtype=np.float64)
        enc_b = np.ones(128, dtype=np.float64)
        backend = _StubBackend()

        result = backend.cluster_faces(
            [(staging / "f1.png", enc_a), (staging / "f2.png", enc_b)],
            tmp_path,
            tolerance=0.6,
        )
        assert len(result) == 2

    def test_reference_name_used_for_matching_face(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc = np.zeros(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)
        backend = _StubBackend()

        result = backend.cluster_faces(
            [(staging / "face1.png", enc)],
            tmp_path,
            tolerance=0.6,
            reference_encodings=[ref_enc],
            reference_names=["alice"],
        )
        assert "alice" in result
        assert (tmp_path / "alice" / "face1.png").exists()

    def test_unknown_face_gets_person_nn_with_refs(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc = np.ones(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)
        backend = _StubBackend()

        result = backend.cluster_faces(
            [(staging / "face1.png", enc)],
            tmp_path,
            tolerance=0.6,
            reference_encodings=[ref_enc],
            reference_names=["alice"],
        )
        assert "person_01" in result
        assert "alice" not in result

    def test_person_nn_numbering_avoids_reference_collision(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc = np.ones(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)
        backend = _StubBackend()

        result = backend.cluster_faces(
            [(staging / "face1.png", enc)],
            tmp_path,
            tolerance=0.6,
            reference_encodings=[ref_enc],
            reference_names=["person_05"],
        )
        assert "person_06" in result

    def test_no_folder_for_unmatched_reference(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_png(staging / "face1.png")

        enc = np.ones(128, dtype=np.float64)
        ref_enc = np.zeros(128, dtype=np.float64)
        backend = _StubBackend()

        backend.cluster_faces(
            [(staging / "face1.png", enc)],
            tmp_path,
            tolerance=0.6,
            reference_encodings=[ref_enc],
            reference_names=["alice"],
        )
        assert not (tmp_path / "alice").exists()


# ---------------------------------------------------------------------------
# load_reference_encodings (as method on backend)
# ---------------------------------------------------------------------------


class TestLoadReferenceEncodings:
    def test_loads_from_identity_dirs(self, tmp_path):
        ref_dir = tmp_path / "classified"
        (ref_dir / "alice").mkdir(parents=True)
        _make_png(ref_dir / "alice" / "a1.png")
        (ref_dir / "bob").mkdir(parents=True)
        _make_png(ref_dir / "bob" / "b1.png")

        enc_a = np.zeros(128, dtype=np.float64)
        enc_b = np.ones(128, dtype=np.float64)
        call_count = {"n": 0}

        def encs(image, locs):
            call_count["n"] += 1
            return [enc_a] if call_count["n"] <= 1 else [enc_b]

        backend = _StubBackend(
            face_locations=[(10, 90, 90, 10)],
        )
        backend.encode_faces = encs

        encodings, names = backend.load_reference_encodings(ref_dir)
        assert len(encodings) == 2
        assert "alice" in names
        assert "bob" in names

    def test_empty_dir_returns_empty(self, tmp_path):
        ref_dir = tmp_path / "classified"
        ref_dir.mkdir()
        backend = _StubBackend()

        encodings, names = backend.load_reference_encodings(ref_dir)
        assert len(encodings) == 0
        assert len(names) == 0

    def test_no_face_in_image_skips(self, tmp_path):
        ref_dir = tmp_path / "classified"
        (ref_dir / "alice").mkdir(parents=True)
        _make_png(ref_dir / "alice" / "noface.png")

        backend = _StubBackend(face_locations=[], face_encs=[])

        encodings, names = backend.load_reference_encodings(ref_dir)
        assert len(encodings) == 0

    def test_max_per_identity(self, tmp_path):
        ref_dir = tmp_path / "classified"
        (ref_dir / "alice").mkdir(parents=True)
        _make_png(ref_dir / "alice" / "a1.png")
        _make_png(ref_dir / "alice" / "a2.png")
        _make_png(ref_dir / "alice" / "a3.png")

        backend = _StubBackend(
            face_locations=[(10, 90, 90, 10)],
            face_encs=[np.zeros(128)],
        )

        encodings, names = backend.load_reference_encodings(
            ref_dir, max_per_identity=2
        )
        assert len(encodings) == 2
        assert names.count("alice") == 2


# ---------------------------------------------------------------------------
# InsightFaceBackend.face_distance (cosine distance)
# ---------------------------------------------------------------------------


class TestInsightFaceDistance:
    """Test the cosine distance computation via the static method."""

    def test_identical_vectors_zero_distance(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dist = InsightFaceBackend._cosine_distance([v], v)
        assert dist[0] == pytest.approx(0.0, abs=0.01)

    def test_orthogonal_vectors_distance_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        dist = InsightFaceBackend._cosine_distance([a], b)
        assert dist[0] == pytest.approx(1.0, abs=0.01)

    def test_opposite_vectors_distance_two(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        dist = InsightFaceBackend._cosine_distance([a], b)
        assert dist[0] == pytest.approx(2.0, abs=0.01)

    def test_empty_known_encodings(self):
        dist = InsightFaceBackend._cosine_distance([], np.zeros(3))
        assert len(dist) == 0


# ---------------------------------------------------------------------------
# InsightFaceBackend extended init & app property
# ---------------------------------------------------------------------------


class TestInsightFaceBackendInit:
    """Test the InsightFaceBackend's configurable __init__ and app property."""

    def test_app_property_returns_face_analysis(self):
        """The ``app`` property exposes the underlying FaceAnalysis."""
        backend = InsightFaceBackend()
        assert backend.app is not None
        # Should have a 'get' method (the FaceAnalysis stub from conftest)
        assert hasattr(backend.app, "get")

    def test_custom_providers_accepted(self):
        """Passing explicit providers bypasses the auto-detection path."""
        backend = InsightFaceBackend(
            providers=[("CPUExecutionProvider", {})],
        )
        assert backend.app is not None

    def test_det_thresh_forwarded(self):
        """The det_thresh parameter is accepted without error."""
        backend = InsightFaceBackend(det_thresh=0.3)
        assert backend.app is not None


class TestInsightFaceDetect:
    """Test InsightFaceBackend.detect() rich results."""

    def test_detect_returns_empty_on_no_faces(self):
        backend = InsightFaceBackend()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = backend.detect(image)
        assert result == []

    def test_detect_returns_detected_face_objects(self):
        """When the stub returns faces, detect returns DetectedFace objects."""
        from face_ops.types import DetectedFace
        import types as _types

        backend = InsightFaceBackend()
        fake_face = _types.SimpleNamespace(
            bbox=np.array([10, 20, 50, 60], dtype=np.float32),
            kps=np.array([[15, 25], [35, 25], [25, 35], [18, 45], [38, 45]], dtype=np.float32),
            normed_embedding=np.ones(512, dtype=np.float32),
        )
        backend._app.get = lambda img: [fake_face]

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = backend.detect(image)
        assert len(result) == 1
        assert isinstance(result[0], DetectedFace)
        # bbox should be in (top, right, bottom, left) format
        top, right, bottom, left = result[0].bbox
        assert left == 10 and top == 20 and right == 50 and bottom == 60
        assert result[0].embedding is not None
        assert result[0].landmarks is not None
        assert result[0].raw is fake_face


class TestGetBackendDetect:
    """Test detect() via the get_backend() factory."""

    def test_get_backend_insightface_has_detect(self):
        backend = get_backend("insightface")
        assert hasattr(backend, "detect")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = backend.detect(image)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# FaceBBox / Encoding type aliases
# ---------------------------------------------------------------------------


class TestTypes:
    def test_face_bbox_is_tuple(self):
        bbox: FaceBBox = (10, 90, 90, 10)
        assert len(bbox) == 4

    def test_encoding_is_ndarray(self):
        enc: Encoding = np.zeros(128)
        assert isinstance(enc, np.ndarray)
