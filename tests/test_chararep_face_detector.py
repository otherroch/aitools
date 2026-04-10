"""Tests for face_detector.py."""

import numpy as np
import pytest

from chararep.config import PipelineConfig
from chararep.face_detector import FaceDetector, TrackedFace, _iou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**kw) -> PipelineConfig:
    defaults = dict(
        input_video="",
        output_video="",
        tracker_max_age=3,
        tracker_iou_threshold=0.3,
        detection_model="buffalo_l",
        detection_threshold=0.5,
        detection_size=(640, 640),
        device_id=0,
    )
    defaults.update(kw)
    return PipelineConfig(**defaults)


def _make_face(bbox, emb=None):
    """Return a minimal InsightFace-like face object."""
    import types
    face = types.SimpleNamespace(
        bbox=np.array(bbox, dtype=np.float32),
        kps=np.array(
            [[10, 10], [20, 10], [15, 20], [12, 28], [22, 28]], dtype=np.float32
        ),
        normed_embedding=emb if emb is not None else np.ones(512, dtype=np.float32),
    )
    return face


def _make_detected_face(bbox, emb=None):
    """Return a DetectedFace matching the protocol from face_ops."""
    from face_ops.types import DetectedFace
    x1, y1, x2, y2 = bbox
    raw = _make_face(bbox, emb)
    embedding = emb if emb is not None else np.ones(512, dtype=np.float32)
    return DetectedFace(
        bbox=(int(y1), int(x2), int(y2), int(x1)),
        embedding=np.array(embedding, dtype=np.float32),
        landmarks=np.array(
            [[10, 10], [20, 10], [15, 20], [12, 28], [22, 28]], dtype=np.float32
        ),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# _iou helper
# ---------------------------------------------------------------------------

class TestIou:
    def test_identical_boxes(self):
        a = np.array([0, 0, 10, 10], dtype=np.float32)
        assert _iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 5, 5], dtype=np.float32)
        b = np.array([10, 10, 20, 20], dtype=np.float32)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10], dtype=np.float32)
        b = np.array([5, 5, 15, 15], dtype=np.float32)
        iou = _iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        a = np.array([5, 5, 5, 5], dtype=np.float32)
        b = np.array([5, 5, 5, 5], dtype=np.float32)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_contained_box(self):
        outer = np.array([0, 0, 20, 20], dtype=np.float32)
        inner = np.array([5, 5, 10, 10], dtype=np.float32)
        iou = _iou(outer, inner)
        assert 0.0 < iou < 1.0


# ---------------------------------------------------------------------------
# TrackedFace dataclass
# ---------------------------------------------------------------------------

class TestTrackedFace:
    def test_defaults(self):
        tf = TrackedFace(
            track_id=0,
            bbox=np.zeros(4),
            landmarks=np.zeros((5, 2)),
        )
        assert tf.age_since_seen == 0
        assert tf.identity_label is None
        assert tf.identity_sim == 0.0
        assert tf.embedding is None
        assert tf.face_obj is None

    def test_custom_values(self):
        emb = np.ones(512, dtype=np.float32)
        bbox = np.array([0, 0, 50, 50], dtype=np.float32)
        tf = TrackedFace(
            track_id=42,
            bbox=bbox,
            landmarks=np.zeros((5, 2)),
            embedding=emb,
            identity_label="hero",
            identity_sim=0.85,
        )
        assert tf.track_id == 42
        assert tf.identity_label == "hero"
        assert tf.identity_sim == pytest.approx(0.85)
        np.testing.assert_array_equal(tf.embedding, emb)


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class TestFaceDetector:
    def test_init(self):
        cfg = _make_cfg()
        det = FaceDetector(cfg)
        assert det.backend is not None

    def test_backend_satisfies_protocol(self):
        """FaceDetector creates a backend that satisfies FaceBackend."""
        from face_ops.backend import FaceBackend as Proto
        cfg = _make_cfg()
        det = FaceDetector(cfg)
        assert isinstance(det.backend, Proto)

    def test_detect_empty_frame_no_faces(self):
        """When the backend returns no faces, detect() returns empty list."""
        cfg = _make_cfg()
        det = FaceDetector(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert result == []

    def test_detect_returns_tracked_faces(self, monkeypatch):
        """Stub returns one fake face; detect assigns a track_id."""
        cfg = _make_cfg()
        det = FaceDetector(cfg)

        df = _make_detected_face([10, 10, 60, 60])
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [df])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)

        assert len(result) == 1
        assert result[0].track_id == 0
        assert result[0].age_since_seen == 0

    def test_track_id_increments(self, monkeypatch):
        cfg = _make_cfg()
        det = FaceDetector(cfg)

        df = _make_detected_face([10, 10, 60, 60])
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [df])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det.detect(frame)
        # Clear tracks so next face gets a new id
        det._tracks = []
        result = det.detect(frame)
        assert result[0].track_id == 1

    def test_tracks_age_out(self, monkeypatch):
        """A face not seen for tracker_max_age+1 frames is evicted."""
        cfg = _make_cfg(tracker_max_age=2)
        det = FaceDetector(cfg)

        df = _make_detected_face([10, 10, 60, 60])
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [df])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det.detect(frame)

        # Now no faces – track should age
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [])

        det.detect(frame)  # age 1
        det.detect(frame)  # age 2
        result = det.detect(frame)  # age 3 → evicted
        assert result == []

    def test_tracks_retained_within_max_age(self, monkeypatch):
        """Tracks survive up to max_age frames without detection."""
        cfg = _make_cfg(tracker_max_age=2)
        det = FaceDetector(cfg)

        df = _make_detected_face([10, 10, 60, 60])
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [df])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det.detect(frame)

        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [])
        result = det.detect(frame)  # age 1 – should still be retained
        assert len(result) == 1
        assert result[0].age_since_seen == 1

    def test_active_tracks_only_current(self, monkeypatch):
        """active_tracks() returns only faces seen in the most recent frame."""
        cfg = _make_cfg(tracker_max_age=2)
        det = FaceDetector(cfg)

        df = _make_detected_face([10, 10, 60, 60])
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [df])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det.detect(frame)

        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [])
        det.detect(frame)  # face ages

        active = det.active_tracks()
        assert active == []

    def test_face_without_embedding(self, monkeypatch):
        """Faces without embedding don't crash detect()."""
        from face_ops.types import DetectedFace
        cfg = _make_cfg()
        det = FaceDetector(cfg)

        import types
        raw_face = types.SimpleNamespace(
            bbox=np.array([10, 10, 60, 60], dtype=np.float32),
            kps=np.array([[10, 10], [20, 10], [15, 20], [12, 28], [22, 28]]),
            normed_embedding=None,
        )
        df = DetectedFace(
            bbox=(10, 60, 60, 10),
            embedding=None,
            landmarks=np.array([[10, 10], [20, 10], [15, 20], [12, 28], [22, 28]], dtype=np.float32),
            raw=raw_face,
        )
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [df])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 1
        assert result[0].embedding is None

    def test_identity_propagated_from_prior_frame(self, monkeypatch):
        """Identity labels are carried forward through track matching."""
        cfg = _make_cfg()
        det = FaceDetector(cfg)

        df = _make_detected_face([10, 10, 60, 60])
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: [df])

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        result[0].identity_label = "villain"
        det._tracks = result

        result2 = det.detect(frame)
        assert result2[0].identity_label == "villain"

    def test_multiple_faces_distinct_track_ids(self, monkeypatch):
        cfg = _make_cfg()
        det = FaceDetector(cfg)

        faces = [
            _make_detected_face([0, 0, 30, 30]),
            _make_detected_face([100, 100, 150, 150]),
        ]
        monkeypatch.setattr(det._backend, "detect", lambda img, **kw: faces)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert len(result) == 2
        ids = {t.track_id for t in result}
        assert len(ids) == 2
