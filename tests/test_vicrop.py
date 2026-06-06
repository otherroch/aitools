"""
tests/test_vicrop.py

Unit tests for vicrop.crop
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vicrop.crop import (
    SUPPORTED_VIDEO_EXTS,
    DEFAULT_EVERY_N_FRAMES,
    DEFAULT_MARGIN_RATIO,
    DEFAULT_CROP_SIZE,
    crop_video,
    crop_folder,
)
from vicrop.segment import (
    DEFAULT_MAX_SEGMENT_LENGTH,
    DEFAULT_MIN_SEGMENT_LENGTH,
    _Segment,
    _build_raw_segments,
    _compute_crop_rect,
    _filter_and_split_segments,
    segment_video,
    segment_folder,
)

from face_ops.testing import MockBackendShim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_png(path: Path, size: tuple = (64, 64), color: tuple = (100, 150, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _fake_video_capture(frames_rgb: list[np.ndarray]):
    """Return a mock cv2.VideoCapture that yields the given frames in order."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    call_count = [0]

    def read_side_effect():
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(frames_rgb):
            bgr = frames_rgb[idx][:, :, ::-1].copy()  # RGB → BGR
            return True, bgr
        return False, None

    mock_cap.read.side_effect = read_side_effect
    return mock_cap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_mp4_in_supported_exts(self):
        assert ".mp4" in SUPPORTED_VIDEO_EXTS

    def test_mov_in_supported_exts(self):
        assert ".mov" in SUPPORTED_VIDEO_EXTS

    def test_avi_in_supported_exts(self):
        assert ".avi" in SUPPORTED_VIDEO_EXTS

    def test_defaults(self):
        assert DEFAULT_EVERY_N_FRAMES == 30
        assert 0 < DEFAULT_MARGIN_RATIO < 1
        assert DEFAULT_CROP_SIZE > 0


# ---------------------------------------------------------------------------
# _cluster_faces
# ---------------------------------------------------------------------------


class TestClusterFaces:
    def _make_face_file(self, path: Path) -> Path:
        make_png(path)
        return path

    def test_single_face_creates_person_01(self, tmp_path):
        encoding = np.zeros(128)
        face_path = self._make_face_file(tmp_path / "staging" / "face1.png")

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([0.3])

        backend = MockBackendShim(fr_mock)
        result = backend.cluster_faces([(face_path, encoding)], tmp_path)

        assert "person_01" in result
        assert (tmp_path / "person_01" / "face1.png").exists()

    def test_two_similar_faces_same_person(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        enc1 = np.zeros(128)
        enc2 = np.zeros(128)
        path1 = self._make_face_file(staging / "face1.png")
        path2 = self._make_face_file(staging / "face2.png")

        fr_mock = MagicMock()
        # First face: no existing encodings → creates person_01
        # Second face: distance 0.2 ≤ 0.6 → same person
        fr_mock.face_distance.return_value = np.array([0.2])

        backend = MockBackendShim(fr_mock)
        result = backend.cluster_faces([(path1, enc1), (path2, enc2)], tmp_path)

        assert len(result) == 1  # Only one person

    def test_two_different_faces_different_persons(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        enc1 = np.zeros(128)
        enc2 = np.ones(128)
        path1 = self._make_face_file(staging / "face1.png")
        path2 = self._make_face_file(staging / "face2.png")

        fr_mock = MagicMock()
        # Distance > tolerance → different person
        fr_mock.face_distance.return_value = np.array([0.8])

        backend = MockBackendShim(fr_mock)
        result = backend.cluster_faces([(path1, enc1), (path2, enc2)], tmp_path)

        assert len(result) == 2


# ---------------------------------------------------------------------------
# crop_video
# ---------------------------------------------------------------------------


class TestCropVideo:
    def _dummy_frame(self, color=(128, 64, 32), size=(100, 100)) -> np.ndarray:
        arr = np.zeros((*size, 3), dtype=np.uint8)
        arr[:, :] = color
        return arr

    def test_unopenable_video_returns_zeros(self, tmp_path):
        bad_video = tmp_path / "bad.mp4"
        bad_video.write_bytes(b"not a video")

        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        with patch("vicrop.crop.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap

            stats = crop_video(bad_video, tmp_path / "out", backend=backend)

        assert stats["frames_processed"] == 0
        assert stats["faces"] == 0

    def test_skip_existing_when_output_has_pngs(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"
        # Pre-create output with a PNG to trigger skip
        stem_dir = out_dir / "clip"
        make_png(stem_dir / "existing.png")

        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        stats = crop_video(video_path, out_dir, skip_existing=True, backend=backend)

        assert stats["frames_processed"] == 0  # skipped

    def test_video_with_no_faces_produces_no_output(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        frame = self._dummy_frame()
        mock_cap = _fake_video_capture([frame, frame])

        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = []
        fr_mock.face_encodings.return_value = []

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.crop.cv2.VideoCapture", return_value=mock_cap):
            with patch("vicrop.crop.cv2.cvtColor", return_value=frame):
                stats = crop_video(video_path, out_dir, every_n=1, classify=False, backend=backend)

        assert stats["faces"] == 0

    def test_video_with_face_saves_png(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        # Single 100×100 frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:40, 10:40] = 200  # fake face region

        mock_cap = _fake_video_capture([frame])

        face_location = (10, 40, 40, 10)  # top, right, bottom, left
        fake_encoding = np.zeros(128)

        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = [face_location]
        fr_mock.face_encodings.return_value = [fake_encoding]

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.crop.cv2.VideoCapture", return_value=mock_cap):
            with patch("vicrop.crop.cv2.cvtColor", return_value=frame):
                stats = crop_video(
                    video_path,
                    out_dir,
                    every_n=1,
                    classify=False,
                    crop_size=64,
                    backend=backend,
                )

        assert stats["faces"] == 1
        saved = list((out_dir / "clip").rglob("*.png"))
        assert len(saved) == 1

    def test_every_n_controls_frame_sampling(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        # 6 identical frames
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap = _fake_video_capture([frame] * 6)

        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = []
        fr_mock.face_encodings.return_value = []

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.crop.cv2.VideoCapture", return_value=mock_cap):
            with patch("vicrop.crop.cv2.cvtColor", return_value=frame):
                stats = crop_video(
                    video_path, out_dir, every_n=2, classify=False, backend=backend,
                )

        # frames 0, 2, 4 → 3 processed
        assert stats["frames_processed"] == 3


# ---------------------------------------------------------------------------
# crop_folder
# ---------------------------------------------------------------------------


class TestCropFolder:
    def test_empty_directory_returns_zeros(self, tmp_path):
        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        stats = crop_folder(tmp_path / "in", tmp_path / "out", backend=backend)

        assert stats["videos_processed"] == 0
        assert stats["frames_processed"] == 0
        assert stats["faces"] == 0

    def test_non_video_files_are_ignored(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        (src / "notes.txt").write_text("ignore me")
        (src / "photo.png").write_bytes(b"fake image")

        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        stats = crop_folder(src, tmp_path / "out", backend=backend)

        assert stats["videos_processed"] == 0

    def test_counts_videos_processed(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        for name in ["a.mp4", "b.mp4"]:
            (src / name).write_bytes(b"fake")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def make_cap():
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.read.side_effect = [(True, frame), (False, None)]
            return cap

        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = []
        fr_mock.face_encodings.return_value = []

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.crop.cv2.VideoCapture", side_effect=lambda _: make_cap()):
            with patch("vicrop.crop.cv2.cvtColor", return_value=frame):
                stats = crop_folder(src, tmp_path / "out", every_n=1, classify=False, backend=backend)

        assert stats["videos_processed"] == 2


# ---------------------------------------------------------------------------
# ref_thresh integration
# ---------------------------------------------------------------------------


class TestRefThreshIntegration:
    """Verify --ref-thresh plumbing through crop_video / crop_folder."""

    def _run_with_ref(self, tmp_path, *, classify, ref_thresh, ref_score_val):
        """Helper: one video, one face, controllable ref score."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:40, 10:40] = 200

        mock_cap = _fake_video_capture([frame])

        face_location = (10, 40, 40, 10)
        fake_encoding = np.zeros(128)

        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = [face_location]
        fr_mock.face_encodings.return_value = [fake_encoding]
        fr_mock.face_landmarks.return_value = []
        fr_mock.face_distance.return_value = np.array([0.2])

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.crop.cv2.VideoCapture", return_value=mock_cap), \
             patch("vicrop.crop.cv2.cvtColor", return_value=frame), \
             patch("vicrop.crop.score_reference_quality", return_value=ref_score_val):
            stats = crop_video(
                video_path,
                out_dir,
                every_n=1,
                classify=classify,
                tolerance=0.6,
                crop_size=64,
                ref_thresh=ref_thresh,
                backend=backend,
            )

        return stats, out_dir

    # -- ref_thresh disabled (0) --

    def test_ref_thresh_zero_no_ref_folder(self, tmp_path):
        stats, out_dir = self._run_with_ref(
            tmp_path, classify=False, ref_thresh=0, ref_score_val=0.9,
        )
        assert stats["ref_photos"] == 0
        assert not list(out_dir.rglob("ref"))

    # -- no classify, score above threshold --

    def test_no_classify_ref_folder_created(self, tmp_path):
        stats, out_dir = self._run_with_ref(
            tmp_path, classify=False, ref_thresh=0.5, ref_score_val=0.9,
        )
        assert stats["ref_photos"] == 1
        ref_dir = out_dir / "clip" / "ref"
        assert ref_dir.is_dir()
        assert len(list(ref_dir.glob("*.png"))) == 1

    # -- no classify, score below threshold --

    def test_no_classify_below_thresh_no_ref_folder(self, tmp_path):
        stats, out_dir = self._run_with_ref(
            tmp_path, classify=False, ref_thresh=0.95, ref_score_val=0.5,
        )
        assert stats["ref_photos"] == 0
        assert not (out_dir / "clip" / "ref").exists()

    # -- classify, score above threshold --

    def test_classify_ref_folder_in_person_dir(self, tmp_path):
        stats, out_dir = self._run_with_ref(
            tmp_path, classify=True, ref_thresh=0.5, ref_score_val=0.9,
        )
        assert stats["ref_photos"] == 1
        person_dirs = list((out_dir / "clip").glob("person_*"))
        assert len(person_dirs) == 1
        assert (person_dirs[0] / "ref").is_dir()

    # -- classify, score below threshold --

    def test_classify_below_thresh_no_ref_folder(self, tmp_path):
        stats, out_dir = self._run_with_ref(
            tmp_path, classify=True, ref_thresh=0.95, ref_score_val=0.5,
        )
        assert stats["ref_photos"] == 0
        person_dirs = list((out_dir / "clip").glob("person_*"))
        assert len(person_dirs) == 1
        assert not (person_dirs[0] / "ref").exists()

    # -- stats key present even when skipped --

    def test_ref_photos_key_in_skip_existing(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"
        stem_dir = out_dir / "clip"
        make_png(stem_dir / "existing.png")

        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        stats = crop_video(video_path, out_dir, skip_existing=True, backend=backend)

        assert "ref_photos" in stats
        assert stats["ref_photos"] == 0


# ---------------------------------------------------------------------------
# _cluster_faces with reference encodings
# ---------------------------------------------------------------------------


class TestClusterFacesWithReferences:
    def _make_face_file(self, path: Path) -> Path:
        make_png(path)
        return path

    def test_face_matches_reference_uses_original_name(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        self._make_face_file(staging / "face1.png")

        enc_new = np.zeros(128)
        ref_enc = np.zeros(128)

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([0.1])

        backend = MockBackendShim(fr_mock)
        result = backend.cluster_faces(
            [(staging / "face1.png", enc_new)],
            tmp_path,
            tolerance=0.6,
            reference_encodings=[ref_enc],
            reference_names=["alice"],
        )

        assert "alice" in result
        assert (tmp_path / "alice" / "face1.png").exists()

    def test_unknown_face_gets_person_nn(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        self._make_face_file(staging / "face1.png")

        enc_new = np.ones(128)
        ref_enc = np.zeros(128)

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([1.5])

        backend = MockBackendShim(fr_mock)
        result = backend.cluster_faces(
            [(staging / "face1.png", enc_new)],
            tmp_path,
            tolerance=0.6,
            reference_encodings=[ref_enc],
            reference_names=["alice"],
        )

        assert "person_01" in result
        assert (tmp_path / "person_01" / "face1.png").exists()
        assert "alice" not in result

    def test_no_folder_for_unmatched_reference(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        self._make_face_file(staging / "face1.png")

        enc_new = np.ones(128)
        ref_enc = np.zeros(128)

        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array([1.5])

        backend = MockBackendShim(fr_mock)
        backend.cluster_faces(
            [(staging / "face1.png", enc_new)],
            tmp_path,
            tolerance=0.6,
            reference_encodings=[ref_enc],
            reference_names=["alice"],
        )

        assert not (tmp_path / "alice").exists()


# ---------------------------------------------------------------------------
# _build_raw_segments
# ---------------------------------------------------------------------------


class TestBuildRawSegments:
    """Unit tests for the segment-building logic (no I/O)."""

    def _backend(self, distances):
        """Return a MockBackendShim whose face_distance always returns *distances*."""
        fr_mock = MagicMock()
        fr_mock.face_distance.return_value = np.array(distances)
        return MockBackendShim(fr_mock)

    def test_empty_records_returns_empty(self):
        backend = self._backend([])
        result = _build_raw_segments([], every_n=30, tolerance=0.6, backend=backend)
        assert result == []

    def test_single_single_person_frame_creates_one_segment(self):
        enc = np.zeros(128)
        bbox = (10, 40, 40, 10)
        records = [(0, enc, bbox)]
        backend = self._backend([])
        result = _build_raw_segments(records, every_n=30, tolerance=0.6, backend=backend)
        assert len(result) == 1
        assert result[0].start_frame == 0
        assert result[0].end_frame == 29  # 0 + 30 - 1
        assert result[0].sample_bboxes == [(0, bbox)]

    def test_none_frame_closes_segment(self):
        enc = np.zeros(128)
        bbox = (10, 40, 40, 10)
        records = [(0, enc, bbox), (30, None, None), (60, enc, bbox)]
        backend = self._backend([0.1])  # within tolerance when checked
        result = _build_raw_segments(records, every_n=30, tolerance=0.6, backend=backend)
        # Two separate segments
        assert len(result) == 2
        assert result[0].start_frame == 0
        assert result[1].start_frame == 60

    def test_consecutive_same_person_extends_segment(self):
        enc = np.zeros(128)
        bbox = (10, 40, 40, 10)
        records = [(0, enc, bbox), (30, enc, bbox), (60, enc, bbox)]
        backend = self._backend([0.1])  # distance within tolerance
        result = _build_raw_segments(records, every_n=30, tolerance=0.6, backend=backend)
        assert len(result) == 1
        assert result[0].start_frame == 0
        assert result[0].end_frame == 89  # 60 + 30 - 1
        assert len(result[0].sample_bboxes) == 3

    def test_different_person_starts_new_segment(self):
        enc_a = np.zeros(128)
        enc_b = np.ones(128)
        bbox_a = (10, 40, 40, 10)
        bbox_b = (50, 80, 80, 50)
        records = [(0, enc_a, bbox_a), (30, enc_b, bbox_b)]
        fr_mock = MagicMock()
        # First call: distance between enc_a and enc_b is large
        fr_mock.face_distance.return_value = np.array([0.9])
        backend = MockBackendShim(fr_mock)
        result = _build_raw_segments(records, every_n=30, tolerance=0.6, backend=backend)
        assert len(result) == 2
        assert result[0].start_frame == 0
        assert result[1].start_frame == 30

    def test_all_none_records_returns_empty(self):
        records = [(0, None, None), (30, None, None), (60, None, None)]
        backend = self._backend([])
        result = _build_raw_segments(records, every_n=30, tolerance=0.6, backend=backend)
        assert result == []


# ---------------------------------------------------------------------------
# _compute_crop_rect
# ---------------------------------------------------------------------------


class TestComputeCropRect:
    """Unit tests for the crop rectangle helper."""

    def test_empty_bboxes_returns_full_frame(self):
        result = _compute_crop_rect([], margin_ratio=0.4, frame_width=100, frame_height=100)
        assert result == (0, 0, 100, 100)

    def test_single_bbox_with_zero_margin(self):
        # bbox: top=10, right=40, bottom=40, left=10
        bboxes = [(0, (10, 40, 40, 10))]
        top, left, bottom, right = _compute_crop_rect(
            bboxes, margin_ratio=0.0, frame_width=100, frame_height=100
        )
        assert top == 10
        assert left == 10
        assert bottom == 40
        assert right == 40

    def test_margin_expands_rect(self):
        # face is 30x30, margin=0.4 → expand by 12 on each side
        bboxes = [(0, (20, 70, 50, 40))]
        top, left, bottom, right = _compute_crop_rect(
            bboxes, margin_ratio=0.4, frame_width=200, frame_height=200
        )
        assert top == 8      # 20 - 12
        assert left == 28    # 40 - 12
        assert bottom == 62  # 50 + 12
        assert right == 82   # 70 + 12

    def test_margin_clamped_at_frame_boundary(self):
        # Face near top-left corner — margin should be clamped to 0
        bboxes = [(0, (2, 10, 10, 2))]
        top, left, bottom, right = _compute_crop_rect(
            bboxes, margin_ratio=0.5, frame_width=100, frame_height=100
        )
        assert top >= 0
        assert left >= 0
        assert bottom <= 100
        assert right <= 100

    def test_union_covers_multiple_bboxes(self):
        # Two faces on opposite sides of a 200x200 frame
        bboxes = [(0, (10, 50, 40, 20)), (30, (100, 180, 150, 120))]
        top, left, bottom, right = _compute_crop_rect(
            bboxes, margin_ratio=0.0, frame_width=200, frame_height=200
        )
        # union: top=10, right=180, bottom=150, left=20
        assert top <= 10
        assert left <= 20
        assert bottom >= 150
        assert right >= 180


# ---------------------------------------------------------------------------
# _filter_and_split_segments
# ---------------------------------------------------------------------------


class TestFilterAndSplitSegments:
    """Unit tests for segment filtering and splitting."""

    def _seg(self, start, end):
        return _Segment(start, end, np.zeros(128))

    def test_segment_meeting_min_length_kept(self):
        # 3 seconds at 10 fps = 30 frames; min = 2 seconds → keep
        segs = [self._seg(0, 29)]
        result = _filter_and_split_segments(segs, fps=10.0, total_frames=100,
                                            min_segment_length=2.0,
                                            max_segment_length=30.0)
        assert len(result) == 1

    def test_segment_below_min_length_discarded(self):
        # 0.5 seconds at 10 fps = 5 frames; min = 2 seconds → discard
        segs = [self._seg(0, 4)]
        result = _filter_and_split_segments(segs, fps=10.0, total_frames=100,
                                            min_segment_length=2.0,
                                            max_segment_length=30.0)
        assert result == []

    def test_segment_over_max_length_split(self):
        # 100 frames at 10 fps = 10 seconds; max = 3 seconds (30 frames) → 4 chunks
        # [0,29] [30,59] [60,89] [90,99] — last chunk is 10 frames = 1 second < min 2s
        segs = [self._seg(0, 99)]
        result = _filter_and_split_segments(segs, fps=10.0, total_frames=200,
                                            min_segment_length=2.0,
                                            max_segment_length=3.0)
        # 3 full chunks of 30 frames; last chunk [90,99] = 10 frames = 1s < 2s → dropped
        assert len(result) == 3
        assert result[0].start_frame == 0 and result[0].end_frame == 29
        assert result[1].start_frame == 30 and result[1].end_frame == 59
        assert result[2].start_frame == 60 and result[2].end_frame == 89

    def test_end_frame_clipped_to_total_frames(self):
        # Segment claims end=150 but total_frames=100
        segs = [self._seg(0, 150)]
        result = _filter_and_split_segments(segs, fps=10.0, total_frames=100,
                                            min_segment_length=2.0,
                                            max_segment_length=30.0)
        assert len(result) == 1
        assert result[0].end_frame == 99  # clipped

    def test_empty_input_returns_empty(self):
        result = _filter_and_split_segments([], fps=25.0, total_frames=1000,
                                            min_segment_length=2.0,
                                            max_segment_length=30.0)
        assert result == []

    def test_defaults_match_constants(self):
        assert DEFAULT_MAX_SEGMENT_LENGTH == 30.0
        assert DEFAULT_MIN_SEGMENT_LENGTH == 2.0


# ---------------------------------------------------------------------------
# segment_video
# ---------------------------------------------------------------------------


def _make_segment_cap(frames_rgb, fps=25.0, total=None, width=100, height=100):
    """Create a mock cv2.VideoCapture for segment_video tests."""
    import cv2 as _cv2

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    prop_map = {
        _cv2.CAP_PROP_FPS: fps,
        _cv2.CAP_PROP_FRAME_COUNT: float(total if total is not None else len(frames_rgb)),
        _cv2.CAP_PROP_FRAME_WIDTH: float(width),
        _cv2.CAP_PROP_FRAME_HEIGHT: float(height),
    }
    mock_cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

    call_count = [0]

    def read_side():
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(frames_rgb):
            bgr = frames_rgb[idx][:, :, ::-1].copy()
            return True, bgr
        return False, None

    mock_cap.read.side_effect = read_side
    return mock_cap


class TestSegmentVideo:
    def _dummy_frame(self, size=(100, 100)):
        return np.zeros((*size, 3), dtype=np.uint8)

    def test_unopenable_video_returns_zeros(self, tmp_path):
        video_path = tmp_path / "bad.mp4"
        video_path.write_bytes(b"not a video")

        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        with patch("vicrop.segment.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap

            stats = segment_video(video_path, tmp_path / "out", backend=backend)

        assert stats["segments"] == 0
        assert stats["persons"] == 0

    def test_skip_existing_when_output_has_mp4(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"
        # Pre-create an MP4 in the expected output location
        mp4_dir = out_dir / "clip" / "person_01"
        mp4_dir.mkdir(parents=True)
        (mp4_dir / "seg_001.mp4").write_bytes(b"fake mp4")

        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        stats = segment_video(video_path, out_dir, skip_existing=True, backend=backend)

        assert stats["segments"] == 0  # skipped

    def test_no_single_person_frames_returns_zeros(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        frame = self._dummy_frame()
        mock_cap = _make_segment_cap([frame, frame])

        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = []  # no faces
        fr_mock.face_encodings.return_value = []

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.segment.cv2.VideoCapture", return_value=mock_cap), \
             patch("vicrop.segment.cv2.cvtColor", return_value=frame):
            stats = segment_video(video_path, out_dir, every_n=1, backend=backend)

        assert stats["segments"] == 0

    def test_multi_face_frames_not_included(self, tmp_path):
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        frame = self._dummy_frame()
        mock_cap = _make_segment_cap([frame])

        fr_mock = MagicMock()
        # Two faces detected → not a single-person frame
        fr_mock.face_locations.return_value = [(10, 40, 40, 10), (50, 80, 80, 50)]
        fr_mock.face_encodings.return_value = [np.zeros(128), np.zeros(128)]

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.segment.cv2.VideoCapture", return_value=mock_cap), \
             patch("vicrop.segment.cv2.cvtColor", return_value=frame):
            stats = segment_video(video_path, out_dir, every_n=1, backend=backend)

        assert stats["segments"] == 0

    def test_single_person_segment_written(self, tmp_path):
        """Happy-path: 5 single-person frames produce one segment file."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        frame = self._dummy_frame()
        frames = [frame] * 5

        # Analysis cap (first VideoCapture call)
        analysis_cap = _make_segment_cap(frames, fps=25.0)
        # Write cap (second VideoCapture call)
        write_cap = _make_segment_cap(frames, fps=25.0)

        enc = np.zeros(128)
        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = [(10, 40, 40, 10)]
        fr_mock.face_encodings.return_value = [enc]
        fr_mock.face_distance.return_value = np.array([0.1])

        backend = MockBackendShim(fr_mock)

        mock_writer = MagicMock()
        with patch("vicrop.segment.cv2.VideoCapture", side_effect=[analysis_cap, write_cap]), \
             patch("vicrop.segment.cv2.cvtColor", return_value=frame), \
             patch("vicrop.segment.cv2.VideoWriter", return_value=mock_writer), \
             patch("vicrop.segment.cv2.VideoWriter_fourcc", return_value=0x7634706d):
            stats = segment_video(
                video_path, out_dir, every_n=1,
                min_segment_length=0.0,  # no minimum so the short clip is kept
                max_segment_length=30.0,
                backend=backend,
            )

        assert stats["segments"] == 1
        assert stats["persons"] == 1
        # VideoWriter.write() should have been called for each frame in the segment
        assert mock_writer.write.call_count >= 1
        mock_writer.release.assert_called_once()

    def test_segment_below_min_length_not_written(self, tmp_path):
        """A segment shorter than min_segment_length is discarded."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        frame = self._dummy_frame()
        frames = [frame] * 3  # 3 frames at 25 fps = 0.12 s — below 2 s minimum

        analysis_cap = _make_segment_cap(frames, fps=25.0)

        enc = np.zeros(128)
        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = [(10, 40, 40, 10)]
        fr_mock.face_encodings.return_value = [enc]
        fr_mock.face_distance.return_value = np.array([0.1])

        backend = MockBackendShim(fr_mock)

        # Only the analysis cap is needed — no write should happen
        with patch("vicrop.segment.cv2.VideoCapture", side_effect=[analysis_cap]), \
             patch("vicrop.segment.cv2.cvtColor", return_value=frame), \
             patch("vicrop.segment.cv2.VideoWriter") as mock_vw_cls:
            stats = segment_video(
                video_path, out_dir, every_n=1,
                min_segment_length=2.0,
                backend=backend,
            )

        assert stats["segments"] == 0
        mock_vw_cls.assert_not_called()

    def test_long_segment_split_into_multiple_files(self, tmp_path):
        """A segment exceeding max_segment_length is split into multiple files."""
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"fake")
        out_dir = tmp_path / "out"

        fps = 10.0
        # 60 frames = 6 seconds; max = 2 seconds (20 frames) → 3 chunks
        frames = [self._dummy_frame()] * 60

        analysis_cap = _make_segment_cap(frames, fps=fps)
        write_cap = _make_segment_cap(frames, fps=fps)

        enc = np.zeros(128)
        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = [(10, 40, 40, 10)]
        fr_mock.face_encodings.return_value = [enc]
        fr_mock.face_distance.return_value = np.array([0.1])

        backend = MockBackendShim(fr_mock)

        writers_created = []

        def make_writer(*args, **kwargs):
            w = MagicMock()
            writers_created.append(w)
            return w

        with patch("vicrop.segment.cv2.VideoCapture", side_effect=[analysis_cap, write_cap]), \
             patch("vicrop.segment.cv2.cvtColor", side_effect=lambda f, _: f), \
             patch("vicrop.segment.cv2.VideoWriter", side_effect=make_writer), \
             patch("vicrop.segment.cv2.VideoWriter_fourcc", return_value=0x7634706d):
            stats = segment_video(
                video_path, out_dir, every_n=1,
                min_segment_length=1.0,
                max_segment_length=2.0,
                backend=backend,
            )

        assert stats["segments"] == 3
        assert len(writers_created) == 3


# ---------------------------------------------------------------------------
# segment_folder
# ---------------------------------------------------------------------------


class TestSegmentFolder:
    def test_empty_directory_returns_zeros(self, tmp_path):
        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        stats = segment_folder(tmp_path / "in", tmp_path / "out", backend=backend)

        assert stats["videos_processed"] == 0
        assert stats["segments"] == 0
        assert stats["persons"] == 0

    def test_non_video_files_ignored(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        (src / "notes.txt").write_text("ignore me")

        fr_mock = MagicMock()
        backend = MockBackendShim(fr_mock)
        stats = segment_folder(src, tmp_path / "out", backend=backend)

        assert stats["videos_processed"] == 0

    def test_counts_videos_processed(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        for name in ["a.mp4", "b.mp4"]:
            (src / name).write_bytes(b"fake")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def make_cap():
            cap = MagicMock()
            cap.isOpened.return_value = True
            cap.read.side_effect = [(True, frame), (False, None)]
            cap.get.return_value = 0.0
            return cap

        fr_mock = MagicMock()
        fr_mock.face_locations.return_value = []
        fr_mock.face_encodings.return_value = []

        backend = MockBackendShim(fr_mock)
        with patch("vicrop.segment.cv2.VideoCapture", side_effect=lambda _: make_cap()), \
             patch("vicrop.segment.cv2.cvtColor", return_value=frame):
            stats = segment_folder(src, tmp_path / "out", every_n=1, backend=backend)

        assert stats["videos_processed"] == 2


# ---------------------------------------------------------------------------
# CLI — --output-type integration
# ---------------------------------------------------------------------------


class TestCLIOutputType:
    """Smoke-test the CLI argument additions."""

    def test_default_output_type_is_photo(self):
        from vicrop.cli import parse_args

        args = parse_args(["--input", "/tmp/v.mp4", "--output-dir", "/tmp/out"])
        assert args.output_type == "photo"

    def test_video_output_type_parsed(self):
        from vicrop.cli import parse_args

        args = parse_args([
            "--input", "/tmp/v.mp4",
            "--output-dir", "/tmp/out",
            "--output-type", "video",
        ])
        assert args.output_type == "video"

    def test_max_segment_length_default(self):
        from vicrop.cli import parse_args

        args = parse_args(["--input", "/tmp/v.mp4", "--output-dir", "/tmp/out"])
        assert args.max_segment_length == 30.0

    def test_min_segment_length_default(self):
        from vicrop.cli import parse_args

        args = parse_args(["--input", "/tmp/v.mp4", "--output-dir", "/tmp/out"])
        assert args.min_segment_length == 2.0

    def test_max_min_segment_lengths_parsed(self):
        from vicrop.cli import parse_args

        args = parse_args([
            "--input", "/tmp/v.mp4",
            "--output-dir", "/tmp/out",
            "--output-type", "video",
            "--max-segment-length", "60",
            "--min-segment-length", "5",
        ])
        assert args.max_segment_length == 60.0
        assert args.min_segment_length == 5.0

    def test_invalid_output_type_raises(self):
        from vicrop.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--input", "/tmp/v.mp4",
                "--output-dir", "/tmp/out",
                "--output-type", "gif",
            ])
