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
