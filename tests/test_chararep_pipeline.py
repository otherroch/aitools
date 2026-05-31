"""Tests for chararep/pipeline.py – parallel and sequential pipeline paths."""

import types
from collections import deque
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chararep.config import PipelineConfig
from chararep.pipeline import CharacterReplacementPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_pipeline(batch_size: int = 1, enable_timers: bool = False):
    """Build a pipeline with every heavy component stubbed out."""
    cfg = PipelineConfig(batch_size=batch_size, enable_timers=enable_timers)

    with patch.object(CharacterReplacementPipeline, "__init__", lambda self, _cfg: None):
        pipe = CharacterReplacementPipeline(cfg)

    pipe._cfg = cfg
    pipe._detector = MagicMock()
    pipe._recognizer = MagicMock()
    pipe._swapper = MagicMock()
    pipe._enhancer = MagicMock()
    pipe._enhancer.available = False
    pipe._blender = MagicMock()
    return pipe


# ---------------------------------------------------------------------------
# _prepare_frame
# ---------------------------------------------------------------------------


class TestPrepareFrame:
    def test_no_faces_returns_empty_pairs(self):
        pipe = _stub_pipeline()
        pipe._detector.detect.return_value = []

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stats = {"frames_detected": 0, "faces_identified": 0}

        result = pipe._prepare_frame(frame, 0, stats)
        assert np.array_equal(result[0], frame)
        assert result[1] == 0
        assert result[2] == []  # tracked_faces
        assert result[3] == []  # swap_pairs
        assert stats["frames_detected"] == 0

    def test_faces_detected_but_no_identity(self):
        pipe = _stub_pipeline()

        tf = MagicMock()
        tf.identity_label = None
        tf.track_id = 1
        pipe._detector.detect.return_value = [tf]
        pipe._recognizer.identify_faces.return_value = [tf]

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stats = {"frames_detected": 0, "faces_identified": 0}

        result = pipe._prepare_frame(frame, 0, stats)
        assert result[3] == []  # swap_pairs empty
        assert stats["frames_detected"] == 1
        assert stats["faces_identified"] == 0

    def test_swap_pair_built_for_identified_face(self):
        pipe = _stub_pipeline()

        tf = MagicMock()
        tf.identity_label = "hero"
        tf.track_id = 1
        tf.face_obj = "face_obj_sentinel"

        target = MagicMock()
        target.label = "hero"
        target.reference_faces = ["ref_face"]

        pipe._detector.detect.return_value = [tf]
        pipe._recognizer.identify_faces.return_value = [tf]
        pipe._recognizer.get_target.return_value = target

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stats = {"frames_detected": 0, "faces_identified": 0}

        result = pipe._prepare_frame(frame, 5, stats)
        assert len(result[3]) == 1  # one swap pair
        assert result[3][0] == ("face_obj_sentinel", "ref_face")
        assert stats["faces_identified"] == 1

    def test_stale_tracks_are_skipped_for_swapping(self):
        pipe = _stub_pipeline()

        stale = MagicMock()
        stale.age_since_seen = 2
        stale.identity_label = "hero"
        stale.track_id = 7
        stale.face_obj = "stale_face"
        stale.landmarks = np.zeros((5, 2), dtype=np.float32)
        stale.bbox = np.array([0, 0, 20, 20], dtype=np.float32)

        active = MagicMock()
        active.age_since_seen = 0
        active.identity_label = "hero"
        active.track_id = 1
        active.face_obj = "active_face"
        active.landmarks = np.array(
            [[10, 10], [20, 10], [15, 15], [11, 20], [19, 20]],
            dtype=np.float32,
        )
        active.bbox = np.array([0, 0, 30, 30], dtype=np.float32)

        target = MagicMock()
        target.label = "hero"
        target.reference_faces = ["ref_face"]

        pipe._detector.detect.return_value = [stale, active]
        pipe._recognizer.get_target.return_value = target

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stats = {"frames_detected": 0, "faces_identified": 0}

        result = pipe._prepare_frame(frame, 0, stats)
        assert result[2] == [active]
        assert result[3] == [("active_face", "ref_face")]
        pipe._recognizer.identify_faces.assert_called_once_with([active])

    def test_landmarks_are_stabilized_per_track(self):
        pipe = _stub_pipeline()

        base_kps = np.array(
            [[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
            dtype=np.float32,
        )
        jumped_kps = base_kps + np.array([18, 0], dtype=np.float32)

        first_face = types.SimpleNamespace(
            age_since_seen=0,
            identity_label=None,
            track_id=3,
            bbox=np.array([20, 30, 100, 120], dtype=np.float32),
            landmarks=base_kps.copy(),
            face_obj=types.SimpleNamespace(kps=base_kps.copy()),
        )
        second_face = types.SimpleNamespace(
            age_since_seen=0,
            identity_label=None,
            track_id=3,
            bbox=np.array([20, 30, 100, 120], dtype=np.float32),
            landmarks=jumped_kps.copy(),
            face_obj=types.SimpleNamespace(kps=jumped_kps.copy()),
        )

        pipe._detector.detect.side_effect = [[first_face], [second_face]]

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stats = {"frames_detected": 0, "faces_identified": 0}

        pipe._prepare_frame(frame, 0, stats)
        result = pipe._prepare_frame(frame, 1, stats)

        smoothed = result[2][0].landmarks
        assert np.all(smoothed[:, 0] > base_kps[:, 0])
        assert np.all(smoothed[:, 0] < jumped_kps[:, 0])
        np.testing.assert_allclose(result[2][0].face_obj.kps, smoothed)

    def test_coherent_landmark_scale_shrink_is_strongly_damped(self):
        pipe = _stub_pipeline()

        base_kps = np.array(
            [[40, 50], [74, 50], [57, 70], [42, 90], [72, 90]],
            dtype=np.float32,
        )
        anchor_center = ((base_kps[0] + base_kps[1]) * 0.5 + base_kps[2]) * 0.5
        shrunk_kps = anchor_center + (base_kps - anchor_center) * 0.90

        first_face = types.SimpleNamespace(
            age_since_seen=0,
            identity_label=None,
            track_id=4,
            bbox=np.array([20, 30, 100, 120], dtype=np.float32),
            landmarks=base_kps.copy(),
            face_obj=types.SimpleNamespace(kps=base_kps.copy()),
        )
        second_face = types.SimpleNamespace(
            age_since_seen=0,
            identity_label=None,
            track_id=4,
            bbox=np.array([20, 30, 100, 120], dtype=np.float32),
            landmarks=shrunk_kps.copy(),
            face_obj=types.SimpleNamespace(kps=shrunk_kps.copy()),
        )

        pipe._detector.detect.side_effect = [[first_face], [second_face]]

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stats = {"frames_detected": 0, "faces_identified": 0}

        pipe._prepare_frame(frame, 0, stats)
        result = pipe._prepare_frame(frame, 1, stats)

        smoothed = result[2][0].landmarks
        base_eye_dist = np.linalg.norm(base_kps[1] - base_kps[0])
        shrunk_eye_dist = np.linalg.norm(shrunk_kps[1] - shrunk_kps[0])
        smoothed_eye_dist = np.linalg.norm(smoothed[1] - smoothed[0])

        assert shrunk_eye_dist < base_eye_dist * 0.91
        assert smoothed_eye_dist > base_eye_dist * 0.97
        assert smoothed_eye_dist < base_eye_dist * 1.01

    def test_timers_updated(self):
        pipe = _stub_pipeline(enable_timers=True)
        pipe._detector.detect.return_value = []

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stats = {
            "frames_detected": 0,
            "faces_identified": 0,
            "timers": {"detect": 0.0, "recognize": 0.0},
        }

        pipe._prepare_frame(frame, 0, stats)
        # detect timer should have been incremented (even if tiny)
        assert stats["timers"]["detect"] >= 0.0


# ---------------------------------------------------------------------------
# _finish_frame
# ---------------------------------------------------------------------------


class TestFinishFrame:
    def test_no_swap_pairs_returns_original(self):
        pipe = _stub_pipeline()
        frame = np.ones((4, 4, 3), dtype=np.uint8) * 42
        result, local, mask = pipe._finish_frame(frame, 0, [], [])
        assert np.array_equal(result, frame)
        assert local["frames_swapped"] == 0

    def test_swap_pairs_triggers_swap_and_blend(self):
        pipe = _stub_pipeline()
        pipe._swapper.swap_multiple.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )
        pipe._blender.blend_all.return_value = (np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))

        frame = np.ones((4, 4, 3), dtype=np.uint8)
        tracked = [MagicMock()]
        pairs = [("src", "tgt")]

        result, local, mask = pipe._finish_frame(frame, 0, tracked, pairs)
        assert local["frames_swapped"] == 1
        assert local["faces_swapped"] == 1
        pipe._swapper.swap_multiple.assert_called_once()
        pipe._blender.blend_all.assert_called_once()

    def test_enhancer_called_when_available(self):
        pipe = _stub_pipeline()
        pipe._enhancer.available = True
        pipe._enhancer.enhance_faces.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )
        pipe._swapper.swap_multiple.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )
        pipe._blender.blend_all.return_value = (np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))

        frame = np.ones((4, 4, 3), dtype=np.uint8)
        result, local, mask = pipe._finish_frame(frame, 0, [MagicMock()], [("s", "t")])
        pipe._enhancer.enhance_faces.assert_called_once()
        assert local["enhance"] >= 0.0

    def test_timer_values_populated(self):
        pipe = _stub_pipeline()
        pipe._swapper.swap_multiple.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )
        pipe._blender.blend_all.return_value = (np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))

        frame = np.ones((4, 4, 3), dtype=np.uint8)
        _, local, mask = pipe._finish_frame(frame, 0, [MagicMock()], [("s", "t")])
        assert local["swap"] >= 0.0
        assert local["blend"] >= 0.0


# ---------------------------------------------------------------------------
# _merge_finish_stats
# ---------------------------------------------------------------------------


class TestMergeFinishStats:
    def test_counters_merged(self):
        stats = {"frames_swapped": 1, "faces_swapped": 2}
        local = {
            "frames_swapped": 1,
            "faces_swapped": 3,
            "swap": 0.0,
            "blend": 0.0,
            "enhance": 0.0,
        }
        CharacterReplacementPipeline._merge_finish_stats(stats, local)
        assert stats["frames_swapped"] == 2
        assert stats["faces_swapped"] == 5

    def test_timers_merged_when_present(self):
        stats = {
            "frames_swapped": 0,
            "faces_swapped": 0,
            "timers": {"swap": 1.0, "blend": 2.0, "enhance": 3.0},
        }
        local = {
            "frames_swapped": 0,
            "faces_swapped": 0,
            "swap": 0.5,
            "blend": 0.5,
            "enhance": 0.5,
        }
        CharacterReplacementPipeline._merge_finish_stats(stats, local)
        assert stats["timers"]["swap"] == pytest.approx(1.5)
        assert stats["timers"]["blend"] == pytest.approx(2.5)
        assert stats["timers"]["enhance"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# _process_frame (sequential composition)
# ---------------------------------------------------------------------------


class TestProcessFrame:
    def test_returns_frame_for_no_faces(self):
        pipe = _stub_pipeline()
        pipe._detector.detect.return_value = []

        frame = np.ones((4, 4, 3), dtype=np.uint8) * 99
        stats = {
            "frames_detected": 0,
            "faces_identified": 0,
            "frames_swapped": 0,
            "faces_swapped": 0,
        }
        result = pipe._process_frame(frame, 0, stats)
        assert np.array_equal(result, frame)
        assert stats["frames_swapped"] == 0


class TestTemporalFaceBlend:
    def test_no_overlap_skips_blending(self):
        pipe = _stub_pipeline()
        pipe._cfg.temporal_smooth_alpha = 0.5

        first = np.zeros((4, 4, 3), dtype=np.uint8)
        first[:2, :2] = 100
        first_mask = np.zeros((4, 4), dtype=np.uint8)
        first_mask[:2, :2] = 255
        pipe._apply_temporal_face_blend(first, first_mask)

        second = np.zeros((4, 4, 3), dtype=np.uint8)
        second[2:, 2:] = 200
        second_mask = np.zeros((4, 4), dtype=np.uint8)
        second_mask[2:, 2:] = 255

        result = pipe._apply_temporal_face_blend(second, second_mask)
        np.testing.assert_array_equal(result, second)

    def test_overlap_blends_only_shared_face_region(self):
        pipe = _stub_pipeline()
        pipe._cfg.temporal_smooth_alpha = 0.5

        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :2] = 255

        first = np.zeros((4, 4, 3), dtype=np.uint8)
        first[:2, :2] = 100
        pipe._apply_temporal_face_blend(first, mask)

        second = np.zeros((4, 4, 3), dtype=np.uint8)
        second[:2, :2] = 200
        result = pipe._apply_temporal_face_blend(second, mask)

        assert result[0, 0, 0] == 150
        assert result[3, 3, 0] == 0


# ---------------------------------------------------------------------------
# _drain_one
# ---------------------------------------------------------------------------


class TestDrainOne:
    def test_writes_result_and_merges(self):
        pipe = _stub_pipeline()
        writer = MagicMock()
        stats = {"frames_total": 0, "frames_swapped": 0, "faces_swapped": 0}
        local = {
            "frames_swapped": 1,
            "faces_swapped": 2,
            "swap": 0.1,
            "blend": 0.2,
            "enhance": 0.0,
        }

        future = Future()
        future.set_result((np.zeros((2, 2, 3), dtype=np.uint8), local, np.zeros((2, 2), dtype=np.uint8)))
        pending = deque([(np.zeros((2, 2, 3), dtype=np.uint8), future)])

        count = pipe._drain_one(pending, writer, stats, 0, 0.0, 100)
        assert count == 1
        writer.write.assert_called_once()
        assert stats["frames_swapped"] == 1
        assert stats["faces_swapped"] == 2
        assert stats["frames_total"] == 1
        assert len(pending) == 0


# ---------------------------------------------------------------------------
# _run_parallel (integration-ish)
# ---------------------------------------------------------------------------


class TestRunParallel:
    def test_parallel_processes_all_frames_in_order(self):
        pipe = _stub_pipeline(batch_size=2)

        # Two frames
        frames = [
            np.ones((4, 4, 3), dtype=np.uint8) * i for i in range(2)
        ]

        # Stub _prepare_frame to pass through
        pipe._prepare_frame = MagicMock(
            side_effect=lambda f, idx, s: (f, idx, [], [])
        )
        # Stub _finish_frame to return the frame unchanged
        pipe._finish_frame = MagicMock(
            side_effect=lambda f, idx, tf, sp: (
                f,
                {
                    "frames_swapped": 0,
                    "faces_swapped": 0,
                    "swap": 0.0,
                    "blend": 0.0,
                    "enhance": 0.0,
                },
                np.zeros((4, 4), dtype=np.uint8)
            )
        )

        writer = MagicMock()
        reader = MagicMock()
        reader.__iter__ = MagicMock(return_value=iter(frames))
        reader.total_frames = 2

        stats = {
            "frames_total": 0,
            "frames_swapped": 0,
            "faces_swapped": 0,
            "frames_detected": 0,
            "faces_identified": 0,
        }

        pipe._run_parallel(reader, writer, stats, 0.0)

        assert writer.write.call_count == 2
        assert stats["frames_total"] == 2
        # Verify ordering: first written frame is frames[0]
        first_written = writer.write.call_args_list[0][0][0]
        assert np.array_equal(first_written, frames[0])


# ---------------------------------------------------------------------------
# run() dispatches correctly
# ---------------------------------------------------------------------------


class TestRunDispatches:
    def test_batch_1_uses_sequential(self):
        pipe = _stub_pipeline(batch_size=1)
        pipe._run_sequential = MagicMock()
        pipe._run_parallel = MagicMock()

        with patch("chararep.pipeline.VideoReader") as MockReader, \
             patch("chararep.pipeline.VideoWriter") as MockWriter:
            reader_inst = MagicMock()
            reader_inst.__enter__ = MagicMock(return_value=reader_inst)
            reader_inst.__exit__ = MagicMock(return_value=False)
            reader_inst.width = 640
            reader_inst.height = 480
            reader_inst.fps = 30.0
            reader_inst.total_frames = 0
            MockReader.return_value = reader_inst

            writer_inst = MagicMock()
            writer_inst.__enter__ = MagicMock(return_value=writer_inst)
            writer_inst.__exit__ = MagicMock(return_value=False)
            MockWriter.return_value = writer_inst

            pipe.run()

        pipe._run_sequential.assert_called_once()
        pipe._run_parallel.assert_not_called()

    def test_batch_gt1_uses_parallel(self):
        pipe = _stub_pipeline(batch_size=4)
        pipe._run_sequential = MagicMock()
        pipe._run_parallel = MagicMock()

        with patch("chararep.pipeline.VideoReader") as MockReader, \
             patch("chararep.pipeline.VideoWriter") as MockWriter:
            reader_inst = MagicMock()
            reader_inst.__enter__ = MagicMock(return_value=reader_inst)
            reader_inst.__exit__ = MagicMock(return_value=False)
            reader_inst.width = 640
            reader_inst.height = 480
            reader_inst.fps = 30.0
            reader_inst.total_frames = 0
            MockReader.return_value = reader_inst

            writer_inst = MagicMock()
            writer_inst.__enter__ = MagicMock(return_value=writer_inst)
            writer_inst.__exit__ = MagicMock(return_value=False)
            MockWriter.return_value = writer_inst

            pipe.run()

        pipe._run_parallel.assert_called_once()
        pipe._run_sequential.assert_not_called()


# ---------------------------------------------------------------------------
# CLI --batch argument
# ---------------------------------------------------------------------------


class TestBatchCliArg:
    def test_default_batch_size(self, monkeypatch):
        """--batch defaults to 4."""
        monkeypatch.setattr("sys.argv", ["chararep", "-i", "in.mp4", "-o", "out.mp4"])
        from chararep.main import _parse_args
        args = _parse_args()
        assert args.batch_size == 4

    def test_custom_batch_size(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["chararep", "-i", "in.mp4", "-o", "out.mp4", "--batch", "8"],
        )
        from chararep.main import _parse_args
        args = _parse_args()
        assert args.batch_size == 8

    def test_batch_1_sequential(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["chararep", "-i", "in.mp4", "-o", "out.mp4", "--batch", "1"],
        )
        from chararep.main import _parse_args
        args = _parse_args()
        assert args.batch_size == 1

    def test_batch_zero_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["chararep", "-i", "in.mp4", "-o", "out.mp4", "--batch", "0"],
        )
        from chararep.main import _parse_args
        with pytest.raises(SystemExit):
            _parse_args()

    def test_batch_negative_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["chararep", "-i", "in.mp4", "-o", "out.mp4", "--batch", "-1"],
        )
        from chararep.main import _parse_args
        with pytest.raises(SystemExit):
            _parse_args()

    def test_default_temporal_smooth_alpha(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["chararep", "-i", "in.mp4", "-o", "out.mp4"])
        from chararep.main import _parse_args
        args = _parse_args()
        assert args.temporal_smooth_alpha == 0.0

    def test_custom_temporal_smooth_alpha(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "chararep",
                "-i",
                "in.mp4",
                "-o",
                "out.mp4",
                "--temporal-smooth-alpha",
                "0.2",
            ],
        )
        from chararep.main import _parse_args
        args = _parse_args()
        assert args.temporal_smooth_alpha == pytest.approx(0.2)

    def test_temporal_smooth_alpha_out_of_range_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "chararep",
                "-i",
                "in.mp4",
                "-o",
                "out.mp4",
                "--temporal-smooth-alpha",
                "1.5",
            ],
        )
        from chararep.main import _parse_args
        with pytest.raises(SystemExit):
            _parse_args()


# ---------------------------------------------------------------------------
# TrackedFace snapshot in parallel mode
# ---------------------------------------------------------------------------


class TestTrackedFaceSnapshot:
    def test_parallel_snapshots_tracked_faces(self):
        """Worker threads receive copies, not the original TrackedFace objects."""
        from chararep.face_detector import TrackedFace

        pipe = _stub_pipeline(batch_size=2)

        tf = TrackedFace(
            track_id=1,
            bbox=np.array([10, 10, 50, 50], dtype=np.float32),
            landmarks=np.zeros((5, 2), dtype=np.float32),
            identity_label="hero",
        )

        # _prepare_frame returns real TrackedFace objects
        pipe._prepare_frame = MagicMock(
            side_effect=lambda f, idx, s: (f, idx, [tf], [])
        )

        # Capture the tracked_faces list passed to _finish_frame
        captured = []

        def capture_finish(f, idx, tracked, pairs):
            captured.append(tracked)
            return f, {
                "frames_swapped": 0,
                "faces_swapped": 0,
                "swap": 0.0,
                "blend": 0.0,
                "enhance": 0.0,
            }, np.zeros((4, 4), dtype=np.uint8)

        pipe._finish_frame = MagicMock(side_effect=capture_finish)

        writer = MagicMock()
        reader = MagicMock()
        reader.__iter__ = MagicMock(
            return_value=iter([np.zeros((4, 4, 3), dtype=np.uint8)])
        )
        reader.total_frames = 1

        stats = {
            "frames_total": 0,
            "frames_swapped": 0,
            "faces_swapped": 0,
            "frames_detected": 0,
            "faces_identified": 0,
        }

        pipe._run_parallel(reader, writer, stats, 0.0)

        assert len(captured) == 1
        # The tracked face passed to worker should be a copy, not the original
        assert captured[0][0] is not tf
        # But should have the same data
        assert captured[0][0].track_id == tf.track_id
        assert captured[0][0].identity_label == tf.identity_label
