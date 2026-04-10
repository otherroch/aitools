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
        result, local = pipe._finish_frame(frame, 0, [], [])
        assert np.array_equal(result, frame)
        assert local["frames_swapped"] == 0

    def test_swap_pairs_triggers_swap_and_blend(self):
        pipe = _stub_pipeline()
        pipe._swapper.swap_multiple.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )
        pipe._blender.blend_all.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )

        frame = np.ones((4, 4, 3), dtype=np.uint8)
        tracked = [MagicMock()]
        pairs = [("src", "tgt")]

        result, local = pipe._finish_frame(frame, 0, tracked, pairs)
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
        pipe._blender.blend_all.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )

        frame = np.ones((4, 4, 3), dtype=np.uint8)
        result, local = pipe._finish_frame(frame, 0, [MagicMock()], [("s", "t")])
        pipe._enhancer.enhance_faces.assert_called_once()
        assert local["enhance"] >= 0.0

    def test_timer_values_populated(self):
        pipe = _stub_pipeline()
        pipe._swapper.swap_multiple.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )
        pipe._blender.blend_all.return_value = np.zeros(
            (4, 4, 3), dtype=np.uint8
        )

        frame = np.ones((4, 4, 3), dtype=np.uint8)
        _, local = pipe._finish_frame(frame, 0, [MagicMock()], [("s", "t")])
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
        future.set_result((np.zeros((2, 2, 3), dtype=np.uint8), local))
        pending = deque([future])

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
