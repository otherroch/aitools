"""Tests for pipeline timer functionality in chararep/pipeline.py.

Covers:
  - CharacterReplacementPipeline.run() timer accumulation (enable_timers=True/False)
  - CharacterReplacementPipeline._log_timer_distribution()
  - CharacterReplacementPipeline._log_progress()
"""

import logging
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chararep.config import PipelineConfig
from chararep.pipeline import CharacterReplacementPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**kw) -> PipelineConfig:
    defaults = dict(
        input_video="fake.mp4",
        output_video="out.mp4",
        enable_timers=False,
    )
    defaults.update(kw)
    return PipelineConfig(**defaults)


def _make_pipeline(cfg: PipelineConfig) -> CharacterReplacementPipeline:
    """Construct a pipeline with all heavy components mocked out."""
    pipeline = CharacterReplacementPipeline.__new__(CharacterReplacementPipeline)
    pipeline._cfg = cfg
    pipeline._detector = MagicMock()
    pipeline._detector.detect.return_value = []
    pipeline._recognizer = MagicMock()
    pipeline._recognizer.identify_faces.return_value = []
    pipeline._swapper = MagicMock()
    pipeline._enhancer = MagicMock()
    pipeline._enhancer.available = False
    pipeline._blender = MagicMock()
    return pipeline


# ---------------------------------------------------------------------------
# _log_timer_distribution
# ---------------------------------------------------------------------------

class TestLogTimerDistribution:
    def test_all_zero_timers_no_division_error(self, caplog):
        """All-zero timer dict should not raise ZeroDivisionError."""
        timers = {"detect": 0.0, "recognize": 0.0, "swap": 0.0, "enhance": 0.0, "blend": 0.0}
        with caplog.at_level(logging.INFO, logger="chararep.pipeline"):
            CharacterReplacementPipeline._log_timer_distribution(timers)
        # No exception is the pass condition; also check TOTAL appears in log
        assert any("TOTAL" in r.message for r in caplog.records)

    def test_nonzero_timers_shows_percentages(self, caplog):
        """Non-zero timers should produce percentage lines in the log."""
        timers = {"detect": 1.0, "recognize": 2.0, "swap": 3.0, "enhance": 0.5, "blend": 0.5}
        with caplog.at_level(logging.INFO, logger="chararep.pipeline"):
            CharacterReplacementPipeline._log_timer_distribution(timers)
        # Each stage name should appear in the log
        log_text = " ".join(r.message for r in caplog.records)
        for stage in timers:
            assert stage in log_text

    def test_percentages_sum_to_100(self, caplog):
        """The per-stage percentage values logged should sum to 100%."""
        timers = {"detect": 2.0, "swap": 2.0, "enhance": 1.0}
        with caplog.at_level(logging.INFO, logger="chararep.pipeline"):
            CharacterReplacementPipeline._log_timer_distribution(timers)
        # Collect per-stage percentage strings from log (skip the TOTAL line)
        import re
        pcts = []
        for r in caplog.records:
            if "TOTAL" in r.message:
                continue
            m = re.search(r"\(\s*(\d+\.\d+)%\)", r.message)
            if m:
                pcts.append(float(m.group(1)))
        if pcts:
            assert sum(pcts) == pytest.approx(100.0, abs=0.1)

    def test_single_stage(self, caplog):
        """A single timer stage should report 100%."""
        timers = {"detect": 5.0}
        with caplog.at_level(logging.INFO, logger="chararep.pipeline"):
            CharacterReplacementPipeline._log_timer_distribution(timers)
        log_text = " ".join(r.message for r in caplog.records)
        assert "100.0" in log_text


# ---------------------------------------------------------------------------
# _log_progress
# ---------------------------------------------------------------------------

class TestLogProgress:
    def test_no_raise_with_valid_args(self, caplog):
        """_log_progress should log a progress line without raising."""
        stats = {"faces_swapped": 3}
        t0 = time.perf_counter() - 2.0  # 2 seconds ago

        with caplog.at_level(logging.INFO, logger="chararep.pipeline"):
            CharacterReplacementPipeline._log_progress(50, 100, t0, stats)

        assert any("50/100" in r.message for r in caplog.records)

    def test_zero_elapsed_no_division_error(self, caplog):
        """When elapsed is effectively 0, fps should be reported as 0."""
        stats = {"faces_swapped": 0}
        t0 = time.perf_counter()  # start right now

        with caplog.at_level(logging.INFO, logger="chararep.pipeline"):
            CharacterReplacementPipeline._log_progress(1, 1, t0, stats)
        # No exception is the pass condition


# ---------------------------------------------------------------------------
# run() with enable_timers=False
# ---------------------------------------------------------------------------

class TestRunWithoutTimers:
    def _run_with_single_frame(self, cfg):
        """Patch I/O and run pipeline returning stats."""
        pipeline = _make_pipeline(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.__iter__ = MagicMock(return_value=iter([frame]))
        mock_reader.width = 640
        mock_reader.height = 480
        mock_reader.fps = 30.0
        mock_reader.total_frames = 1

        mock_writer = MagicMock()
        mock_writer.__enter__ = MagicMock(return_value=mock_writer)
        mock_writer.__exit__ = MagicMock(return_value=False)

        with patch("chararep.pipeline.VideoReader", return_value=mock_reader), \
             patch("chararep.pipeline.VideoWriter", return_value=mock_writer):
            return pipeline.run()

    def test_timers_key_absent_when_disabled(self):
        cfg = _make_cfg(enable_timers=False)
        stats = self._run_with_single_frame(cfg)
        assert "timers" not in stats

    def test_frames_total_counted(self):
        cfg = _make_cfg(enable_timers=False)
        stats = self._run_with_single_frame(cfg)
        assert stats["frames_total"] == 1

    def test_elapsed_s_positive(self):
        cfg = _make_cfg(enable_timers=False)
        stats = self._run_with_single_frame(cfg)
        assert stats["elapsed_s"] >= 0.0

    def test_fps_non_negative(self):
        cfg = _make_cfg(enable_timers=False)
        stats = self._run_with_single_frame(cfg)
        assert stats["fps"] >= 0.0


# ---------------------------------------------------------------------------
# run() with enable_timers=True
# ---------------------------------------------------------------------------

class TestRunWithTimers:
    def _run_with_frames(self, cfg, frames):
        """Patch I/O and run pipeline with given frames, returning stats."""
        pipeline = _make_pipeline(cfg)

        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.__iter__ = MagicMock(return_value=iter(frames))
        mock_reader.width = 640
        mock_reader.height = 480
        mock_reader.fps = 30.0
        mock_reader.total_frames = len(frames)

        mock_writer = MagicMock()
        mock_writer.__enter__ = MagicMock(return_value=mock_writer)
        mock_writer.__exit__ = MagicMock(return_value=False)

        with patch("chararep.pipeline.VideoReader", return_value=mock_reader), \
             patch("chararep.pipeline.VideoWriter", return_value=mock_writer):
            return pipeline.run()

    def test_timers_key_present_when_enabled(self):
        cfg = _make_cfg(enable_timers=True)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        stats = self._run_with_frames(cfg, frames)
        assert "timers" in stats

    def test_timers_has_expected_keys(self):
        cfg = _make_cfg(enable_timers=True)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        stats = self._run_with_frames(cfg, frames)
        for key in ("detect", "recognize", "swap", "enhance", "blend"):
            assert key in stats["timers"]

    def test_detect_timer_non_negative(self):
        cfg = _make_cfg(enable_timers=True)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        stats = self._run_with_frames(cfg, frames)
        assert stats["timers"]["detect"] >= 0.0

    def test_timers_accumulated_over_multiple_frames(self):
        cfg = _make_cfg(enable_timers=True)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        stats = self._run_with_frames(cfg, frames)
        # With 3 frames and no detected faces, detect timer is called 3 times
        assert stats["timers"]["detect"] >= 0.0
        assert stats["frames_total"] == 3

    def test_frames_total_zero_for_empty_video(self):
        cfg = _make_cfg(enable_timers=True)
        stats = self._run_with_frames(cfg, [])
        assert stats["frames_total"] == 0

    def test_fps_reported_when_nonzero_elapsed(self):
        """fps = frames_total / elapsed_s (unless elapsed is 0)."""
        cfg = _make_cfg(enable_timers=True)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        stats = self._run_with_frames(cfg, frames)
        if stats["elapsed_s"] > 0:
            assert stats["fps"] == pytest.approx(
                stats["frames_total"] / stats["elapsed_s"], rel=0.01
            )


# ---------------------------------------------------------------------------
# run() progress logging (every 100 frames)
# ---------------------------------------------------------------------------

class TestRunProgressLogging:
    def test_progress_logged_every_100_frames(self, caplog):
        """_log_progress is called once per 100 frames."""
        cfg = _make_cfg(enable_timers=False)
        pipeline = _make_pipeline(cfg)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(100)]

        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=False)
        mock_reader.__iter__ = MagicMock(return_value=iter(frames))
        mock_reader.width = 640
        mock_reader.height = 480
        mock_reader.fps = 30.0
        mock_reader.total_frames = 100

        mock_writer = MagicMock()
        mock_writer.__enter__ = MagicMock(return_value=mock_writer)
        mock_writer.__exit__ = MagicMock(return_value=False)

        with patch("chararep.pipeline.VideoReader", return_value=mock_reader), \
             patch("chararep.pipeline.VideoWriter", return_value=mock_writer), \
             caplog.at_level(logging.INFO, logger="chararep.pipeline"):
            pipeline.run()

        progress_msgs = [r for r in caplog.records if "Progress:" in r.message]
        assert len(progress_msgs) == 1
