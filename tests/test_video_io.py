"""Tests for chararep/video_io.py — VideoReader and VideoWriter."""

import subprocess
import threading
import time
import types
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_capture(frames=None, width=640, height=480, fps=30.0):
    """Return a mock cv2.VideoCapture that yields *frames* then returns ok=False."""
    if frames is None:
        frames = [np.zeros((height, width, 3), dtype=np.uint8)]

    frames = list(frames)
    state = {"idx": 0}

    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: width,   # CAP_PROP_FRAME_WIDTH
        4: height,  # CAP_PROP_FRAME_HEIGHT
        5: fps,     # CAP_PROP_FPS
        7: len(frames),  # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)

    def _read():
        i = state["idx"]
        if i < len(frames):
            state["idx"] += 1
            return True, frames[i].copy()
        return False, None

    cap.read.side_effect = _read
    return cap


def _make_fake_proc():
    """Return a mock subprocess.Popen with a writable stdin."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write.return_value = None
    proc.stdin.close.return_value = None
    proc.wait.return_value = 0
    return proc


# ---------------------------------------------------------------------------
# VideoReader
# ---------------------------------------------------------------------------

class TestVideoReaderInit:
    def test_open_nonexistent_raises(self):
        """FileNotFoundError when cv2 can't open the path."""
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            cap = MagicMock()
            cap.isOpened.return_value = False
            mock_cap_cls.return_value = cap
            with pytest.raises(FileNotFoundError, match="Cannot open video"):
                VideoReader("/nonexistent/video.mp4")

    def test_properties_populated_from_cap(self):
        """width, height, fps, total_frames, duration_s are set from cap.get()."""
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            mock_cap_cls.return_value = _make_fake_capture(
                frames=[np.zeros((480, 640, 3), dtype=np.uint8)] * 5,
                width=640,
                height=480,
                fps=25.0,
            )
            reader = VideoReader("fake.mp4", queue_size=4)

        assert reader.width == 640
        assert reader.height == 480
        assert reader.fps == pytest.approx(25.0)
        assert reader.total_frames == 5
        assert reader.duration_s == pytest.approx(5 / 25.0)

    def test_fps_defaults_to_30_when_zero(self):
        """When cap.get(FPS) returns 0, fps defaults to 30.0."""
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            cap = _make_fake_capture(fps=0.0)
            mock_cap_cls.return_value = cap
            reader = VideoReader("fake.mp4")

        assert reader.fps == pytest.approx(30.0)


class TestVideoReaderStartStop:
    def test_start_spawns_thread(self):
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            mock_cap_cls.return_value = _make_fake_capture(frames=[])
            reader = VideoReader("fake.mp4", queue_size=4)

        reader.start()
        assert reader._thread is not None
        reader.stop()

    def test_stop_releases_capture(self):
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            cap = _make_fake_capture(frames=[])
            mock_cap_cls.return_value = cap
            reader = VideoReader("fake.mp4", queue_size=4)

        reader.start()
        reader.stop()
        cap.release.assert_called_once()

    def test_context_manager(self):
        """__enter__ starts, __exit__ stops the reader."""
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            cap = _make_fake_capture(frames=[])
            mock_cap_cls.return_value = cap
            reader = VideoReader("fake.mp4", queue_size=4)

        with reader:
            assert reader._thread is not None
        cap.release.assert_called_once()


class TestVideoReaderIteration:
    def test_iter_yields_frames(self):
        """__iter__ yields all frames then stops."""
        from chararep.video_io import VideoReader

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            mock_cap_cls.return_value = _make_fake_capture(frames=[frame1, frame2])
            reader = VideoReader("fake.mp4", queue_size=8)

        collected = []
        with reader:
            for frame in reader:
                collected.append(frame.copy())

        assert len(collected) == 2

    def test_iter_empty_video_yields_nothing(self):
        """An empty video yields zero frames."""
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            mock_cap_cls.return_value = _make_fake_capture(frames=[])
            reader = VideoReader("fake.mp4", queue_size=4)

        with reader:
            collected = [f for f in reader]

        assert collected == []

    def test_read_returns_none_at_eos(self):
        """read() returns None after all frames have been consumed."""
        from chararep.video_io import VideoReader

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            mock_cap_cls.return_value = _make_fake_capture(frames=[frame])
            reader = VideoReader("fake.mp4", queue_size=8)

        with reader:
            first = reader.read()
            second = reader.read()

        assert first is not None
        assert second is None

    def test_read_returns_none_when_stop_event_set(self):
        """read() returns None when stop_event is set before any data arrives."""
        from chararep.video_io import VideoReader

        with patch("chararep.video_io.cv2.VideoCapture") as mock_cap_cls:
            mock_cap_cls.return_value = _make_fake_capture(frames=[])
            reader = VideoReader("fake.mp4", queue_size=4)

        # Set the stop event before starting so the decode thread exits immediately
        # and no EOS is placed in the queue; read() must detect the stopped state.
        reader._stop_event.set()
        reader.start()
        if reader._thread is not None:
            reader._thread.join(timeout=2)
        result = reader.read(timeout=0.2)
        reader._cap.release()
        assert result is None


# ---------------------------------------------------------------------------
# VideoWriter
# ---------------------------------------------------------------------------

class TestVideoWriterInit:
    def test_no_audio_uses_same_path(self):
        """Without audio, _tmp_path == _final_path == path."""
        from chararep.video_io import VideoWriter

        w = VideoWriter("out.mp4", width=640, height=480, fps=30.0)
        assert w._tmp_path == "out.mp4"
        assert w._final_path == "out.mp4"

    def test_with_audio_uses_tmp_path(self):
        """When audio_source is set, a temp path is used for raw video."""
        from chararep.video_io import VideoWriter

        w = VideoWriter(
            "out.mp4",
            width=640,
            height=480,
            fps=30.0,
            audio_source="input.mp4",
        )
        assert w._tmp_path != w._final_path
        assert w._tmp_path.endswith(".tmp.mp4")
        assert w._final_path == "out.mp4"

    def test_ffmpeg_command_contains_codec(self):
        """The ffmpeg command includes the specified codec."""
        from chararep.video_io import VideoWriter

        w = VideoWriter("out.mp4", width=320, height=240, fps=24.0, codec="libx265")
        assert "libx265" in w._cmd

    def test_ffmpeg_command_contains_crf(self):
        """The ffmpeg command includes the CRF value."""
        from chararep.video_io import VideoWriter

        w = VideoWriter("out.mp4", width=320, height=240, fps=24.0, crf=23)
        assert "23" in w._cmd


class TestVideoWriterStartStop:
    def test_start_spawns_thread_and_proc(self):
        from chararep.video_io import VideoWriter

        with patch("chararep.video_io.subprocess.Popen") as mock_popen:
            mock_popen.return_value = _make_fake_proc()
            w = VideoWriter("out.mp4", width=640, height=480, fps=30.0)
            w.start()
            assert w._thread is not None
            assert w._thread.is_alive()
            w.stop()

    def test_context_manager(self):
        """__enter__ starts, __exit__ stops the writer."""
        from chararep.video_io import VideoWriter

        proc = _make_fake_proc()
        with patch("chararep.video_io.subprocess.Popen", return_value=proc):
            w = VideoWriter("out.mp4", width=640, height=480, fps=30.0)
            with w:
                assert w._thread is not None

    def test_write_then_stop(self):
        """write() enqueues a frame; stop() drains and closes stdin."""
        from chararep.video_io import VideoWriter

        proc = _make_fake_proc()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("chararep.video_io.subprocess.Popen", return_value=proc):
            w = VideoWriter("out.mp4", width=640, height=480, fps=30.0)
            with w:
                w.write(frame)

        proc.stdin.close.assert_called()

    def test_multiple_frames_written(self):
        """Writing multiple frames should not raise."""
        from chararep.video_io import VideoWriter

        proc = _make_fake_proc()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]

        with patch("chararep.video_io.subprocess.Popen", return_value=proc):
            w = VideoWriter("out.mp4", width=640, height=480, fps=30.0)
            with w:
                for f in frames:
                    w.write(f)


class TestVideoWriterMuxAudio:
    def test_mux_audio_success(self, tmp_path):
        """When ffmpeg mux succeeds (returncode 0), tmp file is removed."""
        from chararep.video_io import VideoWriter

        tmp_video = tmp_path / "out.mp4.tmp.mp4"
        tmp_video.write_bytes(b"fake")
        final = tmp_path / "out.mp4"

        proc = _make_fake_proc()
        mux_result = MagicMock()
        mux_result.returncode = 0
        mux_result.stderr = ""

        w = VideoWriter(
            str(final),
            width=640,
            height=480,
            fps=30.0,
            audio_source="input.mp4",
        )
        w._tmp_path = str(tmp_video)
        w._final_path = str(final)

        with patch("chararep.video_io.subprocess.run", return_value=mux_result):
            w._mux_audio()

        assert not tmp_video.exists()

    def test_mux_audio_failure_falls_back(self, tmp_path):
        """When ffmpeg mux fails, the tmp file is renamed to the final path."""
        from chararep.video_io import VideoWriter

        tmp_video = tmp_path / "out.mp4.tmp.mp4"
        tmp_video.write_bytes(b"fake")
        final = tmp_path / "out.mp4"

        w = VideoWriter(
            str(final),
            width=640,
            height=480,
            fps=30.0,
            audio_source="input.mp4",
        )
        w._tmp_path = str(tmp_video)
        w._final_path = str(final)

        mux_result = MagicMock()
        mux_result.returncode = 1
        mux_result.stderr = "some error"

        with patch("chararep.video_io.subprocess.run", return_value=mux_result):
            w._mux_audio()

        assert final.exists()
        assert not tmp_video.exists()

    def test_mux_audio_timeout_falls_back(self, tmp_path):
        """When the mux times out, the tmp file is renamed to the final path."""
        from chararep.video_io import VideoWriter

        tmp_video = tmp_path / "out.mp4.tmp.mp4"
        tmp_video.write_bytes(b"fake")
        final = tmp_path / "out.mp4"

        w = VideoWriter(
            str(final),
            width=640,
            height=480,
            fps=30.0,
            audio_source="input.mp4",
        )
        w._tmp_path = str(tmp_video)
        w._final_path = str(final)

        with patch(
            "chararep.video_io.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 300),
        ):
            w._mux_audio()

        assert final.exists()
        assert not tmp_video.exists()
