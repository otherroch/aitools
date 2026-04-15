"""
tests/test_new_coverage.py

Tests to add coverage for previously uncovered modules:
  - videsc/video/messages.py
  - videsc/wd_cli.py
  - videsc/video/info.py
  - chararep/__main__.py
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import pytest


# ---------------------------------------------------------------------------
# videsc.video.messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def _vinfo(self, width=320, height=240, fps=24.0, num_frames=240, tot_time=10.0):
        return {
            "width": width,
            "height": height,
            "FPS": fps,
            "num_frames": num_frames,
            "tot_time": tot_time,
        }

    def test_basic_returns_user_message(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        msgs = build_messages("/tmp/v.mp4", vinfo, 320 * 240, 10, 0.0, "describe it")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert content[0]["type"] == "video"
        assert content[0]["video"] == "/tmp/v.mp4"

    def test_with_system_message(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        msgs = build_messages(
            "/tmp/v.mp4", vinfo, 0, 5, 0.0, "prompt", system="You are helpful."
        )
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert msgs[1]["role"] == "user"

    def test_is_omni_true_omits_pixels_and_nframes(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        msgs = build_messages("/tmp/v.mp4", vinfo, 0, 5, 0.0, "prompt", is_omni=True)
        content = msgs[0]["content"]
        video_block = next(c for c in content if c["type"] == "video")
        assert "total_pixels" not in video_block
        assert "nframes" not in video_block

    def test_audio_transcript_added_to_content(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        msgs = build_messages(
            "/tmp/v.mp4",
            vinfo,
            0,
            5,
            0.0,
            "prompt",
            audio_transcript="hello world",
        )
        content = msgs[0]["content"]
        text_blocks = [c for c in content if c["type"] == "text"]
        assert any("hello world" in b["text"] for b in text_blocks)

    def test_nframes_capped_at_768(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo(fps=1.0, num_frames=2000, tot_time=2000.0)
        msgs = build_messages("/tmp/v.mp4", vinfo, 100, 2000, 0.0, "p")
        video_block = msgs[0]["content"][0]
        assert video_block["nframes"] <= 768

    def test_tot_pixels_zero_uses_frame_dimensions(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo(width=100, height=200)
        msgs = build_messages("/tmp/v.mp4", vinfo, 0, 5, 0.0, "p")
        video_block = msgs[0]["content"][0]
        assert video_block["total_pixels"] == 100 * 200

    def test_spf_positive_recalculates_nframes(self):
        from videsc.video.messages import build_messages

        # spf=1.0, fps=30 → interval=30, new_nframes = 300/30 = 10
        vinfo = self._vinfo(fps=30.0, num_frames=300, tot_time=10.0)
        msgs = build_messages("/tmp/v.mp4", vinfo, 100, 300, 1.0, "p")
        video_block = msgs[0]["content"][0]
        assert video_block["nframes"] == 10

    def test_continue_prompt_appended(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        msgs = build_messages(
            "/tmp/v.mp4", vinfo, 0, 5, 0.0, "p", continue_prompt=True
        )
        content = msgs[0]["content"]
        text_blocks = [c for c in content if c["type"] == "text"]
        assert any("Continue" in b["text"] for b in text_blocks)

    def test_audio_segments_valid_timestamps(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo(fps=24.0, num_frames=240, tot_time=10.0)
        segments = [
            {"timestamp": [0.0, 2.5], "text": "Hello there"},
            {"timestamp": [3.0, 5.0], "text": "World"},
        ]
        msgs = build_messages(
            "/tmp/v.mp4", vinfo, 0, 10, 0.0, "p", audio_segments=segments
        )
        content = msgs[0]["content"]
        text_blocks = [c for c in content if c["type"] == "text"]
        combined = " ".join(b["text"] for b in text_blocks)
        assert "Hello there" in combined
        assert "World" in combined

    def test_audio_segments_skips_invalid_timestamps(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        segments = [
            {"timestamp": None, "text": "bad1"},
            {"timestamp": [None, 2.0], "text": "bad2"},
            {"timestamp": [1.0], "text": "wrong length"},
        ]
        msgs = build_messages(
            "/tmp/v.mp4", vinfo, 0, 5, 0.0, "p", audio_segments=segments
        )
        assert len(msgs) >= 1

    def test_prompt_text_appended_to_content(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        msgs = build_messages("/tmp/v.mp4", vinfo, 0, 5, 0.0, "my custom prompt")
        content = msgs[0]["content"]
        text_blocks = [c for c in content if c["type"] == "text"]
        assert any("my custom prompt" in b["text"] for b in text_blocks)

    def test_empty_prompt_produces_only_video_block(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo()
        msgs = build_messages("/tmp/v.mp4", vinfo, 0, 5, 0.0, "")
        content = msgs[0]["content"]
        assert all(c["type"] == "video" for c in content)

    def test_segment_no_text_uses_placeholder(self):
        from videsc.video.messages import build_messages

        vinfo = self._vinfo(fps=24.0, num_frames=240, tot_time=10.0)
        segments = [{"timestamp": [0.0, 2.0], "text": ""}]
        msgs = build_messages(
            "/tmp/v.mp4", vinfo, 0, 10, 0.0, "p", audio_segments=segments
        )
        content = msgs[0]["content"]
        text_blocks = [c for c in content if c["type"] == "text"]
        combined = " ".join(b["text"] for b in text_blocks)
        assert "<no text>" in combined

    def test_spf_positive_with_zero_fps_keeps_nframes(self):
        from videsc.video.messages import build_messages

        # fps=0.0 → interval = spf * 0 = 0 → fallback: new_nframes = nframes
        vinfo = self._vinfo(fps=0.0, num_frames=10, tot_time=10.0)
        msgs = build_messages("/tmp/v.mp4", vinfo, 100, 5, 1.0, "p")
        video_block = msgs[0]["content"][0]
        # nframes stays at max(1, 5) = 5 since interval was 0
        assert video_block["nframes"] == 5


# ---------------------------------------------------------------------------
# videsc.wd_cli
# ---------------------------------------------------------------------------


class TestWdCliParseArgs:
    def test_input_dir_defaults(self):
        from videsc.wd_cli import parse_args

        args = parse_args(["--input-dir", "/tmp"])
        assert args.input_dir == Path("/tmp")
        assert args.every_n == 30
        assert args.max_frames == 10
        assert args.prefix == ""
        assert args.threshold == 0.35
        assert args.model_repo == "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
        assert args.include_ratings is False
        assert args.no_skip_existing is False
        assert args.output_dir is None

    def test_custom_input_dir_args(self):
        from videsc.wd_cli import parse_args

        args = parse_args([
            "--input-dir", "/data",
            "--every-n", "15",
            "--max-frames", "20",
            "--prefix", "ohwx man",
            "--threshold", "0.25",
            "--model-repo", "myrepo/model",
            "--include-ratings",
            "--no-skip-existing",
            "--output-dir", "/out",
        ])
        assert args.every_n == 15
        assert args.max_frames == 20
        assert args.prefix == "ohwx man"
        assert args.threshold == 0.25
        assert args.model_repo == "myrepo/model"
        assert args.include_ratings is True
        assert args.no_skip_existing is True
        assert args.output_dir == Path("/out")

    def test_youtube_url_without_api_key_exits(self):
        from videsc.wd_cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--youtube-url", "https://youtube.com/watch?v=abc"])

    def test_youtube_url_with_api_key(self):
        from videsc.wd_cli import parse_args

        args = parse_args([
            "--youtube-url", "https://youtube.com/watch?v=abc",
            "--youtube-api-key", "key123",
        ])
        assert args.youtube_url == "https://youtube.com/watch?v=abc"
        assert args.youtube_api_key == "key123"

    def test_mutually_exclusive_input_sources(self):
        from videsc.wd_cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--input-dir", "/tmp",
                "--youtube-url", "https://youtube.com/watch?v=abc",
                "--youtube-api-key", "key",
            ])

    def test_no_input_source_exits(self):
        from videsc.wd_cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([])


class TestWdCliMain:
    def test_main_with_input_dir_calls_describe_folder(self, monkeypatch, tmp_path):
        called = {}

        def fake_describe_folder(*args, **kwargs):
            called["invoked"] = True
            return {"described": 5, "skipped": 2}

        describe_mod = types.SimpleNamespace(describe_folder=fake_describe_folder)
        monkeypatch.setitem(sys.modules, "videsc.describe", describe_mod)

        from videsc.wd_cli import main

        main(["--input-dir", str(tmp_path)])
        assert called.get("invoked") is True

    def test_main_with_youtube_url_calls_describe_youtube(self, monkeypatch):
        called = {}

        def fake_describe_youtube(*args, **kwargs):
            called["invoked"] = True
            return {"described": 1, "skipped": 0}

        describe_mod = types.SimpleNamespace(describe_youtube=fake_describe_youtube)
        monkeypatch.setitem(sys.modules, "videsc.describe", describe_mod)

        from videsc.wd_cli import main

        main([
            "--youtube-url", "https://youtube.com/watch?v=abc",
            "--youtube-api-key", "mykey",
        ])
        assert called.get("invoked") is True


# ---------------------------------------------------------------------------
# videsc.video.info
# ---------------------------------------------------------------------------


class TestGetVideoInfo:
    def _make_mock_cap(
        self, opened=True, frame_count=100, fps=24.0, width=1920, height=1080
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = opened
        prop_map = {
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }
        mock_cap.get.side_effect = lambda p: prop_map.get(p, 0.0)
        return mock_cap

    def test_basic_info_returned(self):
        from videsc.video.info import get_video_info

        mock_cap = self._make_mock_cap()
        with patch("videsc.video.info.cv2.VideoCapture", return_value=mock_cap):
            info = get_video_info("/fake/video.mp4")

        assert info["num_frames"] == 100
        assert info["FPS"] == 24.0
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["tot_time"] == pytest.approx(100 / 24.0)
        assert info["duration_minutes"] == pytest.approx((100 / 24.0) / 60.0)

    def test_fps_zero_sets_duration_to_zero(self):
        from videsc.video.info import get_video_info

        mock_cap = self._make_mock_cap(fps=0.0, frame_count=50)
        with patch("videsc.video.info.cv2.VideoCapture", return_value=mock_cap):
            info = get_video_info("/fake/zero_fps.mp4")

        assert info["tot_time"] == 0.0
        assert info["duration_minutes"] == 0.0

    def test_unopenable_video_calls_sys_exit(self):
        from videsc.video.info import get_video_info

        mock_cap = self._make_mock_cap(opened=False)
        with patch("videsc.video.info.cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(SystemExit):
                get_video_info("/fake/bad.mp4")

    def test_cap_is_released_after_read(self):
        from videsc.video.info import get_video_info

        mock_cap = self._make_mock_cap()
        with patch("videsc.video.info.cv2.VideoCapture", return_value=mock_cap):
            get_video_info("/fake/video.mp4")

        mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# chararep.__main__
# ---------------------------------------------------------------------------


class TestChararepMainModule:
    def test_main_module_invokes_main(self, monkeypatch):
        """chararep/__main__.py should call chararep.main.main() exactly once."""
        import chararep.main as chararep_main_mod

        mock_main = MagicMock()
        monkeypatch.setattr(chararep_main_mod, "main", mock_main)

        # Force reload of __main__ so module-level code runs again
        sys.modules.pop("chararep.__main__", None)
        importlib.import_module("chararep.__main__")

        mock_main.assert_called_once()
