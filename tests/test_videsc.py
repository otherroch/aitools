"""
tests/test_videsc.py

Unit tests for videsc.describe
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image

from videsc.describe import (
    SUPPORTED_VIDEO_EXTS,
    DEFAULT_EVERY_N_FRAMES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_THRESHOLD,
    extract_keyframes,
    _preprocess_frame,
    _build_caption,
    _describe_video_impl,
    _describe_folder_impl,
    extract_youtube_video_id,
    _validate_youtube_video,
    describe_youtube,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_frame(color=(128, 64, 32), size=(100, 100)) -> np.ndarray:
    arr = np.zeros((*size, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def _fake_video_capture(frames_rgb: list[np.ndarray]):
    """Return a mock cv2.VideoCapture that yields the given frames in order."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    call_count = [0]

    def read_side_effect():
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(frames_rgb):
            bgr = frames_rgb[idx][:, :, ::-1].copy()
            return True, bgr
        return False, None

    mock_cap.read.side_effect = read_side_effect
    return mock_cap


def make_video_stub(path: Path) -> Path:
    """Create a zero-byte stub file with a video extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake video data")
    return path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_supported_exts_includes_mp4(self):
        assert ".mp4" in SUPPORTED_VIDEO_EXTS

    def test_supported_exts_includes_mov(self):
        assert ".mov" in SUPPORTED_VIDEO_EXTS

    def test_default_every_n(self):
        assert DEFAULT_EVERY_N_FRAMES > 0

    def test_default_max_frames(self):
        assert DEFAULT_MAX_FRAMES > 0

    def test_default_threshold_range(self):
        assert 0.0 < DEFAULT_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# extract_keyframes
# ---------------------------------------------------------------------------


class TestExtractKeyframes:
    def test_unopenable_video_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        with patch("videsc.describe.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_vc.return_value = mock_cap
            frames = extract_keyframes(bad)
        assert frames == []

    def test_extracts_every_n_frames(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        frame = _dummy_frame()
        # 6 frames, sample every 2 → frames 0, 2, 4
        mock_cap = _fake_video_capture([frame] * 6)

        with patch("videsc.describe.cv2.VideoCapture", return_value=mock_cap):
            with patch("videsc.describe.cv2.cvtColor", side_effect=lambda f, c: f):
                frames = extract_keyframes(video, every_n=2, max_frames=100)

        assert len(frames) == 3

    def test_respects_max_frames(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        frame = _dummy_frame()
        mock_cap = _fake_video_capture([frame] * 30)

        with patch("videsc.describe.cv2.VideoCapture", return_value=mock_cap):
            with patch("videsc.describe.cv2.cvtColor", side_effect=lambda f, c: f):
                frames = extract_keyframes(video, every_n=1, max_frames=5)

        assert len(frames) == 5

    def test_returns_rgb_arrays(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")

        frame_rgb = _dummy_frame(color=(10, 20, 30))
        mock_cap = _fake_video_capture([frame_rgb])

        with patch("videsc.describe.cv2.VideoCapture", return_value=mock_cap):
            with patch(
                "videsc.describe.cv2.cvtColor", side_effect=lambda f, c: f
            ):
                frames = extract_keyframes(video, every_n=1, max_frames=1)

        assert len(frames) == 1
        assert frames[0].shape == frame_rgb.shape


# ---------------------------------------------------------------------------
# _preprocess_frame
# ---------------------------------------------------------------------------


class TestPreprocessFrame:
    def test_output_shape(self):
        frame = _dummy_frame(size=(200, 150))
        result = _preprocess_frame(frame, size=448)
        assert result.shape == (1, 448, 448, 3)

    def test_output_dtype_float32(self):
        frame = _dummy_frame()
        result = _preprocess_frame(frame, size=64)
        assert result.dtype == np.float32

    def test_non_square_frame_padded(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _preprocess_frame(frame, size=64)
        assert result.shape == (1, 64, 64, 3)


# ---------------------------------------------------------------------------
# _build_caption
# ---------------------------------------------------------------------------


class TestBuildCaption:
    def _make_session(self, tag_probs: np.ndarray):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.run.return_value = [tag_probs[np.newaxis, :]]
        return session

    def test_returns_string(self):
        tag_names = ["tag_a", "tag_b", "tag_c"]
        general_indices = [0, 1, 2]
        probs = np.array([0.9, 0.8, 0.1])  # tag_c below default threshold
        session = self._make_session(probs)
        frame = _dummy_frame()

        caption = _build_caption(
            session=session,
            frames=[frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=[],
            threshold=0.35,
            prefix="",
            include_ratings=False,
        )

        assert isinstance(caption, str)
        assert "tag a" in caption
        assert "tag b" in caption
        assert "tag c" not in caption  # below threshold

    def test_prefix_prepended(self):
        tag_names = ["tag_a"]
        general_indices = [0]
        probs = np.array([0.9])
        session = self._make_session(probs)
        frame = _dummy_frame()

        caption = _build_caption(
            session=session,
            frames=[frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=[],
            threshold=0.35,
            prefix="ohwx man",
            include_ratings=False,
        )

        assert caption.startswith("ohwx man,")

    def test_empty_prefix_not_added(self):
        tag_names = ["tag_a"]
        general_indices = [0]
        probs = np.array([0.9])
        session = self._make_session(probs)
        frame = _dummy_frame()

        caption = _build_caption(
            session=session,
            frames=[frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=[],
            threshold=0.35,
            prefix="",
            include_ratings=False,
        )

        assert not caption.startswith(",")

    def test_tags_sorted_by_confidence(self):
        tag_names = ["low_conf", "high_conf"]
        general_indices = [0, 1]
        # high_conf has higher score
        probs = np.array([0.4, 0.9])
        session = self._make_session(probs)
        frame = _dummy_frame()

        caption = _build_caption(
            session=session,
            frames=[frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=[],
            threshold=0.35,
            prefix="",
            include_ratings=False,
        )

        tags = [t.strip() for t in caption.split(",")]
        assert tags[0] == "high conf"
        assert tags[1] == "low conf"

    def test_aggregates_across_frames(self):
        """Tags seen across multiple frames are included."""
        tag_names = ["frame1_only", "frame2_only"]
        general_indices = [0, 1]
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]

        # frame 1: only tag 0 above threshold; frame 2: only tag 1 above threshold
        probs1 = np.array([0.9, 0.1])
        probs2 = np.array([0.1, 0.9])
        session.run.side_effect = [
            [probs1[np.newaxis, :]],
            [probs2[np.newaxis, :]],
        ]

        frame = _dummy_frame()
        caption = _build_caption(
            session=session,
            frames=[frame, frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=[],
            threshold=0.35,
            prefix="",
            include_ratings=False,
        )

        assert "frame1 only" in caption
        assert "frame2 only" in caption

    def test_no_tags_above_threshold_returns_prefix_only(self):
        tag_names = ["rare_tag"]
        general_indices = [0]
        probs = np.array([0.1])  # below any reasonable threshold
        session = self._make_session(probs)
        frame = _dummy_frame()

        caption = _build_caption(
            session=session,
            frames=[frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=[],
            threshold=0.35,
            prefix="my prefix",
            include_ratings=False,
        )

        assert caption == "my prefix"

    def test_no_tags_no_prefix_returns_empty(self):
        tag_names = ["rare_tag"]
        general_indices = [0]
        probs = np.array([0.1])
        session = self._make_session(probs)
        frame = _dummy_frame()

        caption = _build_caption(
            session=session,
            frames=[frame],
            tag_names=tag_names,
            general_indices=general_indices,
            rating_indices=[],
            threshold=0.35,
            prefix="",
            include_ratings=False,
        )

        assert caption == ""


# ---------------------------------------------------------------------------
# _describe_video_impl
# ---------------------------------------------------------------------------


class TestDescribeVideoImpl:
    def _make_ort(self, tag_probs: np.ndarray):
        ort = MagicMock()
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.run.return_value = [tag_probs[np.newaxis, :]]
        ort.InferenceSession.return_value = session
        return ort

    def _make_labels(self, tags=("tag_a", "tag_b")):
        return list(tags), [], list(range(len(tags)))

    def test_writes_txt_file(self, tmp_path):
        video = make_video_stub(tmp_path / "clip.mp4")
        frame = _dummy_frame()
        probs = np.array([0.9, 0.8])
        ort = self._make_ort(probs)

        with patch("videsc.describe.extract_keyframes", return_value=[frame]):
            with patch("videsc.describe._download_model", return_value=(tmp_path / "m.onnx", tmp_path / "t.csv")):
                with patch("videsc.describe._load_labels", return_value=self._make_labels()):
                    stats = _describe_video_impl(
                        ort=ort,
                        video_path=video,
                        output_dir=None,
                        every_n=30,
                        max_frames=10,
                        prefix="",
                        threshold=0.35,
                        model_repo="fake/repo",
                        include_ratings=False,
                        skip_existing=False,
                    )

        assert stats["described"] == 1
        assert stats["skipped"] == 0
        assert (tmp_path / "clip.txt").exists()

    def test_skips_when_txt_exists(self, tmp_path):
        video = make_video_stub(tmp_path / "clip.mp4")
        (tmp_path / "clip.txt").write_text("existing")
        ort = MagicMock()

        stats = _describe_video_impl(
            ort=ort,
            video_path=video,
            output_dir=None,
            every_n=30,
            max_frames=10,
            prefix="",
            threshold=0.35,
            model_repo="fake/repo",
            include_ratings=False,
            skip_existing=True,
        )

        assert stats["skipped"] == 1
        assert stats["described"] == 0
        # Content unchanged
        assert (tmp_path / "clip.txt").read_text() == "existing"

    def test_writes_to_output_dir(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        video = make_video_stub(src / "clip.mp4")
        probs = np.array([0.9])
        ort = self._make_ort(probs)
        frame = _dummy_frame()

        with patch("videsc.describe.extract_keyframes", return_value=[frame]):
            with patch("videsc.describe._download_model", return_value=(tmp_path / "m.onnx", tmp_path / "t.csv")):
                with patch("videsc.describe._load_labels", return_value=(["tag_a"], [], [0])):
                    stats = _describe_video_impl(
                        ort=ort,
                        video_path=video,
                        output_dir=out,
                        every_n=30,
                        max_frames=10,
                        prefix="",
                        threshold=0.35,
                        model_repo="fake/repo",
                        include_ratings=False,
                        skip_existing=False,
                    )

        assert (out / "clip.txt").exists()
        assert stats["described"] == 1

    def test_skips_when_no_frames_extracted(self, tmp_path):
        video = make_video_stub(tmp_path / "clip.mp4")
        ort = MagicMock()

        with patch("videsc.describe.extract_keyframes", return_value=[]):
            with patch("videsc.describe._download_model", return_value=(tmp_path / "m.onnx", tmp_path / "t.csv")):
                with patch("videsc.describe._load_labels", return_value=([], [], [])):
                    stats = _describe_video_impl(
                        ort=ort,
                        video_path=video,
                        output_dir=None,
                        every_n=30,
                        max_frames=10,
                        prefix="",
                        threshold=0.35,
                        model_repo="fake/repo",
                        include_ratings=False,
                        skip_existing=False,
                    )

        assert stats["skipped"] == 1
        assert stats["described"] == 0


# ---------------------------------------------------------------------------
# _describe_folder_impl
# ---------------------------------------------------------------------------


class TestDescribeFolderImpl:
    def _make_ort(self, probs: np.ndarray):
        ort = MagicMock()
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.run.return_value = [probs[np.newaxis, :]]
        ort.InferenceSession.return_value = session
        return ort

    def test_empty_directory_returns_zeros(self, tmp_path):
        ort = MagicMock()
        stats = _describe_folder_impl(
            ort=ort,
            input_dir=tmp_path / "in",
            output_dir=None,
            every_n=30,
            max_frames=10,
            prefix="",
            threshold=0.35,
            model_repo="fake/repo",
            include_ratings=False,
            skip_existing=True,
        )
        assert stats["described"] == 0
        assert stats["skipped"] == 0

    def test_non_video_files_ignored(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        (src / "readme.txt").write_text("ignore")
        (src / "photo.png").write_bytes(b"fake")

        ort = MagicMock()
        stats = _describe_folder_impl(
            ort=ort,
            input_dir=src,
            output_dir=None,
            every_n=30,
            max_frames=10,
            prefix="",
            threshold=0.35,
            model_repo="fake/repo",
            include_ratings=False,
            skip_existing=True,
        )
        assert stats["described"] == 0

    def test_describes_multiple_videos(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        for name in ["a.mp4", "b.mp4"]:
            (src / name).write_bytes(b"fake")

        probs = np.array([0.9])
        ort = self._make_ort(probs)
        frame = _dummy_frame()

        with patch("videsc.describe.extract_keyframes", return_value=[frame]):
            with patch("videsc.describe._download_model", return_value=(tmp_path / "m.onnx", tmp_path / "t.csv")):
                with patch("videsc.describe._load_labels", return_value=(["tag_a"], [], [0])):
                    stats = _describe_folder_impl(
                        ort=ort,
                        input_dir=src,
                        output_dir=None,
                        every_n=30,
                        max_frames=10,
                        prefix="",
                        threshold=0.35,
                        model_repo="fake/repo",
                        include_ratings=False,
                        skip_existing=False,
                    )

        assert stats["described"] == 2

    def test_skips_existing_txt(self, tmp_path):
        src = tmp_path / "in"
        src.mkdir()
        video = src / "clip.mp4"
        video.write_bytes(b"fake")
        (src / "clip.txt").write_text("existing")

        ort = MagicMock()
        with patch("videsc.describe._download_model", return_value=(tmp_path / "m.onnx", tmp_path / "t.csv")):
            with patch("videsc.describe._load_labels", return_value=([], [], [])):
                stats = _describe_folder_impl(
                    ort=ort,
                    input_dir=src,
                    output_dir=None,
                    every_n=30,
                    max_frames=10,
                    prefix="",
                    threshold=0.35,
                    model_repo="fake/repo",
                    include_ratings=False,
                    skip_existing=True,
                )

        assert stats["skipped"] == 1
        assert stats["described"] == 0

    def test_writes_to_output_dir(self, tmp_path):
        src = tmp_path / "in"
        out = tmp_path / "out"
        src.mkdir()
        (src / "clip.mp4").write_bytes(b"fake")

        probs = np.array([0.9])
        ort = self._make_ort(probs)
        frame = _dummy_frame()

        with patch("videsc.describe.extract_keyframes", return_value=[frame]):
            with patch("videsc.describe._download_model", return_value=(tmp_path / "m.onnx", tmp_path / "t.csv")):
                with patch("videsc.describe._load_labels", return_value=(["tag_a"], [], [0])):
                    _describe_folder_impl(
                        ort=ort,
                        input_dir=src,
                        output_dir=out,
                        every_n=30,
                        max_frames=10,
                        prefix="",
                        threshold=0.35,
                        model_repo="fake/repo",
                        include_ratings=False,
                        skip_existing=False,
                    )

        assert (out / "clip.txt").exists()


# ---------------------------------------------------------------------------
# extract_youtube_video_id
# ---------------------------------------------------------------------------


class TestExtractYoutubeVideoId:
    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_youtu_be_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_live_url(self):
        url = "https://www.youtube.com/live/dQw4w9WgXcQ"
        assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self):
        assert extract_youtube_video_id("https://example.com/video") is None

    def test_non_youtube_hostname_returns_none(self):
        assert extract_youtube_video_id("https://vimeo.com/123456") is None

    def test_empty_string_returns_none(self):
        assert extract_youtube_video_id("") is None


# ---------------------------------------------------------------------------
# _validate_youtube_video
# ---------------------------------------------------------------------------


class TestValidateYoutubeVideo:
    def _api_response(self, title="Test Video"):
        import json

        payload = json.dumps(
            {
                "items": [
                    {
                        "snippet": {"title": title},
                        "contentDetails": {"duration": "PT3M33S"},
                    }
                ]
            }
        ).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = payload
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        return mock_response

    def test_returns_metadata_on_success(self):
        with patch(
            "videsc.describe.urllib.request.urlopen",
            return_value=self._api_response(),
        ):
            result = _validate_youtube_video("fake_key", "dQw4w9WgXcQ")
        assert result is not None
        assert result["snippet"]["title"] == "Test Video"

    def test_returns_none_when_no_items(self):
        import json

        payload = json.dumps({"items": []}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("videsc.describe.urllib.request.urlopen", return_value=mock_resp):
            result = _validate_youtube_video("fake_key", "nonexistent")
        assert result is None

    def test_returns_none_on_missing_api_key(self):
        assert _validate_youtube_video("", "dQw4w9WgXcQ") is None

    def test_returns_none_on_missing_video_id(self):
        assert _validate_youtube_video("fake_key", "") is None

    def test_returns_none_on_network_error(self):
        with patch(
            "videsc.describe.urllib.request.urlopen",
            side_effect=Exception("network error"),
        ):
            result = _validate_youtube_video("fake_key", "dQw4w9WgXcQ")
        assert result is None


# ---------------------------------------------------------------------------
# describe_youtube
# ---------------------------------------------------------------------------


class TestDescribeYoutube:
    def _make_ort(self, tag_probs: np.ndarray):
        ort = MagicMock()
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.run.return_value = [tag_probs[np.newaxis, :]]
        ort.InferenceSession.return_value = session
        return ort

    def _meta(self, title="Test Video"):
        return {"snippet": {"title": title}, "contentDetails": {}}

    def test_writes_txt_named_by_video_id(self, tmp_path):
        import sys
        from contextlib import ExitStack

        out = tmp_path / "out"
        video_id = "dQw4w9WgXcQ"
        fake_video = tmp_path / f"{video_id}.mp4"
        fake_video.write_bytes(b"fake")

        with ExitStack() as stack:
            stack.enter_context(
                patch("videsc.describe.extract_youtube_video_id", return_value=video_id)
            )
            stack.enter_context(
                patch("videsc.describe._validate_youtube_video", return_value=self._meta())
            )
            stack.enter_context(
                patch("videsc.describe._download_youtube_video", return_value=fake_video)
            )
            stack.enter_context(patch("videsc.describe.shutil.rmtree"))
            stack.enter_context(
                patch("videsc.describe.tempfile.mkdtemp", return_value=str(tmp_path))
            )
            stack.enter_context(
                patch(
                    "videsc.describe._describe_video_impl",
                    return_value={"described": 1, "skipped": 0},
                )
            )
            stack.enter_context(patch.dict(sys.modules, {"onnxruntime": MagicMock()}))
            stats = describe_youtube(
                f"https://www.youtube.com/watch?v={video_id}",
                "fake_api_key",
                output_dir=out,
                skip_existing=False,
            )

        assert stats["described"] == 1
        assert stats["skipped"] == 0

    def test_skips_when_txt_exists(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        video_id = "dQw4w9WgXcQ"
        (out / f"{video_id}.txt").write_text("existing")

        with patch("videsc.describe.extract_youtube_video_id", return_value=video_id):
            with patch(
                "videsc.describe._validate_youtube_video", return_value=self._meta()
            ):
                with patch(
                    "videsc.describe._download_youtube_video"
                ) as mock_dl:
                    stats = describe_youtube(
                        f"https://www.youtube.com/watch?v={video_id}",
                        "fake_api_key",
                        output_dir=out,
                        skip_existing=True,
                    )
        assert stats["skipped"] == 1
        assert stats["described"] == 0
        mock_dl.assert_not_called()

    def test_returns_skipped_on_invalid_url(self, tmp_path):
        with patch("videsc.describe.extract_youtube_video_id", return_value=None):
            stats = describe_youtube(
                "https://example.com/not-youtube",
                "fake_api_key",
                output_dir=tmp_path,
            )
        assert stats["skipped"] == 1
        assert stats["described"] == 0

    def test_returns_skipped_on_validation_failure(self, tmp_path):
        with patch(
            "videsc.describe.extract_youtube_video_id", return_value="dQw4w9WgXcQ"
        ):
            with patch(
                "videsc.describe._validate_youtube_video", return_value=None
            ):
                stats = describe_youtube(
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "bad_key",
                    output_dir=tmp_path,
                )
        assert stats["skipped"] == 1
        assert stats["described"] == 0

    def test_returns_skipped_on_download_failure(self, tmp_path):
        with patch(
            "videsc.describe.extract_youtube_video_id", return_value="dQw4w9WgXcQ"
        ):
            with patch(
                "videsc.describe._validate_youtube_video", return_value=self._meta()
            ):
                with patch(
                    "videsc.describe._download_youtube_video", return_value=None
                ):
                    with patch("videsc.describe.shutil.rmtree"):
                        with patch(
                            "videsc.describe.tempfile.mkdtemp",
                            return_value=str(tmp_path),
                        ):
                            stats = describe_youtube(
                                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                "fake_key",
                                output_dir=tmp_path,
                                skip_existing=False,
                            )
        assert stats["skipped"] == 1
        assert stats["described"] == 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_input_dir_parsed(self, tmp_path):
        from videsc.cli import parse_args

        args = parse_args(["--input-dir", str(tmp_path)])
        assert args.input_dir == tmp_path
        assert args.youtube_url is None

    def test_youtube_url_with_api_key(self):
        from videsc.cli import parse_args

        args = parse_args(
            [
                "--youtube-url",
                "https://www.youtube.com/watch?v=abc123",
                "--youtube-api-key",
                "MY_KEY",
            ]
        )
        assert args.youtube_url == "https://www.youtube.com/watch?v=abc123"
        assert args.youtube_api_key == "MY_KEY"
        assert args.input_dir is None

    def test_youtube_url_without_api_key_errors(self):
        from videsc.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--youtube-url", "https://www.youtube.com/watch?v=abc123"])

    def test_input_dir_and_youtube_url_are_mutually_exclusive(self, tmp_path):
        from videsc.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--input-dir",
                    str(tmp_path),
                    "--youtube-url",
                    "https://www.youtube.com/watch?v=abc123",
                    "--youtube-api-key",
                    "MY_KEY",
                ]
            )

    def test_no_input_source_errors(self):
        from videsc.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([])
