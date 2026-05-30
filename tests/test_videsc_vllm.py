"""
tests/test_videsc_vllm.py

Unit tests for the vLLM integration in videsc.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_frame(size=(100, 100)) -> Image.Image:
    """Create a dummy PIL image for testing."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    arr[:, :] = (128, 64, 32)
    return Image.fromarray(arr, mode="RGB")


def _make_vllm_args(**overrides) -> argparse.Namespace:
    """Build a minimal argparse.Namespace mimicking --vllm CLI args."""
    defaults = dict(
        vllm=True,
        vl=False,
        vllm_host="localhost",
        vllm_port=8000,
        vllm_model="test-model",
        vllm_api_key="EMPTY",
        vllm_temperature=0.7,
        vllm_fps=1.0,
        vllm_chunk_duration=0.0,
        vllm_max_image_size=1280,
        video="/tmp/test_video.mp4",
        videos=None,
        indir=None,
        filelist=None,
        ext=[],
        outdir=None,
        prompt="Describe this video.",
        system="You are a helpful assistant.",
        max_new_tokens=1024,
        clip_start=0.0,
        clip_end=-1.0,
        consolidate=False,
        consolidate_prompt=None,
        dry=False,
        workers=2,
        dry_run=False,
        log_level="INFO",
        youtube_url=None,
        youtube_api_key=None,
        output_dir=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# VLLMClient tests
# ---------------------------------------------------------------------------


class TestVLLMClient:
    """Tests for videsc.model.vllm_client.VLLMClient."""

    @patch("openai.OpenAI")
    def test_init_creates_client(self, mock_openai_cls):
        from videsc.model.vllm_client import VLLMClient

        client = VLLMClient(host="myhost", port=9000, model="my-model", api_key="key123")
        mock_openai_cls.assert_called_once_with(
            base_url="http://myhost:9000/v1",
            api_key="key123",
        )
        assert client.model == "my-model"
        assert client.max_tokens == 8192

    @patch("openai.OpenAI")
    def test_encode_frame_returns_base64(self, mock_openai_cls):
        from videsc.model.vllm_client import VLLMClient

        client = VLLMClient()
        frame = _dummy_frame()
        b64 = client._encode_frame(frame)
        assert isinstance(b64, str)
        assert len(b64) > 0

    @patch("openai.OpenAI")
    def test_encode_frame_resizes_large_images(self, mock_openai_cls):
        from videsc.model.vllm_client import VLLMClient

        client = VLLMClient()
        large_frame = _dummy_frame(size=(3000, 2000))
        b64 = client._encode_frame(large_frame, max_size=640)
        # Just verify it doesn't crash and produces output
        assert isinstance(b64, str)
        assert len(b64) > 0

    @patch("openai.OpenAI")
    def test_build_content(self, mock_openai_cls):
        from videsc.model.vllm_client import VLLMClient

        client = VLLMClient()
        frames = [_dummy_frame(), _dummy_frame()]
        content = client.build_content(frames, "Describe this.")
        # 2 image entries + 1 text
        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"
        assert content[2]["text"] == "Describe this."

    @patch("openai.OpenAI")
    def test_generate_calls_api(self, mock_openai_cls):
        from videsc.model.vllm_client import VLLMClient

        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_choice = MagicMock()
        mock_choice.message.content = "A video showing a cat."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client_instance.chat.completions.create.return_value = mock_response

        client = VLLMClient(model="test-model")
        messages = [{"role": "user", "content": "Hello"}]
        result = client.generate(messages)

        assert result == "A video showing a cat."
        mock_client_instance.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_describe_frames(self, mock_openai_cls):
        from videsc.model.vllm_client import VLLMClient

        mock_client_instance = MagicMock()
        mock_openai_cls.return_value = mock_client_instance

        mock_choice = MagicMock()
        mock_choice.message.content = "Description of video frames."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client_instance.chat.completions.create.return_value = mock_response

        client = VLLMClient(model="test-model")
        frames = [_dummy_frame()]
        result = client.describe_frames(frames, prompt="Describe this.", system="Be helpful.")

        assert result == "Description of video frames."
        call_kwargs = mock_client_instance.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# vLLM runner tests
# ---------------------------------------------------------------------------


class TestVLLMRunner:
    """Tests for videsc.pipeline.vllm_runner."""

    @patch("videsc.pipeline.vllm_runner._create_vllm_client")
    @patch("videsc.pipeline.vllm_runner.extract_frames_as_pil")
    @patch("videsc.pipeline.vllm_runner.get_video_info")
    def test_run_single_video_vllm_basic(self, mock_vinfo, mock_extract, mock_client_factory, tmp_path):
        from videsc.pipeline.vllm_runner import run_single_video_vllm

        # Setup
        mock_vinfo.return_value = {"tot_time": 10.0, "FPS": 30, "width": 640, "height": 480, "num_frames": 300}
        mock_extract.return_value = [_dummy_frame(), _dummy_frame()]

        mock_client = MagicMock()
        mock_client.describe_frames.return_value = "A test description."
        mock_client_factory.return_value = mock_client

        video_file = tmp_path / "test.mp4"
        video_file.touch()

        args = _make_vllm_args(video=str(video_file), outdir=str(tmp_path / "out"))
        rc = run_single_video_vllm(args)

        assert rc == 0
        mock_client.describe_frames.assert_called_once()
        out_file = tmp_path / "out" / "test.txt"
        assert out_file.exists()
        assert out_file.read_text() == "A test description."

    @patch("videsc.pipeline.vllm_runner._create_vllm_client")
    @patch("videsc.pipeline.vllm_runner.extract_frames_as_pil")
    @patch("videsc.pipeline.vllm_runner.get_video_info")
    def test_run_single_video_vllm_chunked(self, mock_vinfo, mock_extract, mock_client_factory, tmp_path):
        from videsc.pipeline.vllm_runner import run_single_video_vllm

        mock_vinfo.return_value = {"tot_time": 60.0, "FPS": 30, "width": 640, "height": 480, "num_frames": 1800}
        mock_extract.return_value = [_dummy_frame()]

        mock_client = MagicMock()
        mock_client.describe_frames.side_effect = ["Chunk 1.", "Chunk 2."]
        mock_client_factory.return_value = mock_client

        video_file = tmp_path / "long.mp4"
        video_file.touch()

        args = _make_vllm_args(
            video=str(video_file),
            outdir=str(tmp_path / "out"),
            vllm_chunk_duration=30.0,
        )
        rc = run_single_video_vllm(args)

        assert rc == 0
        assert mock_client.describe_frames.call_count == 2
        out_file = tmp_path / "out" / "long.txt"
        assert out_file.exists()
        content = out_file.read_text()
        assert "Chunk 1." in content
        assert "Chunk 2." in content

    @patch("videsc.pipeline.vllm_runner._create_vllm_client")
    @patch("videsc.pipeline.vllm_runner.extract_frames_as_pil")
    @patch("videsc.pipeline.vllm_runner.get_video_info")
    def test_run_single_video_vllm_dry_run(self, mock_vinfo, mock_extract, mock_client_factory, tmp_path):
        from videsc.pipeline.vllm_runner import run_single_video_vllm

        mock_vinfo.return_value = {"tot_time": 10.0, "FPS": 30, "width": 640, "height": 480, "num_frames": 300}
        mock_extract.return_value = [_dummy_frame()]

        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client

        video_file = tmp_path / "test.mp4"
        video_file.touch()

        args = _make_vllm_args(video=str(video_file), outdir=str(tmp_path / "out"), dry=True)
        rc = run_single_video_vllm(args)

        assert rc == 0
        mock_client.describe_frames.assert_not_called()


# ---------------------------------------------------------------------------
# CLI args tests
# ---------------------------------------------------------------------------


class TestVLLMArgs:
    """Tests for vLLM-related CLI argument parsing."""

    def test_vllm_flag_parsed(self):
        from videsc.cli.args import parse_args

        args = parse_args(["--vllm", "--video", "/tmp/v.mp4"])
        assert args.vllm is True
        assert args.vllm_host == "localhost"
        assert args.vllm_port == 8000
        assert args.vllm_model == "default"

    def test_vllm_custom_options(self):
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vllm",
            "--video", "/tmp/v.mp4",
            "--vllm-host", "gpu-server",
            "--vllm-port", "9001",
            "--vllm-model", "Qwen/Qwen2.5-VL-72B",
            "--vllm-api-key", "my-secret",
            "--vllm-temperature", "0.3",
            "--vllm-fps", "2.0",
            "--vllm-chunk-duration", "30",
        ])
        assert args.vllm_host == "gpu-server"
        assert args.vllm_port == 9001
        assert args.vllm_model == "Qwen/Qwen2.5-VL-72B"
        assert args.vllm_api_key == "my-secret"
        assert args.vllm_temperature == 0.3
        assert args.vllm_fps == 2.0
        assert args.vllm_chunk_duration == 30.0

    def test_vllm_requires_video_input(self):
        from videsc.cli.args import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--vllm"])
