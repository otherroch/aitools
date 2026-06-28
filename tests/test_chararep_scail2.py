"""Tests for SCAIL-2 runner."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chararep.scail2_runner import SCAIL2Config, SCAIL2Runner, _ensure_bgr_frame


class TestSCAIL2Config:
    def test_default_values(self):
        cfg = SCAIL2Config()
        assert cfg.enabled is False
        assert cfg.model_path == ""
        assert cfg.mode == "replacement"
        assert cfg.resolution == "704p"
        assert cfg.width == 704
        assert cfg.height == 1280
        assert cfg.steps == 30
        assert cfg.cfg_scale == 3.5
        assert cfg.fps == 24.0
        assert cfg.seed == 42
        assert cfg.device == "cuda"
        assert cfg.use_fp16 is True

    def test_custom_values(self):
        cfg = SCAIL2Config(
            enabled=True,
            model_path="/path/to/model",
            mode="animation",
            resolution="512p",
            width=512,
            height=512,
            steps=50,
            cfg_scale=4.0,
            fps=30.0,
            seed=12345,
        )
        assert cfg.enabled is True
        assert cfg.model_path == "/path/to/model"
        assert cfg.mode == "animation"
        assert cfg.resolution == "512p"
        assert cfg.width == 512
        assert cfg.height == 512
        assert cfg.steps == 50
        assert cfg.cfg_scale == 4.0


class TestSCAIL2Runner:
    def test_default_config(self):
        runner = SCAIL2Runner()
        assert runner._loaded is False
        assert runner._model is None
        assert runner._vae is None
        assert runner._t5_encoder is None

    def test_config_passed_through(self):
        cfg = SCAIL2Config(enabled=True, mode="animation")
        runner = SCAIL2Runner(cfg)
        assert runner.cfg.enabled is True
        assert runner.cfg.mode == "animation"

    def test_is_available_before_load(self):
        runner = SCAIL2Runner()
        assert runner.is_available() is False

    def test_load_missing_directory(self):
        cfg = SCAIL2Config(model_path="/nonexistent/path")
        runner = SCAIL2Runner(cfg)
        result = runner.load()
        assert result is False

    def test_load_nonexistent_path(self):
        cfg = SCAIL2Config(model_path="nonexistent_dir")
        runner = SCAIL2Runner(cfg)
        result = runner.load()
        assert result is False

    def test_load_with_valid_directory(self, tmp_path):
        model_dir = tmp_path / "scail2_model"
        model_dir.mkdir()
        (model_dir / "model").mkdir()
        (model_dir / "Wan21_VAE.pth").touch()

        cfg = SCAIL2Config(model_path=str(model_dir))
        runner = SCAIL2Runner(cfg)
        result = runner.load()
        assert result is True
        assert runner.is_available() is True

    def test_generate_without_load(self):
        runner = SCAIL2Runner()
        with pytest.raises(RuntimeError, match="SCAIL-2 not loaded"):
            runner.generate(
                reference_image=None,
                driving_frames=[],
            )

    def test_generate_no_frames(self):
        model_dir = Path("/tmp/scail2_model")
        cfg = SCAIL2Config(model_path=str(model_dir))
        runner = SCAIL2Runner(cfg)
        # Without loading, generate should raise an error
        with pytest.raises(RuntimeError, match="SCAIL-2 not loaded"):
            runner.generate(reference_image=None, driving_frames=[])

    def test_generate_with_frames(self):
        model_dir = Path("/tmp/scail2_model")
        cfg = SCAIL2Config(model_path=str(model_dir))
        runner = SCAIL2Runner(cfg)
        # Without loading, generate should raise an error
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        with pytest.raises(RuntimeError, match="SCAIL-2 not loaded"):
            runner.generate(reference_image=None, driving_frames=frames)


class TestEnsureBgrFrame:
    def test_none_returns_none(self):
        assert _ensure_bgr_frame(None) is None

    def test_uint8_passthrough(self):
        f = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _ensure_bgr_frame(f)
        assert result.dtype == np.uint8

    def test_rgba_cropped_to_rgb(self):
        f = np.zeros((100, 100, 4), dtype=np.uint8)
        result = _ensure_bgr_frame(f)
        assert result.shape == (100, 100, 3)


class TestSCAIL2ConfigWithResolution:
    def test_invalid_resolution_fallback(self):
        cfg = SCAIL2Config(resolution="invalid")
        assert cfg.resolution == "invalid"

    def test_all_fields(self):
        cfg = SCAIL2Config(
            enabled=True,
            model_path="/path/to/model",
            mode="replacement",
            resolution="704p",
            width=704,
            height=1280,
            steps=30,
            cfg_scale=3.5,
            fps=24.0,
            seed=42,
            device="cuda",
            device_id=0,
            use_fp16=True,
            vae_path="/path/to/vae",
            t5_encoder_path="/path/to/t5",
            environment_mask_path="/path/to/mask",
            num_extra_refs=2,
            mask_video_path="/path/to/mask_video",
        )
        assert cfg.enabled is True
        assert cfg.environment_mask_path == "/path/to/mask"
        assert cfg.num_extra_refs == 2
        assert cfg.mask_video_path == "/path/to/mask_video"
