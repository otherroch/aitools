"""Tests for the SCAIL-2 prepared-assets runner."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chararep.config import PipelineConfig
from chararep.scail2_runner import Scail2PreparedAssetsRunner


def _make_scail2_cfg(tmp_path: Path) -> PipelineConfig:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "generate.py").write_text("print('stub')\n", encoding="utf-8")

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()

    model = tmp_path / "model.safetensors"
    model.write_bytes(b"x")

    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"video")

    ref = tmp_path / "ref.png"
    ref.write_bytes(b"ref")
    ref_mask = tmp_path / "ref_mask.png"
    ref_mask.write_bytes(b"mask")
    mask_video = tmp_path / "mask.mp4"
    mask_video.write_bytes(b"maskvideo")

    return PipelineConfig(
        backend="scail2",
        input_video=str(input_video),
        output_video=str(tmp_path / "output.mp4"),
        scail2_repo_path=str(repo),
        scail2_ckpt_dir=str(ckpt),
        scail2_model_path=str(model),
        scail2_reference_image=str(ref),
        scail2_reference_mask=str(ref_mask),
        scail2_mask_video=str(mask_video),
        scail2_prompt="A detailed prompt",
    )


class TestScail2PreparedAssetsRunner:
    def test_run_invokes_generate_and_finalizes_output(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        runner = Scail2PreparedAssetsRunner(cfg)

        def _run_side_effect(cmd, cwd, capture_output, text):
            output_path = Path(cmd[cmd.index("--save_file") + 1])
            output_path.write_bytes(b"generated")
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch.object(runner, "_probe_video", return_value=(12, 24.0)), \
             patch("chararep.scail2_runner.subprocess.run", side_effect=_run_side_effect) as mock_run, \
             patch("chararep.scail2_runner.finalize_video_output") as mock_finalize:
            stats = runner.run()

        cmd = mock_run.call_args[0][0]
        assert "--replace_flag" in cmd
        assert "--offload_model" in cmd
        mock_finalize.assert_called_once_with(
            mock_finalize.call_args.args[0],
            cfg.output_video,
            audio_source=cfg.input_video,
        )
        assert stats["backend"] == "scail2"
        assert stats["frames_total"] == 12
        assert stats["frames_swapped"] == 12

    def test_prompt_file_used_when_inline_prompt_missing(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Prompt from file", encoding="utf-8")
        cfg.scail2_prompt = None
        cfg.scail2_prompt_file = str(prompt_file)

        runner = Scail2PreparedAssetsRunner(cfg)
        assert runner._resolve_prompt() == "Prompt from file"

    def test_run_raises_on_subprocess_failure(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        runner = Scail2PreparedAssetsRunner(cfg)

        failed = subprocess.CompletedProcess(
            args=["python", "generate.py"],
            returncode=2,
            stdout="",
            stderr="bad things happened",
        )

        with patch.object(runner, "_probe_video", return_value=(0, 0.0)), \
             patch("chararep.scail2_runner.subprocess.run", return_value=failed), \
             patch("chararep.scail2_runner.finalize_video_output"):
            with pytest.raises(RuntimeError, match="generate.py failed"):
                runner.run()