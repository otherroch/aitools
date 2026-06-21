"""Tests for the SCAIL-2 prepared-assets runner."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from chararep.config import CharacterMapping, PipelineConfig
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


def _write_valid_png(path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :] = (32, 64, 128)
    assert cv2.imwrite(str(path), img)


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

    def test_run_auto_prep_from_character_mapping(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "generate.py").write_text("print('stub')\n", encoding="utf-8")
        pose_repo = repo / "SCAIL-Pose"
        (pose_repo / "NLFPoseExtract").mkdir(parents=True)
        (pose_repo / "NLFPoseExtract" / "process_replacement.py").write_text("print('pose')\n", encoding="utf-8")

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        model = tmp_path / "model.safetensors"
        model.write_bytes(b"x")

        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"video")

        source_ref = tmp_path / "source_ref.png"
        _write_valid_png(source_ref)
        portrait = tmp_path / "portrait.png"
        _write_valid_png(portrait)

        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(input_video),
            output_video=str(tmp_path / "output.mp4"),
            characters=[
                CharacterMapping(
                    source_label="hero",
                    reference_paths=[str(source_ref)],
                    portrait_paths=[str(portrait)],
                )
            ],
            scail2_repo_path=str(repo),
            scail2_ckpt_dir=str(ckpt),
            scail2_model_path=str(model),
            scail2_prompt="A detailed prompt",
            scail2_matchnearest=True,
        )
        runner = Scail2PreparedAssetsRunner(cfg)

        def _run_side_effect(cmd, cwd, capture_output, text):
            if "process_replacement.py" in cmd[1]:
                subdir = Path(cmd[cmd.index("--subdir") + 1])
                (subdir / "ref_mask.png").write_bytes(b"mask")
                (subdir / "rendered_v2.mp4").write_bytes(b"rendered")
                (subdir / "replace_mask.mp4").write_bytes(b"replace")
                return MagicMock(returncode=0, stdout="pose ok", stderr="")

            output_path = Path(cmd[cmd.index("--save_file") + 1])
            output_path.write_bytes(b"generated")
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch.object(runner, "_probe_video", return_value=(6, 24.0)), \
             patch("chararep.scail2_runner.subprocess.run", side_effect=_run_side_effect) as mock_run, \
             patch("chararep.scail2_runner.finalize_video_output") as mock_finalize:
            stats = runner.run()

        pose_cmd = mock_run.call_args_list[0][0][0]
        gen_cmd = mock_run.call_args_list[1][0][0]
        assert "process_replacement.py" in pose_cmd[1]
        assert "--matchnearest" in pose_cmd
        assert "--text" in pose_cmd
        assert Path(gen_cmd[gen_cmd.index("--image") + 1]).name == "ref_image.png"
        assert Path(gen_cmd[gen_cmd.index("--mask_image") + 1]).name == "ref_mask.png"
        assert Path(gen_cmd[gen_cmd.index("--mask_video") + 1]).name == "replace_mask.mp4"
        mock_finalize.assert_called_once()
        assert stats["frames_total"] == 6

    def test_read_image_bgr_falls_back_to_pil(self, tmp_path, monkeypatch):
        image_path = tmp_path / "portrait.png"
        _write_valid_png(image_path)

        monkeypatch.setattr("chararep.scail2_runner.cv2.imread", lambda _path: None)

        img = Scail2PreparedAssetsRunner._read_image_bgr(str(image_path))

        assert img.shape == (8, 8, 3)
        assert tuple(int(x) for x in img[0, 0]) == (32, 64, 128)