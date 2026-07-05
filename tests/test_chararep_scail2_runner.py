"""Tests for the SCAIL-2 prepared-assets runner."""

import subprocess
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from chararep.config import CharacterMapping, PipelineConfig
from chararep.face_detector import TrackedFace
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

        def _run_side_effect(cmd, cwd, capture_output, text, **kwargs):
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

    def test_run_includes_additional_references_when_provided(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        extra_ref_a = tmp_path / "extra_ref_a.png"
        extra_ref_b = tmp_path / "extra_ref_b.png"
        extra_mask_a = tmp_path / "extra_mask_a.png"
        extra_mask_b = tmp_path / "extra_mask_b.png"
        for path in [extra_ref_a, extra_ref_b, extra_mask_a, extra_mask_b]:
            _write_valid_png(path)

        cfg.scail2_additional_reference_images = [
            str(extra_ref_a),
            str(extra_ref_b),
        ]
        cfg.scail2_additional_reference_masks = [
            str(extra_mask_a),
            str(extra_mask_b),
        ]

        runner = Scail2PreparedAssetsRunner(cfg)

        def _run_side_effect(cmd, cwd, capture_output, text, **kwargs):
            output_path = Path(cmd[cmd.index("--save_file") + 1])
            output_path.write_bytes(b"generated")
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch.object(runner, "_probe_video", return_value=(12, 24.0)), \
             patch("chararep.scail2_runner.subprocess.run", side_effect=_run_side_effect) as mock_run, \
             patch("chararep.scail2_runner.finalize_video_output"):
            runner.run()

        cmd = mock_run.call_args[0][0]
        image_flag_index = cmd.index("--additional_ref_image")
        mask_flag_index = cmd.index("--additional_ref_mask_image")
        prompt_flag_index = cmd.index("--prompt")

        staged_image_names = [Path(path).name for path in cmd[image_flag_index + 1:mask_flag_index]]
        staged_mask_names = [Path(path).name for path in cmd[mask_flag_index + 1:prompt_flag_index]]
        assert staged_image_names == ["additional_ref_0.png", "additional_ref_1.png"]
        assert staged_mask_names == ["additional_ref_mask_0.png", "additional_ref_mask_1.png"]

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
        )
        runner = Scail2PreparedAssetsRunner(cfg)

        def _run_side_effect(cmd, cwd, capture_output, text, **kwargs):
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
             patch.object(
                 runner,
                 "_scan_driving_identity",
                 return_value={
                     "frames_sampled": 3,
                     "frames_with_target": 2,
                     "max_active_faces_in_frame": 2,
                     "max_target_faces_in_frame": 1,
                 },
             ), \
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
        assert "--matchnearest" in pose_cmd
        mock_finalize.assert_called_once()
        assert stats["frames_total"] == 6

    def test_run_passes_extra_generate_args_and_env(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        cfg.scail2_extra_args = ["--quantize", "fp8"]
        cfg.scail2_env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
        runner = Scail2PreparedAssetsRunner(cfg)

        def _run_side_effect(cmd, cwd, capture_output, text, **kwargs):
            output_path = Path(cmd[cmd.index("--save_file") + 1])
            output_path.write_bytes(b"generated")
            assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
            assert "PATH" in kwargs["env"]
            return MagicMock(returncode=0, stdout="ok", stderr="")

        with patch.object(runner, "_probe_video", return_value=(12, 24.0)), \
             patch("chararep.scail2_runner.subprocess.run", side_effect=_run_side_effect) as mock_run, \
             patch("chararep.scail2_runner.finalize_video_output"):
            runner.run()

        cmd = mock_run.call_args[0][0]
        assert cmd[-2:] == ["--quantize", "fp8"]
        assert cmd[cmd.index("--offload_model") + 1:] == ["--quantize", "fp8"]

    def test_run_fails_fast_on_vram_risk_when_requested(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        cfg.scail2_fail_on_vram_risk = True
        runner = Scail2PreparedAssetsRunner(cfg)

        with patch.object(runner, "_probe_video", return_value=(12, 24.0)), \
             patch.object(runner, "_describe_vram_risk", return_value="oom risk"):
            with pytest.raises(RuntimeError, match="oom risk"):
                runner.run()

    def test_read_image_bgr_falls_back_to_pil(self, tmp_path, monkeypatch):
        image_path = tmp_path / "portrait.png"
        _write_valid_png(image_path)

        monkeypatch.setattr("chararep.scail2_runner.cv2.imread", lambda _path: None)

        img = Scail2PreparedAssetsRunner._read_image_bgr(str(image_path))

        assert img.shape == (8, 8, 3)
        assert tuple(int(x) for x in img[0, 0]) == (32, 64, 128)

    def test_preflight_auto_enables_matchnearest(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        cfg.characters = [
            CharacterMapping(
                source_label="hero",
                reference_paths=[str(tmp_path / "ref-find.png")],
                portrait_paths=[str(tmp_path / "portrait.png")],
            )
        ]
        runner = Scail2PreparedAssetsRunner(cfg)

        with patch.object(
            runner,
            "_scan_driving_identity",
            return_value={
                "frames_sampled": 4,
                "frames_with_target": 2,
                "max_active_faces_in_frame": 2,
                "max_target_faces_in_frame": 1,
            },
        ):
            matchnearest, egocentric = runner._resolve_pose_preprocess_flags(120, 24.0)

        assert matchnearest is True
        assert egocentric is False

    def test_preflight_rejects_when_target_is_ambiguous(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        cfg.characters = [
            CharacterMapping(
                source_label="hero",
                reference_paths=[str(tmp_path / "ref-find.png")],
                portrait_paths=[str(tmp_path / "portrait.png")],
            )
        ]
        runner = Scail2PreparedAssetsRunner(cfg)

        with patch.object(
            runner,
            "_scan_driving_identity",
            return_value={
                "frames_sampled": 4,
                "frames_with_target": 2,
                "max_active_faces_in_frame": 2,
                "max_target_faces_in_frame": 2,
            },
        ):
            with pytest.raises(RuntimeError, match="multiple simultaneous faces matching target"):
                runner._resolve_pose_preprocess_flags(120, 24.0)

    def test_preflight_rejects_when_clip_is_too_crowded(self, tmp_path):
        cfg = _make_scail2_cfg(tmp_path)
        cfg.characters = [
            CharacterMapping(
                source_label="hero",
                reference_paths=[str(tmp_path / "ref-find.png")],
                portrait_paths=[str(tmp_path / "portrait.png")],
            )
        ]
        runner = Scail2PreparedAssetsRunner(cfg)

        with patch.object(
            runner,
            "_scan_driving_identity",
            return_value={
                "frames_sampled": 4,
                "frames_with_target": 2,
                "max_active_faces_in_frame": 3,
                "max_target_faces_in_frame": 1,
            },
        ):
            with pytest.raises(RuntimeError, match="more than two simultaneous faces"):
                runner._resolve_pose_preprocess_flags(120, 24.0)

    def test_scan_driving_identity_uses_detector_and_recognizer(self, tmp_path, monkeypatch):
        cfg = _make_scail2_cfg(tmp_path)
        cfg.characters = [
            CharacterMapping(
                source_label="hero",
                reference_paths=[str(tmp_path / "ref-find.png")],
                portrait_paths=[str(tmp_path / "portrait.png")],
            )
        ]
        runner = Scail2PreparedAssetsRunner(cfg)

        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

        class _FakeCapture:
            def __init__(self, _path):
                self._frames = list(frames)

            def isOpened(self):
                return True

            def read(self):
                if self._frames:
                    return True, self._frames.pop(0)
                return False, None

            def release(self):
                return None

        detections = [
            [
                TrackedFace(1, np.zeros(4, dtype=np.float32), np.zeros((5, 2), dtype=np.float32)),
                TrackedFace(2, np.zeros(4, dtype=np.float32), np.zeros((5, 2), dtype=np.float32)),
            ],
            [
                TrackedFace(1, np.zeros(4, dtype=np.float32), np.zeros((5, 2), dtype=np.float32)),
            ],
        ]

        class _FakeDetector:
            def __init__(self, _cfg):
                self.backend = object()
                self._calls = 0

            def detect(self, _frame):
                idx = min(self._calls, len(detections) - 1)
                self._calls += 1
                return detections[idx]

        class _FakeRecognizer:
            def __init__(self, _cfg, backend):
                self.targets = [types.SimpleNamespace(label="hero")]

            def identify_faces(self, faces):
                for face in faces:
                    if face.track_id == 1:
                        face.identity_label = "hero"
                return faces

        monkeypatch.setattr("chararep.scail2_runner.cv2.VideoCapture", _FakeCapture)
        monkeypatch.setattr("chararep.face_detector.FaceDetector", _FakeDetector)
        monkeypatch.setattr("chararep.face_recognizer.FaceRecognizer", _FakeRecognizer)

        summary = runner._scan_driving_identity(total_frames=3, input_fps=1.0)

        assert summary["frames_sampled"] >= 2
        assert summary["frames_with_target"] >= 2
        assert summary["max_active_faces_in_frame"] == 2
        assert summary["max_target_faces_in_frame"] == 1