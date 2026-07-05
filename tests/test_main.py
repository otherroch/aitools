"""Tests for chararep/main.py — CLI argument parsing and config building."""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chararep.config import CharacterMapping, PipelineConfig


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def _parse(self, args):
        from chararep.main import _parse_args
        with patch("sys.argv", ["chararep"] + args):
            return _parse_args()

    def test_defaults(self):
        """All defaults are set when no arguments are passed."""
        args = self._parse([])
        assert args.backend == "classic"
        assert args.input_video is None
        assert args.output_video is None
        assert args.config_file is None
        assert args.characters == []
        assert args.similarity_threshold == pytest.approx(0.5)
        assert args.detection_model == "buffalo_l"
        assert args.detect_size == 640
        assert args.enhance is False
        assert args.enhance_model == "gfpgan"
        assert args.enhance_weight == pytest.approx(0.7)
        assert args.device == 0
        assert args.no_fp16 is False
        assert args.codec == "libx264"
        assert args.crf == 18
        assert args.no_audio is False
        assert args.blend_mode == "alpha"
        assert args.mask_blur_kernel == 15
        assert args.mask_erode_pixels == 2
        assert args.verbose is False
        assert args.log_file is None
        assert args.timers is False
        assert args.dump_config is False
        assert args.scail2_offload_model is True
        assert args.scail2_memory_preset == "default"
        assert args.scail2_extra_arg == []
        assert args.scail2_env == []
        assert args.scail2_fail_on_vram_risk is False
        assert args.scail2_additional_reference_images == []
        assert args.scail2_additional_reference_masks == []
        assert args.scail2_pose_repo_path is None

    def test_scail2_backend_flag(self):
        args = self._parse(["--backend", "scail2"])
        assert args.backend == "scail2"

    def test_scail2_prompt_file_flag(self):
        args = self._parse(["--scail2-prompt-file", "prompt.txt"])
        assert args.scail2_prompt_file == "prompt.txt"

    def test_scail2_matchnearest_flag(self):
        args = self._parse(["--scail2-matchnearest"])
        assert args.scail2_matchnearest is True

    def test_scail2_memory_preset_flag(self):
        args = self._parse(["--scail2-memory-preset", "low-vram"])
        assert args.scail2_memory_preset == "low-vram"

    def test_scail2_sam_text_flag(self):
        args = self._parse(["--scail2-sam-text", "human", "bear"])
        assert args.scail2_sam_text == ["human", "bear"]

    def test_scail2_additional_reference_flags(self):
        args = self._parse(
            [
                "--scail2-additional-reference-image",
                "ref_a.png",
                "ref_b.png",
                "--scail2-additional-reference-mask",
                "mask_a.png",
                "mask_b.png",
            ]
        )
        assert args.scail2_additional_reference_images == ["ref_a.png", "ref_b.png"]
        assert args.scail2_additional_reference_masks == ["mask_a.png", "mask_b.png"]

    def test_scail2_extra_arg_and_env_flags(self):
        args = self._parse(
            [
                "--scail2-extra-arg=--quantize",
                "--scail2-extra-arg",
                "fp8",
                "--scail2-env",
                "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
                "--scail2-fail-on-vram-risk",
            ]
        )
        assert args.scail2_extra_arg == ["--quantize", "fp8"]
        assert args.scail2_env == [
            "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
        ]
        assert args.scail2_fail_on_vram_risk is True

    def test_scail2_env_requires_key_equals_value_format(self, capsys):
        with pytest.raises(SystemExit):
            self._parse(["--scail2-env", "BROKEN"])
        captured = capsys.readouterr()
        assert "KEY=VALUE" in captured.err

    def test_scail2_target_width_uses_generic_positive_int_error(self, capsys):
        with pytest.raises(SystemExit):
            self._parse(["--scail2-target-width", "0"])

        captured = capsys.readouterr()
        error_line = captured.err.strip().splitlines()[-1]
        assert "expected a positive integer" in error_line
        assert "--batch" not in error_line

    def test_temporal_smooth_alpha_nan_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--temporal-smooth-alpha", "nan"])

    def test_input_output(self):
        args = self._parse(["-i", "in.mp4", "-o", "out.mp4"])
        assert args.input_video == "in.mp4"
        assert args.output_video == "out.mp4"

    def test_config_flag(self):
        args = self._parse(["--config", "config.json"])
        assert args.config_file == "config.json"

    def test_char_flag(self):
        args = self._parse(["--char", "find_dir", "replace_dir"])
        assert len(args.characters) == 1
        assert args.characters[0] == ["find_dir", "replace_dir"]

    def test_multiple_char_flags(self):
        args = self._parse([
            "--char", "find1", "replace1",
            "--char", "find2", "replace2",
        ])
        assert len(args.characters) == 2

    def test_enhance_flag(self):
        args = self._parse(["--enhance"])
        assert args.enhance is True

    def test_no_fp16_flag(self):
        args = self._parse(["--no-fp16"])
        assert args.no_fp16 is True

    def test_no_audio_flag(self):
        args = self._parse(["--no-audio"])
        assert args.no_audio is True

    def test_verbose_flag(self):
        args = self._parse(["-v"])
        assert args.verbose is True

    def test_timers_flag(self):
        args = self._parse(["--timers"])
        assert args.timers is True

    def test_dump_config_flag(self):
        args = self._parse(["--dump-config"])
        assert args.dump_config is True

    def test_device_flag(self):
        args = self._parse(["--device", "1"])
        assert args.device == 1

    def test_crf_flag(self):
        args = self._parse(["--crf", "23"])
        assert args.crf == 23

    def test_blend_mode_seamless(self):
        args = self._parse(["--blend-mode", "seamless"])
        assert args.blend_mode == "seamless"

    def test_similarity_threshold(self):
        args = self._parse(["--similarity-threshold", "0.7"])
        assert args.similarity_threshold == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _scan_image_dir
# ---------------------------------------------------------------------------

class TestScanImageDir:
    def test_returns_sorted_image_paths(self, tmp_path):
        from chararep.main import _scan_image_dir
        (tmp_path / "c.jpg").write_bytes(b"")
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "b.jpeg").write_bytes(b"")
        (tmp_path / "readme.txt").write_bytes(b"")

        paths = _scan_image_dir(str(tmp_path), "find")
        names = [Path(p).name for p in paths]
        assert names == sorted(names)
        assert "readme.txt" not in names
        assert len(names) == 3

    def test_all_image_extensions_accepted(self, tmp_path):
        from chararep.main import _scan_image_dir
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"]:
            (tmp_path / f"face{ext}").write_bytes(b"")
        paths = _scan_image_dir(str(tmp_path), "find")
        assert len(paths) == 7

    def test_nonexistent_dir_exits(self):
        from chararep.main import _scan_image_dir
        with pytest.raises(SystemExit):
            _scan_image_dir("/nonexistent/path", "find")

    def test_empty_dir_exits(self, tmp_path):
        from chararep.main import _scan_image_dir
        with pytest.raises(SystemExit):
            _scan_image_dir(str(tmp_path), "find")


# ---------------------------------------------------------------------------
# _build_config_from_args
# ---------------------------------------------------------------------------

class TestBuildConfigFromArgs:
    def _make_args(self, **kw):
        """Return a minimal argparse.Namespace for _build_config_from_args."""
        import argparse
        defaults = dict(
            backend="classic",
            input_video="input.mp4",
            output_video="output.mp4",
            characters=[],
            similarity_threshold=0.5,
            swap_model_path=None,
            embedding_converter_path=None,
            detection_model="buffalo_l",
            detect_size=640,
            enhance=False,
            enhance_model="gfpgan",
            enhance_model_path=None,
            enhance_weight=0.7,
            batch_size=4,
            device=0,
            no_fp16=False,
            codec="libx264",
            crf=18,
            no_audio=False,
            blend_mode="alpha",
            mask_blur_kernel=15,
            mask_erode_pixels=2,
            verbose=False,
            log_file=None,
            timers=False,
            scene_cut_threshold=PipelineConfig.scene_cut_threshold,
            scail2_repo_path=None,
            scail2_ckpt_dir=None,
            scail2_pose_repo_path=None,
            scail2_model_path=None,
            scail2_model_name=PipelineConfig.scail2_model_name,
            scail2_memory_preset=PipelineConfig.scail2_memory_preset,
            scail2_reference_image=None,
            scail2_reference_mask=None,
            scail2_mask_video=None,
            scail2_additional_reference_images=[],
            scail2_additional_reference_masks=[],
            scail2_prompt=None,
            scail2_prompt_file=None,
            scail2_target_width=PipelineConfig.scail2_target_width,
            scail2_target_height=PipelineConfig.scail2_target_height,
            scail2_sample_steps=PipelineConfig.scail2_sample_steps,
            scail2_sample_shift=PipelineConfig.scail2_sample_shift,
            scail2_sample_guide_scale=PipelineConfig.scail2_sample_guide_scale,
            scail2_sample_solver=PipelineConfig.scail2_sample_solver,
            scail2_matchnearest=False,
            scail2_egocentric=False,
            scail2_sam_text=list(PipelineConfig().scail2_sam_text),
            scail2_sam3_model=None,
            scail2_offload_model=PipelineConfig.scail2_offload_model,
            scail2_extra_arg=[],
            scail2_env=[],
            scail2_fail_on_vram_risk=False,
            scail2_work_dir=None,
            scail2_keep_intermediates=False,
        )
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_basic_config(self):
        from chararep.main import _build_config_from_args
        args = self._make_args()
        cfg = _build_config_from_args(args)
        assert cfg.input_video == "input.mp4"
        assert cfg.output_video == "output.mp4"
        assert cfg.characters == []

    def test_no_fp16_inverted(self):
        from chararep.main import _build_config_from_args
        args = self._make_args(no_fp16=True)
        cfg = _build_config_from_args(args)
        assert cfg.use_fp16 is False

    def test_no_audio_inverted(self):
        from chararep.main import _build_config_from_args
        args = self._make_args(no_audio=True)
        cfg = _build_config_from_args(args)
        assert cfg.copy_audio is False

    def test_verbose_sets_debug_level(self):
        from chararep.main import _build_config_from_args
        args = self._make_args(verbose=True)
        cfg = _build_config_from_args(args)
        assert cfg.log_level == "DEBUG"

    def test_timers_flag(self):
        from chararep.main import _build_config_from_args
        args = self._make_args(timers=True)
        cfg = _build_config_from_args(args)
        assert cfg.enable_timers is True

    def test_detect_size_is_tuple(self):
        from chararep.main import _build_config_from_args
        args = self._make_args(detect_size=1024)
        cfg = _build_config_from_args(args)
        assert cfg.detection_size == (1024, 1024)

    def test_char_args_build_character_mappings(self, tmp_path):
        """--char pairs are converted to CharacterMapping entries."""
        from chararep.main import _build_config_from_args

        find_dir = tmp_path / "find_hero"
        find_dir.mkdir()
        (find_dir / "face.jpg").write_bytes(b"")

        replace_dir = tmp_path / "replace_hero"
        replace_dir.mkdir()
        (replace_dir / "newface.jpg").write_bytes(b"")

        args = self._make_args(
            characters=[[str(find_dir), str(replace_dir)]],
        )
        cfg = _build_config_from_args(args)
        assert len(cfg.characters) == 1
        assert cfg.characters[0].source_label == "find_hero"

    def test_enhance_flag_sets_enhancement(self):
        from chararep.main import _build_config_from_args
        args = self._make_args(enhance=True)
        cfg = _build_config_from_args(args)
        assert cfg.enable_face_enhancement is True

    def test_scail2_fields_passed_through(self):
        from chararep.main import _build_config_from_args
        args = self._make_args(
            backend="scail2",
            scail2_repo_path="repo",
            scail2_pose_repo_path="pose-repo",
            scail2_ckpt_dir="ckpt",
            scail2_model_path="model.safetensors",
            scail2_memory_preset="default",
            scail2_reference_image="ref.png",
            scail2_reference_mask="ref_mask.png",
            scail2_mask_video="mask.mp4",
            scail2_additional_reference_images=["ref_a.png", "ref_b.png"],
            scail2_additional_reference_masks=["mask_a.png", "mask_b.png"],
            scail2_prompt="prompt",
            scail2_target_width=704,
            scail2_target_height=512,
            scail2_sample_steps=28,
            scail2_sample_solver="dpm++",
            scail2_matchnearest=True,
            scail2_sam_text=["human", "bear"],
            scail2_offload_model=False,
            scail2_extra_arg=["--quantize", "fp8"],
            scail2_env=["PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"],
            scail2_fail_on_vram_risk=True,
        )
        cfg = _build_config_from_args(args)
        assert cfg.backend == "scail2"
        assert cfg.scail2_repo_path == "repo"
        assert cfg.scail2_pose_repo_path == "pose-repo"
        assert cfg.scail2_ckpt_dir == "ckpt"
        assert cfg.scail2_model_path == "model.safetensors"
        assert cfg.scail2_memory_preset == "default"
        assert cfg.scail2_reference_image == "ref.png"
        assert cfg.scail2_reference_mask == "ref_mask.png"
        assert cfg.scail2_mask_video == "mask.mp4"
        assert cfg.scail2_additional_reference_images == ["ref_a.png", "ref_b.png"]
        assert cfg.scail2_additional_reference_masks == ["mask_a.png", "mask_b.png"]
        assert cfg.scail2_prompt == "prompt"
        assert cfg.scail2_target_width == 704
        assert cfg.scail2_target_height == 512
        assert cfg.scail2_sample_steps == 28
        assert cfg.scail2_sample_solver == "dpm++"
        assert cfg.scail2_matchnearest is True
        assert cfg.scail2_sam_text == ["human", "bear"]
        assert cfg.scail2_offload_model is False
        assert cfg.scail2_extra_args == ["--quantize", "fp8"]
        assert cfg.scail2_env == {
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
        }
        assert cfg.scail2_fail_on_vram_risk is True

    def test_scail2_low_vram_preset_applies_defaults(self):
        from chararep.main import _build_config_from_args

        args = self._make_args(
            backend="scail2",
            scail2_memory_preset="low-vram",
        )
        cfg = _build_config_from_args(args)
        cfg.apply_runtime_overrides()

        assert cfg.scail2_target_width == 672
        assert cfg.scail2_target_height == 384
        assert cfg.scail2_sample_steps == 28

    def test_scail2_low_vram_preset_preserves_explicit_target_overrides(self):
        from chararep.main import _build_config_from_args

        args = self._make_args(
            backend="scail2",
            scail2_memory_preset="low-vram",
            scail2_target_width=640,
            scail2_target_height=352,
            scail2_sample_steps=20,
        )
        cfg = _build_config_from_args(args)
        cfg.apply_runtime_overrides()

        assert cfg.scail2_target_width == 640
        assert cfg.scail2_target_height == 352
        assert cfg.scail2_sample_steps == 20


# ---------------------------------------------------------------------------
# _build_config_from_json
# ---------------------------------------------------------------------------

class TestBuildConfigFromJson:
    def test_explicit_paths_mode(self, tmp_path):
        """Characters with explicit reference_paths / portrait_paths lists."""
        from chararep.main import _build_config_from_json

        ref = tmp_path / "ref.jpg"
        ref.write_bytes(b"")
        portrait = tmp_path / "portrait.jpg"
        portrait.write_bytes(b"")

        data = {
            "input_video": "in.mp4",
            "output_video": "out.mp4",
            "characters": [
                {
                    "source_label": "hero",
                    "reference_paths": [str(ref)],
                    "portrait_paths": [str(portrait)],
                }
            ],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        cfg = _build_config_from_json(str(config_file))
        assert cfg.input_video == "in.mp4"
        assert len(cfg.characters) == 1
        assert cfg.characters[0].source_label == "hero"

    def test_find_replace_folder_mode(self, tmp_path):
        """Characters with find/replace folder shortcuts."""
        from chararep.main import _build_config_from_json

        find_dir = tmp_path / "originals" / "villain"
        find_dir.mkdir(parents=True)
        (find_dir / "screenshot.jpg").write_bytes(b"")

        replace_dir = tmp_path / "replacements" / "villain"
        replace_dir.mkdir(parents=True)
        (replace_dir / "new_face.jpg").write_bytes(b"")

        data = {
            "input_video": "in.mp4",
            "output_video": "out.mp4",
            "characters": [
                {
                    "find": str(find_dir),
                    "replace": str(replace_dir),
                    "similarity_threshold": 0.6,
                }
            ],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        cfg = _build_config_from_json(str(config_file))
        assert len(cfg.characters) == 1
        assert cfg.characters[0].source_label == "villain"
        assert cfg.characters[0].similarity_threshold == pytest.approx(0.6)

    def test_find_replace_with_explicit_label(self, tmp_path):
        """'label' key in JSON overrides folder name as source_label."""
        from chararep.main import _build_config_from_json

        find_dir = tmp_path / "find"
        find_dir.mkdir()
        (find_dir / "face.jpg").write_bytes(b"")

        replace_dir = tmp_path / "replace"
        replace_dir.mkdir()
        (replace_dir / "new.jpg").write_bytes(b"")

        data = {
            "input_video": "in.mp4",
            "output_video": "out.mp4",
            "characters": [
                {
                    "find": str(find_dir),
                    "replace": str(replace_dir),
                    "label": "custom_label",
                }
            ],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        cfg = _build_config_from_json(str(config_file))
        assert cfg.characters[0].source_label == "custom_label"

    def test_scail2_low_vram_preset_from_json(self, tmp_path):
        from chararep.main import _build_config_from_json

        data = {
            "backend": "scail2",
            "input_video": "in.mp4",
            "output_video": "out.mp4",
            "scail2_memory_preset": "low-vram",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        cfg = _build_config_from_json(str(config_file))
        cfg.apply_runtime_overrides()
        assert cfg.scail2_memory_preset == "low-vram"
        assert cfg.scail2_target_width == 672
        assert cfg.scail2_target_height == 384
        assert cfg.scail2_sample_steps == 28

    def test_label_key_remapped_to_source_label(self, tmp_path):
        """'label' key in explicit-paths mode is mapped to source_label."""
        from chararep.main import _build_config_from_json

        ref = tmp_path / "ref.jpg"
        ref.write_bytes(b"")
        portrait = tmp_path / "portrait.jpg"
        portrait.write_bytes(b"")

        data = {
            "input_video": "in.mp4",
            "output_video": "out.mp4",
            "characters": [
                {
                    "label": "myhero",
                    "reference_paths": [str(ref)],
                    "portrait_paths": [str(portrait)],
                }
            ],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        cfg = _build_config_from_json(str(config_file))
        assert cfg.characters[0].source_label == "myhero"

    def test_empty_characters_list(self, tmp_path):
        from chararep.main import _build_config_from_json

        data = {"input_video": "in.mp4", "output_video": "out.mp4", "characters": []}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        cfg = _build_config_from_json(str(config_file))
        assert cfg.characters == []

    def test_top_level_fields_passed_through(self, tmp_path):
        from chararep.main import _build_config_from_json

        data = {
            "input_video": "in.mp4",
            "output_video": "out.mp4",
            "characters": [],
            "enable_face_enhancement": False,
            "device_id": 1,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        cfg = _build_config_from_json(str(config_file))
        assert cfg.enable_face_enhancement is False
        assert cfg.device_id == 1


# ---------------------------------------------------------------------------
# _setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_no_raise_info_level(self, tmp_path):
        from chararep.main import _setup_logging
        cfg = PipelineConfig(log_level="INFO")
        _setup_logging(cfg)  # should not raise

    def test_no_raise_debug_level(self):
        from chararep.main import _setup_logging
        cfg = PipelineConfig(log_level="DEBUG")
        _setup_logging(cfg)

    def test_log_file_handler_added(self, tmp_path):
        from chararep.main import _setup_logging
        log_file = tmp_path / "test.log"
        cfg = PipelineConfig(log_file=str(log_file))
        with patch("chararep.main.logging.basicConfig") as mock_basic:
            _setup_logging(cfg)
        call_kwargs = mock_basic.call_args[1]
        handlers = call_kwargs.get("handlers", [])
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert str(log_file) in file_handlers[0].baseFilename


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

class TestMain:
    def _make_valid_cfg(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")
        ref = tmp_path / "ref.jpg"
        ref.write_bytes(b"")
        portrait = tmp_path / "portrait.jpg"
        portrait.write_bytes(b"")
        cm = CharacterMapping(
            source_label="hero",
            reference_paths=[str(ref)],
            portrait_paths=[str(portrait)],
        )
        return PipelineConfig(
            input_video=str(video),
            output_video=str(tmp_path / "out.mp4"),
            characters=[cm],
        )

    def test_exits_on_validation_errors(self, tmp_path, monkeypatch):
        """main() calls sys.exit(1) when the config has validation errors."""
        from chararep import main as main_module

        bad_cfg = PipelineConfig()  # empty → validation errors

        monkeypatch.setattr("sys.argv", ["chararep"])
        monkeypatch.setattr(main_module, "_parse_args", lambda: MagicMock(
            config_file=None,
            input_video="",
            output_video="",
            characters=[],
            similarity_threshold=0.5,
            swap_model_path=None,
            embedding_converter_path=None,
            detection_model="buffalo_l",
            detect_size=640,
            enhance=False,
            enhance_model="gfpgan",
            enhance_model_path=None,
            enhance_weight=0.7,
            batch_size=4,
            device=0,
            no_fp16=False,
            codec="libx264",
            crf=18,
            no_audio=False,
            blend_mode="alpha",
            mask_blur_kernel=15,
            mask_erode_pixels=2,
            verbose=False,
            log_file=None,
            timers=False,
            dump_config=False,
        ))

        with pytest.raises(SystemExit) as exc_info:
            main_module.main()
        assert exc_info.value.code == 1

    def test_dump_config_prints_json(self, tmp_path, monkeypatch, capsys):
        """When --dump-config is set, main() prints JSON config before running."""
        from chararep import main as main_module

        cfg = self._make_valid_cfg(tmp_path)

        mock_args = MagicMock()
        mock_args.config_file = None
        mock_args.dump_config = True
        mock_args.input_video = cfg.input_video
        mock_args.output_video = cfg.output_video
        mock_args.characters = []
        mock_args.similarity_threshold = 0.5
        mock_args.swap_model_path = None
        mock_args.embedding_converter_path = None
        mock_args.detection_model = "buffalo_l"
        mock_args.detect_size = 640
        mock_args.enhance = False
        mock_args.enhance_model = "gfpgan"
        mock_args.enhance_model_path = None
        mock_args.enhance_weight = 0.7
        mock_args.batch_size = 4
        mock_args.device = 0
        mock_args.no_fp16 = False
        mock_args.codec = "libx264"
        mock_args.crf = 18
        mock_args.no_audio = False
        mock_args.blend_mode = "alpha"
        mock_args.mask_blur_kernel = 15
        mock_args.mask_erode_pixels = 2
        mock_args.verbose = False
        mock_args.log_file = None
        mock_args.timers = False

        monkeypatch.setattr(main_module, "_parse_args", lambda: mock_args)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "frames_total": 0,
            "frames_swapped": 0,
            "faces_swapped": 0,
            "elapsed_s": 0.1,
            "fps": 0.0,
            "frames_detected": 0,
            "faces_identified": 0,
        }

        with patch("chararep.main.CharacterReplacementPipeline", return_value=mock_pipeline):
            # Config has no characters so will fail validation; patch validate to pass
            with patch.object(PipelineConfig, "validate", return_value=[]):
                main_module.main()

        captured = capsys.readouterr()
        assert captured.out.strip()  # something was printed
        # Check it's valid JSON
        parsed = json.loads(captured.out)
        assert "input_video" in parsed

    def test_runs_pipeline_on_valid_config(self, tmp_path, monkeypatch):
        """main() constructs a pipeline and calls run() once."""
        from chararep import main as main_module

        cfg = self._make_valid_cfg(tmp_path)

        mock_args = MagicMock()
        mock_args.config_file = None
        mock_args.dump_config = False
        mock_args.input_video = cfg.input_video
        mock_args.output_video = cfg.output_video
        mock_args.characters = []
        mock_args.similarity_threshold = 0.5
        mock_args.swap_model_path = None
        mock_args.embedding_converter_path = None
        mock_args.detection_model = "buffalo_l"
        mock_args.detect_size = 640
        mock_args.enhance = False
        mock_args.enhance_model = "gfpgan"
        mock_args.enhance_model_path = None
        mock_args.enhance_weight = 0.7
        mock_args.batch_size = 4
        mock_args.device = 0
        mock_args.no_fp16 = False
        mock_args.codec = "libx264"
        mock_args.crf = 18
        mock_args.no_audio = False
        mock_args.blend_mode = "alpha"
        mock_args.mask_blur_kernel = 15
        mock_args.mask_erode_pixels = 2
        mock_args.verbose = False
        mock_args.log_file = None
        mock_args.timers = False

        monkeypatch.setattr(main_module, "_parse_args", lambda: mock_args)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "frames_total": 10,
            "frames_swapped": 2,
            "faces_swapped": 4,
            "elapsed_s": 1.0,
            "fps": 10.0,
            "frames_detected": 3,
            "faces_identified": 4,
        }

        with patch("chararep.main.CharacterReplacementPipeline", return_value=mock_pipeline):
            with patch.object(PipelineConfig, "validate", return_value=[]):
                main_module.main()

        mock_pipeline.run.assert_called_once()

    def test_build_runner_selects_scail2_backend(self):
        from chararep import main as main_module

        cfg = PipelineConfig(backend="scail2")
        mock_runner = MagicMock()

        with patch("chararep.main.Scail2PreparedAssetsRunner", return_value=mock_runner):
            runner = main_module._build_runner(cfg)

        assert runner is mock_runner

    def test_main_uses_scail2_runner(self, tmp_path, monkeypatch):
        """main() dispatches to the SCAIL-2 runner when backend=scail2."""
        from chararep import main as main_module

        mock_args = MagicMock()
        mock_args.config_file = None
        mock_args.dump_config = False
        monkeypatch.setattr(main_module, "_parse_args", lambda: mock_args)

        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(tmp_path / "in.mp4"),
            output_video=str(tmp_path / "out.mp4"),
            scail2_repo_path=str(tmp_path / "repo"),
            scail2_ckpt_dir=str(tmp_path / "ckpt"),
            scail2_model_path=str(tmp_path / "model.safetensors"),
            scail2_reference_image=str(tmp_path / "ref.png"),
            scail2_reference_mask=str(tmp_path / "ref_mask.png"),
            scail2_mask_video=str(tmp_path / "mask.mp4"),
            scail2_prompt="prompt",
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = {
            "backend": "scail2",
            "frames_total": 8,
            "frames_swapped": 8,
            "faces_swapped": 0,
            "elapsed_s": 1.0,
            "fps": 8.0,
            "frames_detected": 0,
            "faces_identified": 0,
        }

        with patch("chararep.main._build_config_from_args", return_value=cfg), \
             patch("chararep.main.Scail2PreparedAssetsRunner", return_value=mock_runner):
            with patch.object(PipelineConfig, "validate", return_value=[]):
                main_module.main()

        mock_runner.run.assert_called_once()

    def test_main_uses_json_config_when_provided(self, tmp_path, monkeypatch):
        """When --config is set, _build_config_from_json is called."""
        from chararep import main as main_module

        ref = tmp_path / "ref.jpg"
        ref.write_bytes(b"")
        portrait = tmp_path / "portrait.jpg"
        portrait.write_bytes(b"")
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")

        data = {
            "input_video": str(video),
            "output_video": str(tmp_path / "out.mp4"),
            "characters": [
                {
                    "source_label": "hero",
                    "reference_paths": [str(ref)],
                    "portrait_paths": [str(portrait)],
                }
            ],
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(data))

        mock_args = MagicMock()
        mock_args.config_file = str(config_file)
        mock_args.dump_config = False
        mock_args.verbose = False

        monkeypatch.setattr(main_module, "_parse_args", lambda: mock_args)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "frames_total": 0,
            "frames_swapped": 0,
            "faces_swapped": 0,
            "elapsed_s": 0.0,
            "fps": 0.0,
            "frames_detected": 0,
            "faces_identified": 0,
        }

        with patch("chararep.main.CharacterReplacementPipeline", return_value=mock_pipeline):
            main_module.main()

        mock_pipeline.run.assert_called_once()
