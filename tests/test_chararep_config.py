"""Tests for config.py."""

import os
import tempfile

import pytest

from chararep.config import CharacterMapping, PipelineConfig


# ---------------------------------------------------------------------------
# CharacterMapping
# ---------------------------------------------------------------------------

class TestCharacterMapping:
    def test_default_values(self):
        cm = CharacterMapping(source_label="hero")
        assert cm.source_label == "hero"
        assert cm.reference_paths == []
        assert cm.portrait_paths == []
        assert cm.similarity_threshold == 0.5

    def test_custom_values(self):
        cm = CharacterMapping(
            source_label="villain",
            reference_paths=["a.jpg"],
            portrait_paths=["b.jpg"],
            similarity_threshold=0.7,
        )
        assert cm.source_label == "villain"
        assert cm.reference_paths == ["a.jpg"]
        assert cm.portrait_paths == ["b.jpg"]
        assert cm.similarity_threshold == 0.7


# ---------------------------------------------------------------------------
# PipelineConfig defaults
# ---------------------------------------------------------------------------

class TestPipelineConfigDefaults:
    def test_default_values(self):
        cfg = PipelineConfig()
        assert cfg.backend == "classic"
        assert cfg.input_video == ""
        assert cfg.output_video == ""
        assert cfg.characters == []
        assert cfg.detection_model == "buffalo_l"
        assert cfg.detection_threshold == 0.5
        assert cfg.detection_size == (640, 640)
        assert cfg.tracker_max_age == 30
        assert cfg.tracker_iou_threshold == 0.3
        assert cfg.swap_model_path is None
        assert cfg.embedding_converter_path is None
        assert cfg.enable_face_enhancement is True
        assert cfg.enhancement_model == "gfpgan"
        assert cfg.enhancement_weight == 0.7
        assert cfg.device_id == 0
        assert cfg.batch_size == 4
        assert cfg.use_fp16 is True
        assert cfg.pin_memory is True
        assert cfg.num_io_workers == 2
        assert cfg.output_codec == "libx264"
        assert cfg.output_quality == 18
        assert cfg.copy_audio is True
        assert cfg.blend_mode == "alpha"
        assert cfg.mask_blur_kernel == 15
        assert cfg.mask_erode_pixels == 2
        assert cfg.log_level == "INFO"
        assert cfg.log_file is None


# ---------------------------------------------------------------------------
# PipelineConfig.validate
# ---------------------------------------------------------------------------

class TestPipelineConfigValidate:
    def test_empty_config_errors(self):
        cfg = PipelineConfig()
        errors = cfg.validate()
        assert any("input_video" in e for e in errors)
        assert any("output_video" in e for e in errors)
        assert any("character" in e.lower() for e in errors)

    def test_missing_input_video_file(self):
        cfg = PipelineConfig(input_video="/nonexistent/input.mp4", output_video="out.mp4")
        errors = cfg.validate()
        assert any("input_video not found" in e for e in errors)

    def test_missing_output_video(self):
        cfg = PipelineConfig(input_video="", output_video="")
        errors = cfg.validate()
        assert any("output_video is required" in e for e in errors)

    def test_too_many_characters(self):
        chars = [CharacterMapping(source_label=str(i)) for i in range(4)]
        cfg = PipelineConfig(
            input_video="x.mp4", output_video="out.mp4", characters=chars
        )
        errors = cfg.validate()
        assert any("Maximum of 3" in e for e in errors)

    def test_character_missing_reference_paths(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        cm = CharacterMapping(
            source_label="hero",
            reference_paths=[],
            portrait_paths=[],
        )
        cfg = PipelineConfig(
            input_video=str(video),
            output_video="out.mp4",
            characters=[cm],
        )
        errors = cfg.validate()
        assert any("no reference images" in e for e in errors)

    def test_character_missing_portrait_paths(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        ref = tmp_path / "ref.jpg"
        ref.touch()
        cm = CharacterMapping(
            source_label="hero",
            reference_paths=[str(ref)],
            portrait_paths=[],
        )
        cfg = PipelineConfig(
            input_video=str(video),
            output_video="out.mp4",
            characters=[cm],
        )
        errors = cfg.validate()
        assert any("no portrait images" in e for e in errors)

    def test_reference_image_not_found(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        cm = CharacterMapping(
            source_label="hero",
            reference_paths=["/nonexistent/ref.jpg"],
            portrait_paths=["/nonexistent/portrait.jpg"],
        )
        cfg = PipelineConfig(
            input_video=str(video),
            output_video="out.mp4",
            characters=[cm],
        )
        errors = cfg.validate()
        assert any("Reference image not found" in e for e in errors)

    def test_portrait_not_found(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        ref = tmp_path / "ref.jpg"
        ref.touch()
        cm = CharacterMapping(
            source_label="hero",
            reference_paths=[str(ref)],
            portrait_paths=["/nonexistent/portrait.jpg"],
        )
        cfg = PipelineConfig(
            input_video=str(video),
            output_video="out.mp4",
            characters=[cm],
        )
        errors = cfg.validate()
        assert any("Portrait not found" in e for e in errors)

    def test_valid_config(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        ref = tmp_path / "ref.jpg"
        ref.touch()
        portrait = tmp_path / "portrait.jpg"
        portrait.touch()
        cm = CharacterMapping(
            source_label="hero",
            reference_paths=[str(ref)],
            portrait_paths=[str(portrait)],
        )
        cfg = PipelineConfig(
            input_video=str(video),
            output_video="out.mp4",
            characters=[cm],
        )
        errors = cfg.validate()
        assert errors == []

    def test_exactly_three_characters_allowed(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        chars = []
        for i in range(3):
            ref = tmp_path / f"ref{i}.jpg"
            ref.touch()
            portrait = tmp_path / f"portrait{i}.jpg"
            portrait.touch()
            chars.append(
                CharacterMapping(
                    source_label=f"char{i}",
                    reference_paths=[str(ref)],
                    portrait_paths=[str(portrait)],
                )
            )
        cfg = PipelineConfig(
            input_video=str(video),
            output_video="out.mp4",
            characters=chars,
        )
        errors = cfg.validate()
        assert errors == []

    def test_scail2_backend_requires_prepared_assets(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(video),
            output_video="out.mp4",
        )
        errors = cfg.validate()
        assert any("scail2_repo_path" in e for e in errors)
        assert any("scail2_ckpt_dir" in e for e in errors)
        assert any("scail2_model_path" in e for e in errors)
        assert any("scail2_reference_image or one character mapping" in e for e in errors)
        assert any("scail2_prompt" in e for e in errors)
        assert any("scail2_pose_repo_path" in e for e in errors)

    def test_scail2_auto_prep_allows_one_character_mapping(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "generate.py").write_text("print('ok')\n", encoding="utf-8")
        pose_repo = repo / "SCAIL-Pose"
        (pose_repo / "NLFPoseExtract").mkdir(parents=True)
        (pose_repo / "NLFPoseExtract" / "process_replacement.py").write_text("print('ok')\n", encoding="utf-8")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        model = tmp_path / "model.safetensors"
        model.touch()
        ref = tmp_path / "ref.png"
        ref.touch()
        ref_mask = tmp_path / "ref_mask.png"
        ref_mask.touch()
        mask_video = tmp_path / "mask.mp4"
        mask_video.touch()
        portrait = tmp_path / "portrait.jpg"
        portrait.touch()

        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(video),
            output_video="out.mp4",
            characters=[
                CharacterMapping(
                    source_label="hero",
                    reference_paths=[str(ref)],
                    portrait_paths=[str(portrait)],
                )
            ],
            scail2_repo_path=str(repo),
            scail2_ckpt_dir=str(ckpt_dir),
            scail2_model_path=str(model),
            scail2_prompt="prompt",
        )
        assert cfg.validate() == []

    def test_scail2_backend_rejects_multiple_character_mappings(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "generate.py").write_text("print('ok')\n", encoding="utf-8")
        pose_repo = repo / "SCAIL-Pose"
        (pose_repo / "NLFPoseExtract").mkdir(parents=True)
        (pose_repo / "NLFPoseExtract" / "process_replacement.py").write_text("print('ok')\n", encoding="utf-8")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        model = tmp_path / "model.safetensors"
        model.touch()
        ref = tmp_path / "ref.png"
        ref.touch()
        portrait = tmp_path / "portrait.jpg"
        portrait.touch()
        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(video),
            output_video="out.mp4",
            characters=[
                CharacterMapping(
                    source_label="hero",
                    reference_paths=[str(ref)],
                    portrait_paths=[str(portrait)],
                ),
                CharacterMapping(
                    source_label="villain",
                    reference_paths=[str(ref)],
                    portrait_paths=[str(portrait)],
                ),
            ],
            scail2_repo_path=str(repo),
            scail2_ckpt_dir=str(ckpt_dir),
            scail2_model_path=str(model),
            scail2_prompt="prompt",
        )
        errors = cfg.validate()
        assert any("at most one character mapping" in e for e in errors)

    def test_scail2_prepared_assets_must_be_complete(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "generate.py").write_text("print('ok')\n", encoding="utf-8")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        model = tmp_path / "model.safetensors"
        model.touch()
        ref = tmp_path / "ref.png"
        ref.touch()

        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(video),
            output_video="out.mp4",
            scail2_repo_path=str(repo),
            scail2_ckpt_dir=str(ckpt_dir),
            scail2_model_path=str(model),
            scail2_reference_image=str(ref),
            scail2_prompt="prompt",
        )
        errors = cfg.validate()
        assert any("prepared-assets mode requires" in e for e in errors)

    def test_scail2_target_size_must_be_divisible_by_32(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "generate.py").write_text("print('ok')\n", encoding="utf-8")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        model = tmp_path / "model.safetensors"
        model.touch()
        ref = tmp_path / "ref.png"
        ref.touch()
        ref_mask = tmp_path / "ref_mask.png"
        ref_mask.touch()
        mask_video = tmp_path / "mask.mp4"
        mask_video.touch()

        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(video),
            output_video="out.mp4",
            scail2_repo_path=str(repo),
            scail2_ckpt_dir=str(ckpt_dir),
            scail2_model_path=str(model),
            scail2_reference_image=str(ref),
            scail2_reference_mask=str(ref_mask),
            scail2_mask_video=str(mask_video),
            scail2_prompt="prompt",
            scail2_target_width=705,
            scail2_target_height=512,
        )
        errors = cfg.validate()
        assert any("divisible by 32" in e for e in errors)

    def test_valid_scail2_config(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "generate.py").write_text("print('ok')\n", encoding="utf-8")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        model = tmp_path / "model.safetensors"
        model.touch()
        ref = tmp_path / "ref.png"
        ref.touch()
        ref_mask = tmp_path / "ref_mask.png"
        ref_mask.touch()
        mask_video = tmp_path / "mask.mp4"
        mask_video.touch()

        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(video),
            output_video="out.mp4",
            scail2_repo_path=str(repo),
            scail2_ckpt_dir=str(ckpt_dir),
            scail2_model_path=str(model),
            scail2_reference_image=str(ref),
            scail2_reference_mask=str(ref_mask),
            scail2_mask_video=str(mask_video),
            scail2_prompt="prompt",
            scail2_target_width=704,
            scail2_target_height=512,
        )
        assert cfg.validate() == []

    def test_scail2_matchnearest_and_egocentric_are_mutually_exclusive(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.touch()
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "generate.py").write_text("print('ok')\n", encoding="utf-8")
        pose_repo = repo / "SCAIL-Pose"
        (pose_repo / "NLFPoseExtract").mkdir(parents=True)
        (pose_repo / "NLFPoseExtract" / "process_replacement.py").write_text("print('ok')\n", encoding="utf-8")
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        model = tmp_path / "model.safetensors"
        model.touch()
        ref = tmp_path / "ref.png"
        ref.touch()

        cfg = PipelineConfig(
            backend="scail2",
            input_video=str(video),
            output_video="out.mp4",
            scail2_repo_path=str(repo),
            scail2_ckpt_dir=str(ckpt_dir),
            scail2_model_path=str(model),
            scail2_reference_image=str(ref),
            scail2_prompt="prompt",
            scail2_matchnearest=True,
            scail2_egocentric=True,
        )
        errors = cfg.validate()
        assert any("cannot enable both" in e for e in errors)
