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
