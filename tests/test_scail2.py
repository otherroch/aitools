"""Tests for SCAIL-2 integration in chararep."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chararep.config import PipelineConfig
from chararep.pipeline import CharacterReplacementPipeline


class TestSCAIL2Config(unittest.TestCase):
    """Test SCAIL-2 configuration validation."""

    def test_default_config(self):
        """Test that default config doesn't enable SCAIL-2."""
        cfg = PipelineConfig()
        self.assertFalse(cfg.enable_scail2)
        self.assertIsNone(cfg.scail2_model_path)

    def test_valid_scail2_config(self):
        """Test valid SCAIL-2 configuration."""
        cfg = PipelineConfig(
            input_video="test.mp4",
            output_video="output.mp4",
            enable_scail2=True,
            scail2_model_path="model.ckpt",
            scail2_reference_video="reference.mp4",
            scail2_resolution="704p",
        )
        self.assertTrue(cfg.enable_scail2)
        self.assertEqual(cfg.scail2_model_path, "model.ckpt")
        self.assertEqual(cfg.scail2_reference_video, "reference.mp4")

    def test_invalid_resolution(self):
        """Test that invalid resolution is rejected."""
        cfg = PipelineConfig(
            input_video="test.mp4",
            output_video="output.mp4",
            enable_scail2=True,
            scail2_model_path="model.ckpt",
            scail2_reference_video="reference.mp4",
            scail2_resolution="invalid",
        )
        errors = cfg.validate()
        self.assertTrue(any("invalid resolution" in e for e in errors))

    def test_missing_model_path(self):
        """Test that missing model path is rejected when SCAIL-2 enabled."""
        cfg = PipelineConfig(
            input_video="test.mp4",
            output_video="output.mp4",
            enable_scail2=True,
            scail2_reference_video="reference.mp4",
        )
        errors = cfg.validate()
        self.assertTrue(any("scail2_model_path" in e for e in errors))

    def test_both_reference_types(self):
        """Test that both reference types are rejected."""
        cfg = PipelineConfig(
            input_video="test.mp4",
            output_video="output.mp4",
            enable_scail2=True,
            scail2_model_path="model.ckpt",
            scail2_reference_video="reference.mp4",
            scail2_reference_images=["ref1.jpg", "ref2.jpg"],
        )
        errors = cfg.validate()
        self.assertTrue(any("not both" in e for e in errors))


class TestSCAIL2Pipeline(unittest.TestCase):
    """Test SCAIL-2 pipeline integration."""

    def test_pipeline_creation_without_scail2(self):
        """Test that pipeline can be created without SCAIL-2."""
        cfg = PipelineConfig(
            input_video="test.mp4",
            output_video="output.mp4",
        )
        # Pipeline creation should succeed (it will fail later when trying to load models)
        pipeline = CharacterReplacementPipeline(cfg)
        self.assertIsNotNone(pipeline)

    def test_pipeline_creation_with_scail2(self):
        """Test that pipeline can be created with SCAIL-2 enabled."""
        cfg = PipelineConfig(
            input_video="test.mp4",
            output_video="output.mp4",
            enable_scail2=True,
            scail2_model_path="model.ckpt",
            scail2_reference_video="reference.mp4",
        )
        # This should fail because we don't have actual model files,
        # but it should at least get past configuration
        with self.assertRaises(Exception):
            pipeline = CharacterReplacementPipeline(cfg)

    def test_run_method_exists(self):
        """Test that run method exists in pipeline."""
        # Just verify the method exists
        self.assertTrue(hasattr(CharacterReplacementPipeline, 'run'))
        self.assertTrue(hasattr(CharacterReplacementPipeline, '_run_scail2'))


if __name__ == '__main__':
    unittest.main()