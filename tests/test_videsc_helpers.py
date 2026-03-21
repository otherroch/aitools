"""
tests/test_videsc_helpers.py

Unit tests for videsc.utils.helpers (ported from otherroch/videsc).
"""

import argparse
import pytest
from videsc.utils.helpers import (
    _format_time_hhmmss,
    _patch_size_for_model,
    _edge_to_pixels,
    expand_inputs,
    namespace_to_cli,
)


class TestFormatTime:
    def test_zero(self):
        assert _format_time_hhmmss(0.0) == "0:00:00.000"

    def test_one_minute(self):
        assert _format_time_hhmmss(60.0) == "0:01:00.000"

    def test_one_hour(self):
        assert _format_time_hhmmss(3600.0) == "1:00:00.000"

    def test_with_milliseconds(self):
        assert _format_time_hhmmss(1.5) == "0:00:01.500"

    def test_negative_clamps_to_zero(self):
        assert _format_time_hhmmss(-5.0) == "0:00:00.000"

    def test_complex(self):
        # 1h 2m 3.456s
        assert _format_time_hhmmss(3723.456) == "1:02:03.456"


class TestPatchSize:
    def test_qwen3_returns_32(self):
        assert _patch_size_for_model("Qwen3-VL-8B-Instruct") == 32

    def test_qwen3_case_insensitive(self):
        assert _patch_size_for_model("QWEN3-VL") == 32

    def test_qwen2_returns_28(self):
        assert _patch_size_for_model("Qwen2.5-VL-7B") == 28

    def test_empty_returns_28(self):
        assert _patch_size_for_model("") == 28

    def test_qwen35_returns_32(self):
        assert _patch_size_for_model("Qwen3.5-4B") == 32

    def test_qwen35_full_path_returns_32(self):
        assert _patch_size_for_model("Qwen/Qwen3.5-4B") == 32


class TestEdgeToPixels:
    def test_basic(self):
        assert _edge_to_pixels(1, 32) == 1024

    def test_patch_28(self):
        assert _edge_to_pixels(2, 28) == 2 * 28 * 28


class TestExpandInputs:
    def test_empty_returns_empty(self):
        result = expand_inputs(None, None, [], None)
        assert result == []

    def test_directory_scan_with_extensions(self, tmp_path):
        # Create files with various extensions
        file_mp4 = tmp_path / "video.mp4"
        file_mov = tmp_path / "clip.mov"
        file_txt = tmp_path / "notes.txt"
        file_mp4.write_text("mp4")
        file_mov.write_text("mov")
        file_txt.write_text("txt")

        # Scan the directory, filtering by video extensions
        result = expand_inputs(None, str(tmp_path), [".mp4", ".mov"], None)

        result_set = set(map(str, result))
        assert str(file_mp4) in result_set
        assert str(file_mov) in result_set
        assert str(file_txt) not in result_set

    def test_filelist_processing(self, tmp_path):
        # Create some files and a filelist that references them
        file1 = tmp_path / "input1.txt"
        file2 = tmp_path / "input2.txt"
        file1.write_text("data1")
        file2.write_text("data2")

        filelist = tmp_path / "files.txt"
        filelist.write_text(f"{file1}\n{file2}\n")

        result = expand_inputs(None, None, [], str(filelist))

        assert set(map(str, result)) == {str(file1), str(file2)}


class TestNamespaceToCli:
    def test_bool_true_adds_flag(self):
        ns = argparse.Namespace(audio=True, dry=False)
        argv = namespace_to_cli(ns, exclude_keys=set())
        assert "--audio" in argv
        assert "--dry" not in argv

    def test_string_value(self):
        ns = argparse.Namespace(model="Qwen/Qwen3-VL-8B-Instruct")
        argv = namespace_to_cli(ns, exclude_keys=set())
        assert "--model" in argv
        assert "Qwen/Qwen3-VL-8B-Instruct" in argv

    def test_none_value_excluded(self):
        ns = argparse.Namespace(outdir=None)
        argv = namespace_to_cli(ns, exclude_keys=set())
        assert "--outdir" not in argv

    def test_exclude_keys(self):
        ns = argparse.Namespace(workers=4)
        argv = namespace_to_cli(ns, exclude_keys={"workers"})
        assert "--workers" not in argv

    def test_list_value(self):
        ns = argparse.Namespace(ext=[".mp4", ".mov"])
        argv = namespace_to_cli(ns, exclude_keys=set())
        assert "--ext" in argv
        assert ".mp4" in argv
        assert ".mov" in argv
