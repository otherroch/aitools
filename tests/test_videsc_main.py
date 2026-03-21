"""
tests/test_videsc_main.py

Tests for the unified videsc CLI entry point.
"""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
VIDESC_ROOT = REPO_ROOT / "videsc"


class TestVidescUnifiedCommand:
    def test_pyproject_defines_videsc_script(self):
        """pyproject.toml must define a 'videsc' entry-point pointing to videsc.main:main."""
        pyproject = REPO_ROOT / "pyproject.toml"
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        assert "videsc" in scripts, "No 'videsc' script defined in [project.scripts]"
        assert scripts["videsc"] == "videsc.main:main", (
            f"Expected 'videsc.main:main', got '{scripts['videsc']}'"
        )

    def test_pyproject_does_not_define_videsc_vl_script(self):
        """pyproject.toml must not define a separate 'videsc-vl' entry-point."""
        pyproject = REPO_ROOT / "pyproject.toml"
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        assert "videsc-vl" not in scripts, (
            "'videsc-vl' should have been merged into 'videsc'"
        )

    def test_videsc_main_py_has_main_function(self):
        """videsc/main.py must define a callable named 'main'."""
        main_py = VIDESC_ROOT / "main.py"
        tree = ast.parse(main_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "main" in func_names, "videsc/main.py must define a 'main' function"

    def test_videsc_main_py_has_run_vl_function(self):
        """videsc/main.py must define a '_run_vl' helper for VL mode."""
        main_py = VIDESC_ROOT / "main.py"
        tree = ast.parse(main_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "_run_vl" in func_names, "videsc/main.py must define a '_run_vl' function"

    def test_videsc_main_py_has_run_wd14_function(self):
        """videsc/main.py must define a '_run_wd14' helper for WD14 mode."""
        main_py = VIDESC_ROOT / "main.py"
        tree = ast.parse(main_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "_run_wd14" in func_names, "videsc/main.py must define a '_run_wd14' function"

    def test_args_help_exits_zero(self):
        """Running videsc with --help should exit with code 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv=['videsc','--help']; "
                "from videsc.cli.args import parse_args; parse_args()",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--vl" in result.stdout

    def test_args_help_includes_wd14_args(self):
        """The --help output must document WD14-mode arguments."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv=['videsc','--help']; "
                "from videsc.cli.args import parse_args; parse_args()",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--input-dir" in result.stdout
        assert "--threshold" in result.stdout
        assert "--model-repo" in result.stdout

    def test_args_help_includes_vl_args(self):
        """The --help output must document VL-mode arguments."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv=['videsc','--help']; "
                "from videsc.cli.args import parse_args; parse_args()",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--video" in result.stdout
        assert "--quant" in result.stdout
        assert "--audio" in result.stdout

    def test_parse_args_vl_flag_default_false(self):
        """--vl flag defaults to False when not specified."""
        from videsc.cli.args import parse_args

        # Minimal WD14 args (won't actually validate required fields here)
        args = parse_args(["--input-dir", "/tmp"])
        assert args.vl is False

    def test_parse_args_vl_flag_true(self):
        """--vl flag is True when specified."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--video", "/tmp/test.mp4"])
        assert args.vl is True

    def test_parse_args_wd14_defaults(self):
        """WD14 default values are correct."""
        from videsc.cli.args import parse_args

        args = parse_args(["--input-dir", "/tmp"])
        assert args.every_n == 30
        assert args.max_frames == 10
        assert args.threshold == 0.35
        assert args.prefix == ""
        assert args.model_repo == "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
        assert args.include_ratings is False
        assert args.no_skip_existing is False

    def test_parse_args_vl_defaults(self):
        """VL mode default values are correct."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--video", "/tmp/test.mp4"])
        assert args.model == "Qwen/Qwen3-VL-8B-Instruct"
        assert args.quant == "none"
        assert args.attn == "flash_attention_2"
        assert args.spf == 4.0
        assert args.num_frames == 256
        assert args.audio is False

    def test_parse_args_qwen35_flag_default_false(self):
        """--qwen35 flag defaults to False when not specified."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--video", "/tmp/test.mp4"])
        assert args.qwen35 is False

    def test_parse_args_qwen35_flag_true(self):
        """--qwen35 flag is True when specified and defaults model to Qwen3.5-4B."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--qwen35", "--video", "/tmp/test.mp4"])
        assert args.qwen35 is True
        assert args.model == "Qwen/Qwen3.5-4B"
        assert args.model_hf is True

    def test_parse_args_qwen35_with_model(self):
        """--qwen35 flag works with a custom model name."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--qwen35",
            "--model", "Qwen/Qwen3.5-4B",
            "--model_hf",
            "--video", "/tmp/test.mp4",
        ])
        assert args.qwen35 is True
        assert args.model == "Qwen/Qwen3.5-4B"
        assert args.model_hf is True

    def test_parse_args_qwen35_explicit_model_not_overridden(self):
        """--qwen35 with explicit --model keeps user's model choice."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--qwen35",
            "--model", "Qwen/Qwen3.5-9B",
            "--model_full",
            "--video", "/tmp/test.mp4",
        ])
        assert args.qwen35 is True
        assert args.model == "Qwen/Qwen3.5-9B"
        assert args.model_full is True
        assert args.model_hf is False

    def test_args_help_includes_qwen35(self):
        """The --help output must document the --qwen35 argument."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv=['videsc','--help']; "
                "from videsc.cli.args import parse_args; parse_args()",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--qwen35" in result.stdout

    def test_loader_has_load_qwen35_function(self):
        """videsc/model/loader.py must define 'load_qwen35_model_and_processor'."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        tree = ast.parse(loader_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "load_qwen35_model_and_processor" in func_names, (
            "videsc/model/loader.py must define 'load_qwen35_model_and_processor'"
        )
