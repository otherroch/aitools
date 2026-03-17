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


class TestRunnerVLPipeline:
    """Structural tests verifying the VL pipeline metadata handling in runner.py."""

    RUNNER = VIDESC_ROOT / "pipeline" / "runner.py"

    def test_runner_does_not_import_is_qwen35_model(self):
        """runner.py must NOT import _is_qwen35_model.

        Model-class detection is done in loader.py; runner.py uses the same
        two-step metadata path for all VL models (Qwen3-VL and Qwen3.5).
        """
        src = self.RUNNER.read_text()
        assert "_is_qwen35_model" not in src, (
            "runner.py must not import _is_qwen35_model; model-specific "
            "class selection belongs in loader.py"
        )

    def test_runner_unified_metadata_path(self):
        """All VL models use the same metadata path: use_metadata = not args.no_meta."""
        src = self.RUNNER.read_text()
        assert "use_metadata = not args.no_meta" in src, (
            "runner.py must use 'use_metadata = not args.no_meta' for all models"
        )

    def test_runner_video_metadata_passed_conditionally(self):
        """video_metadata must only be passed when video_metadatas is not None."""
        src = self.RUNNER.read_text()
        assert 'video_metadatas is not None' in src, (
            "runner.py must guard video_metadata kwarg with an is-not-None check"
        )

    def test_runner_pops_video_metadata_before_generate(self):
        """runner.py must pop video_metadata from BatchFeature before model.generate().

        The Qwen3VL/Qwen3.5 processor may return video_metadata (a list of
        VideoMetadata named-tuples, not tensors) inside the BatchFeature.
        model.generate(**inputs) would receive it as an unknown kwarg and crash
        (Qwen3-VL: TypeError/crash; Qwen3.5: StopIteration in get_rope_index).
        Popping it before generate() prevents both failure modes.
        """
        src = self.RUNNER.read_text()
        assert 'inputs.data.pop("video_metadata", None)' in src, (
            'runner.py must remove video_metadata from the BatchFeature before '
            'calling model.generate() to avoid an unknown-kwarg / StopIteration crash'
        )

    def test_runner_passes_do_resize_false_to_processor(self):
        """processor must be called with do_resize=False.

        process_vision_info already resizes video frames to the target resolution.
        Without do_resize=False the processor tries to resize again, which causes
        a C-level crash for Qwen3.5 (producing no visible Python traceback because
        stdout is still buffered when the crash occurs).
        """
        src = self.RUNNER.read_text()
        assert "do_resize=False" in src, (
            "runner.py must pass do_resize=False to processor() because "
            "process_vision_info already resized the frames"
        )

    def test_runner_uses_dict_key_access_for_input_ids(self):
        """generated_ids_trimmed must use inputs['input_ids'] (dict-key access)."""
        src = self.RUNNER.read_text()
        assert 'inputs["input_ids"]' in src, (
            "runner.py must use inputs['input_ids'] for trimming (works for "
            "BatchEncoding returned by processor())"
        )

    def test_runner_no_tokenize_true_in_vl_branch(self):
        """runner.py must NOT call apply_chat_template(tokenize=True).

        apply_chat_template(tokenize=True) ignores the nframes/total_pixels
        constraints in the message dict and attempts to load all video frames,
        which causes OOM for large videos.
        """
        import re
        src = self.RUNNER.read_text()
        # Strip single-line comments before searching so the check isn't
        # tripped by explanatory comments that mention the rejected approach.
        src_no_comments = re.sub(r"#[^\n]*", "", src)
        assert "tokenize=True" not in src_no_comments, (
            "runner.py must not use apply_chat_template(tokenize=True); "
            "it ignores nframes constraints and loads all video frames"
        )
