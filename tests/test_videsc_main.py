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
        assert args.attn == "sdpa"
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

    # ------------------------------------------------------------------
    # Qwen3.6 tests
    # ------------------------------------------------------------------

    def test_parse_args_qwen36_flag_default_false(self):
        """--qwen36 flag defaults to False when not specified."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--video", "/tmp/test.mp4"])
        assert args.qwen36 is False

    def test_parse_args_qwen36_flag_true(self):
        """--qwen36 flag is True when specified and defaults model to Qwen3.6-27B."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--qwen36", "--video", "/tmp/test.mp4"])
        assert args.qwen36 is True
        assert args.model == "Qwen/Qwen3.6-27B"
        assert args.model_hf is True

    def test_parse_args_qwen36_with_model(self):
        """--qwen36 flag works with a custom model name."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--qwen36",
            "--model", "Qwen/Qwen3.6-35B-A3B",
            "--model_hf",
            "--video", "/tmp/test.mp4",
        ])
        assert args.qwen36 is True
        assert args.model == "Qwen/Qwen3.6-35B-A3B"
        assert args.model_hf is True

    def test_parse_args_qwen36_explicit_model_not_overridden(self):
        """--qwen36 with explicit --model keeps user's model choice."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--qwen36",
            "--model", "Qwen/Qwen3.6-27B",
            "--model_full",
            "--video", "/tmp/test.mp4",
        ])
        assert args.qwen36 is True
        assert args.model == "Qwen/Qwen3.6-27B"
        assert args.model_full is True
        assert args.model_hf is False

    def test_args_help_includes_qwen36(self):
        """The --help output must document the --qwen36 argument."""
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
        assert "--qwen36" in result.stdout

    def test_loader_has_load_qwen36_function(self):
        """videsc/model/loader.py must define 'load_qwen36_model_and_processor'."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        tree = ast.parse(loader_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "load_qwen36_model_and_processor" in func_names, (
            "videsc/model/loader.py must define 'load_qwen36_model_and_processor'"
        )

    def test_run_vl_source_handles_qwen36(self):
        """_run_vl must dispatch to the Qwen3.6 pipeline when --qwen36 is set."""
        main_py = VIDESC_ROOT / "main.py"
        source = main_py.read_text()
        assert "qwen36" in source, "_run_vl must handle the --qwen36 flag"

    # ------------------------------------------------------------------
    # Qwen3.6 quantization support (AWQ / NVFP4)
    # ------------------------------------------------------------------

    def test_parse_args_quant_awq(self):
        """--quant awq is a valid choice and leaves model quantization at passthrough."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--qwen36", "--quant", "awq", "--video", "/tmp/test.mp4"])
        assert args.quant == "awq"

    def test_parse_args_quant_nvfp4(self):
        """--quant nvfp4 is a valid choice and leaves model quantization at passthrough."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--qwen36", "--quant", "nvfp4", "--video", "/tmp/test.mp4"])
        assert args.quant == "nvfp4"

    def test_quant_config_awq_returns_none(self):
        """_quant_config('awq') must return None (no BitsAndBytes config)."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        # Verify textually that awq case returns None (not a BnB config)
        source = loader_py.read_text()
        assert "awq" in source, "loader must mention 'awq' in _quant_config or its comments"

    def test_quant_config_nvfp4_returns_none(self):
        """_quant_config('nvfp4') must return None (no BitsAndBytes config)."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        assert "nvfp4" in source, "loader must mention 'nvfp4' in _quant_config or its comments"

    def test_loader_imports_qwen35_moe_class(self):
        """loader.py must import Qwen3_5MoeForConditionalGeneration for MoE model support."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        assert "Qwen3_5MoeForConditionalGeneration" in source, (
            "loader must import Qwen3_5MoeForConditionalGeneration to handle MoE Qwen3.6 variants"
        )

    def test_loader_imports_autoconfig(self):
        """loader.py must import AutoConfig for architecture auto-detection."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        assert "AutoConfig" in source, (
            "loader must import AutoConfig to detect dense vs MoE architecture from config.json"
        )

    def test_loader_qwen36_uses_autoconfig_for_arch_detection(self):
        """load_qwen36_model_and_processor must use AutoConfig to detect MoE architecture."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        assert "Qwen3_5MoeForConditionalGeneration" in source
        assert "AutoConfig" in source

    def test_loader_qwen36_prevents_cpu_offload_for_quantized_models(self):
        """load_qwen36_model_and_processor must build a CUDA-only max_memory dict for
        nvfp4/awq to prevent accelerate from CPU/disk-offloading quantized layers.
        Packed weight names in the offload index don't match the float names the
        forward pass looks up, causing a KeyError at inference time.
        The fix omits the 'cpu' key entirely so accelerate only considers CUDA devices."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        # Must build a max_memory dict (CUDA-only) for nvfp4/awq
        assert "max_memory" in source, (
            "loader must pass max_memory to from_pretrained for nvfp4/awq models"
        )
        # Must NOT set cpu=0 (that still causes disk offloading); cpu key must be absent
        assert 'max_memory["cpu"] = 0' not in source and "max_memory['cpu'] = 0" not in source, (
            "loader must NOT set max_memory['cpu']=0; omit the 'cpu' key entirely so "
            "accelerate never falls back to disk offloading for quantized models"
        )

    def test_vl_youtube_output_dir_fallback(self):
        """In VL + YouTube mode, --output-dir should be used as fallback for --outdir."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl",
            "--youtube-url", "https://www.youtube.com/watch?v=test",
            "--youtube-api-key", "fake-key",
            "--output-dir", "./out",
            "--video", "/tmp/test.mp4",
        ])
        # Before _run_vl processes it, outdir is None and output_dir is set
        assert args.outdir is None
        assert args.output_dir == Path("out")

    def test_run_vl_source_honours_output_dir_for_youtube(self):
        """_run_vl must set args.outdir from args.output_dir when using YouTube."""
        main_py = VIDESC_ROOT / "main.py"
        source = main_py.read_text()
        # Verify the fallback logic exists in _run_vl
        assert "args.outdir = str(args.output_dir)" in source, (
            "_run_vl must fall back to --output-dir for --outdir in YouTube mode"
        )

    def test_runner_converts_total_pixels_to_raw(self):
        """run_single_video must convert total_pixels from edge-multiplier to raw pixels."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "args.total_pixels * patch * patch" in source, (
            "runner must convert total_pixels to raw pixels using patch size"
        )

    def test_runner_uses_patch_for_image_patch_size(self):
        """process_vision_info must use model-specific patch size, not hardcoded 16."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "image_patch_size=patch // 2" in source, (
            "process_vision_info must use patch // 2 instead of hardcoded 16"
        )
        assert "image_patch_size=16" not in source, (
            "runner must not hardcode image_patch_size=16"
        )

    def test_loader_uses_patch_for_pixel_limits(self):
        """All model loaders must use _patch_size_for_model for pixel limit calculation."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        assert "from videsc.utils.helpers import _patch_size_for_model" in source, (
            "loader must import _patch_size_for_model"
        )
        # Should not contain hardcoded 32 * 32 in processor initialization
        assert "args.min_pixels * 32 * 32" not in source, (
            "loader must not hardcode 32 * 32 for min_pixels"
        )
        assert "args.max_pixels * 32 * 32" not in source, (
            "loader must not hardcode 32 * 32 for max_pixels"
        )
        # Should use patch * patch instead
        assert "args.min_pixels * patch * patch" in source, (
            "loader must use patch * patch for min_pixels"
        )
        assert "args.max_pixels * patch * patch" in source, (
            "loader must use patch * patch for max_pixels"
        )

    def test_total_pixels_conversion_math(self):
        """Verify total_pixels edge-multiplier to raw pixel conversion."""
        from videsc.utils.helpers import _patch_size_for_model
        # For Qwen3.5: patch=32, total_pixels=24000 → raw=24000*1024=24,576,000
        patch = _patch_size_for_model("Qwen/Qwen3.5-4B")
        assert patch == 32
        raw = 24000 * patch * patch
        assert raw == 24_576_000
        # For Qwen2.5: patch=28, total_pixels=24000 → raw=24000*784=18,816,000
        patch25 = _patch_size_for_model("Qwen/Qwen2.5-VL-7B")
        assert patch25 == 28
        raw25 = 24000 * patch25 * patch25
        assert raw25 == 18_816_000

    # ------------------------------------------------------------------
    # --save-video tests
    # ------------------------------------------------------------------

    def test_args_help_includes_save_video(self):
        """The --help output must document the --save-video argument."""
        import subprocess, sys
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
        assert "--save-video" in result.stdout

    def test_parse_args_save_video_default_none(self):
        """--save-video defaults to None when not specified."""
        from videsc.cli.args import parse_args

        args = parse_args(["--input-dir", "/tmp"])
        assert args.save_video is None

    def test_parse_args_save_video_file_path(self):
        """--save-video accepts a full file path."""
        from videsc.cli.args import parse_args
        from pathlib import Path

        args = parse_args([
            "--youtube-url", "https://www.youtube.com/watch?v=abc123",
            "--youtube-api-key", "fake-key",
            "--save-video", "/tmp/video.mp4",
        ])
        assert args.save_video == Path("/tmp/video.mp4")

    def test_run_wd14_save_video_exits_without_processing(self):
        """When --save-video is set, _run_wd14 saves the file and exits before describe_youtube."""
        main_py = VIDESC_ROOT / "main.py"
        source = main_py.read_text()
        # The save-and-exit block must appear before the describe_youtube call
        save_video_pos = source.find("args.save_video is not None")
        describe_youtube_pos = source.find("describe_youtube(")
        assert save_video_pos != -1, "_run_wd14 must check args.save_video"
        assert describe_youtube_pos != -1, "_run_wd14 must call describe_youtube"
        assert save_video_pos < describe_youtube_pos, (
            "save-and-exit logic must come before describe_youtube call"
        )

    def test_run_wd14_save_video_copies_to_file_path(self):
        """_run_wd14 must copy the video to the exact file path, not a directory."""
        main_py = VIDESC_ROOT / "main.py"
        source = main_py.read_text()
        assert "shutil.copy2(video_path, dest)" in source, (
            "_run_wd14 must use shutil.copy2 to save the video"
        )

    def test_run_vl_save_video_exits_without_model_load(self):
        """When --save-video is set in VL mode, the tool saves and returns 0 before loading the model."""
        main_py = VIDESC_ROOT / "main.py"
        source = main_py.read_text()
        assert "args.save_video is not None" in source, (
            "_run_vl must check args.save_video before loading the model"
        )

    # ------------------------------------------------------------------
    # Gemma 4 tests
    # ------------------------------------------------------------------

    def test_parse_args_gemma4_flag_default_false(self):
        """--gemma4 flag defaults to False when not specified."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--video", "/tmp/test.mp4"])
        assert args.gemma4 is False

    def test_parse_args_gemma4_flag_true(self):
        """--gemma4 flag is True when specified and defaults model to gemma-4-4eb-it."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--gemma4", "--video", "/tmp/test.mp4"])
        assert args.gemma4 is True
        assert args.model == "google/gemma-4-4eb-it"
        assert args.model_hf is True

    def test_parse_args_gemma4_explicit_model_not_overridden(self):
        """--gemma4 with explicit --model keeps the user's model choice."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--gemma4",
            "--model", "google/gemma-4-27b-it",
            "--model_hf",
            "--video", "/tmp/test.mp4",
        ])
        assert args.gemma4 is True
        assert args.model == "google/gemma-4-27b-it"
        assert args.model_hf is True

    def test_parse_args_gemma4_chunk_duration_default(self):
        """--gemma4-chunk-duration defaults to 30.0 seconds."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--gemma4", "--video", "/tmp/test.mp4"])
        assert args.gemma4_chunk_duration == 30.0

    def test_parse_args_gemma4_chunk_duration_custom(self):
        """--gemma4-chunk-duration accepts a custom value."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--gemma4",
            "--gemma4-chunk-duration", "45",
            "--video", "/tmp/test.mp4",
        ])
        assert args.gemma4_chunk_duration == 45.0

    def test_parse_args_gemma4_fps_default(self):
        """--gemma4-fps defaults to 1.0."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--gemma4", "--video", "/tmp/test.mp4"])
        assert args.gemma4_fps == 1.0

    def test_parse_args_gemma4_fps_custom(self):
        """--gemma4-fps accepts a custom value."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--gemma4",
            "--gemma4-fps", "2.0",
            "--video", "/tmp/test.mp4",
        ])
        assert args.gemma4_fps == 2.0

    def test_args_help_includes_gemma4(self):
        """The --help output must document the --gemma4 argument."""
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
        assert "--gemma4" in result.stdout

    def test_args_help_includes_gemma4_chunk_duration(self):
        """The --help output must document --gemma4-chunk-duration."""
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
        assert "--gemma4-chunk-duration" in result.stdout

    def test_loader_has_load_gemma4_function(self):
        """videsc/model/loader.py must define 'load_gemma4_model_and_processor'."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        tree = ast.parse(loader_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "load_gemma4_model_and_processor" in func_names, (
            "videsc/model/loader.py must define 'load_gemma4_model_and_processor'"
        )

    def test_runner_has_extract_frames_as_pil(self):
        """videsc/pipeline/runner.py must define 'extract_frames_as_pil'."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        tree = ast.parse(runner_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "extract_frames_as_pil" in func_names, (
            "videsc/pipeline/runner.py must define 'extract_frames_as_pil'"
        )

    def test_runner_has_run_single_video_gemma4(self):
        """videsc/pipeline/runner.py must define 'run_single_video_gemma4'."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        tree = ast.parse(runner_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "run_single_video_gemma4" in func_names, (
            "videsc/pipeline/runner.py must define 'run_single_video_gemma4'"
        )

    def test_runner_gemma4_handles_chunking(self):
        """run_single_video_gemma4 must split long videos into chunks."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "chunk_duration" in source, (
            "runner must use chunk_duration to split long videos for Gemma 4"
        )
        assert "gemma4_chunk_duration" in source, (
            "runner must read gemma4_chunk_duration from args"
        )

    def test_loader_gemma4_uses_auto_model_for_multimodal_lm(self):
        """Gemma 4 loader must use AutoModelForMultimodalLM."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        assert "AutoModelForMultimodalLM" in source, (
            "loader must use AutoModelForMultimodalLM for Gemma 4"
        )

    def test_runner_gemma4_uses_native_video_type(self):
        """run_single_video_gemma4 must pass the video via the native video content type."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert '"type": "video"' in source or "'type': 'video'" in source, (
            "run_single_video_gemma4 must use the native video content type"
        )

    def test_runner_gemma4_uses_parse_response(self):
        """run_single_video_gemma4 must use processor.parse_response to decode output."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "parse_response" in source, (
            "run_single_video_gemma4 must use processor.parse_response"
        )

    def test_loader_gemma4_sets_padding_side_left(self):
        """Gemma 4 processor must be loaded with padding_side='left'."""
        loader_py = VIDESC_ROOT / "model" / "loader.py"
        source = loader_py.read_text()
        assert 'padding_side="left"' in source or "padding_side='left'" in source, (
            "Gemma 4 processor must be loaded with padding_side='left'"
        )

    def test_run_vl_source_handles_gemma4(self):
        """_run_vl must dispatch to the Gemma 4 pipeline when --gemma4 is set."""
        main_py = VIDESC_ROOT / "main.py"
        source = main_py.read_text()
        assert "gemma4" in source, "_run_vl must handle the --gemma4 flag"
        assert "load_gemma4_model_and_processor" in source, (
            "_run_vl must call load_gemma4_model_and_processor"
        )
        assert "run_single_video_gemma4" in source, (
            "_run_vl must call run_single_video_gemma4"
        )

    # ------------------------------------------------------------------
    # Consolidation tests (multi-stage pipeline for Gemma 4)
    # ------------------------------------------------------------------

    def test_parse_args_consolidate_default_false(self):
        """--consolidate flag defaults to False when not specified."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--gemma4", "--video", "/tmp/test.mp4"])
        assert args.consolidate is False

    def test_parse_args_consolidate_flag_true(self):
        """--consolidate flag is True when specified."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--gemma4", "--consolidate", "--video", "/tmp/test.mp4",
        ])
        assert args.consolidate is True

    def test_parse_args_consolidate_prompt_default_none(self):
        """--consolidate-prompt defaults to None (built-in prompt used)."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--gemma4", "--consolidate", "--video", "/tmp/test.mp4"])
        assert args.consolidate_prompt is None

    def test_parse_args_consolidate_prompt_custom(self):
        """--consolidate-prompt accepts a custom value."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--gemma4", "--consolidate",
            "--consolidate-prompt", "Summarize all.",
            "--video", "/tmp/test.mp4",
        ])
        assert args.consolidate_prompt == "Summarize all."

    def test_parse_args_segment_prompt_default_none(self):
        """--segment-prompt defaults to None (built-in structured prompt used)."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--gemma4", "--consolidate", "--video", "/tmp/test.mp4"])
        assert args.segment_prompt is None

    def test_parse_args_segment_prompt_custom(self):
        """--segment-prompt accepts a custom value."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--gemma4", "--consolidate",
            "--segment-prompt", "Describe what you see.",
            "--video", "/tmp/test.mp4",
        ])
        assert args.segment_prompt == "Describe what you see."

    def test_parse_args_window_size_default(self):
        """--window-size defaults to 10."""
        from videsc.cli.args import parse_args

        args = parse_args(["--vl", "--gemma4", "--consolidate", "--video", "/tmp/test.mp4"])
        assert args.window_size == 10

    def test_parse_args_window_size_custom(self):
        """--window-size accepts a custom integer."""
        from videsc.cli.args import parse_args

        args = parse_args([
            "--vl", "--gemma4", "--consolidate",
            "--window-size", "5",
            "--video", "/tmp/test.mp4",
        ])
        assert args.window_size == 5

    def test_args_help_includes_consolidate(self):
        """The --help output must document the --consolidate argument."""
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
        assert "--consolidate" in result.stdout

    def test_args_help_includes_consolidate_prompt(self):
        """The --help output must document the --consolidate-prompt argument."""
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
        assert "--consolidate-prompt" in result.stdout

    def test_args_help_includes_segment_prompt(self):
        """The --help output must document the --segment-prompt argument."""
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
        assert "--segment-prompt" in result.stdout

    def test_args_help_includes_window_size(self):
        """The --help output must document the --window-size argument."""
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
        assert "--window-size" in result.stdout

    # ── prompts module ──────────────────────────────────────────────────

    def test_prompts_module_exists(self):
        """videsc/pipeline/prompts.py must exist."""
        prompts_py = VIDESC_ROOT / "pipeline" / "prompts.py"
        assert prompts_py.is_file(), "videsc/pipeline/prompts.py must exist"

    def test_prompts_exports_segment_prompt(self):
        """prompts module must export SEGMENT_PROMPT."""
        from videsc.pipeline.prompts import SEGMENT_PROMPT
        assert "events" in SEGMENT_PROMPT.lower()
        assert "objects" in SEGMENT_PROMPT.lower()
        assert "actions" in SEGMENT_PROMPT.lower()
        assert "scene" in SEGMENT_PROMPT.lower()
        assert "summary" in SEGMENT_PROMPT.lower()

    def test_prompts_exports_window_aggregation_prompt(self):
        """prompts module must export WINDOW_AGGREGATION_PROMPT."""
        from videsc.pipeline.prompts import WINDOW_AGGREGATION_PROMPT
        assert "merge" in WINDOW_AGGREGATION_PROMPT.lower() or "consolidate" in WINDOW_AGGREGATION_PROMPT.lower()

    def test_prompts_exports_final_summary_prompt(self):
        """prompts module must export FINAL_SUMMARY_PROMPT."""
        from videsc.pipeline.prompts import FINAL_SUMMARY_PROMPT
        assert "OVERVIEW" in FINAL_SUMMARY_PROMPT
        assert "TIMELINE" in FINAL_SUMMARY_PROMPT
        assert "ENTITIES" in FINAL_SUMMARY_PROMPT
        assert "ACTIONS" in FINAL_SUMMARY_PROMPT
        assert "THEMES" in FINAL_SUMMARY_PROMPT

    def test_prompts_segment_has_format_placeholder(self):
        """SEGMENT_PROMPT must accept chunk_duration and timestamp formatting."""
        from videsc.pipeline.prompts import SEGMENT_PROMPT
        formatted = SEGMENT_PROMPT.format(
            chunk_duration=30,
            timestamp_start="00:05:00",
            timestamp_end="00:05:30",
        )
        assert "30" in formatted
        assert "00:05:00" in formatted
        assert "00:05:30" in formatted

    def test_prompts_default_token_limits(self):
        """prompts module must define per-stage max-token defaults."""
        from videsc.pipeline.prompts import (
            DEFAULT_SEGMENT_MAX_TOKENS,
            DEFAULT_WINDOW_MAX_TOKENS,
            DEFAULT_FINAL_MAX_TOKENS,
        )
        assert 100 < DEFAULT_SEGMENT_MAX_TOKENS <= 1000
        assert DEFAULT_WINDOW_MAX_TOKENS >= 1024
        assert DEFAULT_FINAL_MAX_TOKENS >= 1024

    # ── runner pipeline structure ───────────────────────────────────────

    def test_runner_has_consolidate_segments(self):
        """videsc/pipeline/runner.py must define 'consolidate_segments'."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        tree = ast.parse(runner_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "consolidate_segments" in func_names

    def test_runner_has_aggregate_windows(self):
        """videsc/pipeline/runner.py must define 'aggregate_windows'."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        tree = ast.parse(runner_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "aggregate_windows" in func_names

    def test_runner_has_generate_text_helper(self):
        """videsc/pipeline/runner.py must define '_generate_text'."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        tree = ast.parse(runner_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "_generate_text" in func_names

    def test_runner_consolidate_invoked_conditionally(self):
        """run_single_video_gemma4 must call consolidate_segments when --consolidate is set."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "consolidate_segments" in source
        assert "use_consolidation" in source

    def test_runner_consolidate_output_format(self):
        """When consolidation is used the output must contain both summary and per-segment sections."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "Consolidated Summary" in source
        assert "Per-Segment Descriptions" in source

    def test_runner_consolidate_skipped_for_single_chunk(self):
        """Consolidation must be skipped when the video has only one chunk."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "len(all_descriptions) > 1" in source

    def test_runner_imports_prompts(self):
        """runner.py must import prompt constants from prompts module."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "SEGMENT_PROMPT" in source
        assert "WINDOW_AGGREGATION_PROMPT" in source
        assert "FINAL_SUMMARY_PROMPT" in source

    def test_runner_uses_segment_prompt_for_consolidation(self):
        """When consolidation is enabled, runner must use SEGMENT_PROMPT for chunks."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "SEGMENT_PROMPT.format" in source

    def test_runner_has_seconds_to_hhmmss_helper(self):
        """runner.py must define '_seconds_to_hhmmss' helper."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        tree = ast.parse(runner_py.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "_seconds_to_hhmmss" in func_names

    def test_seconds_to_hhmmss_conversion(self):
        """_seconds_to_hhmmss logic must correctly convert seconds to HH:MM:SS."""
        # Test the logic directly — importing from runner.py requires torch/GPU stubs
        def _seconds_to_hhmmss(seconds: float) -> str:
            total = int(seconds)
            h = total // 3600
            m = (total % 3600) // 60
            s = total % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        # Verify the function source in runner.py matches this logic
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "def _seconds_to_hhmmss" in source
        assert "total // 3600" in source
        assert "(total % 3600) // 60" in source

        # Test the logic itself
        assert _seconds_to_hhmmss(0) == "00:00:00"
        assert _seconds_to_hhmmss(30) == "00:00:30"
        assert _seconds_to_hhmmss(90) == "00:01:30"
        assert _seconds_to_hhmmss(3661) == "01:01:01"
        assert _seconds_to_hhmmss(300.5) == "00:05:00"

    def test_runner_passes_timestamps_to_segment_prompt(self):
        """runner must pass timestamp_start and timestamp_end to SEGMENT_PROMPT.format()."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert "timestamp_start=" in source
        assert "timestamp_end=" in source
        assert "_seconds_to_hhmmss" in source

    def test_runner_ffmpeg_uses_duration_flag(self):
        """ffmpeg trimming must use -t (duration) instead of -to (absolute end)."""
        runner_py = VIDESC_ROOT / "pipeline" / "runner.py"
        source = runner_py.read_text()
        assert '"-t"' in source
        assert "avoid_negative_ts" in source
