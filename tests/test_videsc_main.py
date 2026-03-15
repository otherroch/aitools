"""
tests/test_videsc_main.py

Tests for the videsc-vl CLI entry point (ported from otherroch/videsc test_cli_entry.py).
"""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
VIDESC_ROOT = REPO_ROOT / "videsc"


class TestVidescVlCommand:
    def test_pyproject_defines_videsc_vl_script(self):
        """pyproject.toml must define a 'videsc-vl' entry-point pointing to videsc.main:main."""
        pyproject = REPO_ROOT / "pyproject.toml"
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        assert "videsc-vl" in scripts, "No 'videsc-vl' script defined in [project.scripts]"
        assert scripts["videsc-vl"] == "videsc.main:main", (
            f"Expected 'videsc.main:main', got '{scripts['videsc-vl']}'"
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

    def test_args_help_exits_zero(self):
        """Running videsc.cli.args with --help should exit with code 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv=['videsc-vl','--help']; "
                "from videsc.cli.args import parse_args; parse_args()",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--video" in result.stdout

    def test_pyproject_defines_videsc_script(self):
        """pyproject.toml must keep the legacy 'videsc' WD14-based entry point."""
        pyproject = REPO_ROOT / "pyproject.toml"
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {})
        assert "videsc" in scripts, "No 'videsc' script defined in [project.scripts]"
        assert scripts["videsc"] == "videsc.wd_cli:main", (
            f"Expected 'videsc.wd_cli:main', got '{scripts['videsc']}'"
        )
