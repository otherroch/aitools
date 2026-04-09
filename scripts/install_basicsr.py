#!/usr/bin/env python3
"""Download, patch, and install basicsr 1.4.2.

basicsr 1.4.2 has two bugs that prevent installation on Python 3.13+ with
modern torchvision:

1. setup.py uses ``exec(code, locals())`` which is broken by PEP 667
   (Python 3.13 made ``locals()`` return a snapshot).
2. ``basicsr/data/degradations.py`` imports from
   ``torchvision.transforms.functional_tensor`` which was removed in
   torchvision >= 0.17

This script downloads the sdist, applies both patches, then installs from
the patched source.  It is safe to re-run — it will skip if basicsr is
already importable at the correct version.

Usage
-----
    python scripts/install_basicsr.py          # interactive / CI
    python scripts/install_basicsr.py --force  # reinstall even if present
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

BASICSR_VERSION = "1.4.2"


# ── helpers ────────────────────────────────────────────────────────────
def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def _basicsr_installed() -> bool:
    """Return True if basicsr is already importable at the expected version."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", "import basicsr; print(basicsr.__version__)"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() == BASICSR_VERSION
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _download_sdist(dest_dir: str) -> str:
    """Download the basicsr sdist tarball and return its local path."""
    url = f"https://pypi.org/pypi/basicsr/{BASICSR_VERSION}/json"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    sdist_url = next(
        (u["url"] for u in data["urls"] if u["packagetype"] == "sdist"),
        None,
    )
    if sdist_url is None:
        raise RuntimeError(
            f"sdist package for basicsr {BASICSR_VERSION} could not be found"
        )

    tarball = os.path.join(dest_dir, os.path.basename(sdist_url))
    urllib.request.urlretrieve(sdist_url, tarball)
    return tarball

def _safe_extractall(tf: tarfile.TarFile, dest_dir: str) -> str:

    """Safely extract a tar archive and return the extracted source directory."""

    if sys.version_info >= (3, 12):
        tf.extractall(dest_dir, filter="data")
    else:
        dest_root = os.path.realpath(dest_dir)
        members = tf.getmembers()
        for member in members:
            name = member.name
            if os.path.isabs(name):
                raise tarfile.TarError(f"Refusing to extract absolute path: {name!r}")

            target_path = os.path.realpath(os.path.join(dest_root, name))
            try:
                common_path = os.path.commonpath([dest_root, target_path])
            except ValueError as exc:
                raise tarfile.TarError(
                    f"Refusing to extract path outside destination: {name!r}"
                ) from exc
            if common_path != dest_root:
                raise tarfile.TarError(f"Refusing to extract path outside destination: {name!r}")

            if (
                member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
            ):
                raise tarfile.TarError(f"Refusing to extract special file: {name!r}")

        tf.extractall(dest_root, members=members)

    return os.path.join(dest_dir, f"basicsr-{BASICSR_VERSION}")


def _patch_setup_py(src_dir: str) -> None:
    """Fix exec()/locals() PEP-667 issue in setup.py."""
    path = os.path.join(src_dir, "setup.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    old = (
        "def get_version():\n"
        "    with open(version_file, 'r') as f:\n"
        "        exec(compile(f.read(), version_file, 'exec'))\n"
        "    return locals()['__version__']"
    )
    new = (
        "def get_version():\n"
        "    with open(version_file, 'r') as f:\n"
        "        ns = {}\n"
        "        exec(compile(f.read(), version_file, 'exec'), ns)\n"
        "    return ns['__version__']"
    )

    if old not in text:
        print("  setup.py: patch target not found (already patched?), skipping")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))
    print("  setup.py: patched get_version()")


def _patch_degradations(src_dir: str) -> None:
    """Replace removed torchvision.transforms.functional_tensor import."""
    path = os.path.join(src_dir, "basicsr", "data", "degradations.py")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new = "from torchvision.transforms.functional import rgb_to_grayscale"

    if old not in text:
        print("  degradations.py: patch target not found (already patched?), skipping")
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))
    print("  degradations.py: patched functional_tensor → functional")


# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Reinstall even if basicsr is already present",
    )
    args = parser.parse_args()

    if not args.force and _basicsr_installed():
        print(f"basicsr {BASICSR_VERSION} is already installed, nothing to do.")
        print("Use --force to reinstall.")
        return

    tmpdir = tempfile.mkdtemp(prefix="basicsr_patch_")
    try:
        print(f"[1/4] Downloading basicsr {BASICSR_VERSION} source …")
        tarball = _download_sdist(tmpdir)

        print("[2/4] Extracting …")
        with tarfile.open(tarball, "r:gz") as tf:
            src_dir = _safe_extractall(tf, tmpdir)

        print("[3/4] Patching …")
        _patch_setup_py(src_dir)
        _patch_degradations(src_dir)

        print("[4/4] Installing …")
        # --no-build-isolation requires setuptools+wheel in the environment
        _pip("install", "setuptools", "wheel", "--quiet")
        _pip("install", src_dir, "--no-build-isolation", "--quiet")

        print(f"\nbasicsr {BASICSR_VERSION} installed successfully.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
