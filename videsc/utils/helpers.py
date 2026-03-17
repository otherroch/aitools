import re
import shlex
from pathlib import Path as _P
from typing import List, Any


def _patch_size_for_model(model_id: str) -> int:
    """Return the vision patch size for the given model identifier.

    - 16 for Qwen3.5 (early-fusion architecture, vision patch_size=16)
    - 32 for Qwen3-VL
    - 28 for Qwen2.5-VL and other models
    """
    lower = model_id.lower()
    if "qwen3.5" in lower:
        return 16
    return 32 if "qwen3" in lower else 28


def _is_qwen35_model(model_path: str) -> bool:
    """Return True if *model_path* refers to a Qwen3.5 model.

    Qwen3.5 uses ``Qwen3_5ForConditionalGeneration`` (an early-fusion
    vision-language model), which requires different loading logic compared
    to ``Qwen3VLForConditionalGeneration``.

    Uses a regex with a negative lookahead to avoid matching hypothetical
    future version names like ``qwen3.50`` or ``qwen3.500`` (extra digits or
    decimal points directly after the version number).
    """
    return bool(re.search(r"qwen3\.5(?![.\d])", model_path.lower()))


def _edge_to_pixels(edge: int, patch: int) -> int:
    return int(edge) * (patch * patch)


def _format_time_hhmmss(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm (approx)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"


def expand_inputs(videos, indir, exts, filelist) -> List[_P]:
    paths = []
    if videos:
        for pat in videos:
            paths += sorted(_P().glob(pat))
    if indir:
        base = _P(indir)
        if exts:
            for ext in exts:
                paths += sorted(base.rglob(f"*{ext}"))
        else:
            for ext in [".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"]:
                paths += sorted(base.rglob(f"*{ext}"))
        paths = [p.resolve() for p in paths if p.is_file()]
    if filelist:
        for line in _P(filelist).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                q = _P(line).expanduser().resolve()
                if q.is_file():
                    paths.append(q)
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def namespace_to_cli(args, exclude_keys) -> List[str]:
    argv = []
    for k, v in vars(args).items():
        if k in exclude_keys or v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
            continue
        if isinstance(v, (list, tuple)):
            if not v:
                continue
            argv.append(flag)
            argv += [str(x) for x in v]
            continue
        argv.append(flag)
        argv.append(str(v))
    return argv
