import shlex
from pathlib import Path as _P
from typing import List, Any


def _patch_size_for_model(model_id: str) -> int:
    """Return the patch size (32 for Qwen3, 28 for Qwen2.5)."""
    return 32 if "qwen3" in model_id.lower() else 28


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
