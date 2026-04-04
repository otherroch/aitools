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


def expand_video_grid_thw(inputs: dict) -> dict:
    """Expand ``video_grid_thw`` from per-video ``[[T, H, W]]`` to per-frame
    ``[[1, H, W]] * T`` entries so that ``get_rope_index`` can iterate one
    entry per frame token-group produced by the processor.

    This works around a known bug in ``transformers ≥ 5.3.0``
    (huggingface/transformers#44560) where the Qwen3-VL processor creates
    per-frame video-token groups separated by timestamp tokens, but
    ``video_grid_thw`` still has one row per video.  ``get_rope_index``
    calls ``next()`` on the grid iterator for *every* token-group and raises
    ``StopIteration`` after the first frame.

    The fix is safe for older transformers versions that don't add timestamps
    (they produce a single contiguous token group, ``T == 1`` after
    temporal-patch merging), so the expansion is a no-op.
    """
    import torch

    vg = inputs.get("video_grid_thw")
    mm = inputs.get("mm_token_type_ids")
    if vg is None or mm is None:
        return inputs

    expanded = []
    for row in vg:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        for _ in range(t):
            expanded.append([1, h, w])

    inputs["video_grid_thw"] = torch.tensor(
        expanded, dtype=vg.dtype, device=vg.device
    )
    return inputs


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
