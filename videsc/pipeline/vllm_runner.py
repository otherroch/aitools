"""
videsc.pipeline.vllm_runner – Run video description via a remote vLLM server.

Supports single-video and batch modes.  For each video the pipeline:
1. Extracts frames using the existing video extraction utilities.
2. Sends frames as base64-encoded images to the vLLM OpenAI-compatible API.
3. Writes the resulting description to a text file.

For longer videos the pipeline splits into chunks (similar to Gemma 4 mode)
and optionally consolidates results via the same vLLM server.
"""

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from videsc.video.info import get_video_info

logger = logging.getLogger(__name__)


def _seconds_to_hhmmss(seconds: float) -> str:
    """Convert a time in seconds to HH:MM:SS format."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_frames_as_pil(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 1.0,
) -> list:
    """Extract frames as PIL images from a video segment.

    Args:
        video_path:  Path to the video file.
        start_sec:   Start time in seconds.
        end_sec:     End time in seconds.
        fps:         Frames to sample per second of video (default 1.0).

    Returns:
        List of PIL Images in RGB mode.
    """
    import cv2
    from PIL import Image

    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("extract_frames_as_pil: cannot open %s", video_path)
        return frames

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps <= 0:
        video_fps = 25.0

    start_frame = int(start_sec * video_fps)
    end_frame = min(int(end_sec * video_fps), total_video_frames) if end_sec > start_sec else total_video_frames

    frame_interval = max(1.0, video_fps / fps) if fps > 0 else video_fps

    sample_indices = []
    t = float(start_frame)
    while t < end_frame:
        sample_indices.append(int(t))
        t += frame_interval

    logger.debug(
        "extract_frames_as_pil: %s  [%.1fs–%.1fs]  fps=%.2f  frames=%d",
        video_path, start_sec, end_sec, fps, len(sample_indices),
    )

    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


def _extract_frames(video_path: str, args) -> list:
    """Extract frames from a video for sending to vLLM.

    Uses the same frame extraction approach as the Gemma 4 pipeline:
    sample at a given FPS within the clip bounds.
    """
    clip_start = getattr(args, "clip_start", 0.0)
    clip_end = getattr(args, "clip_end", -1.0)
    fps = getattr(args, "vllm_fps", 1.0)

    vinfo = get_video_info(video_path)
    duration = vinfo["tot_time"]

    start = clip_start
    end = clip_end if clip_end > 0 else duration

    frames = extract_frames_as_pil(video_path, start, end, fps=fps)
    logger.info("_extract_frames: extracted %d frames from %s (%.1fs–%.1fs @ %.2f fps)",
                len(frames), video_path, start, end, fps)
    return frames


def _create_vllm_client(args):
    """Instantiate a VLLMClient from CLI args."""
    from videsc.model.vllm_client import VLLMClient

    return VLLMClient(
        host=args.vllm_host,
        port=args.vllm_port,
        model=args.vllm_model,
        api_key=getattr(args, "vllm_api_key", "EMPTY"),
        max_tokens=args.max_new_tokens,
        temperature=getattr(args, "vllm_temperature", 0.7),
        top_p=getattr(args, "vllm_top_p", 0.95),
        base_url=getattr(args, "vllm_base_url", None),
    )


def run_single_video_vllm(args) -> int:
    """Process a single video through a vLLM server.

    Extracts frames, sends them with the prompt, and writes the output.
    For long videos, splits into chunks based on --vllm-chunk-duration.
    """
    now = datetime.now()
    logger.info("run_single_video_vllm: start %s  video=%s", now.strftime("%H:%M:%S"), args.video)

    client = _create_vllm_client(args)
    video_info = get_video_info(args.video)
    duration = video_info["tot_time"]

    chunk_duration = getattr(args, "vllm_chunk_duration", 0.0)
    fps = getattr(args, "vllm_fps", 1.0)
    max_size = getattr(args, "vllm_max_image_size", 1280)

    # Determine chunks
    if chunk_duration > 0 and duration > chunk_duration:
        chunks = []
        t = 0.0
        while t < duration:
            end = min(t + chunk_duration, duration)
            chunks.append((t, end))
            t = end
    else:
        clip_start = getattr(args, "clip_start", 0.0)
        clip_end = getattr(args, "clip_end", -1.0)
        start = clip_start
        end = clip_end if clip_end > 0 else duration
        chunks = [(start, end)]

    logger.info("run_single_video_vllm: %d chunk(s) for %.1fs video", len(chunks), duration)

    all_descriptions = []

    for chunk_idx, (start, end) in enumerate(chunks):
        logger.info("run_single_video_vllm: chunk %d/%d  %.1fs–%.1fs",
                    chunk_idx + 1, len(chunks), start, end)

        frames = extract_frames_as_pil(args.video, start, end, fps=fps)

        if not frames:
            logger.warning("run_single_video_vllm: no frames extracted for chunk %d", chunk_idx + 1)
            all_descriptions.append(f"[chunk {chunk_idx + 1}: no frames]")
            continue

        # Build prompt with chunk context
        if len(chunks) > 1:
            ts_start = _seconds_to_hhmmss(start)
            ts_end = _seconds_to_hhmmss(end)
            chunk_note = (
                f"[Video segment {chunk_idx + 1}/{len(chunks)}: "
                f"{ts_start}\u2013{ts_end}]\n"
            )
            prompt = chunk_note + args.prompt
        else:
            prompt = args.prompt

        if getattr(args, "dry", False):
            logger.info("[vllm] dry run — skipping generation for chunk %d", chunk_idx + 1)
            all_descriptions.append(f"[chunk {chunk_idx + 1}: dry run]")
            continue

        text = client.describe_frames(
            frames=frames,
            prompt=prompt,
            system=getattr(args, "system", None),
            max_tokens=args.max_new_tokens,
            max_size=max_size,
        )

        all_descriptions.append(text)
        logger.info("[vllm] chunk %d done: %d chars", chunk_idx + 1, len(text))

    # Combine results
    result = "\n\n".join(all_descriptions)

    # Consolidate if requested and multiple chunks
    if getattr(args, "consolidate", False) and len(all_descriptions) > 1:
        result = _consolidate_vllm(all_descriptions, client, args)

    # Write output
    _vid = Path(args.video)
    model_name = args.vllm_model.replace("/", "_")
    desc_dir = f"desc-vllm-{model_name}"
    _default_dir = _vid.parent / desc_dir
    outdir_val = getattr(args, "outdir", None)
    _outdir = Path(outdir_val) if outdir_val else _default_dir
    _outdir.mkdir(parents=True, exist_ok=True)
    out_path = _outdir / f"{_vid.stem}.txt"
    logger.info("run_single_video_vllm: writing result to %s", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)

    now_end = datetime.now()
    logger.info("run_single_video_vllm: done at %s", now_end.strftime("%H:%M:%S"))
    return 0


def _consolidate_vllm(
    segment_texts: List[str],
    client,
    args,
) -> str:
    """Consolidate multiple segment descriptions using the vLLM server.

    Uses a text-only prompt to produce a final structured summary,
    similar to the Gemma 4 consolidation pipeline.
    """
    from videsc.pipeline.prompts import FINAL_SUMMARY_PROMPT

    consolidate_prompt = getattr(args, "consolidate_prompt", None) or FINAL_SUMMARY_PROMPT

    numbered = []
    for idx, text in enumerate(segment_texts, 1):
        numbered.append(f"--- Segment {idx} ---\n{text}")
    body = consolidate_prompt + "\n\n" + "\n\n".join(numbered)

    logger.info("[vllm] consolidating %d segments", len(segment_texts))

    messages: List[Dict[str, Any]] = []
    system = getattr(args, "system", None)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": body})

    consolidated = client.generate(messages, max_tokens=args.max_new_tokens)

    result = (
        "=== Consolidated Summary ===\n\n"
        + consolidated
        + "\n\n"
        + "=== Per-Segment Descriptions ===\n\n"
        + "\n\n".join(segment_texts)
    )
    return result


def run_batch_vllm(args) -> int:
    """Batch mode for vLLM: process multiple videos via the remote server."""
    from videsc.utils.helpers import expand_inputs

    inputs = expand_inputs(
        getattr(args, "videos", None),
        getattr(args, "indir", None),
        getattr(args, "ext", []),
        getattr(args, "filelist", None),
    )
    if not inputs:
        logger.error("[vllm] No input videos matched your criteria.")
        return 3

    logger.info("[vllm-batch] %d video(s) queued  workers=%d", len(inputs), args.workers)

    if getattr(args, "dry_run", False):
        for vid in inputs:
            print(f"[dry-run] would process: {vid}")
        return 0

    print(f"[vllm-batch] total jobs: {len(inputs)}, workers: {args.workers}")

    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for vid in inputs:
            local_args = deepcopy(args)
            local_args.video = str(vid)
            local_args.videos = None
            local_args.indir = None
            local_args.filelist = None
            fut = executor.submit(run_single_video_vllm, local_args)
            futures[fut] = vid

        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                rc = fut.result()
                status = "OK" if rc == 0 else f"EXIT {rc}"
            except Exception as e:
                status = f"ERROR: {e}"
                logger.error("[vllm-batch] %s failed: %s", vid.name, e)
            print(f"[{vid.name}] — done [{status}]")
            results[str(vid)] = status

    print("All jobs complete.")
    return 0
