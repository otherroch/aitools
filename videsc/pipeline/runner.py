import sys
import shlex
import subprocess
import tempfile
import time
import logging
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path as _P
from typing import Dict, Any, List, Optional

import torch

from videsc.model.loader import (
    load_model_and_processor,
    load_omni_model_and_processor,
    load_qwen35_model_and_processor,
    load_gemma4_model_and_processor,
    _maybe_set_reader,
)
from videsc.audio.transcription import transcribe_audio_from_video
from videsc.video.info import get_video_info
from videsc.video.sampling import compute_effective_nframes, compress_audio_segments_to_nframes
from videsc.video.messages import build_messages
from videsc.utils.helpers import expand_inputs, namespace_to_cli, _patch_size_for_model, expand_video_grid_thw
from videsc.pipeline.prompts import (
    SEGMENT_PROMPT,
    WINDOW_AGGREGATION_PROMPT,
    FINAL_SUMMARY_PROMPT,
    DEFAULT_SEGMENT_MAX_TOKENS,
    DEFAULT_WINDOW_MAX_TOKENS,
    DEFAULT_FINAL_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def process_mm_info(*args, **kwargs):
    """
    Lazy import wrapper around ``qwen_omni_utils.process_mm_info`` so tests can
    monkeypatch this symbol on the module.

    This indirection is intentionally used only where we need to:
      * avoid importing heavy / optional multi‑modal dependencies at module import
        time, and
      * make it easy for tests to monkeypatch the processing function at the
        ``videsc.pipeline.runner`` level.

    In general, prefer direct imports at the top of the module for clarity.
    Only introduce wrappers like this when testability or lazy importing of
    optional dependencies is required and should be documented explicitly.
    """
    from qwen_omni_utils import process_mm_info as _process_mm_info

    return _process_mm_info(*args, **kwargs)


def process_vision_info(*args, **kwargs):
    """
    Lazy import wrapper around ``qwen_vl_utils.process_vision_info`` so tests
    can monkeypatch this symbol on the module.

    As with ``process_mm_info``, this pattern is reserved for cases where we
    need lazy imports and convenient test monkeypatching of external
    dependencies.

    For typical usage, prefer direct imports at the top of the module rather
    than adding additional wrapper functions.
    """
    from qwen_vl_utils import process_vision_info as _process_vision_info

    return _process_vision_info(*args, **kwargs)


def run_single_video(args, model, processor) -> int:
    """
    Core pipeline for a single video.
    Receives a preloaded model/processor so we can share them across threads.
    """
    torch.manual_seed(args.seed)
    trace = " "

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    trace += "start time: " + str(current_time)
    logger.debug("run_single_video: start time %s  video=%s", current_time, args.video)
    print(f"✅ start time: {current_time}")

    # Basic video info
    video_info = get_video_info(args.video)
    logger.debug("run_single_video: video_info=%s", video_info)

    transcript = None
    segments: List[Dict[str, Any]] = []
    if args.audio:
        logger.debug("run_single_video: transcribing audio from %s", args.video)
        transcript, segments = transcribe_audio_from_video(args.video, args, model.device)
        if transcript:
            trace += "\n\n audio_transcript_head: " + transcript[:500]
            logger.debug("run_single_video: transcript head: %s", transcript[:200])
        if segments:
            trace += f"\n\n audio_segments_count_raw: {len(segments)}"
            logger.debug("run_single_video: raw audio segments: %d", len(segments))

    # Compute the effective nframes the model will actually see
    effective_nframes = compute_effective_nframes(
        video_info,
        args.num_frames,
        args.spf,
    )
    logger.debug(
        "run_single_video: effective_nframes=%d  requested=%d  spf=%.2f",
        effective_nframes, args.num_frames, args.spf,
    )

    # Compress audio segments so we have exactly one coarse segment per frame bucket
    if segments:
        segments = compress_audio_segments_to_nframes(
            segments,
            effective_nframes,
            video_info["tot_time"],
        )
        trace += f"\n\n audio_segments_count_coarse: {len(segments)}"
        logger.debug("run_single_video: coarse audio segments: %d", len(segments))

    patch = _patch_size_for_model(args.model if getattr(args, "model", None) else "")
    logger.debug("run_single_video: patch_size=%d", patch)
    print("patch:", patch)

    # Convert total_pixels from edge-multiplier units to raw pixels,
    # matching how min_pixels/max_pixels are converted in the processor.
    tot_pixels_raw = args.total_pixels * patch * patch
    logger.debug(
        "run_single_video: total_pixels edge=%d  raw=%d  (patch=%d)",
        args.total_pixels, tot_pixels_raw, patch,
    )

    messages = build_messages(
        video_path=args.video,
        vinfo=video_info,
        tot_pixels=tot_pixels_raw,
        spf=args.spf,
        nframes=effective_nframes,
        prompt=args.prompt,
        system=args.system,
        continue_prompt=args.cont_prompt,
        audio_transcript=transcript,
        audio_segments=segments,
        is_omni=args.omni,
    )

    logger.debug("run_single_video: messages built (%d message(s))", len(messages))
    print("after build messages:", messages)
    trace += "\n\n messages: " + str(messages)

    if args.omni:
        modeltext = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print("after apply chat template, modeltext: ", modeltext)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"✅ Before process mm info: {current_time}")

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        print("before processor")
        inputs = processor(
            text=modeltext,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )

        text = "dry run"
        if args.dry:
            print("dry run, exit early")
        else:
            inputs = inputs.to(model.device).to(model.dtype)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"✅ Before generate: {current_time}")
            trace += "\n\n Before generate " + str(current_time)
            text_ids, audio_output = model.generate(
                **inputs,
                return_audio=False,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=32768,
                thinker_do_sample=True,
                thinker_temperature=0.9,
                thinker_top_p=1.0,
                thinker_top_k=50,
                thinker_repetition_penalty=1.05,
                use_audio_in_video=True,
            )

            print("after generate, before batch decode")
            text = processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

    else:
        modeltext = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("after apply chat template, modeltext: ", modeltext)
        trace += "\n\n messages: " + str(messages)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"✅ Before video decode: {current_time}")

        _maybe_set_reader(args.reader)

        use_metadata = not args.no_meta
        if not use_metadata:
            print("do not use metadata from vision info")

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch // 2,
            return_video_kwargs=True,
            return_video_metadata=use_metadata,
        )

        print("after process vision info")
        print("video_kwargs:", video_kwargs)
        trace += "\n\n video_kwargs: " + str(video_kwargs)

        video_metadatas = None
        if use_metadata:
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

        print("video_metadatas:", video_metadatas)
        trace += "\n\n video_metadatas: " + str(video_metadatas)

        print("before processor")

        inputs = processor(
            text=modeltext,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        )

        # Workaround for transformers ≥ 5.3.0 StopIteration bug
        # (huggingface/transformers#44560): expand video_grid_thw from
        # per-video [[T, H, W]] to per-frame [[1, H, W]] * T so that
        # get_rope_index can match one entry per frame token-group.
        expand_video_grid_thw(inputs)

        inputs = inputs.to("cuda")

        print("before generate")

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"✅ Before generate: {current_time}")
        trace += "\n\n Before generate: " + str(current_time)

        text = "dry run"
        if args.dry:
            print("dry run, exit early")
        else:
            gen_ids = []
            if args.optimize:
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.rep_pen,
                    )
            else:
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.rep_pen,
                )
            print("after generate, before trim")
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
            ]
            text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            if not args.no_think_trim and "</think>" in text:
                text = text.split("</think>", 1)[-1].lstrip()

    print("after generate")
    print(text)
    logger.debug("run_single_video: generation complete  output_length=%d chars", len(text))

    nowGen = datetime.now()
    current_time = nowGen.strftime("%H:%M:%S")
    trace += "\n\n after generate: " + str(current_time)
    diffGen = nowGen - now
    genTime = "generate time: " + str(diffGen.total_seconds()) + " seconds"
    print(f"✅ At end: {current_time}")
    vidTime = video_info["tot_time"]
    genvidRatio = float(diffGen.total_seconds()) / float(vidTime)
    genRatio = genTime + " video time: " + str(vidTime) + " gen ratio: " + str(genvidRatio)
    print(genRatio)
    logger.debug("run_single_video: %s", genRatio)

    result = genRatio + "\n\n" + " input args: " + str(args) + "\n\n" + " result prompt: " + str(text) + "\n\n" + trace

    # Output paths
    _vid = _P(args.video)
    desc_dir = "desc-" + args.model
    if args.audio:
        desc_dir = "desc-with-audio-" + args.model
    _default_dir = _vid.parent / desc_dir
    _outdir = _P(getattr(args, "outdir", None)) if getattr(args, "outdir", None) else _default_dir
    _outdir.mkdir(parents=True, exist_ok=True)
    out_path = str(_outdir / f"{_vid.stem}.txt")
    logger.debug("run_single_video: writing result to %s", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)

    # Save raw transcript separately (unless disabled)
    if transcript and not getattr(args, "no_save_transcript", False):
        transcript_path = _outdir / f"{_vid.stem}.transcript.txt"
        try:
            with open(transcript_path, "w", encoding="utf-8") as tf:
                tf.write(transcript)
            print(f"[audio] transcript saved to: {transcript_path}")
            logger.debug("run_single_video: transcript saved to %s", transcript_path)
        except Exception as e:
            print(f"[audio] failed to save transcript to {transcript_path}: {e}")

    return 0


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

    # Clamp to at least 1 frame so that when the requested fps exceeds the
    # source video_fps the interval never drops below 1, avoiding duplicate
    # frame indices and unnecessary memory / CPU work.
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


def _seconds_to_hhmmss(seconds: float) -> str:
    """Convert a time in seconds to HH:MM:SS format."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _generate_text(
    prompt_body: str,
    model,
    processor,
    args,
    max_new_tokens: int,
) -> str:
    """Send a text-only prompt to the model and return the decoded response.

    This is the shared generation helper used by all consolidation stages
    (window aggregation, final summary).
    """
    messages: List[Dict[str, Any]] = []
    if args.system:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": args.system}],
        })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt_body}],
    })

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    text_output = processor.parse_response(response)
    text = str(text_output["content"])

    if not args.no_think_trim and "</think>" in text:
        text = text.split("</think>", 1)[-1].lstrip()

    return text


def aggregate_windows(
    segment_texts: List[str],
    window_size: int,
    model,
    processor,
    args,
) -> List[str]:
    """Group segment descriptions into windows and aggregate each window.

    When the number of segments exceeds *window_size*, consecutive segments are
    batched into groups of *window_size* and each group is sent to the model
    with the window-aggregation prompt.  If the number of segments is at most
    *window_size*, no aggregation is performed and the original list is returned.

    Args:
        segment_texts: Per-segment descriptions (one string per chunk).
        window_size:   Maximum segments per window.
        model:         The loaded Gemma 4 model.
        processor:     The loaded Gemma 4 processor.
        args:          CLI namespace.

    Returns:
        A list of window-level summaries (or the original segments if
        aggregation was not needed).
    """
    if len(segment_texts) <= window_size:
        return segment_texts

    logger.info("[gemma4] aggregating %d segments into windows of %d", len(segment_texts), window_size) 
    
    windows: List[str] = []
    for win_start in range(0, len(segment_texts), window_size):
        win_segments = segment_texts[win_start : win_start + window_size]
        win_idx = win_start // window_size + 1
        logger.info("[gemma4] aggregating window %d (%d segments)", win_idx, len(win_segments))

        numbered = []
        for idx, text in enumerate(win_segments):
            seg_num = win_start + idx + 1  # 1-based segment number
            numbered.append(f"--- Segment {seg_num} ---\n{text}")
        body = WINDOW_AGGREGATION_PROMPT + "\n\n" + "\n\n".join(numbered)

        logger.debug("[gemma4] window %d prompt:\n%s", win_idx, body)
        logger.info("[gemma4] generating summary for window %d", win_idx)
        result = _generate_text(body, model, processor, args, DEFAULT_WINDOW_MAX_TOKENS)
        windows.append(result)

    logger.info("[gemma4] window aggregation produced %d windows", len(windows))
    return windows


def consolidate_segments(
    segment_texts: List[str],
    model,
    processor,
    args,
) -> str:
    """Multi-stage consolidation of per-segment descriptions.

    Implements the three-stage pipeline from the design document:

    1. **Window aggregation** – If more segments than ``--window-size``,
       consecutive segments are grouped and each group is summarised.
    2. **Final summary** – All window (or segment) texts are fed to the
       final-summary prompt to produce a structured result with OVERVIEW,
       TIMELINE, ENTITIES, ACTIONS and THEMES.

    Callers may override the final-summary prompt via ``--consolidate-prompt``.

    Args:
        segment_texts: The per-segment descriptions (one string per chunk).
        model:         The loaded Gemma 4 model.
        processor:     The loaded Gemma 4 processor.
        args:          The CLI namespace.

    Returns:
        The consolidated summary as a single string.
    """
    window_size = getattr(args, "window_size", 10)

    # Stage 2: window aggregation (only when many segments)
    summaries = aggregate_windows(segment_texts, window_size, model, processor, args)

    logger.info("[gemma4] %d summaries after window aggregation:\n%s", len(summaries), "\n\n".join(summaries))
    
    # Stage 3: final summary
    consolidate_prompt = getattr(args, "consolidate_prompt", None) or FINAL_SUMMARY_PROMPT

    numbered = []
    label = "Window" if len(summaries) < len(segment_texts) else "Segment"
    for idx, text in enumerate(summaries, 1):
        numbered.append(f"--- {label} {idx} ---\n{text}")
    body = consolidate_prompt + "\n\n" + "\n\n".join(numbered)
    
    logger.info("[gemma4] generating final summary from %d %s(s)", len(summaries), label.lower())
    logger.info("[gemma4] final summary prompt:\n%s", body)
    
   
    result = _generate_text(body, model, processor, args, DEFAULT_FINAL_MAX_TOKENS)

    logger.info("[gemma4] consolidation done (%d chars)", len(result))
    return result


def run_single_video_gemma4(args, model, processor) -> int:
    """Run Gemma 4 pipeline for a single video.

    Gemma 4 can only process up to ``--gemma4-chunk-duration`` seconds of video at
    a time (default 60 s).  When the source video is longer the function splits it
    into consecutive chunks, generates a description for each, then concatenates
    the results into a single output file.

    When ``--consolidate`` is set and the video has more than one chunk, the
    pipeline uses a structured segment prompt (requesting JSON output with events,
    objects, actions, scene, summary), optionally groups segments into windows,
    and produces a final structured summary.  The output file contains the
    consolidated summary followed by the raw per-segment descriptions.

    Each chunk is trimmed to a temp file with ffmpeg and passed as a video path
    directly to the processor, matching the gemma4_4b.py approach.  Output is
    decoded with ``processor.parse_response``.
    """
    chunk_duration = args.gemma4_chunk_duration
    gemma4_fps = args.gemma4_fps
    use_consolidation = (
        getattr(args, "consolidate", False)
        and not getattr(args, "dry", False)
    )

    torch.manual_seed(args.seed)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    logger.info("run_single_video_gemma4: start time %s  video=%s", current_time, args.video)

    video_info = get_video_info(args.video)
    duration = video_info["tot_time"]
    logger.debug("run_single_video_gemma4: duration=%.2fs  chunk=%.1fs  fps=%.2f",
                 duration, chunk_duration, gemma4_fps)

    # Build list of (start, end) chunks
    if chunk_duration <= 0:
        raise ValueError(
            f"gemma4_chunk_duration must be > 0, got {chunk_duration}"
        )
    chunks = []
    t = 0.0
    while t < duration:
        end = min(t + chunk_duration, duration)
        chunks.append((t, end))
        t = end

    logger.debug("run_single_video_gemma4: %d chunk(s) for %.1fs video", len(chunks), duration)

    all_descriptions = []

    for chunk_idx, (start, end) in enumerate(chunks):
        logger.info(
            "run_single_video_gemma4: chunk %d/%d  %.1fs–%.1fs",
            chunk_idx + 1, len(chunks), start, end,
        )

        # For a single chunk pass the original path; otherwise trim with ffmpeg.
        tmp_path = None
        if len(chunks) == 1:
            clip_path = args.video
        else:
            suffix = _P(args.video).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.close()
            tmp_path = tmp.name
            seg_duration = end - start
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", args.video,
                "-t", str(seg_duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                tmp_path,
            ]
            logger.debug("run_single_video_gemma4: trimming chunk %d: %s", chunk_idx + 1, " ".join(cmd))
            subprocess.run(cmd, check=True, capture_output=True)
            clip_path = tmp_path

        try:
            ts_start = _seconds_to_hhmmss(start)
            ts_end = _seconds_to_hhmmss(end)

            # When consolidation is enabled and video has multiple chunks,
            # use the structured segment prompt (JSON output) so that
            # downstream aggregation/final-summary stages work reliably.
            # The SEGMENT_PROMPT already embeds timestamps and context, so
            # no additional chunk_note prefix is needed.
            if use_consolidation and len(chunks) > 1:
                custom_seg = getattr(args, "segment_prompt", None)
                if custom_seg:
                    segment_prompt_text = custom_seg
                else:
                    segment_prompt_text = SEGMENT_PROMPT.format(
                        chunk_duration=int(end - start),
                        timestamp_start=ts_start,
                        timestamp_end=ts_end,
                    )
                seg_max_tokens = DEFAULT_SEGMENT_MAX_TOKENS
            else:
                chunk_note = ""
                if len(chunks) > 1:
                    chunk_note = (
                        f"[Video segment {chunk_idx + 1}/{len(chunks)}: "
                        f"{ts_start}\u2013{ts_end}]\n"
                    )
                segment_prompt_text = chunk_note + args.prompt
                seg_max_tokens = args.max_new_tokens

            content: List[Dict[str, Any]] = [
                {"type": "video", "video": clip_path},
                {"type": "text", "text": segment_prompt_text},
            ]

            messages: List[Dict[str, Any]] = []
            if args.system:
                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": args.system}],
                })
            messages.append({"role": "user", "content": content})

            if args.dry:
                logger.info("[gemma4] dry run — skipping generation for this chunk")
                all_descriptions.append(f"[chunk {chunk_idx + 1}: dry run]")
                continue
            
            if chunk_idx == 0:
                logger.info(f"before processor for chunk {chunk_idx + 1} promt: {segment_prompt_text}")
            else:
                logger.debug(f"before processor for chunk {chunk_idx + 1} promt: {segment_prompt_text}")
            chunk_num_frames = max(1, int(round((end - start) * gemma4_fps)))
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                processor_kwargs={
                    "videos_kwargs": {"num_frames": chunk_num_frames},
                },
            ).to(model.device)
        finally:
            if tmp_path is not None:
                _P(tmp_path).unlink(missing_ok=True)

        logger.debug(f"after processor for chunk {chunk_idx + 1} inputs: {inputs} with video num_frames={chunk_num_frames}")
        now_gen = datetime.now()
        current_time = now_gen.strftime("%H:%M:%S")
        logger.info("✅ Before generate (chunk %d): %s", chunk_idx + 1, current_time)

        output_ids = model.generate(**inputs, max_new_tokens=seg_max_tokens)

        input_len = inputs["input_ids"].shape[-1]
        response = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
        text_output = processor.parse_response(response)
        text = str(text_output["content"])

        if not args.no_think_trim and "</think>" in text:
            text = text.split("</think>", 1)[-1].lstrip()

        all_descriptions.append(text)
        now_gen = datetime.now()
        gen_time = now_gen.strftime("%H:%M:%S")
        logger.info("[gemma4] chunk %d done: %s", chunk_idx + 1, gen_time)
        logger.debug("[gemma4] chunk %d description:\n%s", chunk_idx + 1, text)
    # ── Multi-stage consolidation ────────────────────────────────────────────
    consolidated = None
    if use_consolidation and len(all_descriptions) > 1:
        consolidated = consolidate_segments(all_descriptions, model, processor, args)

    if consolidated is not None:
        result = (
            "=== Consolidated Summary ===\n\n"
            + consolidated
            + "\n\n"
            + "=== Per-Segment Descriptions ===\n\n"
            + "\n\n".join(all_descriptions)
        )
    else:
        result = "\n\n".join(all_descriptions)
    logger.debug(result)

    nowEnd = datetime.now()
    current_time = nowEnd.strftime("%H:%M:%S")
    logger.info("✅ At end: %s", current_time)

    # Write result to file
    _vid = _P(args.video)
    desc_dir = "desc-" + args.model
    _default_dir = _vid.parent / desc_dir
    outdir_val = getattr(args, "outdir", None)
    _outdir = _P(outdir_val) if outdir_val else _default_dir
    _outdir.mkdir(parents=True, exist_ok=True)
    out_path = str(_outdir / f"{_vid.stem}.txt")
    logger.info("run_single_video_gemma4: writing result to %s", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)

    return 0


def run_batch_subprocess(args) -> int:
    inputs = expand_inputs(args.videos, args.indir, args.ext, args.filelist)
    if not inputs:
        print("[fatal] No input videos matched your criteria.", file=sys.stderr)
        return 3

    logger.debug("run_batch_subprocess: %d video(s) queued  workers=%d", len(inputs), args.workers)

    exclude = {"videos", "indir", "filelist", "ext", "workers", "sleep", "dry_run", "video", "batch_mode"}
    base_cli = namespace_to_cli(args, exclude)

    if "--prompt" not in base_cli and getattr(args, "prompt", None):
        base_cli += ["--prompt", args.prompt]

    script_path = _P(__file__).resolve().parent.parent / "main.py"
    cmds = [
        (vid.name, [sys.executable, str(script_path), "--video", str(vid)] + base_cli)
        for vid in inputs
    ]

    if args.dry_run:
        for name, cmd in cmds:
            print(f"[dry-run] {name}: {' '.join(shlex.quote(c) for c in cmd)}")
        return 0

    print(f"[supervisor] launching {len(cmds)} jobs with up to {args.workers} workers…")
    running: Dict[int, Any] = {}
    pending = list(cmds)
    completed = 0
    total = len(cmds)

    while pending or running:
        while pending and len(running) < args.workers:
            name, cmd = pending.pop(0)
            print(f"[spawn] {name}: {' '.join(shlex.quote(c) for c in cmd)}")
            proc = subprocess.Popen(cmd)
            running[proc.pid] = (name, proc)

        for pid, (name, proc) in list(running.items()):
            ret = proc.poll()
            if ret is not None:
                print(f"[done] {name} exited with code {ret}")
                running.pop(pid, None)
                completed += 1

        if completed < total:
            time.sleep(args.sleep)

    print(f"All jobs complete: {completed}/{total}")
    return 0


def run_batch_threads(args) -> int:
    """Batch mode using threads that share a single model/processor."""
    inputs = expand_inputs(args.videos, args.indir, args.ext, args.filelist)
    if not inputs:
        print("[fatal] No input videos matched your criteria.", file=sys.stderr)
        return 3

    logger.debug("run_batch_threads: %d video(s) queued  workers=%d", len(inputs), args.workers)

    if args.dry_run:
        for vid in inputs:
            print(f"[dry-run] would process: {vid}")
        return 0

    if args.omni:
        model, processor = load_omni_model_and_processor(args)
    elif args.qwen35:
        model, processor = load_qwen35_model_and_processor(args)
    elif getattr(args, "gemma4", False):
        model, processor = load_gemma4_model_and_processor(args)
    else:
        model, processor = load_model_and_processor(args)

    print(f"[batch-threads] total jobs: {len(inputs)}, workers: {args.workers}")

    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for vid in inputs:
            local_args = deepcopy(args)
            local_args.video = str(vid)
            local_args.videos = None
            local_args.indir = None
            local_args.filelist = None
            if getattr(args, "gemma4", False):
                fut = executor.submit(run_single_video_gemma4, local_args, model, processor)
            else:
                fut = executor.submit(run_single_video, local_args, model, processor)
            futures[fut] = vid

        for fut in as_completed(futures):
            vid = futures[fut]
            try:
                rc = fut.result()
                status = "OK" if rc == 0 else f"EXIT {rc}"
            except Exception as e:
                status = f"ERROR: {e}"
            print(f"[{vid.name}] — done [{status}]")
            results[str(vid)] = status

    print("All jobs complete.")
    return 0


def run_batch(args) -> int:
    """Dispatch to subprocess or threaded batch implementation depending on --batch-mode."""
    mode = getattr(args, "batch_mode", "threads")
    logger.debug("run_batch: mode=%s", mode)
    if mode == "subprocess":
        print("[batch] using subprocess mode (one process per video).")
        return run_batch_subprocess(args)
    else:
        print("[batch] using threaded mode (shared model across videos).")
        return run_batch_threads(args)
