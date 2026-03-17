import sys
import shlex
import subprocess
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path as _P
from typing import Dict, Any, List, Optional

import torch
from qwen_omni_utils import process_mm_info
from qwen_vl_utils import process_vision_info

from videsc.model.loader import (
    load_model_and_processor,
    load_omni_model_and_processor,
    _maybe_set_reader,
)
from videsc.audio.transcription import transcribe_audio_from_video
from videsc.video.info import get_video_info
from videsc.video.sampling import compute_effective_nframes, compress_audio_segments_to_nframes
from videsc.video.messages import build_messages
from videsc.utils.helpers import expand_inputs, namespace_to_cli, _patch_size_for_model, _is_qwen35_model


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
    print(f"✅ start time: {current_time}")

    # Basic video info
    video_info = get_video_info(args.video)

    transcript = None
    segments: List[Dict[str, Any]] = []
    if args.audio:
        transcript, segments = transcribe_audio_from_video(args.video, args, model.device)
        if transcript:
            trace += "\n\n audio_transcript_head: " + transcript[:500]
        if segments:
            trace += f"\n\n audio_segments_count_raw: {len(segments)}"

    # Compute the effective nframes the model will actually see
    effective_nframes = compute_effective_nframes(
        video_info,
        args.num_frames,
        args.spf,
    )

    # Compress audio segments so we have exactly one coarse segment per frame bucket
    if segments:
        segments = compress_audio_segments_to_nframes(
            segments,
            effective_nframes,
            video_info["tot_time"],
        )
        trace += f"\n\n audio_segments_count_coarse: {len(segments)}"

    patch = _patch_size_for_model(args.model if getattr(args, "model", None) else "")
    print("patch:", patch)

    messages = build_messages(
        video_path=args.video,
        vinfo=video_info,
        tot_pixels=args.total_pixels,
        spf=args.spf,
        nframes=effective_nframes,
        prompt=args.prompt,
        system=args.system,
        continue_prompt=args.cont_prompt,
        audio_transcript=transcript,
        audio_segments=segments,
        is_omni=args.omni,
    )

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

        # Qwen3.5 overrides _prepare_position_ids_for_generation which calls
        # get_rope_index and iterates over video_grid_thw.  When the two-step
        # path is used (tokenize=False → process_vision_info → processor())
        # video_grid_thw is not populated correctly and model.generate raises
        # StopIteration.  The single-step apply_chat_template(tokenize=True)
        # path correctly sets all vision tensors in one call.
        is_qwen35 = _is_qwen35_model(getattr(args, "model", ""))

        if is_qwen35:
            print("Qwen3.5 detected: using single-step apply_chat_template(tokenize=True)")
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            trace += "\n\n Qwen3.5 single-step inputs keys: " + str(list(inputs.keys()))
        else:
            use_metadata = not args.no_meta
            if not use_metadata:
                print("do not use metadata from vision info")

            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=patch,
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
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)
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
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)

    # Save raw transcript separately (unless disabled)
    if transcript and not getattr(args, "no_save_transcript", False):
        transcript_path = _outdir / f"{_vid.stem}.transcript.txt"
        try:
            with open(transcript_path, "w", encoding="utf-8") as tf:
                tf.write(transcript)
            print(f"[audio] transcript saved to: {transcript_path}")
        except Exception as e:
            print(f"[audio] failed to save transcript to {transcript_path}: {e}")

    return 0


def run_batch_subprocess(args) -> int:
    inputs = expand_inputs(args.videos, args.indir, args.ext, args.filelist)
    if not inputs:
        print("[fatal] No input videos matched your criteria.", file=sys.stderr)
        return 3

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

    if args.dry_run:
        for vid in inputs:
            print(f"[dry-run] would process: {vid}")
        return 0

    if args.omni:
        model, processor = load_omni_model_and_processor(args)
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
    if mode == "subprocess":
        print("[batch] using subprocess mode (one process per video).")
        return run_batch_subprocess(args)
    else:
        print("[batch] using threaded mode (shared model across videos).")
        return run_batch_threads(args)
