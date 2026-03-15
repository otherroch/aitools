import os
import math
import time
import subprocess
import tempfile
from typing import Optional, Dict, Any, List, Tuple

from transformers import pipeline as asr_pipeline

from videsc.utils.helpers import _format_time_hhmmss


# Shared ASR pipeline (for threaded batch mode)
_ASR_PIPELINE = None


def safe_transcribe_segment(
    audio_path: str,
    model_name: str,
    segment_duration: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Safely transcribe a single audio segment with proper error handling, using
    a Whisper-style ASR pipeline.
    """
    global _ASR_PIPELINE

    try:
        if not os.path.exists(audio_path):
            print(f"[audio] Audio file not found: {audio_path}")
            return None

        if _ASR_PIPELINE is None:
            print(f"[audio] loading Whisper ASR model '{model_name}'")
            _ASR_PIPELINE = asr_pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device="cuda",
                chunk_length_s=segment_duration,
                stride_length_s=max(1, segment_duration // 4),
                ignore_warning=True,
                language="en",
                return_timestamps="word",
            )

        print(f"[audio] transcribing segment: {audio_path}")
        result = _ASR_PIPELINE(
            audio_path,
            max_new_tokens=400,
            return_timestamps="word",
        )
        return result

    except Exception as e:
        print(f"[audio] error transcribing {audio_path}: {e}")
        # Fallback to a tiny Whisper model if the main one fails.
        try:
            fallback_model = "openai/whisper-tiny"
            print(f"[audio] trying fallback ASR model '{fallback_model}'")
            _ASR_PIPELINE = asr_pipeline(
                "automatic-speech-recognition",
                model=fallback_model,
                device="cuda",
                chunk_length_s=segment_duration,
                stride_length_s=max(1, segment_duration // 4),
                ignore_warning=True,
                language="en",
                return_timestamps="word",
            )
            return _ASR_PIPELINE(
                audio_path,
                max_new_tokens=400,
                return_timestamps="word",
            )
        except Exception as e2:
            print(f"[audio] even tiny model failed: {e2}")
            return None


def transcribe_audio_segments(
    segments_meta: List[Dict[str, Any]],
    model_name: str,
    segment_duration: int = 30,
) -> List[Dict[str, Any]]:
    """
    Transcribe multiple audio segments with progress tracking.

    Parameters
    ----------
    segments_meta:
        A list of dicts, each containing at least:
          - "path": path to the audio file
          - "start": start time in seconds (relative to the original video)
          - "end": end time in seconds (relative to the original video)
    model_name:
        Hugging Face ASR model id.
    segment_duration:
        Hint for the Whisper pipeline's internal chunking.

    Returns
    -------
    List[Dict[str, Any]]:
        For each audio segment a dict with:
          - "segment_index"
          - "path"
          - "start"
          - "end"
          - "result" (raw pipeline output)
    """
    results: List[Dict[str, Any]] = []

    for i, seg_meta in enumerate(segments_meta):
        segment_path = seg_meta.get("path")
        base_start = float(seg_meta.get("start", 0.0) or 0.0)
        base_end = float(seg_meta.get("end", base_start + segment_duration) or (base_start + segment_duration))

        print(f"[audio] processing segment {i+1}/{len(segments_meta)}: {segment_path}")
        try:
            time.sleep(1.0)
            result = safe_transcribe_segment(segment_path, model_name, segment_duration)
            if result is not None:
                results.append(
                    {
                        "segment_index": i,
                        "path": segment_path,
                        "start": base_start,
                        "end": base_end,
                        "result": result,
                    }
                )
            else:
                print(f"[audio] segment {i+1} transcription returned no result")
        except Exception as e:
            print(f"[audio] failed to process segment {i+1}: {e}")
            continue

    return results


def combine_transcription_results(
    results: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Combine transcription results from multiple segments into:
      - a single raw text string (no timestamps)
      - a list of small segments with absolute timestamps.
    """
    if not results:
        return "", []

    raw_text_parts: List[str] = []
    segments_with_ts: List[Dict[str, Any]] = []

    for item in results:
        res = item.get("result")
        if not res:
            continue
        try:
            base_start = float(item.get("start", 0.0) or 0.0)

            if isinstance(res, dict):
                chunks = res.get("chunks")
                if isinstance(chunks, list) and chunks:
                    chunk_texts: List[str] = []
                    for chunk in chunks:
                        if not isinstance(chunk, dict):
                            continue
                        txt = chunk.get("text")
                        if not txt:
                            continue

                        ts = chunk.get("timestamp") or chunk.get("timestamps")
                        start_t = end_t = None
                        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                            start_t, end_t = ts[0], ts[1]
                        elif isinstance(ts, dict):
                            start_t = ts.get("start")
                            end_t = ts.get("end")

                        if start_t is not None and end_t is not None:
                            try:
                                start_f = base_start + float(start_t)
                                end_f = base_start + float(end_t)
                            except Exception:
                                start_f = base_start
                                end_f = base_start

                            segments_with_ts.append(
                                {
                                    "timestamp": (start_f, end_f),
                                    "text": str(txt).strip(),
                                }
                            )

                        chunk_texts.append(str(txt).strip())

                    if chunk_texts:
                        raw_text_parts.append(" ".join(chunk_texts))
                elif "text" in res and isinstance(res["text"], str):
                    raw_text_parts.append(res["text"])
            elif isinstance(res, str):
                raw_text_parts.append(res)
        except Exception as e:
            print(f"[audio] error while combining result: {e}")
            continue

    raw_text = " ".join(raw_text_parts).strip()
    return raw_text, segments_with_ts


def transcribe_large_video(
    mp4_path: str,
    model_name: str,
    max_audio_seconds: float = 0.0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    High-level helper that:
      1) Splits the video's audio track into manageable segments using ffmpeg.
      2) Runs Whisper ASR on each segment.
      3) Concatenates the segment transcriptions into a single raw string and
         a list of small segments with absolute timestamps.
    """
    temp_dir: Optional[str] = None
    segment_infos: List[Dict[str, Any]] = []
    audio_files: List[str] = []

    try:
        temp_dir = tempfile.mkdtemp(prefix="whisper_segments_")

        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                mp4_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration_str = result.stdout.strip()
        duration = float(duration_str)

        if duration <= 0.0:
            raise RuntimeError(f"Could not determine valid duration for {mp4_path}")

        if max_audio_seconds > 0.0:
            duration = min(duration, max_audio_seconds)

        if duration < 60:
            segment_duration = max(1, int(duration))
        elif duration < 300:
            segment_duration = 60
        else:
            segment_duration = 120

        print(f"[audio] Using segment duration: {segment_duration} seconds")

        num_segments = max(1, int(math.ceil(duration / segment_duration)))
        for idx in range(num_segments):
            start_time = idx * segment_duration
            if start_time >= duration:
                break
            end_time = min((idx + 1) * segment_duration, duration)
            segment_name = os.path.join(temp_dir, f"temp_segment_{idx+1}.mp3")

            cmd = [
                "ffmpeg",
                "-i", mp4_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-vn",
                "-acodec", "mp3",
                "-ab", "128k",
                "-ar", "44100",
                "-y",
                segment_name,
            ]

            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio_files.append(segment_name)
            segment_infos.append(
                {
                    "path": segment_name,
                    "start": start_time,
                    "end": end_time,
                }
            )

        segment_results = transcribe_audio_segments(
            segment_infos,
            model_name=model_name,
            segment_duration=segment_duration,
        )
        transcription, segments = combine_transcription_results(segment_results)
        return transcription, segments

    except subprocess.CalledProcessError as e:
        print(f"[audio] ffmpeg/ffprobe error while processing {mp4_path}: {e}")
        return "", []
    except Exception as e:
        print(f"[audio] error processing video for transcription: {e}")
        return "", []
    finally:
        for seg in audio_files:
            try:
                if os.path.exists(seg):
                    os.remove(seg)
            except Exception:
                pass
        if temp_dir and os.path.isdir(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass


def transcribe_audio_from_video(
    video_path: str, args, model_device
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Wrapper used by the main pipeline. Delegates to the Whisper-based
    long-video transcription logic and returns both a raw transcript
    (without timestamps) and a list of timestamped segments.
    """
    asr_model_id = getattr(args, "asr_model", None) or ""
    if not asr_model_id.strip():
        return None, []

    try:
        max_secs = float(getattr(args, "max_audio_seconds", 0.0) or 0.0)
    except Exception:
        max_secs = 0.0

    transcript, segments = transcribe_large_video(
        video_path,
        model_name=asr_model_id,
        max_audio_seconds=max_secs,
    )

    if not transcript and not segments:
        print("[audio] empty transcript.")
        return None, []

    return transcript, segments


def format_transcript_with_timestamps(
    raw_transcript: str,
    segments: List[Dict[str, Any]],
) -> str:
    """
    Format a transcript together with timestamped segments.

    - If ``segments`` is empty, returns ``raw_transcript``.
    - Otherwise, returns a human-readable block containing the raw transcript
      followed by an SRT-like listing of segments with start/end times and text.
    """
    raw_transcript = (raw_transcript or "").strip()
    if not segments:
        return raw_transcript

    lines: List[str] = []
    if raw_transcript:
        lines.append("Raw transcript (no timestamps):")
        lines.append(raw_transcript)
        lines.append("")

    lines.append("Transcript with timestamps:")
    for idx, seg in enumerate(segments, start=1):
        ts = seg.get("timestamp")
        text = (seg.get("text") or "").strip()
        if not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
            continue
        start_t, end_t = ts
        try:
            start_f = float(start_t)
            end_f = float(end_t)
        except Exception:
            start_f = 0.0
            end_f = 0.0

        start_str = _format_time_hhmmss(start_f)
        end_str = _format_time_hhmmss(end_f)

        lines.append(str(idx))
        lines.append(f"{start_str} --> {end_str}")
        if text:
            lines.append(text)
        lines.append("")

    return "\n".join(lines).strip()
