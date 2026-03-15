from typing import Optional, Dict, Any, List

from videsc.utils.helpers import _format_time_hhmmss


def build_messages(
    video_path: str,
    vinfo: Dict[str, Any],
    tot_pixels: int,
    nframes: int,
    spf: float,
    prompt: str,
    system: Optional[str] = None,
    continue_prompt: Optional[bool] = False,
    audio_transcript: Optional[str] = None,
    audio_segments: Optional[List[Dict[str, Any]]] = None,
    is_omni: Optional[bool] = False,
):
    """
    Build Qwen3-VL chat messages, including:
      - video block with total_pixels / nframes
      - optional full transcript text block
      - optional timestamp-aligned segments with approximate frame ranges
      - user prompt text
    """
    width_p = vinfo["width"]
    height_p = vinfo["height"]
    frame_rate = vinfo["FPS"]
    num_frames = vinfo["num_frames"]
    total_time = vinfo["tot_time"]

    print("width: ", width_p, "height: ", height_p)
    print("max_pixels: ", width_p * height_p, " frame_rate: ", frame_rate)
    print("num_frames: ", num_frames, " total time: ", total_time)
    print("called with nframes: ", nframes, " called with total_pixels: ", tot_pixels)

    if spf > 0.0:
        interval = spf * frame_rate
        if interval > 0:
            new_nframes = int(num_frames / interval)
        else:
            new_nframes = nframes
        alt_nframes = int(total_time / spf) if spf > 0 else nframes
        print("spf:", spf, " new_nframes: ", new_nframes, " alt nframes: ", alt_nframes)
        nframes = max(1, new_nframes)

    if tot_pixels <= 0:
        tot_pixels = width_p * height_p

    if nframes > 768:
        print("nframes", nframes, "CAPPED to 768")
        nframes = 768

    print("nframes: ", nframes, " total_pixels: ", tot_pixels)

    content: List[Dict[str, Any]] = [
        {
            "type": "video",
            "video": str(video_path),
            "total_pixels": tot_pixels,
            "nframes": nframes,
        }
    ]
    if is_omni:
        content = [
            {
                "type": "video",
                "video": str(video_path),
            }
        ]

    # Full raw transcript (for semantic context)
    if audio_transcript:
        content.append(
            {
                "type": "text",
                "text": (
                    "Here is an automatic transcription of the video's audio track. "
                    "Use it together with the visual frames to better understand the scene, "
                    "dialogue, speakers, and any on-screen text they mention:\n\n"
                    f"{audio_transcript}"
                ),
            }
        )

    # Timestamp + frame-aligned segment list
    if audio_segments:
        fps = frame_rate if frame_rate and frame_rate > 0 else 30.0
        seg_lines = ["Timestamp-aligned audio segments with approximate frame ranges:"]
        for idx, seg in enumerate(audio_segments, start=1):
            ts = seg.get("timestamp")
            seg_text = (seg.get("text") or "").strip()
            if not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
                continue
            start_t, end_t = ts
            if start_t is None or end_t is None:
                continue
            start_frame = int(round(max(0.0, start_t) * fps))
            end_frame = int(round(max(start_t, end_t) * fps))
            if num_frames > 0:
                start_frame = min(max(0, start_frame), num_frames - 1)
                end_frame = min(max(start_frame, end_frame), num_frames - 1)

            t_start_str = _format_time_hhmmss(start_t)
            t_end_str = _format_time_hhmmss(end_t)
            if not seg_text:
                seg_text = "<no text>"

            seg_lines.append(
                f"Segment {idx}: {t_start_str} – {t_end_str} "
                f"(frames {start_frame}–{end_frame})\n{seg_text}"
            )

        if len(seg_lines) > 1:
            content.append({"type": "text", "text": "\n".join(seg_lines)})

    if prompt:
        content.append({"type": "text", "text": prompt})

    if continue_prompt:
        content.append(
            {
                "type": "text",
                "text": (
                    "Continue describing the video in more detail, focusing on any aspects "
                    "you haven't mentioned yet."
                ),
            }
        )

    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages
