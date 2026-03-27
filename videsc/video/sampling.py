from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def compute_effective_nframes(
    vinfo: Dict[str, Any],
    requested_nframes: int,
    spf: float,
) -> int:
    """
    Reproduce the nframes logic used in build_messages so we know
    the final number of frames the model will see.

    This lets us align audio segments to the same count.
    """
    nframes = requested_nframes
    frame_rate = vinfo["FPS"]
    num_frames = vinfo["num_frames"]

    if spf > 0.0:
        interval = spf * frame_rate
        if interval > 0:
            new_nframes = int(num_frames / interval)
        else:
            new_nframes = nframes
        nframes = max(1, new_nframes)

    if nframes > 768:
        nframes = 768

    logger.debug(
        "compute_effective_nframes: requested=%d  fps=%.2f  num_frames=%d  spf=%.2f  effective=%d",
        requested_nframes, frame_rate, num_frames, spf, nframes,
    )
    return nframes


def compress_audio_segments_to_nframes(
    segments: List[Dict[str, Any]],
    nframes: int,
    video_duration: float,
) -> List[Dict[str, Any]]:
    """
    Reduce / aggregate Whisper segments into exactly `nframes` coarse segments.

    We split [0, video_duration] into `nframes` equal time buckets.
    For each bucket we concatenate all segment texts that overlap the bucket
    and assign the bucket [start, end] as the coarse timestamp.

    This guarantees len(return_value) == nframes.
    """
    if not segments or nframes <= 0 or video_duration <= 0:
        logger.debug(
            "compress_audio_segments_to_nframes: skipped  segments=%d  nframes=%d  duration=%.2f",
            len(segments) if segments else 0, nframes, video_duration,
        )
        return segments

    logger.debug(
        "compress_audio_segments_to_nframes: compressing %d segment(s) into %d bucket(s)  duration=%.2f",
        len(segments), nframes, video_duration,
    )

    def _seg_start(s):
        ts = s.get("timestamp") or (0.0, 0.0)
        if isinstance(ts, (list, tuple)) and len(ts) >= 1:
            try:
                return float(ts[0])
            except Exception:
                return 0.0
        return 0.0

    segments_sorted = sorted(segments, key=_seg_start)
    bucket_size = video_duration / float(nframes)

    coarse: List[Dict[str, Any]] = []
    for i in range(nframes):
        bucket_start = i * bucket_size
        bucket_end = video_duration if i == nframes - 1 else (i + 1) * bucket_size

        texts: List[str] = []
        for seg in segments_sorted:
            ts = seg.get("timestamp")
            if not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
                continue
            try:
                s_start = float(ts[0])
                s_end = float(ts[1])
            except Exception:
                continue

            if s_end <= bucket_start or s_start >= bucket_end:
                continue

            t = (seg.get("text") or "").strip()
            if t:
                texts.append(t)

        merged_text = " ".join(texts).strip()
        coarse.append(
            {
                "timestamp": (bucket_start, bucket_end),
                "text": merged_text,
            }
        )

    return coarse
