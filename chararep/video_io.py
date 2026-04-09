"""Threaded video reader / writer with buffered frame I/O."""

import logging
import subprocess
import threading
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sentinel object to signal end-of-stream
_EOS = object()


class VideoReader:
    """Reads frames from a video file in a background thread.

    Decoded frames are placed into a bounded queue so the GPU pipeline
    can consume them without waiting on I/O.
    """

    def __init__(self, path: str, queue_size: int = 16):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_s = (
            self.total_frames / self.fps if self.fps > 0 else 0
        )

        self._queue: Queue = Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── public API ───────────────────────────────────────────────────────

    def start(self) -> "VideoReader":
        """Spawn the background decode thread."""
        self._thread = threading.Thread(
            target=self._decode_loop, daemon=True, name="VideoReader"
        )
        self._thread.start()
        logger.info(
            "VideoReader started: %dx%d @ %.1f fps, %d frames (%.1fs)",
            self.width,
            self.height,
            self.fps,
            self.total_frames,
            self.duration_s,
        )
        return self

    def read(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Return the next frame (BGR, uint8) or None at end-of-stream.

        A queue timeout does not indicate EOF: the decoder may simply be
        temporarily delayed or the queue may be briefly empty. Keep polling
        until we receive a frame, observe the EOS sentinel, or know that the
        reader has been stopped / the decode thread has exited and no more
        frames can arrive.
        """
        while True:
            try:
                item = self._queue.get(timeout=timeout)
            except Empty:
                if self._stop_event.is_set():
                    return None
                if (
                    self._thread is not None
                    and not self._thread.is_alive()
                    and self._queue.empty()
                ):
                    return None
                continue
            if item is _EOS:
                return None
            return item
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    # ── internal ─────────────────────────────────────────────────────────

    def _decode_loop(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                break
            # Use a timeout so we can react to stop requests even when the
            # consumer is slow and the queue is full.
            while not self._stop_event.is_set():
                try:
                    self._queue.put(frame, timeout=0.5)
                    break
                except Full:
                    pass
            else:
                # Stop event fired before we could enqueue the frame.
                break
            idx += 1
        # Best-effort EOS signal; ignore if consumer already stopped.
        try:
            self._queue.put(_EOS, timeout=2.0)
        except Full:
            pass
        logger.debug("VideoReader decoded %d frames total.", idx)


class VideoWriter:
    """Writes frames in a background thread via ffmpeg subprocess.

    Using ffmpeg directly gives us better codec control and lets us
    mux the original audio track without re-encoding it.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        crf: int = 18,
        audio_source: Optional[str] = None,
        queue_size: int = 32,
    ):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self._audio_source = audio_source
        self._queue: Queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()

        # Build ffmpeg command
        self._tmp_path = path if audio_source is None else path + ".tmp.mp4"
        self._final_path = path

        self._cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", codec,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-preset", "slow",
            self._tmp_path,
        ]
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "VideoWriter":
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(
            target=self._write_loop, daemon=True, name="VideoWriter"
        )
        self._thread.start()
        logger.info(
            "VideoWriter started: %s (%dx%d @ %.1f fps)",
            self.path,
            self.width,
            self.height,
            self.fps,
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        self._queue.put(frame)

    def stop(self) -> None:
        self._stop_event.set()

        eos_enqueued = False
        while not eos_enqueued:
            try:
                self._queue.put(_EOS, timeout=0.1)
                eos_enqueued = True
            except Full:
                if self._thread is None or not self._thread.is_alive():
                    break
                try:
                    while True:
                        item = self._queue.get_nowait()
                        if item is _EOS:
                            eos_enqueued = True
                            break
                except Empty:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=60)
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "ffmpeg did not finish within 120 s – terminating."
                )
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        # Mux audio if requested
        if self._audio_source and Path(self._tmp_path).exists():
            self._mux_audio()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ── internal ─────────────────────────────────────────────────────────

    def _write_loop(self) -> None:
        count = 0
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is _EOS:
                break
            try:
                self._proc.stdin.write(item.tobytes())
                count += 1
            except (BrokenPipeError, OSError):
                logger.error("ffmpeg pipe broke – stopping writer.")
                break
        logger.debug("VideoWriter wrote %d frames.", count)

    def _mux_audio(self) -> None:
        """Mux original audio into the final output using ffmpeg."""
        logger.info("Muxing original audio into output video...")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", self._tmp_path,
            "-i", self._audio_source,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            self._final_path,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
        except subprocess.TimeoutExpired:
            logger.warning("Audio mux timed out – output will have no audio.")
            Path(self._tmp_path).rename(self._final_path)
            return
        if result.returncode == 0:
            Path(self._tmp_path).unlink(missing_ok=True)
            logger.info("Audio muxed successfully.")
        else:
            logger.warning(
                "Audio mux failed (output will have no audio): %s",
                result.stderr[-500:] if result.stderr else "",
            )
            # Fall back to video-only output
            Path(self._tmp_path).rename(self._final_path)
