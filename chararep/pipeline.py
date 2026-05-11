"""Main pipeline orchestrator – ties all modules together.

Flow per frame:
  detect → track → identify → swap → enhance → blend → write

The :class:`FaceBackend` (from :mod:`face_ops`) is loaded once and
shared between the detector and recognizer to avoid duplicated VRAM
usage.

When ``batch_size > 1`` the pipeline runs in *parallel* mode: detection
and tracking (which are stateful) stay on the main thread while the
heavier swap / blend / enhance stages are dispatched to a
:class:`~concurrent.futures.ThreadPoolExecutor` so multiple frames
can be processed concurrently.
"""

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import numpy as np


from .config import PipelineConfig
from .face_blender import FaceBlender
from .face_detector import FaceDetector
from .face_enhancer import FaceEnhancer
from .face_recognizer import FaceRecognizer
from .face_swapper import FaceSwapper
from .gpu_utils import gpu_mem_info, log_gpu_info, warmup_cuda
from .video_io import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


class CharacterReplacementPipeline:
    """End-to-end pipeline: read → detect → recognise → swap → enhance → blend → write.

    Usage::

        cfg = PipelineConfig(...)
        pipeline = CharacterReplacementPipeline(cfg)
        pipeline.run()
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg

        # GPU diagnostics & warm-up
        log_gpu_info()
        warmup_cuda(cfg.device_id)

        logger.info("Initialising pipeline components. ..")

        # Detector owns the shared FaceBackend instance
        self._detector = FaceDetector(cfg)
        # Recognizer re-uses the same backend (no extra VRAM)
        self._recognizer = FaceRecognizer(cfg, backend=self._detector.backend)
        self._swapper = FaceSwapper(cfg)
        self._enhancer = FaceEnhancer(cfg)
        self._blender = FaceBlender(cfg)

        # Temporal smoothing state: previous blended frame for EMA
        self._prev_frame: np.ndarray | None = None

        used, total = gpu_mem_info(cfg.device_id)
        logger.info(
            "All pipeline components ready.  GPU memory: %.2f / %.2f GB",
            used,
            total,
        )

    # ── public entry point ────────────────────────────────────────────

    def run(self) -> dict:
        """Process the entire video and return run statistics."""
        stats = {
            "frames_total": 0,
            "frames_swapped": 0,
            "faces_swapped": 0,
            "elapsed_s": 0.0,
            "frames_detected": 0,
            "faces_identified": 0,
            "fps": 0.0,
        }

        if self._cfg.enable_timers:
            stats["timers"] = {
                "detect": 0.0,
                "recognize": 0.0,
                "swap": 0.0,
                "enhance": 0.0,
                "blend": 0.0,
            }

        t0 = time.perf_counter()

        with VideoReader(
            self._cfg.input_video, queue_size=self._cfg.batch_size * 2
        ) as reader:
            writer = VideoWriter(
                path=self._cfg.output_video,
                width=reader.width,
                height=reader.height,
                fps=reader.fps,
                codec=self._cfg.output_codec,
                crf=self._cfg.output_quality,
                audio_source=(
                    self._cfg.input_video if self._cfg.copy_audio else None
                ),
            )
            with writer:
                if self._cfg.batch_size > 1:
                    self._run_parallel(reader, writer, stats, t0)
                else:
                    self._run_sequential(reader, writer, stats, t0)

        stats["elapsed_s"] = time.perf_counter() - t0
        stats["fps"] = (
            stats["frames_total"] / stats["elapsed_s"]
            if stats["elapsed_s"] > 0
            else 0
        )

        logger.info(
            "Pipeline complete: %d frames in %.1fs (%.1f fps), %d faces swapped.",
            stats["frames_total"],
            stats["elapsed_s"],
            stats["fps"],
            stats["faces_swapped"],
        )
        logger.info("Frames with detected faces: %d, total faces identified: %d",
            stats["frames_detected"], stats["faces_identified"])

        if "timers" in stats:
            self._log_timer_distribution(stats["timers"])

        return stats

    # ── sequential processing ───────────────────────────────────────────

    def _run_sequential(self, reader, writer, stats: dict, t0: float) -> None:
        """Process frames one at a time (batch_size=1)."""
        for frame_idx, frame in enumerate(reader):
            processed = self._process_frame(frame, frame_idx, stats)
            writer.write(processed)
            stats["frames_total"] = frame_idx + 1

            if (frame_idx + 1) % 100 == 0:
                self._log_progress(
                    frame_idx + 1, reader.total_frames, t0, stats
                )

    # ── parallel processing ───────────────────────────────────────────

    def _run_parallel(self, reader, writer, stats: dict, t0: float) -> None:
        """Process frames with a thread pool.

        Detection and tracking run sequentially on the main thread
        (the IoU tracker is stateful), while swap / blend / enhance
        are dispatched to worker threads so multiple frames overlap.
        """
        batch_size = self._cfg.batch_size
        logger.info("Parallel pipeline: %d workers", batch_size)

        with ThreadPoolExecutor(
            max_workers=batch_size,
            thread_name_prefix="ChararepWorker",
        ) as pool:
            pending: deque = deque()
            frame_count = 0

            for frame_idx, frame in enumerate(reader):
                # Sequential: detect + track + identify
                prep = self._prepare_frame(frame, frame_idx, stats)

                # Snapshot tracked faces so the worker thread sees
                # immutable per-frame state even if the tracker mutates
                # the original objects on subsequent frames.
                frame_data, fidx, tracked, pairs = prep
                tracked = [copy(tf) for tf in tracked]
                prep = (frame_data, fidx, tracked, pairs)

                # Submit parallel: swap + blend + enhance
                future = pool.submit(self._finish_frame, *prep)
                pending.append(future)

                # Once we hit capacity, drain the oldest result to
                # keep at most *batch_size* frames in flight.
                if len(pending) >= batch_size:
                    frame_count = self._drain_one(
                        pending, writer, stats, frame_count, t0,
                        reader.total_frames,
                    )

            # Drain remaining futures in order.
            while pending:
                frame_count = self._drain_one(
                    pending, writer, stats, frame_count, t0,
                    reader.total_frames,
                )

    def _drain_one(
        self,
        pending: deque,
        writer,
        stats: dict,
        frame_count: int,
        t0: float,
        total_frames: int,
    ) -> int:
        """Wait for the oldest pending future and write its result."""
        result, local = pending.popleft().result()
        writer.write(result)
        frame_count += 1
        self._merge_finish_stats(stats, local)
        stats["frames_total"] = frame_count
        if frame_count % 100 == 0:
            self._log_progress(frame_count, total_frames, t0, stats)
        return frame_count

    # ── per-frame processing ───────────────────────────────────────────

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        stats: dict,
    ) -> np.ndarray:
        """Detect → recognise → swap → enhance → blend for one frame."""
        prep = self._prepare_frame(frame, frame_idx, stats)
        result, local = self._finish_frame(*prep)
        self._merge_finish_stats(stats, local)
        return result

    def _prepare_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        stats: dict,
    ) -> tuple:
        """Detect, track, and identify faces (must run sequentially).

        Returns ``(frame, frame_idx, tracked_faces, swap_pairs)`` for
        :meth:`_finish_frame`.
        """
        timers = stats.get("timers")

        # 1. Detect & track faces
        _t = time.perf_counter()
        tracked_faces = self._detector.detect(frame)
        if timers is not None:
            timers["detect"] += time.perf_counter() - _t

        if not tracked_faces:
            return frame, frame_idx, [], []

        logger.debug("Frame %d: Detected %d faces", frame_idx, len(tracked_faces))

        stats["frames_detected"] += 1

        # 2. Identify new faces against the target gallery
        #    (already-identified tracks keep their label via the tracker)
        _t = time.perf_counter()
        self._recognizer.identify_faces(tracked_faces)
        if timers is not None:
            timers["recognize"] += time.perf_counter() - _t

        # 3. Build swap pairs: (source_face_obj, target_reference_face)
        swap_pairs: list[tuple] = []
        for tf in tracked_faces:
            if tf.identity_label is None:
                continue
            logger.debug("Track %d has identity label '%s'", tf.track_id, tf.identity_label)

            target = self._recognizer.get_target(tf.identity_label)

            if target is None:
                logger.debug("No target found for label '%s'", tf.identity_label)
                continue

            if not target.reference_faces:
                logger.debug("Target '%s' has no reference faces", tf.identity_label)
                continue

            logger.debug("Track %d -> target '%s' with %d reference faces", tf.track_id, target.label, len(target.reference_faces))

            swap_pairs.append((tf.face_obj, target.reference_faces[0]))

        logger.debug("Frame %d: Identified %d faces for swapping", frame_idx, len(swap_pairs))

        if swap_pairs:
            stats["faces_identified"] += len(swap_pairs)

        return frame, frame_idx, tracked_faces, swap_pairs

    def _finish_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        tracked_faces: list,
        swap_pairs: list,
    ) -> tuple:
        """Swap, blend, and enhance faces (safe to run in worker threads).

        Returns ``(processed_frame, local_stats)`` where *local_stats*
        contains per-frame counters and timer deltas to merge back into
        the run stats.
        """
        local: dict = {
            "frames_swapped": 0,
            "faces_swapped": 0,
            "swap": 0.0,
            "blend": 0.0,
            "enhance": 0.0,
        }

        if not swap_pairs:
            return frame, local

        # 4. Run face swaps (inswapper with paste_back)
        original = frame.copy()
        _t = time.perf_counter()
        result = self._swapper.swap_multiple(frame, swap_pairs, frame_idx)
        local["swap"] = time.perf_counter() - _t
        local["frames_swapped"] = 1
        local["faces_swapped"] = len(swap_pairs)

        # 5. Seam blending – single pass over all swapped faces
        #    Run BEFORE enhancement so GFPGAN sees a smoothly-blended
        #    image rather than the hard inswapper paste-back boundary,
        #    and can smooth over the transition zone itself.
        _t = time.perf_counter()
        result = self._blender.blend_all(original, result, tracked_faces, frame_idx)
        local["blend"] = time.perf_counter() - _t

        # 6. Optional face enhancement (GFPGAN) – after blend so it
        #    operates on the already-composited output and does not
        #    introduce a second hard-edged boundary.
        if self._enhancer.available:
            _t = time.perf_counter()
            result = self._enhancer.enhance_faces(result, tracked_faces, frame_idx)
            local["enhance"] = time.perf_counter() - _t

        # 7. Temporal smoothing: exponential moving average to reduce
        #    frame-to-frame jitter.  Blends the current result with the
        #    previous frame using alpha from config.
        alpha = self._cfg.temporal_smooth_alpha
        if alpha > 0.0 and frame_idx > 0:
            if self._prev_frame is None:
                self._prev_frame = result.copy()
            # EMA: smoothed = alpha * current + (1 - alpha) * previous
            result = (
                result.astype(np.float32) * alpha
                + self._prev_frame.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)
            self._prev_frame = result.copy()

        return result, local

    @staticmethod
    def _merge_finish_stats(stats: dict, local: dict) -> None:
        """Fold per-frame counters from :meth:`_finish_frame` into *stats*."""
        stats["frames_swapped"] += local["frames_swapped"]
        stats["faces_swapped"] += local["faces_swapped"]
        timers = stats.get("timers")
        if timers is not None:
            timers["swap"] += local["swap"]
            timers["blend"] += local["blend"]
            timers["enhance"] += local["enhance"]

    @staticmethod
    def _log_timer_distribution(timers: dict) -> None:
        """Log cumulative per-stage timing and percentage distribution."""
        total = sum(timers.values())
        logger.info("=" * 60)
        logger.info("Pipeline stage timing distribution:")
        for stage, elapsed in timers.items():
            pct = elapsed / total * 100 if total > 0 else 0.0
            logger.info("  %-12s  %8.3f s  (%5.1f%%)", stage, elapsed, pct)
        logger.info("  %-12s  %8.3f s  (100.0%%)", "TOTAL", total)
        logger.info("=" * 60)

    @staticmethod
    def _log_progress(
        done: int, total: int, t0: float, stats: dict
    ) -> None:
        elapsed = time.perf_counter() - t0
        fps = done / elapsed if elapsed > 0 else 0
        pct = done / total * 100 if total > 0 else 0
        logger.info(
            "Progress: %d/%d frames (%.1f%%)  %.1f fps  swapped %d faces",
            done,
            total,
            pct,
            fps,
            stats["faces_swapped"],
        )