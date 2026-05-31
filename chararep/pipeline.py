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

        # Temporal smoothing state: for localized EMA on face regions
        self._prev_face_part: np.ndarray | None = None
        self._prev_face_mask: np.ndarray | None = None
        self._landmark_history: dict[int, tuple[int, np.ndarray]] = {}

        used, total = gpu_mem_info(cfg.device_id)
        logger.info(
            "All pipeline components ready.  GPU memory: %.2f / %.2f GB",
            used,
            total,
        )

    @staticmethod
    def _normalize_landmarks(landmarks: np.ndarray | None) -> np.ndarray | None:
        """Return landmarks as a finite ``(5, 2)`` float32 array."""
        if landmarks is None:
            return None

        pts = np.array(landmarks, dtype=np.float32, copy=True)
        if pts.shape == (2, 5):
            pts = pts.T
        if pts.shape != (5, 2) or not np.isfinite(pts).all():
            return None

        if pts[0, 0] > pts[1, 0]:
            pts[[0, 1]] = pts[[1, 0]]
        if pts[3, 0] > pts[4, 0]:
            pts[[3, 4]] = pts[[4, 3]]
        return pts

    @staticmethod
    def _is_active_track(track) -> bool:
        """Treat tracks without an explicit age marker as active."""
        age = getattr(track, "age_since_seen", 0)
        if isinstance(age, (int, np.integer)):
            return int(age) == 0
        return True

    def _smooth_track_landmarks(
        self,
        track_id: int,
        frame_idx: int,
        landmarks: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        """Apply per-track landmark damping on the sequential pipeline path."""
        history = getattr(self, "_landmark_history", None)
        if history is None:
            self._landmark_history = {}
            history = self._landmark_history

        current = landmarks.astype(np.float32, copy=True)
        prev_state = history.get(track_id)
        if prev_state is None:
            history[track_id] = (frame_idx, current)
            return current

        prev_frame_idx, prev_landmarks = prev_state
        if frame_idx <= prev_frame_idx or frame_idx - prev_frame_idx > 1:
            history[track_id] = (frame_idx, current)
            return current

        # Dampen coherent similarity-scale pulses before pointwise smoothing.
        # The detector occasionally moves the whole 5-point constellation
        # inward/outward together for a frame, which makes the aligned swap
        # crop briefly "breathe" even when the head is otherwise stable.
        anchor_prev = prev_landmarks[[0, 1, 2]]
        anchor_current = current[[0, 1, 2]]
        prev_eye_mid = (anchor_prev[0] + anchor_prev[1]) * 0.5
        current_eye_mid = (anchor_current[0] + anchor_current[1]) * 0.5
        prev_anchor_center = (prev_eye_mid + anchor_prev[2]) * 0.5
        current_anchor_center = (current_eye_mid + anchor_current[2]) * 0.5

        prev_anchor_offsets = anchor_prev - prev_anchor_center
        current_anchor_offsets = anchor_current - current_anchor_center
        prev_anchor_scale = float(
            np.sqrt(np.mean(np.sum(prev_anchor_offsets * prev_anchor_offsets, axis=1)))
        )
        current_anchor_scale = float(
            np.sqrt(np.mean(np.sum(current_anchor_offsets * current_anchor_offsets, axis=1)))
        )

        bbox_f = np.array(bbox, dtype=np.float32, copy=False)
        face_size = max(
            float(bbox_f[2] - bbox_f[0]),
            float(bbox_f[3] - bbox_f[1]),
            float(np.linalg.norm(current[1] - current[0]) * 2.5),
            1.0,
        )

        if prev_anchor_scale > 1e-6 and current_anchor_scale > 1e-6:
            max_scale_step = float(np.clip(3.0 / face_size, 0.010, 0.028))
            clipped_anchor_scale = float(
                np.clip(
                    current_anchor_scale,
                    prev_anchor_scale * (1.0 - max_scale_step),
                    prev_anchor_scale * (1.0 + max_scale_step),
                )
            )
            stabilized_anchor_scale = prev_anchor_scale + (
                clipped_anchor_scale - prev_anchor_scale
            ) * 0.35
            scale_ratio = stabilized_anchor_scale / current_anchor_scale
            current = current_anchor_center + (current - current_anchor_center) * scale_ratio

        max_step = max(1.5, face_size * 0.08)

        delta = current - prev_landmarks
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_step / np.maximum(delta_norm, 1e-6))
        capped = prev_landmarks + delta * scale

        current_weight = np.array(
            [0.45, 0.45, 0.55, 0.65, 0.65], dtype=np.float32
        )[:, np.newaxis]
        smoothed = prev_landmarks + (capped - prev_landmarks) * current_weight

        history[track_id] = (frame_idx, smoothed.astype(np.float32, copy=True))
        return smoothed.astype(np.float32)

    def _stabilize_track_landmarks(
        self,
        tracked_faces: list,
        frame_idx: int,
    ) -> None:
        """Stabilize active face landmarks before worker threads start swapping."""
        active_ids: set[int] = set()

        for tf in tracked_faces:
            if not self._is_active_track(tf):
                continue

            pts = self._normalize_landmarks(getattr(tf, "landmarks", None))
            if pts is None:
                continue

            track_id = int(getattr(tf, "track_id", -1))
            active_ids.add(track_id)
            smoothed = self._smooth_track_landmarks(track_id, frame_idx, pts, tf.bbox)
            tf.landmarks = smoothed

            face_obj = getattr(tf, "face_obj", None)
            if face_obj is not None:
                try:
                    face_obj.kps = smoothed.copy()
                except Exception:
                    logger.debug(
                        "Could not store stabilized keypoints on face object for track %s",
                        track_id,
                    )

        history = getattr(self, "_landmark_history", None)
        if not history:
            return

        max_age = max(1, int(self._cfg.tracker_max_age))
        stale_tracks = [
            track_id
            for track_id, (last_frame_idx, _) in history.items()
            if track_id not in active_ids and frame_idx - last_frame_idx > max_age
        ]
        for track_id in stale_tracks:
            history.pop(track_id, None)

    def _apply_temporal_face_blend(
        self,
        result: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend only the overlap between consecutive face masks when enabled."""
        if mask is None or not np.any(mask):
            self._prev_face_part = None
            self._prev_face_mask = None
            return result

        history_weight = float(np.clip(self._cfg.temporal_smooth_alpha, 0.0, 1.0))
        current = result.astype(np.float32)
        mask_f32 = mask.astype(np.float32) / 255.0

        prev_face_part = getattr(self, "_prev_face_part", None)
        prev_face_mask = getattr(self, "_prev_face_mask", None)
        if (
            history_weight > 0.0
            and prev_face_part is not None
            and prev_face_mask is not None
        ):
            overlap_mask = np.minimum(mask_f32, prev_face_mask)
            overlap_area = float(overlap_mask.sum())
            current_area = float(mask_f32.sum())
            overlap_ratio = overlap_area / current_area if current_area > 0.0 else 0.0

            if overlap_ratio >= 0.35:
                overlap_mask = overlap_mask[:, :, np.newaxis]
                blended = (
                    current * (1.0 - history_weight)
                    + prev_face_part * history_weight
                )
                current = blended * overlap_mask + current * (1.0 - overlap_mask)

        self._prev_face_part = current.copy()
        self._prev_face_mask = mask_f32.copy()
        return np.clip(current, 0, 255).astype(np.uint8)

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
                # We store the frame with the future to allow localized EMA in _drain_one
                pending.append((frame_data, future))

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
        frame, future = pending.popleft()
        result, local, mask = future.result()

        result = self._apply_temporal_face_blend(result, mask)

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
        result, local, mask = self._finish_frame(*prep)

        result = self._apply_temporal_face_blend(result, mask)

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
        tracked_faces = [
            tf for tf in self._detector.detect(frame)
            if self._is_active_track(tf)
        ]
        if timers is not None:
            timers["detect"] += time.perf_counter() - _t

        if not tracked_faces:
            return frame, frame_idx, [], []

        self._stabilize_track_landmarks(tracked_faces, frame_idx)

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

        Returns ``(processed_frame, local_stats, mask)`` where *local_stats*
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
            return frame, local, np.zeros(frame.shape[:2], dtype=np.uint8)

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
        result, mask = self._blender.blend_all(original, result, tracked_faces, frame_idx)
        local["blend"] = time.perf_counter() - _t

        # 6. Optional face enhancement (GFPGAN) – after blend so it
        #    operates on the already-composited output and does not
        #    introduce a second hard-edged boundary.
        if self._enhancer.available:
            _t = time.perf_counter()
            result = self._enhancer.enhance_faces(result, tracked_faces, frame_idx)
            local["enhance"] = time.perf_counter() - _t

        return result, local, mask

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