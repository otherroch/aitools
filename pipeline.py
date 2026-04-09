"""Main pipeline orchestrator – ties all modules together.

Flow per frame:
  detect → track → identify → swap → enhance → blend → write

The ``FaceAnalysis`` ONNX model is loaded once and shared between
the detector and recognizer to avoid duplicated VRAM usage.
"""

import logging
import time

import numpy as np


from config import PipelineConfig
from face_blender import FaceBlender
from face_detector import FaceDetector
from face_enhancer import FaceEnhancer
from face_recognizer import FaceRecognizer
from face_swapper import FaceSwapper
from gpu_utils import gpu_mem_info, log_gpu_info, warmup_cuda
from video_io import VideoReader, VideoWriter

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

        logger.info("Initialising pipeline components...")

        # Detector owns the shared FaceAnalysis instance
        self._detector = FaceDetector(cfg)
        # Recognizer re-uses the same model (no extra VRAM)
        self._recognizer = FaceRecognizer(cfg, app=self._detector.app)
        self._swapper = FaceSwapper(cfg)
        self._enhancer = FaceEnhancer(cfg)
        self._blender = FaceBlender(cfg)

        used, total = gpu_mem_info(cfg.device_id)
        logger.info(
            "All pipeline components ready.  GPU memory: %.2f / %.2f GB",
            used,
            total,
        )

    # ── public entry point ───────────────────────────────────────────────

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
                for frame_idx, frame in enumerate(reader):
                    processed = self._process_frame(frame, frame_idx, stats)
                    writer.write(processed)
                    stats["frames_total"] = frame_idx + 1

                    if (frame_idx + 1) % 100 == 0:
                        self._log_progress(frame_idx + 1, reader.total_frames, t0, stats)

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

    # ── per-frame processing ─────────────────────────────────────────────

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        stats: dict,
    ) -> np.ndarray:
        """Detect → recognise → swap → enhance → blend for one frame."""
        timers = stats.get("timers")

        # 1. Detect & track faces
        _t = time.perf_counter()
        tracked_faces = self._detector.detect(frame)
        if timers is not None:
            timers["detect"] += time.perf_counter() - _t

        if not tracked_faces:
            return frame
        
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
            
            logger.debug("Track %d → target '%s' with %d reference faces", tf.track_id, target.label, len(target.reference_faces))  
            
            swap_pairs.append((tf.face_obj, target.reference_faces[0]))

        
        logger.debug("Frame %d: Identified %d faces for swapping", frame_idx, len(swap_pairs))
        
        if not swap_pairs:
            return frame

        stats["faces_identified"] += len(swap_pairs)    
        
        # 4. Run face swaps (inswapper with paste_back)
        original = frame.copy()
        _t = time.perf_counter()
        result = self._swapper.swap_multiple(frame, swap_pairs, frame_idx)
        if timers is not None:
            timers["swap"] += time.perf_counter() - _t
        stats["frames_swapped"] += 1
        stats["faces_swapped"] += len(swap_pairs)

        # 5. Seam blending – single pass over all swapped faces
        #    Run BEFORE enhancement so GFPGAN sees a smoothly-blended
        #    image rather than the hard inswapper paste-back boundary,
        #    and can smooth over the transition zone itself.
        _t = time.perf_counter()
        result = self._blender.blend_all(original, result, tracked_faces, frame_idx)
        if timers is not None:
            timers["blend"] += time.perf_counter() - _t

        # 6. Optional face enhancement (GFPGAN) – after blend so it
        #    operates on the already-composited output and does not
        #    introduce a second hard-edged boundary.
        if self._enhancer.available:
            _t = time.perf_counter()
            result = self._enhancer.enhance_faces(result, tracked_faces, frame_idx)
            if timers is not None:
                timers["enhance"] += time.perf_counter() - _t

        return result

    # ── helpers ──────────────────────────────────────────────────────────

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
