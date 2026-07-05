"""SCAIL-2 prepared-assets runner for whole-video character replacement."""

from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .video_io import finalize_video_output

logger = logging.getLogger(__name__)
_BYTES_PER_GIB = 1024 ** 3


@dataclass(frozen=True)
class _StagedInputs:
    image: Path
    mask_image: Path
    pose: Path
    mask_video: Path
    additional_images: tuple[Path, ...] = ()
    additional_masks: tuple[Path, ...] = ()


class Scail2PreparedAssetsRunner:
    """Run SCAIL-2 replacement in prepared-assets or auto-prep mode.

    This runner intentionally stays separate from the classic per-frame
    face-swap pipeline. It either stages explicit SCAIL-2 assets or derives
    them through SCAIL-Pose auto-preprocessing, then invokes the upstream
    ``generate.py`` entrypoint and finalizes the resulting video for chararep's
    output contract.
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg

    _PREFLIGHT_WINDOW_SECONDS: float = 8.0
    _PREFLIGHT_MIN_SAMPLED_FRAMES: int = 8
    _PREFLIGHT_MAX_SAMPLED_FRAMES: int = 24
    _HIGH_RISK_MODEL_GIB: float = 28.0
    _WARN_RISK_MODEL_GIB: float = 24.0
    _HIGH_RISK_TARGET_PIXELS: int = 896 * 512
    _WARN_RISK_TARGET_PIXELS: int = 672 * 384

    def run(self) -> dict:
        start = time.perf_counter()
        prompt = self._resolve_prompt()
        total_frames, _input_fps = self._probe_video(self._cfg.input_video)
        self._warn_or_raise_vram_risk()

        job_dir = self._create_job_dir()
        cleanup_job_dir = not self._cfg.scail2_keep_intermediates

        try:
            staged = self._prepare_inputs(job_dir, total_frames, _input_fps)
            output_path = job_dir / "output.mp4"
            cmd = self._build_command(staged, prompt, output_path)
            env = self._build_generate_env()

            logger.info("Running SCAIL-2 generate job in %s", job_dir)
            run_kwargs = {
                "cwd": self._cfg.scail2_repo_path,
                "capture_output": True,
                "text": True,
            }
            if env is not None:
                run_kwargs["env"] = env
            result = subprocess.run(cmd, **run_kwargs)

            if result.returncode != 0:
                raise RuntimeError(self._format_subprocess_failure(result))
            if not output_path.is_file():
                raise FileNotFoundError(
                    f"SCAIL-2 did not produce an output video: {output_path}"
                )

            finalize_video_output(
                str(output_path),
                self._cfg.output_video,
                audio_source=self._cfg.input_video if self._cfg.copy_audio else None,
            )
        finally:
            if cleanup_job_dir:
                shutil.rmtree(job_dir, ignore_errors=True)

        elapsed_s = time.perf_counter() - start
        fps = total_frames / elapsed_s if elapsed_s > 0 and total_frames > 0 else 0.0
        return {
            "backend": "scail2",
            "frames_total": total_frames,
            "frames_swapped": total_frames,
            "faces_swapped": 0,
            "elapsed_s": elapsed_s,
            "frames_detected": 0,
            "faces_identified": 0,
            "fps": fps,
        }

    def _prepare_inputs(
        self,
        job_dir: Path,
        total_frames: int,
        input_fps: float,
    ) -> _StagedInputs:
        """Return generate.py inputs, either from explicit assets or auto-prep."""
        if self._cfg.scail2_has_prepared_assets():
            return self._stage_prepared_assets(job_dir)
        return self._stage_inputs_from_scail_pose(job_dir, total_frames, input_fps)

    def _resolve_prompt(self) -> str:
        prompt = (self._cfg.scail2_prompt or "").strip()
        if prompt:
            return prompt

        prompt_file = self._cfg.scail2_prompt_file
        if not prompt_file:
            raise ValueError("SCAIL-2 prompt is required")

        text = Path(prompt_file).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"SCAIL-2 prompt file is empty: {prompt_file}")
        return text

    def _create_job_dir(self) -> Path:
        return Path(
            tempfile.mkdtemp(
                prefix="chararep_scail2_",
                dir=self._cfg.scail2_work_dir,
            )
        )

    def _stage_prepared_assets(self, job_dir: Path) -> _StagedInputs:
        additional_images, additional_masks = self._stage_additional_references(job_dir)
        return _StagedInputs(
            image=self._copy_to_job_dir(job_dir, self._cfg.scail2_reference_image, "ref"),
            mask_image=self._copy_to_job_dir(job_dir, self._cfg.scail2_reference_mask, "ref_mask"),
            pose=self._copy_to_job_dir(job_dir, self._cfg.input_video, "rendered_v2"),
            mask_video=self._copy_to_job_dir(job_dir, self._cfg.scail2_mask_video, "rendered_mask_v2"),
            additional_images=additional_images,
            additional_masks=additional_masks,
        )

    def _stage_inputs_from_scail_pose(
        self,
        job_dir: Path,
        total_frames: int,
        input_fps: float,
    ) -> _StagedInputs:
        """Derive SCAIL-2 masks and intermediates from SCAIL-Pose."""
        ref_image = self._stage_auto_reference_image(job_dir)
        self._stage_driving_video(job_dir)
        additional_images, additional_masks = self._stage_additional_references(job_dir)

        matchnearest, egocentric = self._resolve_pose_preprocess_flags(
            total_frames,
            input_fps,
        )
        cmd = self._build_pose_command(
            job_dir,
            matchnearest=matchnearest,
            egocentric=egocentric,
        )
        logger.info("Running SCAIL-Pose replacement preprocessing in %s", job_dir)
        result = subprocess.run(
            cmd,
            cwd=self._resolve_pose_repo_path(),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(self._format_pose_failure(result))

        mask_image = job_dir / "ref_mask.png"
        pose_video = job_dir / "rendered_v2.mp4"
        mask_video = job_dir / "replace_mask.mp4"
        for path in [mask_image, pose_video, mask_video]:
            if not path.is_file():
                raise FileNotFoundError(
                    f"SCAIL-Pose auto-prep did not produce expected output: {path}"
                )

        return _StagedInputs(
            image=ref_image,
            mask_image=mask_image,
            pose=pose_video,
            mask_video=mask_video,
            additional_images=additional_images,
            additional_masks=additional_masks,
        )

    @staticmethod
    def _copy_to_job_dir(job_dir: Path, source_path: str, stem: str) -> Path:
        source = Path(source_path)
        target = job_dir / f"{stem}{source.suffix}"
        shutil.copy2(source, target)
        return target

    def _stage_additional_references(
        self,
        job_dir: Path,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        """Stage optional multi-reference inputs into the isolated job dir."""
        additional_images = tuple(
            self._copy_to_job_dir(job_dir, source_path, f"additional_ref_{index}")
            for index, source_path in enumerate(
                self._cfg.scail2_additional_reference_images
            )
        )
        additional_masks = tuple(
            self._copy_to_job_dir(job_dir, source_path, f"additional_ref_mask_{index}")
            for index, source_path in enumerate(
                self._cfg.scail2_additional_reference_masks
            )
        )
        return additional_images, additional_masks

    def _stage_auto_reference_image(self, job_dir: Path) -> Path:
        """Write a PNG ref image for SCAIL-Pose auto-prep."""
        source_path = self._resolve_reference_image_source()
        img = self._read_image_bgr(source_path)
        target = job_dir / "ref_image.png"
        if not cv2.imwrite(str(target), img):
            raise RuntimeError(f"Could not write staged SCAIL-2 ref image: {target}")
        return target

    def _stage_driving_video(self, job_dir: Path) -> Path:
        """Stage the input clip under the filename expected by SCAIL-Pose."""
        target = job_dir / "driving.mp4"
        if Path(self._cfg.input_video).suffix.lower() != ".mp4":
            logger.warning(
                "SCAIL-Pose auto-prep expects driving.mp4; staging non-mp4 input %s as %s",
                self._cfg.input_video,
                target,
            )
        shutil.copy2(self._cfg.input_video, target)
        return target

    def _resolve_reference_image_source(self) -> str:
        """Return the image used as SCAIL-2's replacement reference."""
        if self._cfg.scail2_reference_image:
            return self._cfg.scail2_reference_image
        if self._cfg.characters:
            portraits = self._cfg.characters[0].portrait_paths
            if portraits:
                return portraits[0]
        raise ValueError("No SCAIL-2 reference image available for auto-prep")

    def _resolve_pose_repo_path(self) -> str:
        """Return the SCAIL-Pose checkout directory."""
        path = self._cfg._resolved_scail2_pose_repo_path()
        if not path:
            raise ValueError("SCAIL-Pose repo path is not configured")
        return path

    def _resolve_pose_preprocess_flags(
        self,
        total_frames: int,
        input_fps: float,
    ) -> tuple[bool, bool]:
        """Resolve the SCAIL-Pose replacement-mode flags for this clip.

        When auto-prep is driven from a chararep character mapping, use the
        existing detector/recognizer stack to decide whether `--matchnearest`
        is necessary and to reject clips that remain ambiguous.
        """
        matchnearest = bool(self._cfg.scail2_matchnearest)
        egocentric = bool(self._cfg.scail2_egocentric)

        if egocentric or not self._cfg.characters:
            return matchnearest, egocentric

        summary = self._scan_driving_identity(total_frames, input_fps)
        target_label = self._cfg.characters[0].source_label

        if summary["frames_with_target"] == 0:
            raise RuntimeError(
                "SCAIL-2 auto-prep could not identify target "
                f"'{target_label}' in sampled driving frames. "
                "Use clearer reference images, prepared assets, or a clip where "
                "the target face is visible earlier."
            )

        if summary["max_target_faces_in_frame"] > 1:
            raise RuntimeError(
                "SCAIL-2 auto-prep found multiple simultaneous faces matching "
                f"target '{target_label}' in sampled driving frames. "
                "Use prepared assets or a less ambiguous clip."
            )

        if summary["max_active_faces_in_frame"] > 2:
            raise RuntimeError(
                "SCAIL-2 auto-prep saw more than two simultaneous faces in the "
                "sampled driving frames. Upstream matchnearest only supports "
                "two-track selection, so refuse the clip early instead of "
                "guessing."
            )

        if summary["max_active_faces_in_frame"] > 1 and not matchnearest:
            matchnearest = True
            logger.info(
                "Auto-enabling scail2_matchnearest: preflight saw %d faces in a "
                "frame and exactly one matched target '%s'.",
                summary["max_active_faces_in_frame"],
                target_label,
            )

        return matchnearest, egocentric

    def _scan_driving_identity(
        self,
        total_frames: int,
        input_fps: float,
    ) -> dict[str, int]:
        """Sample early frames with the classic detector/recognizer stack."""
        from .face_detector import FaceDetector
        from .face_recognizer import FaceRecognizer

        detector = FaceDetector(self._cfg)
        recognizer = FaceRecognizer(self._cfg, backend=detector.backend)
        if not recognizer.targets:
            raise RuntimeError(
                "SCAIL-2 auto-prep could not build a usable recognition gallery "
                "from the provided character mapping."
            )

        read_limit, sample_stride, sample_budget = self._preflight_sampling_plan(
            total_frames,
            input_fps,
        )
        cap = cv2.VideoCapture(self._cfg.input_video)
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Cannot open input video for SCAIL-2 identity preflight: {self._cfg.input_video}"
            )

        summary = {
            "frames_sampled": 0,
            "frames_with_target": 0,
            "max_active_faces_in_frame": 0,
            "max_target_faces_in_frame": 0,
        }
        target_label = self._cfg.characters[0].source_label

        frame_idx = 0
        sampled = 0
        try:
            while frame_idx < read_limit and sampled < sample_budget:
                ok, frame = cap.read()
                if not ok:
                    break

                should_sample = (frame_idx % sample_stride) == 0
                frame_idx += 1
                if not should_sample:
                    continue

                tracked = [
                    tf
                    for tf in detector.detect(frame)
                    if getattr(tf, "age_since_seen", 0) == 0
                ]
                recognizer.identify_faces(tracked)

                target_faces = [
                    tf for tf in tracked if tf.identity_label == target_label
                ]
                summary["frames_sampled"] += 1
                sampled += 1
                summary["max_active_faces_in_frame"] = max(
                    summary["max_active_faces_in_frame"],
                    len(tracked),
                )
                summary["max_target_faces_in_frame"] = max(
                    summary["max_target_faces_in_frame"],
                    len(target_faces),
                )
                if target_faces:
                    summary["frames_with_target"] += 1

                if (
                    summary["max_target_faces_in_frame"] > 1
                    or summary["max_active_faces_in_frame"] > 2
                ):
                    break
        finally:
            cap.release()

        logger.info(
            "SCAIL-2 identity preflight: sampled=%d target_frames=%d "
            "max_faces=%d max_target_faces=%d",
            summary["frames_sampled"],
            summary["frames_with_target"],
            summary["max_active_faces_in_frame"],
            summary["max_target_faces_in_frame"],
        )
        return summary

    def _preflight_sampling_plan(
        self,
        total_frames: int,
        input_fps: float,
    ) -> tuple[int, int, int]:
        """Return (read_limit, sample_stride, sample_budget) for preflight."""
        fps = input_fps if input_fps > 0 else 24.0
        window = max(
            self._PREFLIGHT_MIN_SAMPLED_FRAMES,
            int(round(fps * self._PREFLIGHT_WINDOW_SECONDS)),
        )
        if total_frames > 0:
            read_limit = min(total_frames, window)
        else:
            read_limit = window

        sample_budget = min(self._PREFLIGHT_MAX_SAMPLED_FRAMES, read_limit)
        sample_budget = max(1, sample_budget)
        sample_stride = max(1, read_limit // sample_budget)
        return read_limit, sample_stride, sample_budget

    def _build_pose_command(
        self,
        job_dir: Path,
        *,
        matchnearest: bool,
        egocentric: bool,
    ) -> list[str]:
        """Build the SCAIL-Pose replacement preprocessing command."""
        pose_script = Path(self._resolve_pose_repo_path()) / "NLFPoseExtract" / "process_replacement.py"
        cmd = [
            sys.executable,
            str(pose_script),
            "--subdir",
            str(job_dir),
        ]
        if matchnearest:
            cmd.append("--matchnearest")
        if egocentric:
            cmd.append("--egocentric")
        if self._cfg.scail2_sam_text:
            cmd.extend(["--text", *self._cfg.scail2_sam_text])
        if self._cfg.scail2_sam3_model:
            cmd.extend(["--sam3_model", str(self._cfg.scail2_sam3_model)])
        return cmd

    @staticmethod
    def _read_image_bgr(image_path: str) -> cv2.typing.MatLike:
        """Read an image as BGR uint8, using PIL as a fallback when needed."""
        img = cv2.imread(str(image_path))
        if img is not None:
            return img
        from PIL import Image

        pil = Image.open(image_path).convert("RGB")
        return np.array(pil)[:, :, ::-1].copy()

    def _build_command(self, staged: _StagedInputs, prompt: str, output_path: Path) -> list[str]:
        generate_script = Path(self._cfg.scail2_repo_path) / "generate.py"
        cmd = [
            sys.executable,
            str(generate_script),
            "--model",
            self._cfg.scail2_model_name,
            "--ckpt_dir",
            str(self._cfg.scail2_ckpt_dir),
            "--scail_path",
            str(self._cfg.scail2_model_path),
            "--target_w",
            str(self._cfg.scail2_target_width),
            "--target_h",
            str(self._cfg.scail2_target_height),
            "--image",
            str(staged.image),
            "--mask_image",
            str(staged.mask_image),
            "--pose",
            str(staged.pose),
            "--mask_video",
            str(staged.mask_video),
        ]
        if staged.additional_images:
            cmd.append("--additional_ref_image")
            cmd.extend(str(path) for path in staged.additional_images)
        if staged.additional_masks:
            cmd.append("--additional_ref_mask_image")
            cmd.extend(str(path) for path in staged.additional_masks)
        cmd.extend(
            [
                "--prompt",
                prompt,
                "--save_file",
                str(output_path),
                "--replace_flag",
                "--sample_steps",
                str(self._cfg.scail2_sample_steps),
                "--sample_shift",
                str(self._cfg.scail2_sample_shift),
                "--sample_guide_scale",
                str(self._cfg.scail2_sample_guide_scale),
                "--sample_solver",
                self._cfg.scail2_sample_solver,
            ]
        )
        if self._cfg.scail2_offload_model:
            cmd.append("--offload_model")
        if self._cfg.scail2_extra_args:
            cmd.extend(str(arg) for arg in self._cfg.scail2_extra_args)
        return cmd

    def _build_generate_env(self) -> dict[str, str] | None:
        """Return a subprocess environment for generate.py when overrides exist."""
        if not self._cfg.scail2_env:
            return None
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self._cfg.scail2_env.items()})
        return env

    def _warn_or_raise_vram_risk(self) -> None:
        """Warn or fail fast for obviously risky model-size/resolution pairs."""
        risk_message = self._describe_vram_risk()
        if not risk_message:
            return
        if self._cfg.scail2_fail_on_vram_risk:
            raise RuntimeError(risk_message)
        logger.warning(risk_message)

    def _describe_vram_risk(self) -> str | None:
        """Return a warning message for high-risk VRAM combinations."""
        model_path = self._cfg.scail2_model_path
        if not model_path:
            return None

        try:
            model_size_bytes = Path(model_path).stat().st_size
        except OSError:
            return None

        model_gib = model_size_bytes / _BYTES_PER_GIB
        target_pixels = self._cfg.scail2_target_width * self._cfg.scail2_target_height
        risk_label: str | None = None
        risk_text = ""
        if (
            model_gib >= self._HIGH_RISK_MODEL_GIB
            and target_pixels >= self._HIGH_RISK_TARGET_PIXELS
        ):
            risk_label = "high"
            risk_text = "is likely to exceed available VRAM on many 24-32 GB class GPUs"
        elif (
            model_gib >= self._WARN_RISK_MODEL_GIB
            and target_pixels > self._WARN_RISK_TARGET_PIXELS
        ):
            risk_label = "elevated"
            risk_text = "may exceed available VRAM on some GPUs"

        if not risk_label:
            return None

        hint = (
            "Try --scail2-memory-preset low-vram, lower "
            "--scail2-target-width/--scail2-target-height, keep model offload "
            "enabled, or pass upstream memory flags via --scail2-extra-arg/--scail2-env."
        )
        if not self._cfg.scail2_offload_model:
            hint += " Re-enable offload if upstream supports it."
        return (
            "SCAIL-2 VRAM preflight (%s risk): %.1f GiB checkpoint at %dx%d %s %s"
            % (
                risk_label,
                model_gib,
                self._cfg.scail2_target_width,
                self._cfg.scail2_target_height,
                risk_text,
                hint,
            )
        )

    @staticmethod
    def _probe_video(path: str) -> tuple[int, float]:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return 0, 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            return total_frames, fps
        finally:
            cap.release()

    @staticmethod
    def _format_subprocess_failure(result: subprocess.CompletedProcess) -> str:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or "no subprocess output captured"
        if len(details) > 1000:
            details = details[-1000:]
        return f"SCAIL-2 generate.py failed with exit code {result.returncode}: {details}"

    @staticmethod
    def _format_pose_failure(result: subprocess.CompletedProcess) -> str:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or "no subprocess output captured"
        if len(details) > 1000:
            details = details[-1000:]
        return f"SCAIL-Pose process_replacement.py failed with exit code {result.returncode}: {details}"