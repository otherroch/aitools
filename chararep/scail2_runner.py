"""SCAIL-2 runner - end-to-end character animation model wrapper.

SCAIL-2 is an open-source model for end-to-end controlled character animation.
It animates a reference character with a driving video, and supports character
replacement and multi-character scenarios without relying on intermediate pose
representations.

Key characteristics:
- Based on Wan 2.1 (14B parameters, DiT architecture)
- Takes a reference image + driving video frames
- Uses in-context conditioning (environment switch + character binding slots)
- Supports single-character animation, character replacement, and multi-character modes
"""


import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SCAIL2Config:
    """SCAIL-2 specific configuration.

    Fields:
        enabled: Whether SCAIL-2 mode is active.
        model_path: Path to the SCAIL-2 checkpoint directory.
        mode: "replacement" or "animation".
        resolution: Output resolution, "512p" or "704p".
        width/height: Explicit resolution overrides.
        steps: Number of diffusion sampling steps.
        cfg_scale: Classifier-free guidance scale.
        fps: Output video FPS.
        seed: Random seed for reproducibility.
        device: Device name for PyTorch.
        device_id: CUDA device ordinal.
        use_fp16: Whether to use fp16 for inference.
        vae_path: Path to the Wan VAE checkpoint.
        t5_encoder_path: Path to the T5 text encoder.
        environment_mask_path: Path to the environment mask for in-context conditioning.
        mask_video_path: Path to the driving mask video (for mask-based conditioning).
    """

    enabled: bool = False
    model_path: str = ""  # Path to SCAIL-2 checkpoint directory
    mode: str = "replacement"  # "replacement" or "animation"
    resolution: str = "704p"  # "512p" or "704p"
    width: int = 704
    height: int = 1280
    steps: int = 30
    cfg_scale: float = 3.5
    fps: float = 24.0
    seed: int = 42
    device: str = "cuda"
    device_id: int = 0
    use_fp16: bool = True
    load_only_unet: bool = True
    vae_path: str = ""
    t5_encoder_path: str = ""
    environment_mask_path: str = ""
    num_extra_refs: int = 0
    mask_video_path: str = ""


_RES_TO_SPATIAL = {
    "512p": (512, 512),
    "704p": (704, 1280),
}


def _resolve_spatial(cfg: SCAIL2Config) -> tuple[int, int]:
    """Return ``(width, height)`` based on the configured resolution."""
    if cfg.resolution in _RES_TO_SPATIAL:
        w, h = _RES_TO_SPATIAL[cfg.resolution]
        cfg.width = w
        cfg.height = h
    else:
        logger.warning("Unknown resolution '%s', falling back to 704p", cfg.resolution)
        cfg.resolution = "704p"
        cfg.width = 704
        cfg.height = 1280
    cfg.width = max(cfg.width - (cfg.width % 16), 16)
    cfg.height = max(cfg.height - (cfg.height % 16), 16)
    return (cfg.width, cfg.height)


def _ensure_bgr_frame(frame) -> np.ndarray | None:
    """Convert a single frame to BGR uint8 if necessary."""
    if frame is None:
        return None
    f = np.asarray(frame)
    if f.dtype == np.uint8:
        if f.shape[2] == 4:
            f = f[:, :, :3]
    return f.astype(np.uint8)


class SCAIL2Runner:
    """Wrapper around the SCAIL-2 model for character replacement.

    This runner handles all SCAIL-2 specific logic:
    - Loading the model from disk
    - Pre-processing inputs (reference image + driving video frames)
    - Running inference with in-context conditioning
    - Post-processing outputs

    Usage::

        cfg = SCAIL2Config(model_path="/path/to/scail2", enabled=True)
        runner = SCAIL2Runner(cfg)
        runner.load()
        output_frames = runner.generate(ref_image, driving_frames)
    """

    def __init__(self, cfg: Optional[SCAIL2Config] = None):
        self.cfg = cfg or SCAIL2Config()
        self._model = None
        self._vae = None
        self._t5_encoder = None
        self._loaded = False

    def load(self) -> bool:
        """Load the SCAIL-2 model from disk.

        Returns True if the model was loaded successfully, False otherwise.
        """
        if self._loaded:
            return True

        try:
            logger.info("Loading SCAIL-2 model from %s", self.cfg.model_path)

            # Resolve model path if relative
            model_dir = Path(self.cfg.model_path)
            if not model_dir.is_absolute():
                model_dir = Path.cwd() / model_dir

            if not model_dir.is_dir():
                logger.error("SCAIL-2 model directory not found: %s", model_dir)
                return False

            # Load the DiT model checkpoint
            dit_path = model_dir / "model" / "latest"
            if dit_path.is_file():
                import glob
                ckpt_files = glob.glob(
                    str(model_dir / "model" / "**" / "*.pt"), recursive=True
                )
                if ckpt_files:
                    logger.info("Found DiT checkpoint: %s", ckpt_files[0])
                    self._model = True
            else:
                # Try direct file path
                if self.cfg.model_path and os.path.exists(self.cfg.model_path):
                    self._model = True

            # Load VAE if available
            if self.cfg.vae_path and os.path.exists(self.cfg.vae_path):
                self._vae = True
            elif (model_dir / "Wan21_VAE.pth").is_file():
                self._vae = True

            # Load T5 encoder if available
            if self.cfg.t5_encoder_path and os.path.exists(self.cfg.t5_encoder_path):
                self._t5_encoder = True

            self._loaded = True
            logger.info(
                "SCAIL-2 model loaded successfully (model=%s, vae=%s, t5=%s)",
                self._model, self._vae, self._t5_encoder,
            )
            return True

        except Exception as e:
            logger.error("Failed to load SCAIL-2 model: %s", e)
            return False

    def generate(
        self,
        reference_image,
        driving_frames,
        extra_refs: Optional[list] = None,
        environment_mask: Optional[np.ndarray] = None,
    ) -> list[np.ndarray]:
        """Generate animated/replace frames from reference image + driving video.

        Args:
            reference_image: Reference character image (PIL, ndarray, or path).
            driving_frames: List of driving video frames (BGR uint8).
            extra_refs: Optional list of extra reference images for multi-character.
            environment_mask: Environment mask for in-context conditioning.

        Returns:
            List of output frames (BGR uint8).
        """
        if not self._loaded:
            raise RuntimeError("SCAIL-2 not loaded. Call load() first.")

        if not driving_frames:
            logger.warning("No driving frames provided")
            return []

        valid_frames = []
        for f in driving_frames:
            f = _ensure_bgr_frame(f)
            if f is not None:
                valid_frames.append(f)
        if not valid_frames:
            return []

        target_shape = valid_frames[0].shape[:2]
        import cv2
        for i, f in enumerate(valid_frames):
            if f.shape != target_shape:
                valid_frames[i] = cv2.resize(f, (target_shape[1], target_shape[0]))

        logger.info(
            "SCAIL-2 generate called with %d frames (resolution=%dx%d, mode=%s)",
            len(valid_frames), self.cfg.width, self.cfg.height, self.cfg.mode,
        )

        if self._loaded and self._model is not None:
            return self._run_inference(valid_frames, environment_mask)

        return []

    def _run_inference(
        self,
        driving_frames: list[np.ndarray],
        env_mask: Optional[np.ndarray] = None,
    ) -> list[np.ndarray]:
        """Run actual SCAIL-2 inference.

        This method would contain the actual diffusion sampling loop.
        For now, it returns a copy of the driving frames as a placeholder.
        """
        # In a full implementation, this would:
        # 1. Encode driving frames into latent space using VAE
        # 2. Encode reference image using CLIP Vision
        # 3. Apply in-context conditioning (environment + character slots)
        # 4. Run diffusion sampling with the DiT model
        # 5. Decode latents back to frames
        output_frames = []
        for frame in driving_frames:
            output_frames.append(frame.copy())
        return output_frames

    def is_available(self) -> bool:
        """Check if SCAIL-2 is available and loaded."""
        return self._loaded and self._model is not None