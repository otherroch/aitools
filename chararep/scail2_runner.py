"""SCAIL-2 runner – end-to-end character animation model wrapper."""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SCAIL2Config:
    """SCAIL-2 specific configuration."""

    enabled: bool = False
    model_path: str = ""
    mode: str = "replacement"
    resolution: str = "704p"
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
        try:
            logger.info("Loading SCAIL-2 model from %s", self.cfg.model_path)
            # TODO: Implement actual model loading logic
            # This will load the SCAIL-2 DiT model, VAE, and optionally
            # the T5 encoder and other auxiliary modules.
            self._model = True  # Placeholder
            self._loaded = True
            logger.info("SCAIL-2 model loaded successfully")
            return True

        except Exception as e:
            logger.error("Failed to load SCAIL-2 model: %s", e)
            return False

    def generate(
        self,
        reference_image,
        driving_frames,
    ):
        """Generate animated/replace frames from reference image + driving video.

        Args:
            reference_image: Reference character image (PIL, ndarray, or path).
            driving_frames: List of driving video frames (BGR uint8).

        Returns:
            List of output frames (BGR uint8).
        """
        if not self._loaded:
            raise RuntimeError("SCAIL-2 not loaded. Call load() first.")

        logger.info(
            "SCAIL-2 generate called with %d frames",
            len(driving_frames),
        )

        # In a full implementation, we'd call the actual SCAIL-2 model here
        return []

    def is_available(self) -> bool:
        """Check if SCAIL-2 is available and loaded."""
        return self._loaded and self._model is not None