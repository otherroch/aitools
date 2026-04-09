"""chararep – Video character (face) replacement pipeline.

Replaces up to 3 characters in a video with different identities using
portrait photos as the source of the new face.  Optimised for NVIDIA
GPUs with CUDA support.

Usage::

    chararep -i input.mp4 -o output.mp4 \\
        --char originals/villain replacements/villain \\
        --char originals/hero   replacements/hero

See ``chararep --help`` for the full CLI reference.
"""

from .config import CharacterMapping, PipelineConfig

__all__ = [
    "CharacterMapping",
    "PipelineConfig",
    "CharacterReplacementPipeline",
]


def __getattr__(name: str):
    """Lazy import of heavy pipeline class to avoid requiring GPU libs at import time."""
    if name == "CharacterReplacementPipeline":
        from .pipeline import CharacterReplacementPipeline
        return CharacterReplacementPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

