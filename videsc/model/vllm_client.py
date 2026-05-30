"""
videsc.model.vllm_client – OpenAI-compatible client for vLLM servers.

vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint that accepts
vision messages with base64-encoded images or image URLs.  This module wraps
the ``openai`` library to provide a simple interface for sending video frames
(as images) and prompts to a remote vLLM server.
"""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for communicating with a vLLM server via its OpenAI-compatible API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        model: str = "default",
        api_key: str = "EMPTY",
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ):
        from openai import OpenAI as _OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        base_url = f"http://{host}:{port}/v1"
        self.client = _OpenAI(base_url=base_url, api_key=api_key)
        logger.info("VLLMClient: connected to %s  model=%s", base_url, model)

    def _encode_frame(self, frame: Image.Image, max_size: int = 1280) -> str:
        """Encode a PIL Image to a base64 JPEG string, resizing if needed."""
        # Resize to fit within max_size while preserving aspect ratio
        w, h = frame.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = BytesIO()
        frame.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def build_content(
        self,
        frames: List[Image.Image],
        prompt: str,
        max_size: int = 1280,
    ) -> List[Dict[str, Any]]:
        """Build the OpenAI-compatible content list with images and text.

        Args:
            frames: List of PIL Images (video frames).
            prompt: The text prompt to send.
            max_size: Maximum edge size for encoding frames.

        Returns:
            List of content parts suitable for the messages API.
        """
        content: List[Dict[str, Any]] = []

        for frame in frames:
            b64 = self._encode_frame(frame, max_size=max_size)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                },
            })

        content.append({"type": "text", "text": prompt})
        return content

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send messages to the vLLM server and return the generated text.

        Args:
            messages: OpenAI-style chat messages.
            max_tokens: Override instance max_tokens.
            temperature: Override instance temperature.

        Returns:
            The generated text string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        text = response.choices[0].message.content or ""
        logger.debug("VLLMClient.generate: received %d chars", len(text))
        return text

    def describe_frames(
        self,
        frames: List[Image.Image],
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_size: int = 1280,
    ) -> str:
        """High-level helper: encode frames + prompt and generate a description.

        Args:
            frames: Video frames as PIL Images.
            prompt: User prompt text.
            system: Optional system prompt.
            max_tokens: Override max tokens for generation.
            temperature: Override temperature.
            max_size: Max image edge for encoding.

        Returns:
            Generated description text.
        """
        content = self.build_content(frames, prompt, max_size=max_size)

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        return self.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
