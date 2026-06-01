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

import requests as _requests
from PIL import Image

logger = logging.getLogger(__name__)


def _append_error_text(parts: List[str], value: object) -> None:
    """Collect string fragments from an exception payload."""
    if value is None:
        return
    if isinstance(value, str):
        parts.append(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _append_error_text(parts, item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _append_error_text(parts, item)


def _extract_error_text(exc: Exception) -> str:
    """Flatten common exception fields into one searchable string."""
    parts: List[str] = []
    _append_error_text(parts, str(exc))
    _append_error_text(parts, getattr(exc, "message", None))
    _append_error_text(parts, getattr(exc, "body", None))

    unique_parts: List[str] = []
    for part in parts:
        if part and part not in unique_parts:
            unique_parts.append(part)
    return "\n".join(unique_parts).lower()


def _should_retry_with_fewer_frames(exc: Exception) -> bool:
    """Return True when the server rejected the request due to multimodal truncation."""
    message = _extract_error_text(exc)
    return (
        "token count" in message
        and ("truncation='max_length'" in message or 'truncation="max_length"' in message)
        and ("image" in message or "video" in message)
    ) or (
        "failed to apply qwen" in message
        and "processor" in message
        and "image" in message
    )


def _downsample_frames(frames: List[Image.Image], target_count: int) -> List[Image.Image]:
    """Keep an even spread of frames while reducing the request size."""
    if target_count >= len(frames):
        return list(frames)
    if target_count <= 0 or not frames:
        return []
    if target_count == 1:
        return [frames[len(frames) // 2]]

    last_index = len(frames) - 1
    step = last_index / float(target_count - 1)
    indices = []
    seen = set()

    for position in range(target_count):
        idx = int(round(position * step))
        idx = max(0, min(last_index, idx))
        if idx in seen:
            continue
        indices.append(idx)
        seen.add(idx)

    if len(indices) < target_count:
        for idx in range(len(frames)):
            if idx in seen:
                continue
            indices.append(idx)
            seen.add(idx)
            if len(indices) == target_count:
                break

    indices.sort()
    return [frames[idx] for idx in indices]


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
        top_p: float = 0.95,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI as _OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = f"http://{host}:{port}/v1"
        self.api_key = api_key
        self.client = _OpenAI(base_url=self.base_url, api_key=api_key)
        self._verify_connection()
        logger.info("VLLMClient: connected to %s  model=%s", self.base_url, model)

    def _verify_connection(self) -> None:
        """Verify the vLLM server is reachable by querying the /models endpoint."""
        try:
            models_url = f"{self.base_url}/models"
            headers = {}
            if self.api_key and self.api_key != "EMPTY":
                headers["Authorization"] = "Bearer " + self.api_key
            resp = _requests.get(models_url, headers=headers, timeout=10)
            resp.raise_for_status()
            logger.debug("VLLMClient: server verification OK (%s)", models_url)
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to vLLM server at {self.base_url}: {e}"
            ) from e

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
        top_p: Optional[float] = None,
    ) -> str:
        """Send messages to the vLLM server and return the generated text.

        Args:
            messages: OpenAI-style chat messages.
            max_tokens: Override instance max_tokens.
            temperature: Override instance temperature.
            top_p: Override instance top_p.

        Returns:
            The generated text string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
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
        current_frames = list(frames)

        while True:
            content = self.build_content(current_frames, prompt, max_size=max_size)

            messages: List[Dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})

            try:
                return self.generate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                if not _should_retry_with_fewer_frames(exc) or len(current_frames) <= 1:
                    raise

                next_count = max(1, len(current_frames) // 2)
                logger.warning(
                    "VLLMClient.describe_frames: retrying with %d frame(s) after multimodal truncation at %d frame(s)",
                    next_count,
                    len(current_frames),
                )
                current_frames = _downsample_frames(current_frames, next_count)
