"""Bounded, cancellable ingestion of current-turn Agent images."""

from __future__ import annotations

import asyncio
import hashlib
import io
import warnings
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, UnidentifiedImageError

from cywl_oopz.core.errors import UserRequestError
from cywl_oopz.features.chat.progress import (
    ConversationProgressEvent,
    ProgressKind,
    ProgressSink,
    emit_progress,
)

from .input import AgentUserInput, ImageInputPart


@dataclass(frozen=True, slots=True)
class AgentImagePolicy:
    """Explicit resource limits for one user-supplied image input."""

    max_images: int = 4
    max_image_bytes: int = 8 * 1024 * 1024
    max_total_bytes: int = 16 * 1024 * 1024
    max_pixels: int = 25_000_000
    max_parallel_downloads: int = 2

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.max_images,
                self.max_image_bytes,
                self.max_total_bytes,
                self.max_pixels,
                self.max_parallel_downloads,
            )
        ):
            raise ValueError("Agent image policy limits must be positive")


class ImageContentLoader(Protocol):
    """Integration-owned async source for one trusted OOPZ image reference."""

    async def load(self, image: ImageInputPart, *, maximum_bytes: int) -> bytes:
        """Return bounded source bytes or raise a safe request error."""


class ImageContentValidator:
    """Validate image bytes away from the event loop and normalize media metadata."""

    _MEDIA_TYPES = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }

    def __init__(self, *, max_pixels: int) -> None:
        if max_pixels <= 0:
            raise ValueError("Maximum image pixels must be positive")
        self._max_pixels = max_pixels

    async def validate(self, data: bytes) -> ImageInputPart:
        """Return only validated image bytes and content-derived metadata."""
        return await asyncio.to_thread(self._validate_sync, data)

    def _validate_sync(self, data: bytes) -> ImageInputPart:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    image_format = source.format or ""
                    width, height = source.size
                    frames = getattr(source, "n_frames", 1)
                    source.verify()
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise UserRequestError(
                "unsupported_image",
                "这张图片无法识别，请换一张 PNG、JPEG、WebP 或静态 GIF。",
            ) from exc
        media_type = self._MEDIA_TYPES.get(image_format.upper())
        if media_type is None or frames != 1:
            raise UserRequestError(
                "unsupported_image",
                "这张图片无法识别，请换一张 PNG、JPEG、WebP 或静态 GIF。",
            )
        if width <= 0 or height <= 0 or width * height > self._max_pixels:
            raise UserRequestError(
                "image_too_large",
                "图片尺寸太大了，请换一张更小的图片。",
            )
        return ImageInputPart(
            data=data,
            media_type=media_type,
            width=width,
            height=height,
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )


class AgentMediaIngestService:
    """Resolve and validate images for the current run without persistence."""

    def __init__(
        self,
        loader: ImageContentLoader,
        policy: AgentImagePolicy,
        validator: ImageContentValidator | None = None,
    ) -> None:
        self._loader = loader
        self._policy = policy
        self._validator = validator or ImageContentValidator(max_pixels=policy.max_pixels)

    async def ingest(
        self,
        user_input: AgentUserInput,
        progress: ProgressSink | None = None,
    ) -> AgentUserInput:
        """Resolve image references concurrently while preserving input order."""
        images = user_input.images
        if not images:
            return user_input
        if len(images) > self._policy.max_images:
            raise UserRequestError(
                "image_limit_exceeded",
                f"这条消息里的图片太多了，最多支持 {self._policy.max_images} 张。",
            )
        await emit_progress(progress, ConversationProgressEvent(ProgressKind.MEDIA_LOADING))
        semaphore = asyncio.Semaphore(self._policy.max_parallel_downloads)

        async def resolve(image: ImageInputPart) -> ImageInputPart:
            if image.resolved:
                return image
            async with semaphore:
                data = await self._loader.load(
                    image,
                    maximum_bytes=self._policy.max_image_bytes,
                )
            if len(data) > self._policy.max_image_bytes:
                raise UserRequestError(
                    "image_too_large",
                    "图片文件太大了，请换一张更小的图片。",
                )
            return await self._validator.validate(data)

        resolved_images = await asyncio.gather(*(resolve(image) for image in images))
        total_bytes = sum(image.actual_byte_size for image in resolved_images)
        if total_bytes > self._policy.max_total_bytes:
            raise UserRequestError(
                "image_total_too_large",
                "这条消息里的图片总大小太大了，请减少图片数量或换更小的图片。",
            )
        iterator = iter(resolved_images)
        resolved_parts = tuple(
            next(iterator) if isinstance(part, ImageInputPart) else part
            for part in user_input.parts
        )
        result = user_input.with_parts(resolved_parts)
        await emit_progress(progress, ConversationProgressEvent(ProgressKind.MEDIA_READY))
        return result
