"""Provider-neutral user input parts for Agent conversations.

The integration layer may attach an OOPZ image URL while it is still an
unresolved reference.  Media ingestion replaces that reference with bytes
before the input reaches a provider adapter.  Keeping both states in one
small value object avoids leaking SDK models into the Agent domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
from uuid import UUID


IMAGE_ONLY_PROMPT = (
    "请查看用户附带的图片，并根据图片内容自然回应；"
    "如果用户没有提出问题，先简要描述你看到的内容。"
)


@dataclass(frozen=True, slots=True)
class TextInputPart:
    """One text segment in the original message order."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Text input part must not be empty")


@dataclass(frozen=True, slots=True)
class ImageInputPart:
    """An image reference, optionally resolved to validated runtime bytes."""

    data: bytes | None = field(default=None, repr=False)
    media_type: str = ""
    width: int = 0
    height: int = 0
    byte_size: int = 0
    sha256: str = ""
    asset_id: UUID | None = None
    source_file_key: str = ""
    source_url: str = ""
    animated: bool = False

    _MAX_SOURCE_URL_LENGTH: ClassVar[int] = 4096

    def __post_init__(self) -> None:
        if self.data is None and not (self.source_file_key.strip() or self.source_url.strip()):
            raise ValueError("Image input part needs a source reference or data")
        if self.data is not None and not isinstance(self.data, bytes):
            raise TypeError("Image input bytes must be bytes")
        if self.source_url and len(self.source_url) > self._MAX_SOURCE_URL_LENGTH:
            raise ValueError("Image input source URL is too long")
        if any(value < 0 for value in (self.width, self.height, self.byte_size)):
            raise ValueError("Image input dimensions and byte size must not be negative")
        if self.data is not None and self.byte_size not in {0, len(self.data)}:
            raise ValueError("Image input byte size does not match data")

    @property
    def resolved(self) -> bool:
        """Whether the image is ready to be encoded for a provider request."""
        return self.data is not None and bool(self.media_type)

    @property
    def actual_byte_size(self) -> int:
        """Return the trusted runtime size when bytes are available."""
        return len(self.data) if self.data is not None else self.byte_size


InputPart = TextInputPart | ImageInputPart


@dataclass(frozen=True, slots=True)
class AgentUserInput:
    """Ordered user content shared by handlers, Agent runs, and providers."""

    parts: tuple[InputPart, ...]
    implicit_prompt: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.parts, tuple):
            raise TypeError("Agent input parts must be a tuple")
        if not self.parts:
            raise ValueError("Agent input must contain text or an image")
        if not any(isinstance(part, (TextInputPart, ImageInputPart)) for part in self.parts):
            raise ValueError("Agent input contains no supported parts")
        if self.implicit_prompt and self.text:
            raise ValueError("Implicit image prompt is only valid for image-only input")

    @classmethod
    def from_parts(cls, parts: list[InputPart] | tuple[InputPart, ...]) -> "AgentUserInput":
        """Build an input while dropping empty text-only SDK segments."""
        normalized = tuple(
            part
            for part in parts
            if not isinstance(part, TextInputPart) or part.text.strip()
        )
        if not normalized:
            raise ValueError("Agent input must contain text or an image")
        has_image = any(isinstance(part, ImageInputPart) for part in normalized)
        has_text = any(isinstance(part, TextInputPart) for part in normalized)
        return cls(normalized, implicit_prompt=has_image and not has_text)

    @property
    def text(self) -> str:
        """Return only user-authored text, excluding the image-only helper prompt."""
        return "".join(part.text for part in self.parts if isinstance(part, TextInputPart)).strip()

    @property
    def prompt(self) -> str:
        """Return the provider-facing text envelope for this input."""
        return self.text or (IMAGE_ONLY_PROMPT if self.implicit_prompt else "")

    @property
    def images(self) -> tuple[ImageInputPart, ...]:
        """Return image parts in their original order."""
        return tuple(part for part in self.parts if isinstance(part, ImageInputPart))

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def resolved_images(self) -> bool:
        return all(part.resolved for part in self.images)

    def with_parts(self, parts: tuple[InputPart, ...]) -> "AgentUserInput":
        """Return a copy retaining the image-only prompt marker."""
        return AgentUserInput(parts=parts, implicit_prompt=self.implicit_prompt)
