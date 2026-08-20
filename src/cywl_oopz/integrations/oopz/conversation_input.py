"""Adapt OOPZ message segments into project-owned Agent input parts."""

from __future__ import annotations

from typing import Any

from oopz_sdk.models.segment import Image, Text

from cywl_oopz.features.agent.input import AgentUserInput, ImageInputPart, TextInputPart


class OopzConversationInputFactory:
    """Project-owned boundary for the SDK's ordered message segment model."""

    def from_message(self, message: Any) -> AgentUserInput:
        """Convert text and image segments without downloading external media."""
        parts: list[TextInputPart | ImageInputPart] = []
        segments = getattr(message, "segments", None)
        if segments is None:
            text = str(
                getattr(message, "plain_text", "")
                or getattr(message, "text", "")
                or getattr(message, "content", "")
            )
            return AgentUserInput.from_parts([TextInputPart(text)])
        for segment in segments:
            if isinstance(segment, Text):
                text = segment.plain_text or segment.text
                if text.strip():
                    parts.append(TextInputPart(text))
                continue
            if isinstance(segment, Image):
                parts.append(
                    ImageInputPart(
                        source_file_key=segment.file_key,
                        source_url=segment.url,
                        width=segment.width,
                        height=segment.height,
                        byte_size=segment.file_size,
                        sha256=segment.hash,
                        animated=segment.animated,
                    )
                )
        return AgentUserInput.from_parts(parts)
