"""Adapt OOPZ message segments into project-owned Agent input parts."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from oopz_sdk.models.attachment import Attachment, ImageAttachment
from oopz_sdk.models.segment import Image, Text, parse_message_segments

from cywl_oopz.features.agent.input import AgentUserInput, ImageInputPart, TextInputPart

logger = logging.getLogger(__name__)


class OopzConversationInputFactory:
    """Project-owned boundary for the SDK's ordered message segment model."""

    def from_message(self, message: Any, *, source_text: str | None = None) -> AgentUserInput:
        """Convert text and image segments without downloading external media."""
        parts: list[TextInputPart | ImageInputPart] = []
        source_text_override = source_text
        source_text = source_text if source_text is not None else self._source_text(message)
        segments = getattr(message, "segments", None)
        if (
            source_text_override is not None
            or segments is None
            or (
                "![IMAGE" in source_text
                and not any(isinstance(segment, Image) for segment in segments)
            )
        ):
            segments = self._parse_segments(message, source_text)
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
        user_input = AgentUserInput.from_parts(parts)
        logger.debug(
            "Projected OOPZ conversation input: parts=%s images=%s has_image_token=%s",
            len(user_input.parts),
            len(user_input.images),
            "![IMAGE" in source_text,
        )
        if "![IMAGE" in source_text and not user_input.has_images:
            logger.warning(
                "OOPZ image token was projected as text: attachments=%s",
                len(getattr(message, "attachments", ()) or ()),
            )
        return user_input

    @staticmethod
    def _source_text(message: Any) -> str:
        """Prefer raw message text so OOPZ image tokens remain parseable."""
        for attribute in ("text", "content", "plain_text"):
            value = getattr(message, attribute, "")
            if value:
                return str(value)
        return ""

    @classmethod
    def _parse_segments(cls, message: Any, source_text: str) -> list[Any]:
        """Rebuild segments for event proxies that omit the SDK cached property."""
        attachments = cls._attachments(message)
        mentions = list(getattr(message, "mention_list", ()) or ())
        try:
            segments = parse_message_segments(
                source_text,
                attachments=attachments,
                mention_list=mentions,
            )
        except Exception:
            logger.warning(
                "Could not reconstruct OOPZ message segments: has_image_token=%s attachments=%s",
                "![IMAGE" in source_text,
                len(attachments),
                exc_info=True,
            )
            return []
        logger.debug(
            "Reconstructed OOPZ message segments: segments=%s images=%s",
            len(segments),
            sum(isinstance(segment, Image) for segment in segments),
        )
        return segments

    @staticmethod
    def _attachments(message: Any) -> list[Any]:
        """Normalize raw attachment mappings before calling the SDK parser."""
        result: list[Any] = []
        for attachment in list(getattr(message, "attachments", ()) or ()):
            if isinstance(attachment, Mapping):
                try:
                    attachment = Attachment.parse(attachment)
                except Exception:
                    logger.debug(
                        "Skipping malformed OOPZ attachment during input parsing", exc_info=True
                    )
                    continue
            elif not isinstance(attachment, ImageAttachment):
                attachment_type = str(
                    getattr(attachment, "attachment_type", "")
                    or getattr(attachment, "attachmentType", "")
                ).upper()
                if attachment_type == "IMAGE":
                    try:
                        payload = (
                            attachment.model_dump(by_alias=True)
                            if callable(getattr(attachment, "model_dump", None))
                            else {
                                "attachmentType": "IMAGE",
                                "fileKey": getattr(attachment, "file_key", ""),
                                "url": getattr(attachment, "url", ""),
                                "fileSize": getattr(attachment, "file_size", 0),
                                "hash": getattr(attachment, "hash", ""),
                                "width": getattr(attachment, "width", 0),
                                "height": getattr(attachment, "height", 0),
                                "animated": getattr(attachment, "animated", False),
                            }
                        )
                        attachment = ImageAttachment.model_validate(payload)
                    except Exception:
                        logger.debug(
                            "Skipping foreign OOPZ image attachment during input parsing",
                            exc_info=True,
                        )
                        continue
            result.append(attachment)
        return result
