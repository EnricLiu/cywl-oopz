"""OOPZ message projection into framework-neutral command requests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cywl_oopz.commands.models import (
    CommandActor,
    CommandLocation,
    CommandMention,
    CommandRequest,
    CommandScope,
    CommandSource,
    CommandTarget,
    CommandText,
    CommandTrigger,
)
from cywl_oopz.commands.parsing import CommandTextParser

from .command_responses import OopzCommandResponder


class OopzCommandRequestFactory:
    """Build one trusted request before feature dispatch starts."""

    def __init__(
        self,
        parser: CommandTextParser,
        reference_projector: Callable[[object], object | None] | None = None,
    ) -> None:
        self._parser = parser
        self._reference_projector = reference_projector

    def from_message(self, message: Any, context: Any) -> CommandRequest | None:
        """Read raw OOPZ text once and retain structured message metadata."""
        text = self._parser.parse(self._raw_text(message))
        if text is None:
            return None
        return self._project(message, context, text)

    def _project(
        self,
        message: Any,
        context: Any,
        text: CommandText,
    ) -> CommandRequest:

        event = getattr(context, "event", None)
        is_private = bool(getattr(event, "is_private", False))
        actor_id = str(getattr(message, "sender_id", "")).strip()
        area_id = str(getattr(message, "area", "")).strip()
        channel_id = str(getattr(message, "channel", "")).strip()
        location = (
            CommandLocation(
                CommandScope.PRIVATE,
                channel_id=channel_id,
                target_person_id=actor_id,
            )
            if is_private
            else CommandLocation(
                CommandScope.CHANNEL,
                area_id=area_id,
                channel_id=channel_id,
            )
        )
        reference_id = str(getattr(message, "reference_message_id", "")).strip()
        return CommandRequest(
            trigger=CommandTrigger.TEXT,
            actor=CommandActor(actor_id),
            location=location,
            source=CommandSource(
                message_id=str(getattr(message, "message_id", "")),
                client_message_id=str(getattr(message, "client_message_id", "")),
                timestamp=str(getattr(message, "timestamp", "")),
            ),
            responder=OopzCommandResponder(context),
            text=text,
            target=(
                CommandTarget(
                    reference_id,
                    evidence=(
                        self._reference_projector(getattr(message, "reference_message", None))
                        if self._reference_projector is not None
                        else None
                    ),
                )
                if reference_id
                else None
            ),
            mentions=self._mentions(message),
        )

    @staticmethod
    def _raw_text(message: Any) -> str:
        return str(
            getattr(message, "text", "")
            or getattr(message, "content", "")
            or getattr(message, "plain_text", "")
        )

    @staticmethod
    def _mentions(message: Any) -> tuple[CommandMention, ...]:
        mentions: list[CommandMention] = []
        for mention in getattr(message, "mention_list", ()) or ():
            person_id = str(getattr(mention, "person", "")).strip()
            if not person_id:
                continue
            mentions.append(
                CommandMention(
                    person_id,
                    is_bot=bool(getattr(mention, "is_bot", False)),
                    bot_type=str(getattr(mention, "bot_type", "")),
                )
            )
        return tuple(mentions)
