"""Trusted OOPZ mention extraction for Agent invocations."""

from __future__ import annotations

from typing import Any

from cywl_oopz.commands.models import CommandRequest
from cywl_oopz.features.chat.models import ChatInvocation


class OopzChatInvocationFactory:
    """Exclude the bot and sender while preserving stable mentioned person IDs."""

    def __init__(self, bot_person_id: str) -> None:
        normalized = bot_person_id.strip()
        if not normalized:
            raise ValueError("OOPZ bot person ID must not be empty")
        self._bot_person_id = normalized

    def from_context(self, context: Any) -> ChatInvocation:
        return ChatInvocation.from_oopz_context(
            context,
            excluded_person_ids=(self._bot_person_id,),
        )

    def from_request(self, request: CommandRequest) -> ChatInvocation:
        excluded = {request.actor.person_id, self._bot_person_id}
        mentioned: list[str] = []
        for mention in request.mentions:
            if mention.person_id in excluded or mention.person_id in mentioned:
                continue
            mentioned.append(mention.person_id)
        return ChatInvocation(
            source_message_id=request.source.message_id,
            transport_channel_id=request.location.channel_id,
            mentioned_person_ids=tuple(mentioned),
        )
