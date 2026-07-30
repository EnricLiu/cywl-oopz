"""Trusted OOPZ mention extraction for Agent invocations."""

from __future__ import annotations

from typing import Any

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
