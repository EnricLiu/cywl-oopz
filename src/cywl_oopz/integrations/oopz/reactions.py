"""OOPZ reaction gateway used by the low-risk Agent write tool."""

from __future__ import annotations

from typing import Any

from cywl_oopz.features.agent.models import AgentIdentity


class OopzReactionGateway:
    """Translate trusted project identity into OOPZ SDK reaction calls."""

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def add_reaction(self, identity: AgentIdentity, emoji: str) -> None:
        """React to the invocation message without exposing SDK context to the tool."""
        key = identity.conversation
        message_id = identity.source_message_id
        if not message_id or not identity.transport_channel_id:
            raise ValueError("The Agent invocation has no reaction target")
        if key.scope == "private":
            await self._bot.messages.add_private_reaction(
                message_id=message_id,
                channel=identity.transport_channel_id,
                target=identity.person_id,
                emoji=emoji,
            )
            return
        await self._bot.messages.add_channel_reaction(
            message_id=message_id,
            area=key.area_id,
            channel=key.channel_id,
            emoji=emoji,
        )
