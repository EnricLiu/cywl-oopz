"""OOPZ reaction gateway used by the low-risk Agent write tool."""

from __future__ import annotations

import logging
from typing import Any

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.agent.models import AgentIdentity

logger = logging.getLogger(__name__)


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
        logger.info(
            "Adding OOPZ reaction: conversation=%s scope=%s",
            opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id),
            key.scope,
        )
        if key.scope == "private":
            await self._bot.messages.add_private_reaction(
                message_id=message_id,
                channel=identity.transport_channel_id,
                target=identity.person_id,
                emoji=emoji,
            )
            logger.debug("Added OOPZ private reaction")
            return
        await self._bot.messages.add_channel_reaction(
            message_id=message_id,
            area=key.area_id,
            channel=key.channel_id,
            emoji=emoji,
        )
        logger.debug("Added OOPZ channel reaction")
