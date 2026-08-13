"""Resolve quoted OOPZ messages into durable Bot-owned receipts."""

from __future__ import annotations

from cywl_oopz.features.admin.models import (
    OopzMessageAddress,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
    ReferencedMessageCandidate,
)
from cywl_oopz.features.admin.ports import OutboundMessageRepository, RecentBotMessageLookup


class ReferencedMessageResolver:
    """Prefer durable ownership evidence, with one strict legacy fallback."""

    def __init__(
        self,
        receipts: OutboundMessageRepository,
        recent_messages: RecentBotMessageLookup,
        bot_person_id: str,
    ) -> None:
        bot_person_id = bot_person_id.strip()
        if not bot_person_id:
            raise ValueError("Bot person ID is required for reference resolution")
        self._receipts = receipts
        self._recent_messages = recent_messages
        self._bot_person_id = bot_person_id

    async def resolve(
        self,
        message_id: str,
        address: OopzMessageAddress,
        embedded: ReferencedMessageCandidate | None,
    ) -> OutboundMessageReceipt | None:
        """Resolve only an exact current-address message proven to belong to this Bot."""
        message_id = message_id.strip()
        if not message_id:
            return None
        receipt = await self._receipts.get_by_message(message_id, address)
        if receipt is not None:
            return receipt

        candidate = embedded if self._matches(embedded, message_id, address) else None
        if candidate is None:
            candidate = await self._recent_messages.find(message_id, address)
        if not self._matches(candidate, message_id, address):
            return None
        assert candidate is not None
        legacy = OutboundMessageReceipt(
            message_id=candidate.message_id,
            message_timestamp=candidate.message_timestamp,
            kind=OutboundMessageKind.COMMAND_REPLY,
            state=OutboundMessageState.FINAL,
            address=candidate.address,
        )
        if await self._receipts.create(legacy):
            return legacy
        return await self._receipts.get_by_message(message_id, address)

    def _matches(
        self,
        candidate: ReferencedMessageCandidate | None,
        message_id: str,
        address: OopzMessageAddress,
    ) -> bool:
        return (
            candidate is not None
            and candidate.message_id == message_id
            and candidate.sender_person_id == self._bot_person_id
            and candidate.address == address
        )
