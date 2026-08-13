"""OOPZ adapters for strict reference parsing, lookup, and recall."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.admin.models import (
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageReceipt,
    ReferencedMessageCandidate,
)
from cywl_oopz.features.admin.recall import BotMessageRecallTransportError

logger = logging.getLogger(__name__)


class OopzReferencedMessageParser:
    """Project-owned parser for the small stable subset needed by recall."""

    @classmethod
    def parse(cls, value: object) -> ReferencedMessageCandidate | None:
        if value is None or not isinstance(value, Mapping) and not hasattr(value, "message_id"):
            return None
        try:
            area_id = cls._field(value, "area")
            channel_id = cls._field(value, "channel")
            target_person_id = cls._field(value, "target")
            scope = OopzMessageScope.CHANNEL if area_id else OopzMessageScope.PRIVATE
            return ReferencedMessageCandidate(
                message_id=cls._field(value, "message_id", "messageId"),
                message_timestamp=cls._field(value, "timestamp"),
                sender_person_id=cls._field(value, "sender_id", "person"),
                address=OopzMessageAddress(
                    scope,
                    area_id=area_id,
                    channel_id=channel_id,
                    target_person_id=target_person_id if scope is OopzMessageScope.PRIVATE else "",
                ),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _field(value: object, *names: str) -> str:
        for name in names:
            if isinstance(value, Mapping) and name in value:
                raw = value[name]
            elif hasattr(value, name):
                raw = getattr(value, name)
            else:
                continue
            return str(raw or "").strip()
        return ""


class OopzRecentBotMessageLookup:
    """Inspect at most 50 current-channel messages for a legacy receipt."""

    history_size = 50

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def find(
        self,
        message_id: str,
        address: OopzMessageAddress,
    ) -> ReferencedMessageCandidate | None:
        if address.scope is OopzMessageScope.PRIVATE:
            return None
        try:
            messages = await self._bot.messages.get_channel_messages(
                area=address.area_id,
                channel=address.channel_id,
                size=self.history_size,
            )
        except Exception as exc:
            raise BotMessageRecallTransportError(type(exc).__name__) from exc
        for message in messages:
            if str(getattr(message, "message_id", "")) != message_id:
                continue
            return OopzReferencedMessageParser.parse(message)
        return None


class OopzBotMessageRecallGateway:
    """Use the SDK's distinct channel/private recall operations."""

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def recall(self, receipt: OutboundMessageReceipt) -> None:
        address = receipt.address
        try:
            if address.scope is OopzMessageScope.PRIVATE:
                result = await self._bot.messages.recall_private_message(
                    message_id=receipt.message_id,
                    channel=address.channel_id,
                    target=address.target_person_id,
                    timestamp=receipt.message_timestamp or None,
                )
            else:
                result = await self._bot.messages.recall_message(
                    message_id=receipt.message_id,
                    area=address.area_id,
                    channel=address.channel_id,
                    timestamp=receipt.message_timestamp or None,
                )
        except Exception as exc:
            raise BotMessageRecallTransportError(type(exc).__name__) from exc
        if not bool(getattr(result, "ok", False)):
            logger.warning(
                "OOPZ rejected Bot message recall: message=%s scope=%s",
                opaque_ref(receipt.message_id),
                address.scope,
            )
            raise BotMessageRecallTransportError(str(getattr(result, "message", "")))
