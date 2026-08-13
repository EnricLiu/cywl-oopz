"""Best-effort receipt tracking around one OOPZ message context."""

from __future__ import annotations

import logging
from typing import Any

from oopz_sdk.models import MessageSendResult

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.admin.models import (
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
)
from cywl_oopz.features.admin.ports import OutboundMessageRepository

logger = logging.getLogger(__name__)


class TrackedMessageContext:
    """Preserve the SDK context API while recording successful reply/send calls."""

    def __init__(self, context: Any, repository: OutboundMessageRepository) -> None:
        self._context = context
        self._repository = repository

    @property
    def event(self) -> Any:
        return self._context.event

    @property
    def bot(self) -> Any:
        return self._context.bot

    @property
    def config(self) -> Any:
        return self._context.config

    async def reply(self, *text: str, **kwargs: Any) -> MessageSendResult:
        result = await self._context.reply(*text, **kwargs)
        await self._track(result, in_reply_to=self._source_message_id())
        return result

    async def send(self, *text: Any, **kwargs: Any) -> MessageSendResult:
        result = await self._context.send(*text, **kwargs)
        await self._track(result, in_reply_to="")
        return result

    async def react(self, emoji: str) -> Any:
        return await self._context.react(emoji)

    async def recall(self, **kwargs: Any) -> Any:
        return await self._context.recall(**kwargs)

    async def _track(self, result: MessageSendResult, *, in_reply_to: str) -> None:
        try:
            message = self.event.message
            private = bool(getattr(self.event, "is_private", False))
            receipt = OutboundMessageReceipt(
                message_id=str(result.message_id),
                message_timestamp=str(result.timestamp),
                kind=OutboundMessageKind.COMMAND_REPLY,
                state=OutboundMessageState.FINAL,
                address=OopzMessageAddress(
                    OopzMessageScope.PRIVATE if private else OopzMessageScope.CHANNEL,
                    area_id="" if private else str(getattr(message, "area", "")),
                    channel_id=str(getattr(message, "channel", "")),
                    target_person_id=(str(getattr(message, "sender_id", "")) if private else ""),
                ),
                in_reply_to_message_id=in_reply_to,
                owner_person_id=str(getattr(message, "sender_id", "")),
            )
            await self._repository.create(receipt)
        except Exception as exc:
            logger.warning(
                "Outbound message tracking degraded: message=%s error=%s",
                opaque_ref(str(getattr(result, "message_id", ""))),
                type(exc).__name__,
            )

    def _source_message_id(self) -> str:
        return str(getattr(self.event.message, "message_id", "")).strip()
