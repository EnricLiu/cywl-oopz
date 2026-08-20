"""Serialized Bot-owned message recall orchestration."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import opaque_ref

from .models import (
    MessageRecallOutcome,
    OopzMessageAddress,
    OutboundMessageReceipt,
    OutboundMessageState,
    ReferencedMessageCandidate,
)
from .ports import (
    ActivePresentationDismissal,
    BotMessageRecallGateway,
    OutboundConversationCanceller,
    OutboundMessageRepository,
)
from .references import ReferencedMessageResolver

logger = logging.getLogger(__name__)


class ReferencedBotMessageNotFoundError(LookupError):
    """The quoted message could not be proven to belong to this Bot."""


class BotMessageRecallTransportError(RuntimeError):
    """OOPZ did not confirm a requested message recall."""


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class _MessageLockPool:
    """Small ref-counted keyed lock pool without permanent message-ID retention."""

    def __init__(self) -> None:
        self._entries: dict[str, _LockEntry] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, message_id: str):
        async with self._guard:
            entry = self._entries.setdefault(message_id, _LockEntry(asyncio.Lock()))
            entry.users += 1
        await entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            async with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(message_id, None)


class MessageRecallService:
    """Dismiss exact active work, recall, then persist one message transition."""

    cancel_grace_seconds = 2.0

    def __init__(
        self,
        references: ReferencedMessageResolver,
        receipts: OutboundMessageRepository,
        active_presentations: ActivePresentationDismissal,
        conversations: OutboundConversationCanceller,
        gateway: BotMessageRecallGateway,
    ) -> None:
        self._references = references
        self._receipts = receipts
        self._active_presentations = active_presentations
        self._conversations = conversations
        self._gateway = gateway
        self._locks = _MessageLockPool()

    async def recall(
        self,
        message_id: str,
        address: OopzMessageAddress,
        embedded: ReferencedMessageCandidate | None = None,
    ) -> MessageRecallOutcome:
        message_id = message_id.strip()
        async with self._locks.hold(message_id):
            receipt = await self._references.resolve(message_id, address, embedded)
            if receipt is None:
                raise ReferencedBotMessageNotFoundError(message_id)
            if receipt.state is OutboundMessageState.RECALLED:
                return MessageRecallOutcome.ALREADY_RECALLED

            presentation_was_active = await self._active_presentations.dismiss(message_id)
            if presentation_was_active:
                await self._cancel_conversation(receipt)
            else:
                logger.debug(
                    "Recalled message has no active presentation; preserving current "
                    "conversation task: message=%s",
                    opaque_ref(message_id),
                )
            await self._gateway.recall(receipt)
            marked = await self._receipts.mark_recalled(message_id)
            if not marked:
                current = await self._receipts.get_by_message(message_id, address)
                if current is None or current.state is not OutboundMessageState.RECALLED:
                    raise DatabaseError("Recalled message state could not be persisted")
            logger.info(
                "Recalled Bot-owned OOPZ message: message=%s scope=%s",
                opaque_ref(message_id),
                address.scope,
            )
            return MessageRecallOutcome.RECALLED

    async def _cancel_conversation(self, receipt: OutboundMessageReceipt) -> None:
        try:
            async with asyncio.timeout(self.cancel_grace_seconds):
                await self._conversations.cancel_for_message(receipt)
        except TimeoutError:
            logger.warning(
                "Timed out awaiting recalled message task cancellation: message=%s",
                opaque_ref(receipt.message_id),
            )
        except Exception as exc:
            logger.warning(
                "Recalled message task cancellation degraded: message=%s error=%s",
                opaque_ref(receipt.message_id),
                type(exc).__name__,
            )
